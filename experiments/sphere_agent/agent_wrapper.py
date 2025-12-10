"""
Sphere-Guarded Agent Wrapper
============================

Provides wrappers that enforce sphere constraints on arbitrary agents.

Key features:
- Pre-execution validation (reject if would violate sphere)
- Post-execution tracking (log fuel consumption)
- Mode-aware stepping (validate mode constraints)
- Automatic history logging

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Optional, Any, List
from abc import ABC, abstractmethod
import logging

# Handle both package imports and direct execution
try:
    from .sphere import (
        GlobalProfile, SphereConstraint, StepResult, StepType,
        evaluate_step, ExecutionHistory, StepRecord, Mode,
    )
    from .monitors import SphereMonitor, Alert, AlertLevel
except ImportError:
    from sphere import (
        GlobalProfile, SphereConstraint, StepResult, StepType,
        evaluate_step, ExecutionHistory, StepRecord, Mode,
    )
    from monitors import SphereMonitor, Alert, AlertLevel

logger = logging.getLogger(__name__)

# Type variables for generic agent interface
State = TypeVar('State')
Action = TypeVar('Action')
Observation = TypeVar('Observation')


# =============================================================================
# § 1. Abstract Agent Interface
# =============================================================================

class Agent(ABC, Generic[State, Action]):
    """Abstract base class for agents to be wrapped."""
    
    @abstractmethod
    def act(self, state: State) -> Action:
        """Select an action given current state."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return agent identifier."""
        pass


class Environment(ABC, Generic[State, Action]):
    """Abstract base class for environments."""
    
    @abstractmethod
    def step(self, action: Action) -> State:
        """Apply action and return new state."""
        pass
    
    @abstractmethod
    def current_state(self) -> State:
        """Return current state."""
        pass
    
    @abstractmethod
    def reset(self) -> State:
        """Reset environment to initial state."""
        pass


# =============================================================================
# § 2. Profile Extractor
# =============================================================================

class ProfileExtractor(ABC, Generic[State]):
    """Extracts a GlobalProfile from a state."""
    
    @abstractmethod
    def extract(self, state: State) -> GlobalProfile:
        """Extract profile from state."""
        pass
    
    @abstractmethod
    def dimension_names(self) -> List[str]:
        """Return names of profile dimensions."""
        pass


@dataclass
class SimpleProfileExtractor(ProfileExtractor[dict]):
    """
    Simple extractor for dict-based states.
    
    Expects state to have keys matching dimension names.
    """
    names: List[str]
    defaults: Optional[List[int]] = None
    
    def extract(self, state: dict) -> GlobalProfile:
        if self.defaults is None:
            defaults = [0] * len(self.names)
        else:
            defaults = self.defaults
        
        values = [state.get(name, default) for name, default in zip(self.names, defaults)]
        return GlobalProfile(values, self.names.copy())
    
    def dimension_names(self) -> List[str]:
        return self.names.copy()


# =============================================================================
# § 3. Sphere-Guarded Agent
# =============================================================================

class VetoError(Exception):
    """Raised when an action is vetoed by the sphere constraint."""
    
    def __init__(self, message: str, action: Any, projected_result: StepResult):
        super().__init__(message)
        self.action = action
        self.projected_result = projected_result


class SphereViolationError(Exception):
    """Raised when sphere constraint is violated (should never happen with proper guards)."""
    
    def __init__(self, message: str, profile: GlobalProfile, constraint: SphereConstraint):
        super().__init__(message)
        self.profile = profile
        self.constraint = constraint


@dataclass
class GuardedStepResult:
    """Result of a guarded step execution."""
    new_state: Any
    action: Any
    step_result: StepResult
    alerts: List[Alert]
    vetoed: bool = False
    veto_reason: Optional[str] = None


class SphereGuardedAgent(Generic[State, Action]):
    """
    Wrapper that enforces sphere constraints on an agent.
    
    Guarantees:
    - No action can increase any profile coordinate (WeakMono)
    - Total fuel consumption ≤ R (sphere constraint)
    - Strict steps are counted and bounded
    
    From Lean theorem max_trajectory_length:
    The number of strict steps is guaranteed ≤ initial fuel ≤ R.
    """
    
    def __init__(
        self,
        agent: Agent[State, Action],
        env: Environment[State, Action],
        extractor: ProfileExtractor[State],
        constraint: SphereConstraint,
        monitor: Optional[SphereMonitor] = None,
        simulate_action: Optional[Callable[[State, Action], State]] = None,
        veto_on_violation: bool = True,
        veto_on_critical: bool = False,
    ):
        """
        Initialize the guarded agent.
        
        Args:
            agent: The agent to wrap
            env: The environment
            extractor: Extracts profiles from states
            constraint: The sphere constraint to enforce
            monitor: Optional monitor for alerts (created if not provided)
            simulate_action: Optional function to simulate action effects for pre-checking
            veto_on_violation: If True, veto actions that violate WeakMono
            veto_on_critical: If True, veto actions when fuel is critical
        """
        self.agent = agent
        self.env = env
        self.extractor = extractor
        self.constraint = constraint
        self.monitor = monitor or SphereMonitor(constraint)
        self.simulate_action = simulate_action
        self.veto_on_violation = veto_on_violation
        self.veto_on_critical = veto_on_critical
        
        # State tracking
        self.history = ExecutionHistory(constraint)
        self.current_profile: Optional[GlobalProfile] = None
        self.step_count = 0
        self.current_mode: Optional[Mode] = None
    
    def reset(self) -> State:
        """Reset the environment and tracking."""
        state = self.env.reset()
        self.current_profile = self.extractor.extract(state)
        self.history = ExecutionHistory(self.constraint)
        self.step_count = 0
        self.monitor.reset()
        
        # Validate initial state is in sphere
        if not self.constraint.validate(self.current_profile):
            raise SphereViolationError(
                f"Initial state outside sphere: fuel={self.current_profile.fuel} > R={self.constraint.R}",
                self.current_profile,
                self.constraint,
            )
        
        logger.info(f"Reset: {self.current_profile}")
        return state
    
    def set_mode(self, mode: Mode):
        """Set the current operational mode."""
        self.current_mode = mode
        logger.debug(f"Mode set to: {mode.name}")
    
    def step(self, state: Optional[State] = None) -> GuardedStepResult:
        """
        Execute one guarded step.
        
        1. Get action from agent
        2. Optionally pre-validate via simulation
        3. Execute action
        4. Evaluate step type
        5. Check constraints, generate alerts
        6. Log to history
        
        Returns:
            GuardedStepResult with new state, action, step result, and alerts
        
        Raises:
            VetoError: if action is vetoed and veto_on_* is True
            SphereViolationError: if sphere constraint is violated
        """
        if state is None:
            state = self.env.current_state()
        
        profile_before = self.extractor.extract(state)
        
        # Get action from agent
        action = self.agent.act(state)
        
        # Pre-validation via simulation (if available)
        if self.simulate_action is not None:
            simulated_state = self.simulate_action(state, action)
            simulated_profile = self.extractor.extract(simulated_state)
            simulated_result = evaluate_step(profile_before, simulated_profile)
            
            # Check for veto conditions
            veto, reason = self._check_veto(profile_before, simulated_result)
            if veto:
                return GuardedStepResult(
                    new_state=state,
                    action=action,
                    step_result=simulated_result,
                    alerts=self.monitor.check(simulated_result, simulated_profile),
                    vetoed=True,
                    veto_reason=reason,
                )
        
        # Execute action
        new_state = self.env.step(action)
        profile_after = self.extractor.extract(new_state)
        
        # Evaluate step
        step_result = evaluate_step(profile_before, profile_after)
        
        # Post-validation
        veto, reason = self._check_veto(profile_before, step_result)
        if veto and self.veto_on_violation:
            raise VetoError(reason, action, step_result)
        
        # Validate sphere constraint
        if not self.constraint.validate(profile_after):
            raise SphereViolationError(
                f"Action caused sphere violation: fuel={profile_after.fuel} > R={self.constraint.R}",
                profile_after,
                self.constraint,
            )
        
        # Generate alerts
        alerts = self.monitor.check(step_result, profile_after)
        
        # Log to history
        record = StepRecord(
            step_number=self.step_count,
            profile_before=profile_before,
            profile_after=profile_after,
            result=step_result,
            mode=self.current_mode.name if self.current_mode else None,
            action=str(action),
        )
        self.history.add(record)
        
        # Update state
        self.current_profile = profile_after
        self.step_count += 1
        
        # Log
        if step_result.is_strict:
            logger.info(f"Step {self.step_count}: STRICT, consumed {step_result.fuel_consumed} fuel, remaining={profile_after.fuel}")
        else:
            logger.debug(f"Step {self.step_count}: {step_result.step_type.name}")
        
        for alert in alerts:
            logger.warning(f"Alert: {alert}")
        
        return GuardedStepResult(
            new_state=new_state,
            action=action,
            step_result=step_result,
            alerts=alerts,
        )
    
    def _check_veto(self, profile_before: GlobalProfile, result: StepResult) -> tuple[bool, Optional[str]]:
        """Check if action should be vetoed."""
        # Violation veto
        if result.is_violation and self.veto_on_violation:
            return True, f"WeakMono violated: coords {result.increased_coords} increased"
        
        # Critical fuel veto
        if self.veto_on_critical and result.fuel_after < self.constraint.R * self.constraint.critical_threshold:
            return True, f"Would enter critical fuel zone: {result.fuel_after}"
        
        # Mode validation
        if self.current_mode is not None and result.is_strict:
            if not self.current_mode.validate_step(result):
                return True, f"Mode {self.current_mode.name} forbids change to coords {result.decreased_coords}"
        
        return False, None
    
    def remaining_budget(self) -> int:
        """
        Remaining strict step budget.
        
        By max_trajectory_length theorem: remaining strict steps ≤ current fuel.
        """
        if self.current_profile is None:
            return self.constraint.R
        return self.current_profile.fuel
    
    def summary(self) -> dict:
        """Get execution summary."""
        result = self.history.summary()
        result['remaining_budget'] = self.remaining_budget()
        result['alerts'] = self.monitor.alert_count()
        return result
    
    def run(self, max_steps: int = 1000) -> List[GuardedStepResult]:
        """
        Run the agent for up to max_steps.
        
        Stops early if:
        - Agent signals done
        - Fuel reaches 0
        - Veto occurs
        """
        results = []
        state = self.reset()
        
        for _ in range(max_steps):
            try:
                result = self.step(state)
                results.append(result)
                
                if result.vetoed:
                    logger.info(f"Execution stopped: action vetoed - {result.veto_reason}")
                    break
                
                state = result.new_state
                
                # Check if in valley (fuel exhausted)
                if self.current_profile.fuel == 0:
                    logger.info("Execution complete: reached minimal valley (fuel=0)")
                    break
                
            except VetoError as e:
                logger.warning(f"Execution stopped: veto - {e}")
                break
            except SphereViolationError as e:
                logger.error(f"Execution stopped: sphere violation - {e}")
                break
        
        return results


# =============================================================================
# § 4. Convenient Factory
# =============================================================================

def create_guarded_agent(
    agent_fn: Callable[[Any], Any],
    step_fn: Callable[[Any, Any], Any],
    profile_fn: Callable[[Any], GlobalProfile],
    R: int,
    D: int,
    names: List[str],
    initial_state: Any,
) -> SphereGuardedAgent:
    """
    Quick factory for creating a guarded agent from functions.
    
    Args:
        agent_fn: function(state) -> action
        step_fn: function(state, action) -> new_state
        profile_fn: function(state) -> GlobalProfile
        R: sphere radius
        D: dimensions
        names: coordinate names
        initial_state: starting state
    
    Returns:
        Configured SphereGuardedAgent
    """
    
    class FnAgent(Agent):
        def act(self, state):
            return agent_fn(state)
        def name(self):
            return "FunctionAgent"
    
    class FnEnv(Environment):
        def __init__(self):
            self._state = initial_state
        def step(self, action):
            self._state = step_fn(self._state, action)
            return self._state
        def current_state(self):
            return self._state
        def reset(self):
            self._state = initial_state
            return self._state
    
    class FnExtractor(ProfileExtractor):
        def extract(self, state):
            return profile_fn(state)
        def dimension_names(self):
            return names
    
    constraint = SphereConstraint(R=R, D=D, names=names)
    
    return SphereGuardedAgent(
        agent=FnAgent(),
        env=FnEnv(),
        extractor=FnExtractor(),
        constraint=constraint,
    )

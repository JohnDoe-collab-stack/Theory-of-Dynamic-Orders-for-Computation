"""
Sphere Framework for AI Safety
==============================

Core abstractions implementing the Global Sphere framework from Sphere.lean.

This module provides:
- GlobalProfile: D-dimensional resource vector
- StrictStep evaluation: detecting fuel-consuming transitions
- Mode: local transition types with active coordinates
- Valley: absorbing stable regions

Mathematical guarantees (from Lean formalization):
- max_trajectory_length: chains of strict steps have length ≤ R
- zero_fuel_stable: states with fuel=0 are stable
- fuel_lyapunov: fuel strictly decreases on strict steps

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Optional, Callable, TypeVar, Generic
from enum import Enum, auto
import numpy as np

# =============================================================================
# § 1. Global Profile
# =============================================================================

@dataclass
class GlobalProfile:
    """
    A D-dimensional profile vector representing state resources.
    
    Corresponds to `GlobalProfile D := Fin D → Nat` in Lean.
    
    Attributes:
        values: numpy array of shape (D,) with non-negative integers
        names: optional names for each coordinate
    """
    values: np.ndarray
    names: Optional[List[str]] = None
    
    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.int64)
        if np.any(self.values < 0):
            raise ValueError("Profile values must be non-negative")
        if self.names is None:
            self.names = [f"L{i}" for i in range(len(self.values))]
    
    @property
    def D(self) -> int:
        """Number of dimensions."""
        return len(self.values)
    
    @property
    def fuel(self) -> int:
        """
        Total fuel = sum of all components (ℓ₁-norm).
        
        Corresponds to `Fuel L x := (L x).sum` in Lean.
        """
        return int(self.values.sum())
    
    def in_sphere(self, R: int) -> bool:
        """
        Check if profile is in the global sphere B_R.
        
        Corresponds to `InSphere R v := v.sum ≤ R` in Lean.
        """
        return self.fuel <= R
    
    def __getitem__(self, i: int) -> int:
        return int(self.values[i])
    
    def __repr__(self) -> str:
        parts = [f"{n}={v}" for n, v in zip(self.names, self.values)]
        return f"Profile({', '.join(parts)}, fuel={self.fuel})"
    
    @classmethod
    def zero(cls, D: int, names: Optional[List[str]] = None) -> GlobalProfile:
        """Create zero profile (all coordinates = 0)."""
        return cls(np.zeros(D, dtype=np.int64), names)
    
    @classmethod
    def uniform(cls, D: int, value: int, names: Optional[List[str]] = None) -> GlobalProfile:
        """Create uniform profile (all coordinates = value)."""
        return cls(np.full(D, value, dtype=np.int64), names)


# =============================================================================
# § 2. Step Evaluation
# =============================================================================

class StepType(Enum):
    """Classification of a transition step."""
    STRICT = auto()      # At least one coord decreased, none increased
    PLATEAU = auto()     # No change
    VIOLATION = auto()   # Some coord increased (WeakMono violated)


@dataclass
class StepResult:
    """
    Result of evaluating a state transition.
    
    Corresponds to `StrictStep Step L x y` in Lean.
    """
    step_type: StepType
    fuel_before: int
    fuel_after: int
    decreased_coords: List[int]    # indices where L decreased
    increased_coords: List[int]    # indices where L increased (violation)
    delta: np.ndarray              # L_before - L_after
    
    @property
    def is_strict(self) -> bool:
        """True if this was a strict step (consumed fuel)."""
        return self.step_type == StepType.STRICT
    
    @property
    def is_plateau(self) -> bool:
        """True if this was a plateau step (no fuel change)."""
        return self.step_type == StepType.PLATEAU
    
    @property
    def is_violation(self) -> bool:
        """True if WeakMono was violated."""
        return self.step_type == StepType.VIOLATION
    
    @property
    def fuel_consumed(self) -> int:
        """Amount of fuel consumed (positive for strict steps)."""
        return max(0, self.fuel_before - self.fuel_after)


def evaluate_step(L_before: GlobalProfile, L_after: GlobalProfile) -> StepResult:
    """
    Evaluate a transition from L_before to L_after.
    
    Implements the classification:
    - STRICT: WeakMono holds and ∃i. L_after[i] < L_before[i]
    - PLATEAU: L_after = L_before
    - VIOLATION: ∃i. L_after[i] > L_before[i] (WeakMono violated)
    
    Args:
        L_before: profile before transition
        L_after: profile after transition
    
    Returns:
        StepResult with classification and details
    """
    if L_before.D != L_after.D:
        raise ValueError(f"Dimension mismatch: {L_before.D} vs {L_after.D}")
    
    delta = L_before.values - L_after.values
    decreased = np.where(delta > 0)[0].tolist()
    increased = np.where(delta < 0)[0].tolist()
    
    if increased:
        step_type = StepType.VIOLATION
    elif decreased:
        step_type = StepType.STRICT
    else:
        step_type = StepType.PLATEAU
    
    return StepResult(
        step_type=step_type,
        fuel_before=L_before.fuel,
        fuel_after=L_after.fuel,
        decreased_coords=decreased,
        increased_coords=increased,
        delta=delta,
    )


# =============================================================================
# § 3. Modes
# =============================================================================

@dataclass
class Mode:
    """
    A mode defines a type of transition with designated active coordinates.
    
    Corresponds to `Mode State D` in Lean:
    - Label: the mode identifier
    - activeCoords: which coordinates this mode affects
    
    In a ModeMono step, only activeCoords can decrease.
    """
    name: str
    active_coords: Set[int]
    description: str = ""
    
    def is_active(self, coord: int) -> bool:
        """Check if a coordinate is active in this mode."""
        return coord in self.active_coords
    
    def validate_step(self, result: StepResult) -> bool:
        """
        Check if a step result is valid for this mode.
        
        A valid mode step only decreases active coordinates.
        """
        for i in result.decreased_coords:
            if i not in self.active_coords:
                return False  # Decreased a non-active coord
        return True


def disjoint_modes(m1: Mode, m2: Mode) -> bool:
    """
    Check if two modes have disjoint active coordinates.
    
    Corresponds to `DisjointModes M₁ M₂` in Lean.
    """
    return len(m1.active_coords & m2.active_coords) == 0


# =============================================================================
# § 4. Stability and Valleys
# =============================================================================

@dataclass
class Valley:
    """
    A valley is an absorbing set with internal stability.
    
    Corresponds to `Valley Step L V` in Lean:
    - absorb: x ∈ V → Step x y → y ∈ V
    - stable: x ∈ V → Stable x (no strict successors)
    
    In practice: a valley is a "safe zone" where the system
    can't do any more strict steps.
    """
    name: str
    fuel_threshold: int  # fuel ≤ threshold → in valley
    
    def contains(self, profile: GlobalProfile) -> bool:
        """Check if a profile is in the valley."""
        return profile.fuel <= self.fuel_threshold
    
    @classmethod
    def minimal(cls) -> Valley:
        """The minimal valley: fuel = 0."""
        return cls("minimal", 0)


# =============================================================================
# § 5. Sphere Constraint
# =============================================================================

@dataclass
class SphereConstraint:
    """
    A complete sphere constraint configuration.
    
    Combines:
    - R: maximum fuel (sphere radius)
    - D: number of dimensions
    - names: coordinate names
    - modes: optional mode restrictions
    - critical_threshold: fuel level triggering alerts
    """
    R: int
    D: int
    names: List[str]
    modes: List[Mode] = field(default_factory=list)
    critical_threshold: float = 0.1  # fraction of R
    
    def __post_init__(self):
        if len(self.names) != self.D:
            raise ValueError(f"Expected {self.D} names, got {len(self.names)}")
    
    def initial_profile(self, values: Optional[List[int]] = None) -> GlobalProfile:
        """Create an initial profile, optionally with custom values."""
        if values is None:
            # Distribute R evenly across coordinates
            base = self.R // self.D
            remainder = self.R % self.D
            vals = [base + (1 if i < remainder else 0) for i in range(self.D)]
            return GlobalProfile(np.array(vals), self.names.copy())
        return GlobalProfile(np.array(values), self.names.copy())
    
    def validate(self, profile: GlobalProfile) -> bool:
        """Check if profile satisfies the sphere constraint."""
        return profile.in_sphere(self.R)
    
    def max_strict_steps(self, profile: GlobalProfile) -> int:
        """
        Maximum number of strict steps possible from this profile.
        
        By theorem max_trajectory_length: len ≤ fuel ≤ R
        """
        return profile.fuel
    
    def is_critical(self, profile: GlobalProfile) -> bool:
        """Check if fuel is below critical threshold."""
        return profile.fuel < self.R * self.critical_threshold
    
    @property
    def valley(self) -> Valley:
        """The minimal valley for this sphere."""
        return Valley.minimal()


# =============================================================================
# § 6. History Tracking
# =============================================================================

@dataclass
class StepRecord:
    """Record of a single step in execution history."""
    step_number: int
    profile_before: GlobalProfile
    profile_after: GlobalProfile
    result: StepResult
    mode: Optional[str] = None
    action: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'step': self.step_number,
            'fuel_before': self.result.fuel_before,
            'fuel_after': self.result.fuel_after,
            'type': self.result.step_type.name,
            'consumed': self.result.fuel_consumed,
            'decreased': self.result.decreased_coords,
            'mode': self.mode,
            'action': self.action,
        }


class ExecutionHistory:
    """
    Tracks execution history for analysis and auditing.
    """
    
    def __init__(self, constraint: SphereConstraint):
        self.constraint = constraint
        self.records: List[StepRecord] = []
    
    def add(self, record: StepRecord):
        self.records.append(record)
    
    @property
    def total_steps(self) -> int:
        return len(self.records)
    
    @property
    def strict_steps(self) -> int:
        return sum(1 for r in self.records if r.result.is_strict)
    
    @property
    def plateau_steps(self) -> int:
        return sum(1 for r in self.records if r.result.is_plateau)
    
    @property
    def violation_count(self) -> int:
        return sum(1 for r in self.records if r.result.is_violation)
    
    @property
    def total_fuel_consumed(self) -> int:
        return sum(r.result.fuel_consumed for r in self.records)
    
    def summary(self) -> dict:
        return {
            'total_steps': self.total_steps,
            'strict_steps': self.strict_steps,
            'plateau_steps': self.plateau_steps,
            'violations': self.violation_count,
            'fuel_consumed': self.total_fuel_consumed,
            'max_budget': self.constraint.R,
            'budget_ratio': self.total_fuel_consumed / self.constraint.R if self.constraint.R > 0 else 0,
        }
    
    def fuel_trace(self) -> List[int]:
        """Get fuel values over time."""
        if not self.records:
            return []
        trace = [self.records[0].result.fuel_before]
        for r in self.records:
            trace.append(r.result.fuel_after)
        return trace


# =============================================================================
# § 7. Convenience Functions
# =============================================================================

def create_safety_constraint(
    risk_budget: int = 10,
    deviation_budget: int = 5,
    env_modification_budget: int = 5,
) -> SphereConstraint:
    """
    Create a standard safety constraint for agent control.
    
    Dimensions:
    - L0: risk_budget - budget for risky actions
    - L1: deviation_budget - allowed deviation from reference policy
    - L2: env_modification_budget - budget for environment changes
    """
    names = ["risk", "deviation", "env_mod"]
    R = risk_budget + deviation_budget + env_modification_budget
    
    # Define modes
    modes = [
        Mode("risk_action", {0}, "Actions that consume risk budget"),
        Mode("policy_change", {1}, "Actions that deviate from reference"),
        Mode("env_change", {2}, "Actions that modify environment"),
        Mode("combined", {0, 1, 2}, "Actions affecting multiple dimensions"),
    ]
    
    return SphereConstraint(R=R, D=3, names=names, modes=modes)


def create_training_constraint(
    distribution_shift_budget: int = 5,
    regularization_debt: int = 10,
    safety_constraint_budget: int = 5,
) -> SphereConstraint:
    """
    Create a constraint for training/fine-tuning processes.
    
    Dimensions:
    - L0: distribution_shift - number of major distribution changes
    - L1: regularization_debt - accumulated regularization violations
    - L2: safety_constraints - safety margins consumed
    """
    names = ["dist_shift", "reg_debt", "safety_margin"]
    R = distribution_shift_budget + regularization_debt + safety_constraint_budget
    
    return SphereConstraint(R=R, D=3, names=names)

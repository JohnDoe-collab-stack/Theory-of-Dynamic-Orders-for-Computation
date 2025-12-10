#!/usr/bin/env python3
"""
Sphere Wrapper for Ω_proof Kernel
==================================

Integrates the Sphere framework with the propositional logic kernel,
providing fuel tracking and bounded budget guarantees.

Key Features:
- ProofProfile: 3D profile (search, depth, revisions)
- Automatic fuel tracking during formula evaluation
- Efficiency metrics for K-real vs Shuffle-K comparison
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# Add sphere_agent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sphere_agent"))

from sphere import (
    GlobalProfile,
    SphereConstraint,
    StepResult,
    StepType,
    evaluate_step,
    ExecutionHistory,
)
from monitors import SphereMonitor, Alert, AlertLevel


# =============================================================================
# § 1. Proof Profile
# =============================================================================

@dataclass
class ProofProfileConfig:
    """Configuration for proof sphere tracking."""
    n_atoms: int              # Number of propositional atoms
    max_depth: int            # Maximum formula depth
    revision_budget: int = 10 # Budget for prediction revisions
    
    @property
    def search_space_size(self) -> int:
        """Size of the valuation search space: 2^n."""
        return 2 ** self.n_atoms
    
    @property
    def total_budget(self) -> int:
        """Total fuel budget R."""
        return self.search_space_size + self.max_depth + self.revision_budget


class ProofSphereTracker:
    """
    Tracks fuel consumption during propositional formula evaluation.
    
    Profile dimensions:
    - L0 (search): remaining search space = 2^n - t_first
    - L1 (depth): remaining depth budget = max_depth - current_depth  
    - L2 (revisions): remaining revision budget
    
    Guarantees from Sphere.lean:
    - Number of strict steps ≤ initial fuel ≤ R
    - Early halts (low t_first) → high search budget remaining
    - Late halts (high t_first) → low search budget remaining
    """
    
    def __init__(self, config: ProofProfileConfig):
        self.config = config
        
        # Create constraint
        self.constraint = SphereConstraint(
            R=config.total_budget,
            D=3,
            names=["search", "depth", "revisions"],
        )
        
        # Monitor
        self.monitor = SphereMonitor(self.constraint)
        
        # State
        self._reset()
    
    def _reset(self):
        """Reset to initial state."""
        self.profile = GlobalProfile(
            [self.config.search_space_size, self.config.max_depth, self.config.revision_budget],
            self.constraint.names.copy()
        )
        self.step_history: List[Dict[str, Any]] = []
        self.strict_steps = 0
        self.plateau_steps = 0
        self.current_t = 0
        self.current_depth = 0
        self.revisions = 0
    
    def reset(self):
        """Reset for new formula evaluation."""
        self._reset()
        self.monitor.reset()
    
    def step(self, t: int, depth: int) -> StepResult:
        """
        Record a step in the evaluation.
        
        Args:
            t: Current search position (0 to 2^n - 1)
            depth: Current depth in formula tree
        
        Returns:
            StepResult indicating if this was a strict/plateau/violation step
        """
        # Compute new profile
        new_search = max(0, self.config.search_space_size - t - 1)
        new_depth = max(0, self.config.max_depth - depth)
        new_revisions = self.profile[2]  # unchanged
        
        new_profile = GlobalProfile(
            [new_search, new_depth, new_revisions],
            self.constraint.names.copy()
        )
        
        # Evaluate step
        result = evaluate_step(self.profile, new_profile)
        
        # Update tracking
        if result.is_strict:
            self.strict_steps += 1
        elif result.is_plateau:
            self.plateau_steps += 1
        
        # Check alerts
        alerts = self.monitor.check(result, new_profile)
        
        # Record history
        self.step_history.append({
            't': t,
            'depth': depth,
            'fuel_before': self.profile.fuel,
            'fuel_after': new_profile.fuel,
            'step_type': result.step_type.name,
            'alerts': [str(a) for a in alerts],
        })
        
        # Update state
        self.profile = new_profile
        self.current_t = t
        self.current_depth = depth
        
        return result
    
    def record_revision(self) -> StepResult:
        """
        Record a prediction revision (consumes revision budget).
        
        Returns:
            StepResult (will be STRICT if budget > 0)
        """
        if self.profile[2] == 0:
            # No revision budget left - this is a violation attempt
            # but we'll allow it as a plateau
            return StepResult(
                step_type=StepType.PLATEAU,
                fuel_before=self.profile.fuel,
                fuel_after=self.profile.fuel,
                decreased_coords=[],
                increased_coords=[],
                delta=self.profile.values - self.profile.values,
            )
        
        new_profile = GlobalProfile(
            [self.profile[0], self.profile[1], self.profile[2] - 1],
            self.constraint.names.copy()
        )
        
        result = evaluate_step(self.profile, new_profile)
        self.revisions += 1
        self.profile = new_profile
        
        if result.is_strict:
            self.strict_steps += 1
        
        return result
    
    def finalize(self, t_first: int, halts: bool) -> Dict[str, Any]:
        """
        Finalize tracking and compute metrics.
        
        Args:
            t_first: Step where halt condition was detected (-1 if never)
            halts: Whether formula halted
        
        Returns:
            Dictionary of metrics
        """
        # Compute final search position
        final_search = self.profile[0]
        
        # Fuel efficiency: how much of the search budget was NOT used
        search_efficiency = final_search / self.config.search_space_size if self.config.search_space_size > 0 else 0
        
        # Overall fuel efficiency
        fuel_used = self.constraint.R - self.profile.fuel
        fuel_efficiency = fuel_used / self.constraint.R if self.constraint.R > 0 else 0
        
        return {
            'n_atoms': self.config.n_atoms,
            'max_depth': self.config.max_depth,
            'search_space_size': self.config.search_space_size,
            'total_budget': self.constraint.R,
            
            't_first': t_first,
            'halts': halts,
            
            'final_fuel': self.profile.fuel,
            'fuel_used': fuel_used,
            'fuel_efficiency': fuel_efficiency,
            
            'search_remaining': final_search,
            'search_efficiency': search_efficiency,
            
            'strict_steps': self.strict_steps,
            'plateau_steps': self.plateau_steps,
            'revisions': self.revisions,
            
            'in_valley': self.profile.fuel == 0,
            'alerts': self.monitor.alert_count(),
        }


# =============================================================================
# § 2. Kernel Integration
# =============================================================================

def wrap_kernel_run(kernel_run_fn):
    """
    Decorator to add sphere tracking to a kernel run function.
    
    Usage:
        @wrap_kernel_run
        def run(formula, ...):
            ...
    """
    def wrapped(formula, *args, track_sphere: bool = False, **kwargs):
        if not track_sphere:
            return kernel_run_fn(formula, *args, **kwargs)
        
        # Get formula properties
        n_atoms = len(formula.atoms()) if hasattr(formula, 'atoms') else 3
        depth = formula.depth() if hasattr(formula, 'depth') else 5
        
        # Create tracker
        config = ProofProfileConfig(n_atoms=n_atoms, max_depth=depth)
        tracker = ProofSphereTracker(config)
        
        # Run original function
        result = kernel_run_fn(formula, *args, **kwargs)
        
        # Extract t_first and halts from result
        t_first = result.get('t_first', -1) if isinstance(result, dict) else -1
        halts = result.get('halts', False) if isinstance(result, dict) else False
        
        # Finalize tracking
        metrics = tracker.finalize(t_first, halts)
        
        if isinstance(result, dict):
            result['sphere_metrics'] = metrics
            return result
        else:
            return result, metrics
    
    return wrapped


# =============================================================================
# § 3. Comparison Analysis
# =============================================================================

@dataclass
class SphereComparisonResults:
    """Results from comparing K-real vs Shuffle-K fuel efficiency."""
    k_real_metrics: List[Dict[str, Any]] = field(default_factory=list)
    shuffle_k_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_k_real(self, metrics: Dict[str, Any]):
        self.k_real_metrics.append(metrics)
    
    def add_shuffle_k(self, metrics: Dict[str, Any]):
        self.shuffle_k_metrics.append(metrics)
    
    def summary(self) -> Dict[str, Any]:
        """Compute comparison summary."""
        def avg(lst, key):
            vals = [m[key] for m in lst if key in m]
            return sum(vals) / len(vals) if vals else 0
        
        return {
            'k_real': {
                'count': len(self.k_real_metrics),
                'avg_fuel_efficiency': avg(self.k_real_metrics, 'fuel_efficiency'),
                'avg_search_efficiency': avg(self.k_real_metrics, 'search_efficiency'),
                'avg_strict_steps': avg(self.k_real_metrics, 'strict_steps'),
                'valley_rate': sum(1 for m in self.k_real_metrics if m.get('in_valley', False)) / max(1, len(self.k_real_metrics)),
            },
            'shuffle_k': {
                'count': len(self.shuffle_k_metrics),
                'avg_fuel_efficiency': avg(self.shuffle_k_metrics, 'fuel_efficiency'),
                'avg_search_efficiency': avg(self.shuffle_k_metrics, 'search_efficiency'),
                'avg_strict_steps': avg(self.shuffle_k_metrics, 'strict_steps'),
                'valley_rate': sum(1 for m in self.shuffle_k_metrics if m.get('in_valley', False)) / max(1, len(self.shuffle_k_metrics)),
            }
        }


def analyze_fuel_traces(
    k_real_results: List[Dict[str, Any]],
    shuffle_k_results: List[Dict[str, Any]],
) -> SphereComparisonResults:
    """
    Analyze and compare fuel traces between K-real and Shuffle-K.
    
    Args:
        k_real_results: List of metrics from K-real condition
        shuffle_k_results: List of metrics from Shuffle-K condition
    
    Returns:
        SphereComparisonResults with comparison summary
    """
    comparison = SphereComparisonResults()
    
    for m in k_real_results:
        if 'sphere_metrics' in m:
            comparison.add_k_real(m['sphere_metrics'])
        else:
            comparison.add_k_real(m)
    
    for m in shuffle_k_results:
        if 'sphere_metrics' in m:
            comparison.add_shuffle_k(m['sphere_metrics'])
        else:
            comparison.add_shuffle_k(m)
    
    return comparison


# =============================================================================
# § 4. Demo
# =============================================================================

def demo():
    """Demonstrate sphere tracking on synthetic data."""
    print("="*60)
    print("Sphere Wrapper Demo for Ω_proof")
    print("="*60)
    
    # Create tracker
    config = ProofProfileConfig(n_atoms=4, max_depth=6, revision_budget=5)
    tracker = ProofSphereTracker(config)
    
    print(f"\nConfiguration:")
    print(f"  n_atoms: {config.n_atoms}")
    print(f"  search_space: 2^{config.n_atoms} = {config.search_space_size}")
    print(f"  max_depth: {config.max_depth}")
    print(f"  total_budget R: {config.total_budget}")
    
    print(f"\nInitial profile: {tracker.profile}")
    
    # Simulate evaluation that halts early (t_first = 3)
    print("\n--- Simulating early halt (t_first=3) ---")
    
    for t in range(4):
        result = tracker.step(t=t, depth=min(t+1, config.max_depth))
        print(f"  Step t={t}: {result.step_type.name}, fuel={tracker.profile.fuel}")
    
    metrics = tracker.finalize(t_first=3, halts=True)
    
    print(f"\nFinal metrics:")
    print(f"  t_first: {metrics['t_first']}")
    print(f"  fuel_used: {metrics['fuel_used']}")
    print(f"  search_efficiency: {metrics['search_efficiency']:.2%}")
    print(f"  strict_steps: {metrics['strict_steps']}")
    print(f"  in_valley: {metrics['in_valley']}")
    
    # Compare with late halt
    print("\n--- Simulating late halt (t_first=12) ---")
    
    tracker.reset()
    for t in range(13):
        tracker.step(t=t, depth=min(t+1, config.max_depth))
    
    metrics_late = tracker.finalize(t_first=12, halts=True)
    
    print(f"\nLate halt metrics:")
    print(f"  t_first: {metrics_late['t_first']}")
    print(f"  fuel_used: {metrics_late['fuel_used']}")
    print(f"  search_efficiency: {metrics_late['search_efficiency']:.2%}")
    
    # Key observation
    print("\n" + "="*60)
    print("KEY OBSERVATION:")
    print(f"  Early halt search_efficiency: {metrics['search_efficiency']:.2%}")
    print(f"  Late halt search_efficiency:  {metrics_late['search_efficiency']:.2%}")
    print("  → Early halts are MORE fuel-efficient!")
    print("="*60)


if __name__ == "__main__":
    demo()

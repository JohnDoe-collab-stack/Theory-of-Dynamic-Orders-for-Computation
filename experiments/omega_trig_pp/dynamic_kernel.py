"""
Dynamic Kernel for Ω_trig++

Implements:
- Interval refinement over time (approx_state)
- 3-valued formula evaluation at each time step
- t_first^K computation (first time formula is decided)
- halt_rank classification
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from trig_pp_kernel import (
    Angle, Interval, TVal, Formula, N_ANGLES, N_FORMULAS,
    get_formula, get_formula_desc, FORMULAS
)


# ============================================================================
# Halt Rank Classification
# ============================================================================

class HaltRank(Enum):
    """Difficulty classification based on t_first^K."""
    EARLY = 0   # t_first <= 2
    MID = 1     # 3 <= t_first <= 5
    LATE = 2    # 6 <= t_first <= T_MAX
    NEVER = 3   # Not decided within T_MAX


def halt_rank_of_tfirst(t_first: Optional[int], T_max: int = 10) -> HaltRank:
    """Classify t_first into halt rank."""
    if t_first is None or t_first > T_max:
        return HaltRank.NEVER
    if t_first <= 2:
        return HaltRank.EARLY
    if t_first <= 5:
        return HaltRank.MID
    return HaltRank.LATE


# ============================================================================
# Dynamic Kernel
# ============================================================================

class DynamicTrigPPKernel:
    """
    Dynamic kernel for Ω_trig++.
    
    Refines interval approximations over time and evaluates
    logical formulas using 3-valued logic.
    """
    
    def __init__(self, T_max: int = 10):
        self.T_max = T_max
    
    def approx_state(self, angle: Angle, t: int) -> Tuple[Interval, Interval]:
        """
        Get interval approximations for sin and cos at time t.
        
        At t=0: wide intervals
        As t increases: intervals shrink towards true values
        """
        # Width decreases as 2/(t+1)
        width = 2.0 / (t + 1)
        half_width = width / 2
        
        true_sin = angle.sin()
        true_cos = angle.cos()
        
        # Intervals centered on true value (simulating refinement)
        # At t=0: width=2, so interval = [true-1, true+1]
        # At t=9: width≈0.2, tight around true value
        sin_I = Interval(
            max(-1.0, true_sin - half_width),
            min(1.0, true_sin + half_width)
        )
        cos_I = Interval(
            max(-1.0, true_cos - half_width),
            min(1.0, true_cos + half_width)
        )
        
        return sin_I, cos_I
    
    def eval_formula_at_t(self, formula: Formula, angle: Angle, t: int) -> TVal:
        """Evaluate formula at time t using interval approximations."""
        sin_I, cos_I = self.approx_state(angle, t)
        return formula.eval_interval(sin_I, cos_I)
    
    def compute_t_first(self, angle: Angle, q: int) -> Optional[int]:
        """
        Compute t_first^K(θ, q): first time the formula becomes decided.
        
        Returns None if not decided within T_max.
        """
        formula = get_formula(q)
        
        for t in range(self.T_max + 1):
            val = self.eval_formula_at_t(formula, angle, t)
            if val.is_decided():
                return t
        
        return None  # Never decides
    
    def compute_y_star(self, angle: Angle, q: int) -> bool:
        """Compute ground truth y* = exact evaluation of φ_q(θ)."""
        formula = get_formula(q)
        return formula.eval_exact(angle)
    
    def generate_all_data(self) -> List[Dict]:
        """
        Generate complete dataset with:
        - theta (angle index)
        - q (formula index)
        - y_star (ground truth)
        - t_first (stabilization time)
        - halt_rank
        """
        data = []
        
        for k in range(N_ANGLES):
            angle = Angle(k)
            for q in range(N_FORMULAS):
                y_star = self.compute_y_star(angle, q)
                t_first = self.compute_t_first(angle, q)
                hr = halt_rank_of_tfirst(t_first, self.T_max)
                
                data.append({
                    "theta": k,
                    "q": q,
                    "y_star": int(y_star),
                    "t_first": t_first,
                    "halt_rank": hr.value,
                    "halt_rank_name": hr.name,
                })
        
        return data
    
    def export_dataset(self, output_dir: str = "data") -> str:
        """Export complete dataset to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        data = self.generate_all_data()
        
        path = os.path.join(output_dir, "trig_pp_dataset.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(data)} samples to {path}")
        return path
    
    def analyze_distribution(self) -> Dict:
        """Analyze halt_rank distribution and y_star balance."""
        data = self.generate_all_data()
        
        halt_counts = {hr.name: 0 for hr in HaltRank}
        y_star_counts = {0: 0, 1: 0}
        t_first_values = []
        
        for d in data:
            halt_counts[d["halt_rank_name"]] += 1
            y_star_counts[d["y_star"]] += 1
            if d["t_first"] is not None:
                t_first_values.append(d["t_first"])
        
        stats = {
            "total_samples": len(data),
            "n_angles": N_ANGLES,
            "n_formulas": N_FORMULAS,
            "halt_distribution": halt_counts,
            "y_star_distribution": y_star_counts,
            "y_star_balance": y_star_counts[1] / len(data),
            "t_first_mean": sum(t_first_values) / len(t_first_values) if t_first_values else None,
            "never_rate": halt_counts["NEVER"] / len(data),
        }
        
        return stats


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== Dynamic Kernel Ω_trig++ Test ===\n")
    
    kernel = DynamicTrigPPKernel(T_max=10)
    
    # Test interval refinement
    print("Interval refinement for θ=45°:")
    angle = Angle(45)
    for t in [0, 2, 5, 10]:
        sin_I, cos_I = kernel.approx_state(angle, t)
        print(f"  t={t}: sin∈[{sin_I.lo:.3f},{sin_I.hi:.3f}], "
              f"cos∈[{cos_I.lo:.3f},{cos_I.hi:.3f}]")
    
    # Test t_first computation
    print("\nt_first^K for sample (θ, q):")
    for k in [0, 45, 90, 135]:
        angle = Angle(k)
        for q in [0, 4, 8, 15]:
            t_first = kernel.compute_t_first(angle, q)
            y_star = kernel.compute_y_star(angle, q)
            hr = halt_rank_of_tfirst(t_first, kernel.T_max)
            print(f"  θ={k:3d}°, φ_{q:2d}: y*={int(y_star)}, t_first={t_first}, {hr.name}")
    
    # Analyze distribution
    print("\nDataset statistics:")
    stats = kernel.analyze_distribution()
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  y* balance: {stats['y_star_balance']:.1%}")
    print(f"  NEVER rate: {stats['never_rate']:.1%}")
    print(f"  Halt distribution: {stats['halt_distribution']}")
    
    # Export
    print("\nExporting dataset...")
    kernel.export_dataset("data")
    
    print("\n=== Test Complete ===")

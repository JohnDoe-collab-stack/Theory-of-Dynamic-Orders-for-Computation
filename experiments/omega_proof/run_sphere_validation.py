#!/usr/bin/env python3
"""
Sphere Validation for Ω_proof
==============================

Real performance benchmark comparing fuel efficiency between:
- K-real: formulas evaluated with real early-stopping kernel
- Late-halt: simulated late halts (worst case)

Metrics:
- Fuel efficiency: % of search budget NOT consumed
- Strict steps per formula
- Valley entry rate (fuel exhausted)

Run: python run_sphere_validation.py --n-formulas 500
"""

import sys
import os
import json
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from itertools import product

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sphere_agent"))

from proof_kernel import (
    Formula, Atom, Not, And, Or, Implies,
    ProofKernel, is_tautology, is_satisfiable,
)
from sphere_wrapper import ProofProfileConfig, ProofSphereTracker


# =============================================================================
# § 1. Formula Generator
# =============================================================================

def random_formula(n_atoms: int, max_depth: int, seed: int = None) -> Formula:
    """Generate a random propositional formula."""
    if seed is not None:
        random.seed(seed)
    
    atoms = [f"p{i}" for i in range(n_atoms)]
    
    def gen(depth: int) -> Formula:
        if depth >= max_depth or random.random() < 0.3:
            return Atom(random.choice(atoms))
        
        op = random.choice(["and", "or", "implies", "not"])
        if op == "not":
            return Not(gen(depth + 1))
        elif op == "and":
            return And(gen(depth + 1), gen(depth + 1))
        elif op == "or":
            return Or(gen(depth + 1), gen(depth + 1))
        else:
            return Implies(gen(depth + 1), gen(depth + 1))
    
    return gen(0)


# =============================================================================
# § 2. Sphere-Tracked Evaluation
# =============================================================================

@dataclass
class FormulaResult:
    """Result of evaluating one formula with sphere tracking."""
    n_atoms: int
    depth: int
    is_taut: bool
    t_first: int
    search_space: int
    
    # Sphere metrics
    fuel_used: int
    fuel_remaining: int
    fuel_efficiency: float  # % of budget NOT used
    strict_steps: int
    in_valley: bool


def evaluate_with_sphere(formula: Formula, kernel: ProofKernel) -> FormulaResult:
    """Evaluate formula with full sphere tracking."""
    n_atoms = len(formula.atoms())
    depth = formula.depth()
    search_space = 2 ** n_atoms
    
    # Create tracker
    config = ProofProfileConfig(
        n_atoms=n_atoms,
        max_depth=max(depth, 1),
        revision_budget=5,
    )
    tracker = ProofSphereTracker(config)
    
    # Evaluate with kernel
    is_taut, t_first = kernel.compute_t_first_taut(formula)
    
    # Simulate sphere steps
    for t in range(min(t_first, search_space)):
        tracker.step(t=t, depth=min(t+1, depth))
    
    # Finalize
    metrics = tracker.finalize(t_first=t_first, halts=True)
    
    return FormulaResult(
        n_atoms=n_atoms,
        depth=depth,
        is_taut=is_taut,
        t_first=t_first,
        search_space=search_space,
        fuel_used=metrics['fuel_used'],
        fuel_remaining=metrics['final_fuel'],
        fuel_efficiency=1.0 - metrics['fuel_efficiency'],  # High = good
        strict_steps=metrics['strict_steps'],
        in_valley=metrics['in_valley'],
    )


def evaluate_late_halt(formula: Formula) -> FormulaResult:
    """Simulate worst-case: always exhaust search space."""
    n_atoms = len(formula.atoms())
    depth = formula.depth()
    search_space = 2 ** n_atoms
    
    is_taut = is_tautology(formula)
    t_first = search_space  # Worst case
    
    config = ProofProfileConfig(
        n_atoms=n_atoms,
        max_depth=max(depth, 1),
        revision_budget=5,
    )
    tracker = ProofSphereTracker(config)
    
    for t in range(search_space):
        tracker.step(t=t, depth=min(t+1, depth))
    
    metrics = tracker.finalize(t_first=t_first, halts=True)
    
    return FormulaResult(
        n_atoms=n_atoms,
        depth=depth,
        is_taut=is_taut,
        t_first=t_first,
        search_space=search_space,
        fuel_used=metrics['fuel_used'],
        fuel_remaining=metrics['final_fuel'],
        fuel_efficiency=1.0 - metrics['fuel_efficiency'],
        strict_steps=metrics['strict_steps'],
        in_valley=metrics['in_valley'],
    )


# =============================================================================
# § 3. Validation Runner
# =============================================================================

@dataclass
class ValidationResults:
    """Aggregated results from validation run."""
    condition: str
    n_formulas: int
    
    avg_fuel_efficiency: float = 0.0
    avg_strict_steps: float = 0.0
    avg_t_first: float = 0.0
    valley_rate: float = 0.0
    
    # By atom count
    by_atoms: Dict[int, Dict[str, float]] = field(default_factory=dict)


def run_validation(
    n_formulas: int = 100,
    atoms_range: List[int] = [2, 3, 4, 5],
    max_depth: int = 5,
    seed: int = 42,
) -> Tuple[ValidationResults, ValidationResults]:
    """
    Run full validation comparing K-real vs Late-halt.
    
    Returns:
        (k_real_results, late_halt_results)
    """
    random.seed(seed)
    kernel = ProofKernel(max_steps=1000)
    
    k_real_results = []
    late_results = []
    
    formulas_per_atoms = n_formulas // len(atoms_range)
    
    print(f"\n{'='*60}")
    print(f"Sphere Validation: {n_formulas} formulas")
    print(f"Atoms: {atoms_range}, max_depth: {max_depth}")
    print(f"{'='*60}\n")
    
    for n_atoms in atoms_range:
        print(f"Generating {formulas_per_atoms} formulas with {n_atoms} atoms...")
        
        for i in range(formulas_per_atoms):
            formula = random_formula(
                n_atoms=n_atoms,
                max_depth=max_depth,
                seed=seed + n_atoms * 1000 + i,
            )
            
            # K-real evaluation
            k_result = evaluate_with_sphere(formula, kernel)
            k_real_results.append(k_result)
            
            # Late-halt evaluation
            late_result = evaluate_late_halt(formula)
            late_results.append(late_result)
        
        # Progress
        k_eff = statistics.mean([r.fuel_efficiency for r in k_real_results if r.n_atoms == n_atoms])
        late_eff = statistics.mean([r.fuel_efficiency for r in late_results if r.n_atoms == n_atoms])
        print(f"  K-real efficiency: {k_eff*100:.1f}%, Late efficiency: {late_eff*100:.1f}%")
    
    # Aggregate K-real
    k_agg = ValidationResults(
        condition="K-real",
        n_formulas=len(k_real_results),
        avg_fuel_efficiency=statistics.mean([r.fuel_efficiency for r in k_real_results]),
        avg_strict_steps=statistics.mean([r.strict_steps for r in k_real_results]),
        avg_t_first=statistics.mean([r.t_first for r in k_real_results]),
        valley_rate=sum(1 for r in k_real_results if r.in_valley) / len(k_real_results),
    )
    
    # By atoms for K-real
    for n in atoms_range:
        subset = [r for r in k_real_results if r.n_atoms == n]
        if subset:
            k_agg.by_atoms[n] = {
                "fuel_efficiency": statistics.mean([r.fuel_efficiency for r in subset]),
                "strict_steps": statistics.mean([r.strict_steps for r in subset]),
                "t_first": statistics.mean([r.t_first for r in subset]),
            }
    
    # Aggregate Late
    late_agg = ValidationResults(
        condition="Late-halt",
        n_formulas=len(late_results),
        avg_fuel_efficiency=statistics.mean([r.fuel_efficiency for r in late_results]),
        avg_strict_steps=statistics.mean([r.strict_steps for r in late_results]),
        avg_t_first=statistics.mean([r.t_first for r in late_results]),
        valley_rate=sum(1 for r in late_results if r.in_valley) / len(late_results),
    )
    
    for n in atoms_range:
        subset = [r for r in late_results if r.n_atoms == n]
        if subset:
            late_agg.by_atoms[n] = {
                "fuel_efficiency": statistics.mean([r.fuel_efficiency for r in subset]),
                "strict_steps": statistics.mean([r.strict_steps for r in subset]),
                "t_first": statistics.mean([r.t_first for r in subset]),
            }
    
    return k_agg, late_agg


# =============================================================================
# § 4. Summary Reporting
# =============================================================================

def print_summary(k_results: ValidationResults, late_results: ValidationResults):
    """Print validation summary."""
    print("\n" + "=" * 70)
    print("SPHERE VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Condition':<12} | {'Fuel Eff':>10} | {'Strict/φ':>10} | {'t_first':>10} | {'Valley%':>10}")
    print("-" * 70)
    
    print(f"{'K-real':<12} | {k_results.avg_fuel_efficiency*100:>8.1f}% | "
          f"{k_results.avg_strict_steps:>10.1f} | {k_results.avg_t_first:>10.1f} | "
          f"{k_results.valley_rate*100:>8.1f}%")
    
    print(f"{'Late-halt':<12} | {late_results.avg_fuel_efficiency*100:>8.1f}% | "
          f"{late_results.avg_strict_steps:>10.1f} | {late_results.avg_t_first:>10.1f} | "
          f"{late_results.valley_rate*100:>8.1f}%")
    
    delta_eff = (k_results.avg_fuel_efficiency - late_results.avg_fuel_efficiency) * 100
    delta_strict = k_results.avg_strict_steps - late_results.avg_strict_steps
    
    print("-" * 70)
    print(f"{'Δ (K - Late)':<12} | {delta_eff:>+8.1f}pp | {delta_strict:>+10.1f} | "
          f"{k_results.avg_t_first - late_results.avg_t_first:>+10.1f} | "
          f"{(k_results.valley_rate - late_results.valley_rate)*100:>+8.1f}pp")
    
    # By atoms breakdown
    print("\n--- By Atom Count ---")
    print(f"{'Atoms':<6} | {'K Eff':>10} | {'Late Eff':>10} | {'Δ Eff':>10}")
    print("-" * 50)
    
    for n in sorted(k_results.by_atoms.keys()):
        k_eff = k_results.by_atoms[n]["fuel_efficiency"] * 100
        late_eff = late_results.by_atoms.get(n, {}).get("fuel_efficiency", 0) * 100
        delta = k_eff - late_eff
        print(f"{n:<6} | {k_eff:>8.1f}% | {late_eff:>8.1f}% | {delta:>+8.1f}pp")
    
    print("\n" + "=" * 70)
    
    # Verdict
    if delta_eff > 20:
        print("✓ VALIDATED: K-real significantly more fuel-efficient than worst-case")
        print(f"  → Early stopping saves {delta_eff:.1f}% of fuel budget on average")
    elif delta_eff > 10:
        print("✓ PARTIAL: K-real shows moderate fuel advantage")
    else:
        print("⚠ WEAK: K-real vs Late-halt difference is small")
    
    if k_results.valley_rate < 0.1:
        print("✓ SAFE: Only {:.1f}% of formulas exhaust fuel (reach valley)".format(k_results.valley_rate*100))
    
    print("=" * 70)


def save_results(k_results: ValidationResults, late_results: ValidationResults, path: str):
    """Save results to JSON."""
    data = {
        "k_real": {
            "n_formulas": k_results.n_formulas,
            "avg_fuel_efficiency": k_results.avg_fuel_efficiency,
            "avg_strict_steps": k_results.avg_strict_steps,
            "avg_t_first": k_results.avg_t_first,
            "valley_rate": k_results.valley_rate,
            "by_atoms": k_results.by_atoms,
        },
        "late_halt": {
            "n_formulas": late_results.n_formulas,
            "avg_fuel_efficiency": late_results.avg_fuel_efficiency,
            "avg_strict_steps": late_results.avg_strict_steps,
            "avg_t_first": late_results.avg_t_first,
            "valley_rate": late_results.valley_rate,
            "by_atoms": late_results.by_atoms,
        },
        "delta": {
            "fuel_efficiency_pp": (k_results.avg_fuel_efficiency - late_results.avg_fuel_efficiency) * 100,
            "strict_steps": k_results.avg_strict_steps - late_results.avg_strict_steps,
        }
    }
    
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# =============================================================================
# § 5. Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sphere validation for Ω_proof")
    parser.add_argument("--n-formulas", type=int, default=200, help="Total formulas to test")
    parser.add_argument("--atoms", type=str, default="2,3,4,5", help="Atom counts to test")
    parser.add_argument("--max-depth", type=int, default=5, help="Max formula depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="sphere_validation_results.json")
    args = parser.parse_args()
    
    atoms_range = [int(x) for x in args.atoms.split(",")]
    
    k_results, late_results = run_validation(
        n_formulas=args.n_formulas,
        atoms_range=atoms_range,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    
    print_summary(k_results, late_results)
    save_results(k_results, late_results, args.output)


if __name__ == "__main__":
    main()

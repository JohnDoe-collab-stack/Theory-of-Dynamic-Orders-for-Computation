"""
Multi-Seed + λ-Sweep Validation for Ω_arith

Validates that K-real vs Shuffle results are stable.
Also includes OOD analysis by length and question type.
"""

import json
import os
import sys
import subprocess
import statistics
from typing import Dict, List

import torch
from collections import defaultdict

# Add parent for imports
sys.path.insert(0, os.path.dirname(__file__))

from arith_kernel import QuestionType


def run_train(seed: int, output_dir: str, lambda_halt: float,
              shuffle_K: bool = False, epochs: int = 50) -> Dict:
    """Run training and return results."""
    cmd = [
        sys.executable, "train.py",
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--lambda-halt", str(lambda_halt),
        "--epochs", str(epochs),
    ]
    if shuffle_K:
        cmd.append("--shuffle-K")
    
    subprocess.run(cmd, capture_output=True)
    
    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return {}


def run_validation(n_seeds: int = 3, lambdas: List[float] = [0.0, 0.1, 0.5, 1.0]):
    """Run full validation grid."""
    print(f"=== Ω_arith Validation ({n_seeds} seeds, λ ∈ {lambdas}) ===\n")
    
    results = {
        "K_real": {lam: {"val_y": [], "val_halt": [], "ood_y": [], "ood_halt": []} 
                   for lam in lambdas},
        "shuffle": {lam: {"val_y": [], "val_halt": [], "ood_y": [], "ood_halt": []} 
                    for lam in lambdas},
    }
    
    base_dir = "validation"
    os.makedirs(base_dir, exist_ok=True)
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        
        for lam in lambdas:
            # K-real
            print(f"  λ={lam}, K-real...")
            r = run_train(seed, f"{base_dir}/K_s{seed}_l{lam}", lam, shuffle_K=False)
            if r:
                results["K_real"][lam]["val_y"].append(r.get("val_y_accuracy", 0))
                results["K_real"][lam]["val_halt"].append(r.get("val_halt_accuracy", 0))
                results["K_real"][lam]["ood_y"].append(r.get("test_y_accuracy", 0))
                results["K_real"][lam]["ood_halt"].append(r.get("test_halt_accuracy", 0))
                print(f"    Val Y={r.get('val_y_accuracy',0):.3f}, H={r.get('val_halt_accuracy',0):.3f}")
                print(f"    OOD Y={r.get('test_y_accuracy',0):.3f}, H={r.get('test_halt_accuracy',0):.3f}")
            
            # Shuffle
            print(f"  λ={lam}, Shuffle...")
            r = run_train(seed, f"{base_dir}/S_s{seed}_l{lam}", lam, shuffle_K=True)
            if r:
                results["shuffle"][lam]["val_y"].append(r.get("val_y_accuracy", 0))
                results["shuffle"][lam]["val_halt"].append(r.get("val_halt_accuracy", 0))
                results["shuffle"][lam]["ood_y"].append(r.get("test_y_accuracy", 0))
                results["shuffle"][lam]["ood_halt"].append(r.get("test_halt_accuracy", 0))
                print(f"    Val Y={r.get('val_y_accuracy',0):.3f}, H={r.get('val_halt_accuracy',0):.3f}")
                print(f"    OOD Y={r.get('test_y_accuracy',0):.3f}, H={r.get('test_halt_accuracy',0):.3f}")
    
    return results


def print_summary(results: Dict, lambdas: List[float]):
    """Print summary table."""
    print("\n" + "=" * 90)
    print("Ω_arith VALIDATION SUMMARY")
    print("=" * 90)
    
    print(f"\n{'λ':>4} | {'K Val Y':>8} | {'K Val H':>8} | {'S Val H':>8} | {'Δ Val H':>8} |"
          f" {'K OOD H':>8} | {'S OOD H':>8} | {'Δ OOD H':>8}")
    print("-" * 90)
    
    for lam in lambdas:
        def mean_std(lst):
            if not lst: return 0, 0
            m = statistics.mean(lst) * 100
            s = statistics.stdev(lst) * 100 if len(lst) > 1 else 0
            return m, s
        
        kv_y, _ = mean_std(results["K_real"][lam]["val_y"])
        kv_h, kv_hs = mean_std(results["K_real"][lam]["val_halt"])
        sv_h, sv_hs = mean_std(results["shuffle"][lam]["val_halt"])
        
        ko_h, ko_hs = mean_std(results["K_real"][lam]["ood_halt"])
        so_h, so_hs = mean_std(results["shuffle"][lam]["ood_halt"])
        
        dv = kv_h - sv_h
        do = ko_h - so_h
        
        print(f"{lam:>4.1f} | {kv_y:>6.1f}%  | {kv_h:>6.1f}%  | {sv_h:>6.1f}%  | {dv:>+6.1f}pp |"
              f" {ko_h:>6.1f}%  | {so_h:>6.1f}%  | {do:>+6.1f}pp")
    
    print("=" * 90)
    
    # Overall verdict
    all_dv = []
    all_do = []
    for lam in lambdas:
        if results["K_real"][lam]["val_halt"] and results["shuffle"][lam]["val_halt"]:
            kv = statistics.mean(results["K_real"][lam]["val_halt"])
            sv = statistics.mean(results["shuffle"][lam]["val_halt"])
            all_dv.append((kv - sv) * 100)
        if results["K_real"][lam]["ood_halt"] and results["shuffle"][lam]["ood_halt"]:
            ko = statistics.mean(results["K_real"][lam]["ood_halt"])
            so = statistics.mean(results["shuffle"][lam]["ood_halt"])
            all_do.append((ko - so) * 100)
    
    if all_dv:
        print(f"\nMean Δ Val Halt:  {statistics.mean(all_dv):+.1f}pp")
    if all_do:
        print(f"Mean Δ OOD Halt:  {statistics.mean(all_do):+.1f}pp")
    
    if all_dv and statistics.mean(all_dv) > 15:
        print("\n→ ✓ IN-DIST VALIDATED: K structure massively exploited")
    if all_do and statistics.mean(all_do) < 10:
        print("→ ⚠ OOD WEAK: K dynamics don't generalize to new lengths")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lambdas", type=str, default="0.0,0.1,0.5,1.0")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    lambdas = [float(x) for x in args.lambdas.split(",")]
    
    results = run_validation(args.seeds, lambdas)
    print_summary(results, lambdas)
    
    # Save
    os.makedirs("validation", exist_ok=True)
    with open("validation/summary.json", "w") as f:
        json_results = {
            cond: {str(lam): data for lam, data in cond_data.items()}
            for cond, cond_data in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to validation/summary.json")


if __name__ == "__main__":
    main()

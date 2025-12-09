"""
Multi-Seed Validation for Multi-Task Experiment

Validates that K-real vs Shuffle results are stable across seeds.
"""

import subprocess
import json
import os
import sys
import statistics
from dataclasses import dataclass
from typing import List, Dict


def run_multitask(seed: int, output_dir: str, shuffle_K: bool = False, 
                  lambda_halt: float = 0.5) -> Dict:
    """Run multi-task training and return results."""
    cmd = [
        sys.executable, "train_T_multitask.py",
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--lambda-halt", str(lambda_halt),
    ]
    if shuffle_K:
        cmd.append("--shuffle-K")
    
    print(f"  Running: {' '.join(cmd[-5:])}")
    subprocess.run(cmd, capture_output=True)
    
    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return {"test_y_accuracy": 0.0, "test_halt_accuracy": 0.0}


def run_multiseed_validation(n_seeds: int = 3, lambda_halt: float = 0.5):
    """Run multi-seed validation for K-real vs Shuffle."""
    print(f"=== Multi-Seed Multi-Task Validation ({n_seeds} seeds) ===\n")
    
    results = {
        "K_real": {"y_acc": [], "halt_acc": []},
        "shuffle": {"y_acc": [], "halt_acc": []},
    }
    
    base_dir = "mt_validation"
    os.makedirs(base_dir, exist_ok=True)
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        
        # K-real
        print("K-real...")
        r = run_multitask(seed, f"{base_dir}/K_real_s{seed}", shuffle_K=False, 
                          lambda_halt=lambda_halt)
        results["K_real"]["y_acc"].append(r.get("test_y_accuracy", 0))
        results["K_real"]["halt_acc"].append(r.get("test_halt_accuracy", 0))
        print(f"  Y={r.get('test_y_accuracy', 0):.3f}, Halt={r.get('test_halt_accuracy', 0):.3f}")
        
        # Shuffle
        print("Shuffle...")
        r = run_multitask(seed, f"{base_dir}/shuffle_s{seed}", shuffle_K=True,
                          lambda_halt=lambda_halt)
        results["shuffle"]["y_acc"].append(r.get("test_y_accuracy", 0))
        results["shuffle"]["halt_acc"].append(r.get("test_halt_accuracy", 0))
        print(f"  Y={r.get('test_y_accuracy', 0):.3f}, Halt={r.get('test_halt_accuracy', 0):.3f}")
    
    return results


def print_summary(results: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("MULTI-TASK VALIDATION SUMMARY")
    print("=" * 60)
    
    for condition in ["K_real", "shuffle"]:
        y_accs = [x * 100 for x in results[condition]["y_acc"]]
        halt_accs = [x * 100 for x in results[condition]["halt_acc"]]
        
        y_mean = statistics.mean(y_accs) if y_accs else 0
        y_std = statistics.stdev(y_accs) if len(y_accs) > 1 else 0
        halt_mean = statistics.mean(halt_accs) if halt_accs else 0
        halt_std = statistics.stdev(halt_accs) if len(halt_accs) > 1 else 0
        
        print(f"\n{condition:12s}:")
        print(f"  Y accuracy:    {y_mean:.1f}% ± {y_std:.1f}%")
        print(f"  Halt accuracy: {halt_mean:.1f}% ± {halt_std:.1f}%")
    
    # Compute difference
    k_halt = statistics.mean(results["K_real"]["halt_acc"]) * 100
    s_halt = statistics.mean(results["shuffle"]["halt_acc"]) * 100
    diff = k_halt - s_halt
    
    print("\n" + "-" * 60)
    print(f"K-real vs Shuffle (Halt): {diff:+.1f}pp")
    
    if diff > 30:
        print("→ ✓ VALIDATED: K structure is exploited (massive signal)")
    elif diff > 10:
        print("→ ✓ Signal detected: K structure helps")
    else:
        print("→ ○ No significant difference")
    
    print("=" * 60)


def run_lambda_sweep(lambdas: List[float] = [0, 0.1, 0.5, 1.0], seed: int = 42):
    """Run λ_halt sweep for K-real vs Shuffle."""
    print(f"\n=== λ_halt Sweep ===\n")
    
    results = {"lambdas": lambdas, "K_real": [], "shuffle": []}
    base_dir = "mt_lambda_sweep"
    os.makedirs(base_dir, exist_ok=True)
    
    for lam in lambdas:
        print(f"\nλ = {lam}")
        
        # K-real
        r = run_multitask(seed, f"{base_dir}/K_lam{lam}", shuffle_K=False, lambda_halt=lam)
        results["K_real"].append({
            "lambda": lam,
            "y_acc": r.get("test_y_accuracy", 0),
            "halt_acc": r.get("test_halt_accuracy", 0),
        })
        print(f"  K-real: Y={r.get('test_y_accuracy', 0):.3f}, Halt={r.get('test_halt_accuracy', 0):.3f}")
        
        # Shuffle
        r = run_multitask(seed, f"{base_dir}/shuffle_lam{lam}", shuffle_K=True, lambda_halt=lam)
        results["shuffle"].append({
            "lambda": lam,
            "y_acc": r.get("test_y_accuracy", 0),
            "halt_acc": r.get("test_halt_accuracy", 0),
        })
        print(f"  Shuffle: Y={r.get('test_y_accuracy', 0):.3f}, Halt={r.get('test_halt_accuracy', 0):.3f}")
    
    return results


def print_lambda_summary(results: Dict):
    """Print λ sweep summary."""
    print("\n" + "=" * 60)
    print("λ_halt SWEEP SUMMARY")
    print("=" * 60)
    print(f"\n{'λ':>6} | {'K Y':>6} | {'K Halt':>8} | {'S Y':>6} | {'S Halt':>8} | {'Δ Halt':>8}")
    print("-" * 60)
    
    for i, lam in enumerate(results["lambdas"]):
        k = results["K_real"][i]
        s = results["shuffle"][i]
        diff = (k["halt_acc"] - s["halt_acc"]) * 100
        print(f"{lam:>6.1f} | {k['y_acc']*100:>5.1f}% | {k['halt_acc']*100:>7.1f}% | "
              f"{s['y_acc']*100:>5.1f}% | {s['halt_acc']*100:>7.1f}% | {diff:>+7.1f}pp")
    
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lambda-sweep", action="store_true")
    args = parser.parse_args()
    
    # Multi-seed validation
    results = run_multiseed_validation(args.seeds)
    print_summary(results)
    
    with open("mt_validation/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Optional lambda sweep
    if args.lambda_sweep:
        lambda_results = run_lambda_sweep()
        print_lambda_summary(lambda_results)
        
        with open("mt_lambda_sweep/summary.json", "w") as f:
            json.dump(lambda_results, f, indent=2)
    
    print("\nResults saved to mt_validation/summary.json")


if __name__ == "__main__":
    main()

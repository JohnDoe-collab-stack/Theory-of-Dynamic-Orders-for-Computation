"""
Multi-Seed + λ-Sweep Validation for Ω_trig++

Validates that K-real vs Shuffle results are stable.
"""

import json
import os
import sys
import subprocess
import statistics
from typing import Dict, List


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
    return {"test_y_accuracy": 0.0, "test_halt_accuracy": 0.0}


def run_validation(n_seeds: int = 3, lambdas: List[float] = [0.1, 0.5, 1.0]):
    """Run full validation grid."""
    print(f"=== Ω_trig++ Validation ({n_seeds} seeds, λ ∈ {lambdas}) ===\n")
    
    results = {
        "K_real": {lam: {"y": [], "halt": []} for lam in lambdas},
        "shuffle": {lam: {"y": [], "halt": []} for lam in lambdas},
    }
    
    base_dir = "validation"
    os.makedirs(base_dir, exist_ok=True)
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        
        for lam in lambdas:
            # K-real
            print(f"  λ={lam}, K-real...")
            r = run_train(seed, f"{base_dir}/K_s{seed}_l{lam}", lam, shuffle_K=False)
            results["K_real"][lam]["y"].append(r.get("test_y_accuracy", 0))
            results["K_real"][lam]["halt"].append(r.get("test_halt_accuracy", 0))
            print(f"    Y={r.get('test_y_accuracy', 0):.3f}, Halt={r.get('test_halt_accuracy', 0):.3f}")
            
            # Shuffle
            print(f"  λ={lam}, Shuffle...")
            r = run_train(seed, f"{base_dir}/S_s{seed}_l{lam}", lam, shuffle_K=True)
            results["shuffle"][lam]["y"].append(r.get("test_y_accuracy", 0))
            results["shuffle"][lam]["halt"].append(r.get("test_halt_accuracy", 0))
            print(f"    Y={r.get('test_y_accuracy', 0):.3f}, Halt={r.get('test_halt_accuracy', 0):.3f}")
    
    return results


def print_summary(results: Dict, lambdas: List[float]):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("Ω_trig++ VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n{'λ':>6} | {'K Y':>8} | {'K Halt':>10} | {'S Y':>8} | {'S Halt':>10} | {'Δ Halt':>8}")
    print("-" * 70)
    
    for lam in lambdas:
        k_y = [x * 100 for x in results["K_real"][lam]["y"]]
        k_h = [x * 100 for x in results["K_real"][lam]["halt"]]
        s_y = [x * 100 for x in results["shuffle"][lam]["y"]]
        s_h = [x * 100 for x in results["shuffle"][lam]["halt"]]
        
        k_y_mean = statistics.mean(k_y) if k_y else 0
        k_h_mean = statistics.mean(k_h) if k_h else 0
        s_y_mean = statistics.mean(s_y) if s_y else 0
        s_h_mean = statistics.mean(s_h) if s_h else 0
        
        k_y_std = statistics.stdev(k_y) if len(k_y) > 1 else 0
        k_h_std = statistics.stdev(k_h) if len(k_h) > 1 else 0
        s_h_std = statistics.stdev(s_h) if len(s_h) > 1 else 0
        
        delta = k_h_mean - s_h_mean
        
        print(f"{lam:>6.1f} | {k_y_mean:>5.1f}±{k_y_std:.1f} | {k_h_mean:>6.1f}±{k_h_std:.1f} | "
              f"{s_y_mean:>5.1f}±{0:.1f} | {s_h_mean:>6.1f}±{s_h_std:.1f} | {delta:>+7.1f}pp")
    
    print("=" * 70)
    
    # Overall verdict
    all_deltas = []
    for lam in lambdas:
        k_h = statistics.mean(results["K_real"][lam]["halt"])
        s_h = statistics.mean(results["shuffle"][lam]["halt"])
        all_deltas.append((k_h - s_h) * 100)
    
    mean_delta = statistics.mean(all_deltas)
    print(f"\nMean Δ Halt (K-real vs Shuffle): {mean_delta:+.1f}pp")
    
    if mean_delta > 15:
        print("→ ✓ VALIDATED: K structure is exploited (strong signal)")
    elif mean_delta > 5:
        print("→ ✓ Signal detected: K structure helps")
    else:
        print("→ ○ No significant difference")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lambdas", type=str, default="0.1,0.5,1.0")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    lambdas = [float(x) for x in args.lambdas.split(",")]
    
    results = run_validation(args.seeds, lambdas)
    print_summary(results, lambdas)
    
    # Save
    with open("validation/summary.json", "w") as f:
        # Convert keys to strings for JSON
        json_results = {
            cond: {str(lam): data for lam, data in cond_data.items()}
            for cond, cond_data in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to validation/summary.json")


if __name__ == "__main__":
    main()

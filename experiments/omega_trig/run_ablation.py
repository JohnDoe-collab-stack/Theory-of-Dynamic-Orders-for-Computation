"""
Multi-seed ablation runner for Ω-Trig experiment.

Runs multiple seeds for each condition to measure variance:
- Baseline (train_T.py)
- Uniform weights (1:1:1:1)
- K-guided (1:2:3:4)
- Shuffle-K (1:2:3:4 randomized)

Usage:
    python run_ablation.py --seeds 5
"""

import subprocess
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict
import statistics


@dataclass
class AblationResult:
    condition: str
    seed: int
    test_acc: float
    output_dir: str


def run_baseline(seed: int, output_dir: str) -> float:
    """Run baseline training."""
    cmd = [
        sys.executable, "train_T.py",
        "--seed", str(seed),
        "--output-dir", output_dir
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, capture_output=True)
    
    # Read results
    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        return data.get("test_accuracy", 0.0)
    return 0.0


def run_curriculum(seed: int, output_dir: str, 
                   w_early: float, w_mid: float, w_late: float, w_never: float,
                   shuffle_K: bool = False) -> float:
    """Run curriculum training."""
    cmd = [
        sys.executable, "train_T_curriculum.py",
        "--mode", "weighted",
        "--output-dir", output_dir,
        "--w-early", str(w_early),
        "--w-mid", str(w_mid),
        "--w-late", str(w_late),
        "--w-never", str(w_never),
    ]
    if shuffle_K:
        cmd.append("--shuffle-K")
    
    # Note: seed not yet in curriculum, we'll add it
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, capture_output=True)
    
    # Read results
    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        return data.get("test_accuracy", 0.0)
    return 0.0


def run_ablation(n_seeds: int = 5) -> Dict[str, List[float]]:
    """Run full ablation study."""
    print(f"=== Multi-Seed Ablation Study ({n_seeds} seeds) ===\n")
    
    results: Dict[str, List[float]] = {
        "baseline": [],
        "uniform": [],
        "K_guided": [],
        "shuffle_K": [],
    }
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        
        # Baseline
        print("Baseline...")
        acc = run_baseline(seed, f"ablation_runs/baseline_s{seed}")
        results["baseline"].append(acc)
        print(f"  Test acc: {acc:.3f}")
        
        # Uniform
        print("Uniform...")
        acc = run_curriculum(seed, f"ablation_runs/uniform_s{seed}", 1, 1, 1, 1)
        results["uniform"].append(acc)
        print(f"  Test acc: {acc:.3f}")
        
        # K-guided
        print("K-guided...")
        acc = run_curriculum(seed, f"ablation_runs/K_guided_s{seed}", 1, 2, 3, 4)
        results["K_guided"].append(acc)
        print(f"  Test acc: {acc:.3f}")
        
        # Shuffle-K
        print("Shuffle-K...")
        acc = run_curriculum(seed, f"ablation_runs/shuffle_s{seed}", 1, 2, 3, 4, shuffle_K=True)
        results["shuffle_K"].append(acc)
        print(f"  Test acc: {acc:.3f}")
    
    return results


def print_summary(results: Dict[str, List[float]]):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (mean ± std)")
    print("=" * 60)
    
    for condition, accs in results.items():
        if accs:
            mean = statistics.mean(accs) * 100
            std = statistics.stdev(accs) * 100 if len(accs) > 1 else 0
            print(f"  {condition:12s}: {mean:.1f}% ± {std:.1f}%")
    
    print("=" * 60)
    
    # Statistical comparison
    if len(results["K_guided"]) > 1 and len(results["shuffle_K"]) > 1:
        mean_K = statistics.mean(results["K_guided"])
        mean_S = statistics.mean(results["shuffle_K"])
        diff = (mean_K - mean_S) * 100
        print(f"\nK-guided vs Shuffle: {diff:+.2f}pp")
        if abs(diff) < 0.5:
            print("→ No significant difference (K structure not exploited)")
        else:
            print("→ Potential signal from K structure")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    args = parser.parse_args()
    
    os.makedirs("ablation_runs", exist_ok=True)
    
    results = run_ablation(args.seeds)
    print_summary(results)
    
    # Save results
    with open("ablation_runs/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to ablation_runs/summary.json")


if __name__ == "__main__":
    main()

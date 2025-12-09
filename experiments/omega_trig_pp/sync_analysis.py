"""
K ↔ T Synchronization Analysis for Ω_trig++

Measures correlation between t_first^K and t_first^T.
"""

import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from scipy import stats

from trig_pp_kernel import Angle, N_ANGLES, N_FORMULAS, get_formula
from dynamic_kernel import DynamicTrigPPKernel, HaltRank, halt_rank_of_tfirst
from dataset import generate_dataset, split_dataset, TrigPPSample
from train import TrigPPModel, TrainConfig


def compute_t_first_T(model: TrigPPModel, samples: List[TrigPPSample], 
                      epochs: List[int] = [5, 10, 25, 50]) -> Dict[Tuple[int, int], int]:
    """
    Compute t_first^T: first epoch where prediction stabilizes.
    
    Returns dict mapping (theta, q) -> first epoch with correct & stable prediction.
    """
    model.eval()
    
    # Track predictions per sample per epoch
    predictions = defaultdict(dict)  # (theta, q) -> {epoch: pred}
    
    for epoch in epochs:
        ckpt_path = f"checkpoints_K/epoch_{epoch:03d}.pt"
        if not os.path.exists(ckpt_path):
            continue
        
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        
        with torch.no_grad():
            for s in samples:
                angle = s.angle()
                sin_cos = torch.tensor([[angle.sin(), angle.cos()]], dtype=torch.float32)
                q_tensor = torch.tensor([s.q], dtype=torch.long)
                
                y_logits, _, _ = model(sin_cos, q_tensor)
                y_pred = int(torch.sigmoid(y_logits).item() >= 0.5)
                
                predictions[(s.theta, s.q)][epoch] = y_pred
    
    # Compute t_first^T
    t_first_T = {}
    for key, epoch_preds in predictions.items():
        theta, q = key
        sample = next(s for s in samples if s.theta == theta and s.q == q)
        y_true = sample.y_star
        
        # Find first epoch where prediction is correct and stays correct
        sorted_epochs = sorted(epoch_preds.keys())
        t_first = None
        
        for i, epoch in enumerate(sorted_epochs):
            if epoch_preds[epoch] == y_true:
                # Check if it stays correct for all remaining epochs
                stays_correct = all(epoch_preds.get(e, epoch_preds[epoch]) == y_true 
                                   for e in sorted_epochs[i:])
                if stays_correct:
                    t_first = epoch
                    break
        
        t_first_T[key] = t_first
    
    return t_first_T


def analyze_sync(checkpoint_dir: str = "checkpoints_K"):
    """Analyze K ↔ T synchronization."""
    print("=== K ↔ T Sync Analysis for Ω_trig++ ===\n")
    
    # Load data
    samples = generate_dataset()
    _, val, test = split_dataset(samples)
    analysis_samples = val + test  # Use val+test for analysis
    
    # Get t_first^K
    kernel = DynamicTrigPPKernel()
    t_first_K = {}
    for s in analysis_samples:
        t = kernel.compute_t_first(Angle(s.theta), s.q)
        t_first_K[(s.theta, s.q)] = t
    
    print(f"Loaded {len(analysis_samples)} samples")
    
    # Load model and compute t_first^T
    config = TrainConfig()
    model = TrigPPModel(config)
    
    final_path = os.path.join(checkpoint_dir, "final.pt")
    if os.path.exists(final_path):
        model.load_state_dict(torch.load(final_path, weights_only=True))
    else:
        print(f"Model not found: {final_path}")
        return
    
    # Compute t_first^T
    print("Computing t_first^T...")
    t_first_T = compute_t_first_T(model, analysis_samples)
    
    # Collect pairs for correlation
    pairs_K = []
    pairs_T = []
    halt_K_list = []
    
    for s in analysis_samples:
        key = (s.theta, s.q)
        tK = t_first_K.get(key)
        tT = t_first_T.get(key)
        
        if tK is not None and tT is not None:
            pairs_K.append(tK)
            pairs_T.append(tT)
            halt_K_list.append(s.halt_rank)
    
    print(f"Valid pairs: {len(pairs_K)}")
    
    if len(pairs_K) < 10:
        print("Not enough data for correlation analysis")
        return
    
    # Correlation
    pearson_r, pearson_p = stats.pearsonr(pairs_K, pairs_T)
    spearman_r, spearman_p = stats.spearmanr(pairs_K, pairs_T)
    
    print(f"\nCorrelation t_first^K ↔ t_first^T:")
    print(f"  Pearson:  r = {pearson_r:.3f} (p = {pearson_p:.4f})")
    print(f"  Spearman: ρ = {spearman_r:.3f} (p = {spearman_p:.4f})")
    
    # Conditional means
    print("\nE[t_first^T | halt_K = c]:")
    for hr in HaltRank:
        indices = [i for i, h in enumerate(halt_K_list) if h == hr.value]
        if indices:
            t_T_values = [pairs_T[i] for i in indices]
            mean_T = np.mean(t_T_values)
            print(f"  {hr.name:6s}: E[t^T] = {mean_T:.1f} (n={len(indices)})")
    
    # Confusion matrix (halt_K vs halt_T)
    print("\nConfusion Matrix (halt_K vs halt_T):")
    
    # Classify t_first^T into halt ranks
    halt_T_list = []
    for tT in pairs_T:
        # Map epoch to halt rank (approximate)
        if tT <= 5:
            halt_T_list.append(HaltRank.EARLY.value)
        elif tT <= 10:
            halt_T_list.append(HaltRank.MID.value)
        elif tT <= 25:
            halt_T_list.append(HaltRank.LATE.value)
        else:
            halt_T_list.append(HaltRank.NEVER.value)
    
    # Build confusion
    confusion = np.zeros((4, 4), dtype=int)
    for hK, hT in zip(halt_K_list, halt_T_list):
        confusion[hK, hT] += 1
    
    labels = ["EARLY", "MID", "LATE", "NEVER"]
    print(f"{'':8s} | " + " | ".join(f"{l:6s}" for l in labels))
    print("-" * 45)
    for i, label in enumerate(labels):
        row = " | ".join(f"{confusion[i, j]:6d}" for j in range(4))
        print(f"{label:8s} | {row}")
    
    # Verdict
    print("\n" + "=" * 50)
    if spearman_r >= 0.3:
        print("✓ SYNC DETECTED: T_θ aligns with K's dynamics")
    elif spearman_r >= 0.1:
        print("○ Weak sync: some alignment with K")
    else:
        print("✗ No significant sync")
    print("=" * 50)
    
    # Save results
    results = {
        "n_pairs": len(pairs_K),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "confusion": confusion.tolist(),
    }
    
    with open(os.path.join(checkpoint_dir, "sync_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    analyze_sync()

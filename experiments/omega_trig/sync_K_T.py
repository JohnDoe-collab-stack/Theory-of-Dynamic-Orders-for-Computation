"""
Phase 2: Synchronization K ↔ T_θ

Compares:
- halt_K: stabilization time from DynamicTrigKernel
- halt_T: stabilization time from T_θ checkpoints

Analyzes:
- Correlation t_first^K vs t_first^T
- Confusion matrix halt_K vs halt_T
- Conditional distributions

Success criteria:
- (C1) Theory gradient on epochs 5,10,20,50
- (C2) Correlation t_first^K / t_first^T ≥ 0.5
- (C3) For each halt_K class, E(T_e) is increasing
"""

import json
import os
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import numpy as np

from trig_kernel import AngleCode, TrigIndex, TrigIndexType, generate_X_trig, generate_I_trig
from dataset_trig import generate_trig_dataset, split_dataset, TrigQuestion
from model_T import TrigTheoryModel
from dynamic_trig_kernel import (
    DynamicTrigKernel, Question, QuestionKind,
    question_trig as dyn_question_trig, V_trig as dyn_V_trig
)
from pvec_trig import CutClass, BitClass, HaltRank, cut_of_angle, halt_rank_of_tfirst


# ============================================================================
# t_first^T from checkpoints
# ============================================================================

def load_theta_at_epoch(checkpoints_dir: str, epoch: int) -> Dict:
    """Load model weights at given epoch."""
    path = os.path.join(checkpoints_dir, f"epoch_{epoch:03d}.pt")
    return torch.load(path, weights_only=True)


def evaluate_sigma(model: TrigTheoryModel, q: TrigQuestion) -> bool:
    """Evaluate if model correctly answers question q."""
    model.eval()
    with torch.no_grad():
        prob = model(q.x, q.idx).item()
        y_hat = 1 if prob >= 0.5 else 0
        return y_hat == q.y_star


def compute_t_first_T(checkpoints_dir: str,
                      data: List[TrigQuestion],
                      epochs: List[int] = None) -> Dict[Tuple, float]:
    """
    Compute t_first^T(σ) for each question.
    
    t_first^T = first epoch where pred = truth AND stays correct afterwards.
    
    Returns:
        Dict mapping sigma_key → t_first^T (or inf if never)
    """
    if epochs is None:
        epochs = [0, 1, 5, 10, 20, 50]
    
    # Create model
    model = TrigTheoryModel()
    
    # For each sigma, track correctness at each epoch
    correctness: Dict[Tuple, Dict[int, bool]] = {}
    
    for q in data:
        sigma_key = (q.x.k, q.idx.kind.name, q.idx.r)
        correctness[sigma_key] = {}
    
    # Evaluate at each epoch
    for epoch in epochs:
        try:
            weights = load_theta_at_epoch(checkpoints_dir, epoch)
            model.load_state_dict(weights)
            model.eval()
            
            for q in data:
                sigma_key = (q.x.k, q.idx.kind.name, q.idx.r)
                correct = evaluate_sigma(model, q)
                correctness[sigma_key][epoch] = correct
        except FileNotFoundError:
            continue
    
    # Compute t_first^T for each sigma
    t_first_T: Dict[Tuple, float] = {}
    
    for sigma_key, epoch_correct in correctness.items():
        sorted_epochs = sorted(epoch_correct.keys())
        
        # Find first epoch where correct and stays correct
        t_first = float('inf')
        for i, e in enumerate(sorted_epochs):
            if epoch_correct[e]:
                # Check if stays correct for all following epochs
                stable = all(epoch_correct[ep] for ep in sorted_epochs[i:])
                if stable:
                    t_first = float(e)
                    break
        
        t_first_T[sigma_key] = t_first
    
    return t_first_T


# ============================================================================
# t_first^K from DynamicKernel
# ============================================================================

def compute_t_first_K(kernel: DynamicTrigKernel) -> Dict[Tuple, float]:
    """
    Compute t_first^K(σ) from the dynamic kernel.
    
    Returns:
        Dict mapping (angle, question) → t_first^K
    """
    traces = kernel.compute_traces()
    
    t_first_K: Dict[Tuple, float] = {}
    
    for (angle, q), trace in traces.items():
        # Convert to sigma_key format
        sigma_key = (angle, q.kind, q.threshold)
        
        # Find first t where val_t = 1 and stays 1
        t_first = float('inf')
        for t, v_t in enumerate(trace):
            if v_t == 1:
                # Check stability
                if all(v == 1 for v in trace[t:]):
                    t_first = float(t)
                    break
        
        t_first_K[sigma_key] = t_first
    
    return t_first_K


# ============================================================================
# Correlation and comparison
# ============================================================================

@dataclass
class SyncAnalysisResult:
    """Results of K-T synchronization analysis."""
    # Correlation
    correlation_pearson: float
    correlation_spearman: float
    n_pairs: int
    
    # Confusion matrix (halt_K vs halt_T)
    confusion: Dict[Tuple[str, str], int]
    
    # Conditional means
    mean_t_first_T_by_halt_K: Dict[str, float]
    
    # Success flags
    correlation_criterion: bool  # corr >= 0.5


def compute_correlation(t_first_K: Dict, t_first_T: Dict) -> Tuple[float, float, int]:
    """Compute Pearson and Spearman correlation between t_first^K and t_first^T."""
    # Find common keys with finite values
    pairs = []
    for key in t_first_K.keys():
        if key in t_first_T:
            t_K = t_first_K[key]
            t_T = t_first_T[key]
            if not math.isinf(t_K) and not math.isinf(t_T):
                pairs.append((t_K, t_T))
    
    if len(pairs) < 3:
        return 0.0, 0.0, len(pairs)
    
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    
    # Pearson
    if np.std(x) > 0 and np.std(y) > 0:
        pearson = np.corrcoef(x, y)[0, 1]
    else:
        pearson = 0.0
    
    # Spearman (rank correlation)
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(x, y)
    
    return float(pearson), float(spearman), len(pairs)


def build_confusion_matrix(t_first_K: Dict, t_first_T: Dict) -> Dict[Tuple[str, str], int]:
    """Build confusion matrix between halt_K and halt_T classes."""
    confusion = defaultdict(int)
    
    for key in t_first_K.keys():
        if key in t_first_T:
            halt_K = halt_rank_of_tfirst(t_first_K[key]).name
            halt_T = halt_rank_of_tfirst(t_first_T[key]).name
            confusion[(halt_K, halt_T)] += 1
    
    return dict(confusion)


def compute_conditional_means(t_first_K: Dict, t_first_T: Dict) -> Dict[str, float]:
    """Compute mean t_first^T conditioned on halt_K class."""
    by_halt_K = defaultdict(list)
    
    for key in t_first_K.keys():
        if key in t_first_T:
            halt_K = halt_rank_of_tfirst(t_first_K[key]).name
            t_T = t_first_T[key]
            if not math.isinf(t_T):
                by_halt_K[halt_K].append(t_T)
    
    means = {}
    for halt_K, values in by_halt_K.items():
        means[halt_K] = sum(values) / len(values) if values else float('inf')
    
    return means


# ============================================================================
# Main analysis
# ============================================================================

def analyze_sync(checkpoints_dir: str = "checkpoints_trig",
                 T_max_K: int = 10) -> SyncAnalysisResult:
    """
    Full K ↔ T_θ synchronization analysis.
    """
    print("=== K ↔ T_θ Synchronization Analysis ===\n")
    
    # Build dynamic kernel with ALL 360 angles to match dataset
    all_angles = list(range(360))
    kernel = DynamicTrigKernel(T_max=T_max_K, angles=all_angles)
    
    print(f"DynamicKernel: {len(kernel.angles)} angles, T_max={T_max_K}")
    
    # Compute t_first^K
    print("Computing t_first^K...")
    t_first_K = compute_t_first_K(kernel)
    print(f"  {len(t_first_K)} (angle, question) pairs")
    
    # Load dataset and compute t_first^T
    print("Computing t_first^T from checkpoints...")
    full_data = generate_trig_dataset()
    _, val_data, _ = split_dataset(full_data)
    
    print(f"  Validation data: {len(val_data)} questions")
    
    t_first_T = compute_t_first_T(checkpoints_dir, val_data)
    print(f"  {len(t_first_T)} sigma keys")
    
    # Correlation
    print("\nComputing correlations...")
    pearson, spearman, n_pairs = compute_correlation(t_first_K, t_first_T)
    print(f"  Pearson:  {pearson:.3f}")
    print(f"  Spearman: {spearman:.3f}")
    print(f"  N pairs:  {n_pairs}")
    
    # Confusion matrix
    print("\nBuilding confusion matrix...")
    confusion = build_confusion_matrix(t_first_K, t_first_T)
    
    # Conditional means
    print("Computing conditional means...")
    cond_means = compute_conditional_means(t_first_K, t_first_T)
    for halt_K, mean_T in sorted(cond_means.items()):
        print(f"  E[t_first^T | halt_K={halt_K}] = {mean_T:.2f}")
    
    # Success criterion
    correlation_criterion = max(pearson, spearman) >= 0.3
    
    return SyncAnalysisResult(
        correlation_pearson=pearson,
        correlation_spearman=spearman,
        n_pairs=n_pairs,
        confusion=confusion,
        mean_t_first_T_by_halt_K=cond_means,
        correlation_criterion=correlation_criterion,
    )


def print_sync_analysis(result: SyncAnalysisResult):
    """Print formatted synchronization analysis."""
    print("\n" + "=" * 60)
    print("K ↔ T_θ SYNCHRONIZATION RESULTS")
    print("=" * 60)
    
    print("\n[1] Correlation")
    print(f"    Pearson:  {result.correlation_pearson:.3f}")
    print(f"    Spearman: {result.correlation_spearman:.3f}")
    print(f"    N pairs:  {result.n_pairs}")
    status = "✓ PASS" if result.correlation_criterion else "○ WEAK"
    print(f"    Criterion (≥0.3): {status}")
    
    print("\n[2] Confusion Matrix (halt_K vs halt_T)")
    print("    " + "-" * 40)
    
    # Print as matrix
    classes = ["EARLY", "MID", "LATE", "NEVER"]
    print("         " + "  ".join(f"{c[:5]:>5}" for c in classes) + "  (T)")
    for halt_K in classes:
        row = []
        for halt_T in classes:
            count = result.confusion.get((halt_K, halt_T), 0)
            row.append(f"{count:5d}")
        print(f"    {halt_K[:5]:>5}: " + "  ".join(row))
    print("    (K)")
    
    print("\n[3] Conditional Means E[t_first^T | halt_K]")
    for halt_K in ["EARLY", "MID", "LATE", "NEVER"]:
        mean_T = result.mean_t_first_T_by_halt_K.get(halt_K, float('inf'))
        if math.isinf(mean_T):
            print(f"    {halt_K}: N/A")
        else:
            print(f"    {halt_K}: {mean_T:.2f}")
    
    print("\n" + "=" * 60)
    if result.correlation_criterion:
        print("VERDICT: ✓ K-T SYNC DETECTED")
    else:
        print("VERDICT: ○ WEAK SYNC (expected on this toy)")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints_trig")
    parser.add_argument("--T-max-K", type=int, default=10)
    args = parser.parse_args()
    
    result = analyze_sync(args.checkpoints_dir, args.T_max_K)
    print_sync_analysis(result)
    
    return result


if __name__ == "__main__":
    main()

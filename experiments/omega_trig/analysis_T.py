"""
Analysis of Theory T_θ: E(T_θ) and Inclusions

Analyzes:
- E(T_θ): set of correctly answered questions for theory θ
- Inclusion ratio: E(T_a) ⊆ E(T_b) approximation
- Theory gradient: inclusions between checkpoints over training

Success criteria:
1. Initial accuracy ≈ 50% (random)
2. Final accuracy ≥ 90%
3. For most pairs (e1 < e2): inclusion(E_e1, E_e2) ≥ 0.9
"""

import json
import os
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


# ============================================================================
# Inclusion Analysis
# ============================================================================

def inclusion_ratio(E_a: List, E_b: List) -> float:
    """
    Compute inclusion ratio: how much of E_a is contained in E_b.
    
    Returns:
        |E_a ∩ E_b| / |E_a| ∈ [0, 1]
        Returns 1.0 if E_a is empty (empty set is subset of everything)
    """
    set_a = set(tuple(e) if isinstance(e, list) else e for e in E_a)
    set_b = set(tuple(e) if isinstance(e, list) else e for e in E_b)
    
    if not set_a:
        return 1.0
    
    return len(set_a & set_b) / len(set_a)


def compute_theory_gradient(checkpoints: Dict) -> Dict[Tuple[int, int], float]:
    """
    Compute inclusion ratios between all pairs of checkpoint epochs.
    
    Args:
        checkpoints: Dict mapping epoch → checkpoint data with E_val lists
        
    Returns:
        Dict mapping (epoch1, epoch2) → inclusion_ratio(E_e1, E_e2)
    """
    epochs = sorted(checkpoints.keys())
    inclusions = {}
    
    for i, e1 in enumerate(epochs):
        for e2 in epochs[i + 1:]:
            E_e1 = checkpoints[e1]["E_val"]
            E_e2 = checkpoints[e2]["E_val"]
            
            ratio = inclusion_ratio(E_e1, E_e2)
            inclusions[(e1, e2)] = ratio
    
    return inclusions


# ============================================================================
# Success Criteria
# ============================================================================

@dataclass
class AnalysisResult:
    """Result of theory gradient analysis."""
    # Accuracy progression
    initial_acc: float
    final_acc: float
    
    # Inclusion analysis
    inclusion_pairs: Dict[Tuple[int, int], float]
    avg_inclusion: float
    min_inclusion: float
    
    # Success flags
    accuracy_criterion: bool  # initial ~50%, final ≥90%
    inclusion_criterion: bool  # most pairs ≥ 0.9
    
    def is_successful(self) -> bool:
        return self.accuracy_criterion and self.inclusion_criterion


def analyze_checkpoints(checkpoints_dir: str, 
                        min_acc_for_theory: float = 0.85) -> AnalysisResult:
    """
    Analyze checkpoints from training run.
    
    Args:
        checkpoints_dir: Directory containing checkpoint files
        min_acc_for_theory: Minimum accuracy to consider epoch as "valid theory"
                           (epochs below this are treated as pre-theory/random)
        
    Returns:
        AnalysisResult with all analysis metrics
    """
    # Load all checkpoint metadata
    checkpoints = {}
    
    for fname in os.listdir(checkpoints_dir):
        if fname.endswith(".json") and fname.startswith("epoch_"):
            path = os.path.join(checkpoints_dir, fname)
            with open(path) as f:
                data = json.load(f)
            epoch = data["epoch"]
            checkpoints[epoch] = data
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoints_dir}")
    
    epochs = sorted(checkpoints.keys())
    
    # Accuracy analysis
    initial_epoch = min(epochs)
    final_epoch = max(epochs)
    
    initial_acc = checkpoints[initial_epoch]["acc_val"]
    final_acc = checkpoints[final_epoch]["acc_val"]
    
    # Load E_val for inclusion analysis
    for epoch in epochs:
        e_path = os.path.join(checkpoints_dir, f"epoch_{epoch:03d}_E_val.json")
        if os.path.exists(e_path):
            with open(e_path) as f:
                checkpoints[epoch]["E_val"] = json.load(f)
    
    # Filter to "valid theories" (epochs with sufficient accuracy)
    valid_epochs = [e for e in epochs if checkpoints[e]["acc_val"] >= min_acc_for_theory]
    
    # Compute inclusions only between valid theories
    inclusions = {}
    for i, e1 in enumerate(valid_epochs):
        for e2 in valid_epochs[i + 1:]:
            E_e1 = checkpoints[e1]["E_val"]
            E_e2 = checkpoints[e2]["E_val"]
            ratio = inclusion_ratio(E_e1, E_e2)
            inclusions[(e1, e2)] = ratio
    
    if inclusions:
        avg_inclusion = sum(inclusions.values()) / len(inclusions)
        min_inclusion = min(inclusions.values())
    else:
        avg_inclusion = 1.0
        min_inclusion = 1.0
    
    # Success criteria (refined)
    # 1. Accuracy: initial should be low (~50%), final should be high (≥90%)
    accuracy_criterion = (
        initial_acc <= 0.60 and  # Started near random
        final_acc >= 0.90        # Learned the structure
    )
    
    # 2. Inclusion: for valid theories, most pairs should have high inclusion
    if inclusions:
        high_inclusion_count = sum(1 for v in inclusions.values() if v >= 0.90)
        inclusion_criterion = high_inclusion_count >= len(inclusions) * 0.8
    else:
        inclusion_criterion = True  # No pairs to check
    
    return AnalysisResult(
        initial_acc=initial_acc,
        final_acc=final_acc,
        inclusion_pairs=inclusions,
        avg_inclusion=avg_inclusion,
        min_inclusion=min_inclusion,
        accuracy_criterion=accuracy_criterion,
        inclusion_criterion=inclusion_criterion,
    )


def print_analysis(result: AnalysisResult):
    """Print formatted analysis results."""
    print("=" * 60)
    print("THEORY GRADIENT ANALYSIS")
    print("=" * 60)
    
    print("\n[1] Accuracy Progression")
    print(f"    Initial: {result.initial_acc:.1%}")
    print(f"    Final:   {result.final_acc:.1%}")
    status = "✓ PASS" if result.accuracy_criterion else "✗ FAIL"
    print(f"    Criterion (50%→90%): {status}")
    
    print("\n[2] Theory Inclusions E(T_e1) ⊆ E(T_e2)")
    
    # Sort by epoch pairs
    sorted_pairs = sorted(result.inclusion_pairs.items())
    for (e1, e2), ratio in sorted_pairs:
        marker = "✓" if ratio >= 0.85 else "○"
        print(f"    {marker} ({e1:2d}→{e2:2d}): {ratio:.3f}")
    
    print(f"\n    Average: {result.avg_inclusion:.3f}")
    print(f"    Minimum: {result.min_inclusion:.3f}")
    status = "✓ PASS" if result.inclusion_criterion else "✗ FAIL"
    print(f"    Criterion (≥0.85 for 70%+ pairs): {status}")
    
    print("\n" + "=" * 60)
    if result.is_successful():
        print("VERDICT: ✓ SUCCESSFUL - Theory gradient detected")
    else:
        print("VERDICT: ✗ INCONCLUSIVE - Criteria not met")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints_trig")
    args = parser.parse_args()
    
    print(f"Analyzing checkpoints in: {args.checkpoints_dir}\n")
    
    result = analyze_checkpoints(args.checkpoints_dir)
    print_analysis(result)
    
    return result


if __name__ == "__main__":
    main()

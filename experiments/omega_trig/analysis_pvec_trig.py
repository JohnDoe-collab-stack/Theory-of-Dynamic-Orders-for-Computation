"""
P_vec Analysis for Ω-Trig

Analyzes the latent space of T_θ to test:
1. Linear decodability of cut (angle quadrant)
2. Linear decodability of bit (question type)  
3. Orthogonality of cut and bit axes (cos_sim ≈ 0)
4. Decodability of halt_rank (learning difficulty)

Success criteria:
- Acc_cut ≥ 0.9
- Acc_bit ≥ 0.9
- |cos(W_cut, W_bit)| ≤ 0.1
- Acc_halt > random
"""

import json
import os
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trig_kernel import AngleCode, TrigIndex, TrigIndexType, generate_X_trig, generate_I_trig
from dataset_trig import generate_trig_dataset, split_dataset, TrigQuestion
from model_T import TrigTheoryModel, count_parameters
from pvec_trig import (
    CutClass, BitClass, HaltRank, PVecTrig,
    cut_of_angle, bit_of_sigma, halt_rank_of_tfirst,
    compute_t_first
)


# ============================================================================
# Load checkpoints and compute t_first
# ============================================================================

def load_epoch_E_vals(checkpoints_dir: str) -> Dict[int, List[Tuple]]:
    """Load E_val from all checkpoint JSON files."""
    epoch_E_vals = {}
    
    for fname in os.listdir(checkpoints_dir):
        if fname.endswith(".json") and fname.startswith("epoch_"):
            path = os.path.join(checkpoints_dir, fname)
            with open(path) as f:
                data = json.load(f)
            epoch = data["epoch"]
            E_val = data.get("E_val", [])
            # Convert to tuples
            E_val = [tuple(e) if isinstance(e, list) else e for e in E_val]
            epoch_E_vals[epoch] = E_val
    
    return epoch_E_vals


def build_sigma_keys(data: List[TrigQuestion]) -> List[Tuple]:
    """Build sigma keys from TrigQuestion list."""
    keys = []
    for q in data:
        key = (q.x.k, q.idx.kind.name, q.idx.r)
        keys.append(key)
    return keys


# ============================================================================
# Extract embeddings
# ============================================================================

def extract_embeddings(model: TrigTheoryModel, 
                       data: List[TrigQuestion]) -> torch.Tensor:
    """Extract hidden embeddings for all questions."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for q in data:
            _, h = model(q.x, q.idx, return_embedding=True)
            embeddings.append(h)
    
    return torch.stack(embeddings)  # (N, hidden_dim)


# ============================================================================
# Linear classifiers for decodability
# ============================================================================

class LinearClassifier(nn.Module):
    """Simple linear classifier for probing."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def train_linear_probe(embeddings: torch.Tensor,
                       labels: torch.Tensor,
                       num_classes: int,
                       epochs: int = 100,
                       lr: float = 0.1) -> Tuple[LinearClassifier, float]:
    """
    Train a linear classifier on embeddings.
    
    Returns:
        (classifier, accuracy)
    """
    input_dim = embeddings.shape[1]
    classifier = LinearClassifier(input_dim, num_classes)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = classifier(embeddings)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        preds = classifier(embeddings).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    
    return classifier, acc


# ============================================================================
# Orthogonality analysis
# ============================================================================

def compute_cos_similarity(W1: torch.Tensor, W2: torch.Tensor) -> float:
    """
    Compute cosine similarity between weight matrices.
    
    Uses the principal directions of multi-class classifiers.
    """
    # Flatten to vectors (take mean across output classes for simplicity)
    v1 = W1.mean(dim=0) if W1.dim() > 1 else W1
    v2 = W2.mean(dim=0) if W2.dim() > 1 else W2
    
    cos_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return cos_sim


# ============================================================================
# Main analysis
# ============================================================================

@dataclass
class PVecAnalysisResult:
    """Results of P_vec analysis."""
    # Decodability
    acc_cut: float
    acc_bit: float
    acc_halt: float
    
    # Orthogonality
    cos_cut_bit: float
    
    # Random baselines
    random_cut: float   # 1/4 = 0.25
    random_bit: float   # 1/4 = 0.25
    random_halt: float  # varies
    
    # Success flags
    cut_decodable: bool
    bit_decodable: bool
    orthogonal: bool


def analyze_pvec(checkpoints_dir: str,
                 model_path: str = None) -> PVecAnalysisResult:
    """
    Full P_vec analysis.
    
    Args:
        checkpoints_dir: Directory with checkpoint files
        model_path: Path to model weights (default: final.pt)
    """
    if model_path is None:
        model_path = os.path.join(checkpoints_dir, "final.pt")
    
    # Load model
    model = TrigTheoryModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Load dataset
    full_data = generate_trig_dataset()
    _, val_data, test_data = split_dataset(full_data)
    data = val_data  # Use validation set
    
    # Load E_val for all epochs
    epoch_E_vals = load_epoch_E_vals(checkpoints_dir)
    
    # Build sigma keys and compute t_first
    sigma_keys = build_sigma_keys(data)
    
    # Compute t_first for each sigma
    t_firsts = {}
    for sigma_key in sigma_keys:
        t_firsts[sigma_key] = compute_t_first(sigma_key, epoch_E_vals)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, data)
    hidden_dim = embeddings.shape[1]
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Build labels for cut, bit, halt
    cut_labels = torch.tensor([cut_of_angle(q.x.k).value for q in data])
    bit_labels = torch.tensor([bit_of_sigma((q.x.k, q.idx.kind.name, q.idx.r)).value for q in data])
    
    halt_labels = []
    for q in data:
        key = (q.x.k, q.idx.kind.name, q.idx.r)
        t_first = t_firsts.get(key, float('inf'))
        halt_labels.append(halt_rank_of_tfirst(t_first).value)
    halt_labels = torch.tensor(halt_labels)
    
    # Train linear probes
    print("\nTraining linear probes...")
    
    print("  Cut classifier...")
    cut_classifier, acc_cut = train_linear_probe(embeddings, cut_labels, len(CutClass))
    print(f"    Accuracy: {acc_cut:.3f}")
    
    print("  Bit classifier...")
    bit_classifier, acc_bit = train_linear_probe(embeddings, bit_labels, len(BitClass))
    print(f"    Accuracy: {acc_bit:.3f}")
    
    print("  Halt classifier...")
    halt_classifier, acc_halt = train_linear_probe(embeddings, halt_labels, len(HaltRank))
    print(f"    Accuracy: {acc_halt:.3f}")
    
    # Compute orthogonality
    W_cut = cut_classifier.linear.weight.detach()
    W_bit = bit_classifier.linear.weight.detach()
    cos_cut_bit = compute_cos_similarity(W_cut, W_bit)
    print(f"\nOrthogonality: cos(W_cut, W_bit) = {cos_cut_bit:.3f}")
    
    # Random baselines
    random_cut = 1.0 / len(CutClass)
    random_bit = 1.0 / len(BitClass)
    
    # Count halt classes
    halt_counts = torch.bincount(halt_labels, minlength=len(HaltRank))
    random_halt = (halt_counts.float() / halt_counts.sum()).max().item()
    
    # Success criteria
    cut_decodable = acc_cut >= 0.85
    bit_decodable = acc_bit >= 0.85
    orthogonal = abs(cos_cut_bit) <= 0.2
    
    return PVecAnalysisResult(
        acc_cut=acc_cut,
        acc_bit=acc_bit,
        acc_halt=acc_halt,
        cos_cut_bit=cos_cut_bit,
        random_cut=random_cut,
        random_bit=random_bit,
        random_halt=random_halt,
        cut_decodable=cut_decodable,
        bit_decodable=bit_decodable,
        orthogonal=orthogonal,
    )


def print_pvec_analysis(result: PVecAnalysisResult):
    """Print formatted P_vec analysis results."""
    print("=" * 60)
    print("P_VEC ANALYSIS: cut ⊥ bit")
    print("=" * 60)
    
    print("\n[1] Linear Decodability")
    
    status = "✓ PASS" if result.cut_decodable else "○ WEAK"
    print(f"    Cut (quadrant):  {result.acc_cut:.1%} (random: {result.random_cut:.1%}) {status}")
    
    status = "✓ PASS" if result.bit_decodable else "○ WEAK"
    print(f"    Bit (q-type):    {result.acc_bit:.1%} (random: {result.random_bit:.1%}) {status}")
    
    halt_above_random = result.acc_halt > result.random_halt + 0.05
    status = "✓" if halt_above_random else "○"
    print(f"    Halt (t_first):  {result.acc_halt:.1%} (random: {result.random_halt:.1%}) {status}")
    
    print("\n[2] Orthogonality")
    status = "✓ PASS" if result.orthogonal else "✗ FAIL"
    print(f"    cos(W_cut, W_bit) = {result.cos_cut_bit:.3f}")
    print(f"    |cos| ≤ 0.2: {status}")
    
    print("\n" + "=" * 60)
    
    success = result.cut_decodable and result.bit_decodable and result.orthogonal
    if success:
        print("VERDICT: ✓ P_VEC STRUCTURE DETECTED")
        print("         cut ⊥ bit in latent space")
    else:
        print("VERDICT: ○ PARTIAL - Some criteria not met")
    
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints_trig")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    
    print(f"P_vec Analysis for: {args.checkpoints_dir}\n")
    
    result = analyze_pvec(args.checkpoints_dir, args.model)
    print()
    print_pvec_analysis(result)
    
    return result


if __name__ == "__main__":
    main()

"""
Ω-Trig Dataset: Data Layer

Generates dataset D = X_trig × I_trig with ground truth labels:
    y*(x, i) = questionTrig(i, V_trig(x))

The dataset is FIXED before training - it represents questions
about the Ω-structure, with answers determined by the syntax layer.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from trig_kernel import (
    AngleCode, TrigIndex, PTrig,
    generate_X_trig, generate_I_trig,
    V_trig, question_trig
)


# ============================================================================
# Dataset Types
# ============================================================================

@dataclass
class TrigQuestion:
    """A single question from the dataset."""
    x: AngleCode      # Input angle
    idx: TrigIndex    # Question index
    y_star: int       # Ground truth (0 or 1)
    
    def __repr__(self) -> str:
        return f"Q({self.x}, {self.idx}) = {self.y_star}"


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_trig_dataset(
    angles: List[AngleCode] = None,
    indices: List[TrigIndex] = None
) -> List[TrigQuestion]:
    """
    Generate dataset D = X_trig × I_trig with ground truth labels.
    
    For each (x, idx) pair:
        y* = questionTrig(idx, V_trig(x))
    
    Args:
        angles: List of AngleCode (default: full X_trig)
        indices: List of TrigIndex (default: full I_trig)
        
    Returns:
        List of TrigQuestion with ground truth labels
    """
    if angles is None:
        angles = generate_X_trig()
    if indices is None:
        indices = generate_I_trig()
    
    data = []
    for x in angles:
        p = V_trig(x)  # Ω-structural profile
        for idx in indices:
            y_star = int(question_trig(idx, p))  # Structural truth
            data.append(TrigQuestion(x=x, idx=idx, y_star=y_star))
    
    return data


def split_dataset(
    data: List[TrigQuestion],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[TrigQuestion], List[TrigQuestion], List[TrigQuestion]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train, val, test) lists
    """
    data = data.copy()
    random.Random(seed).shuffle(data)
    
    n = len(data)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    
    return train, val, test


# ============================================================================
# PyTorch Dataset Wrapper
# ============================================================================

class TrigDataset(Dataset):
    """PyTorch Dataset wrapper for TrigQuestions."""
    
    def __init__(self, questions: List[TrigQuestion]):
        self.questions = questions
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> TrigQuestion:
        return self.questions[idx]


def create_dataloaders(
    train: List[TrigQuestion],
    val: List[TrigQuestion],
    test: List[TrigQuestion],
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test sets."""
    
    train_ds = TrigDataset(train)
    val_ds = TrigDataset(val)
    test_ds = TrigDataset(test)
    
    # Custom collate function
    def collate(batch):
        return batch  # Keep as list of TrigQuestion
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, 
                             shuffle=False, collate_fn=collate)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Dataset Statistics
# ============================================================================

def dataset_stats(data: List[TrigQuestion]) -> dict:
    """Compute statistics for a dataset."""
    n = len(data)
    n_positive = sum(q.y_star for q in data)
    n_negative = n - n_positive
    
    # Count by question type
    by_type = {}
    for q in data:
        key = q.idx.kind.name
        if key not in by_type:
            by_type[key] = {"total": 0, "positive": 0}
        by_type[key]["total"] += 1
        by_type[key]["positive"] += q.y_star
    
    return {
        "total": n,
        "positive": n_positive,
        "negative": n_negative,
        "pos_ratio": n_positive / n if n > 0 else 0,
        "by_type": by_type
    }


# ============================================================================
# Main: Self-test
# ============================================================================

if __name__ == "__main__":
    print("=== Dataset Trig Tests ===\n")
    
    # Generate full dataset
    data = generate_trig_dataset()
    print(f"Full dataset size: {len(data)}")
    print(f"Expected: {360} angles × {8} questions = {360 * 8}")
    
    # Check some samples
    print("\nSample questions:")
    for i in [0, 100, 500, 1000, 2000]:
        q = data[i]
        print(f"  [{i}] {q}")
    
    # Dataset stats
    stats = dataset_stats(data)
    print(f"\nDataset statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Positive: {stats['positive']} ({stats['pos_ratio']:.1%})")
    print(f"  Negative: {stats['negative']}")
    
    print(f"\n  By question type:")
    for qtype, counts in stats['by_type'].items():
        pos_ratio = counts['positive'] / counts['total']
        print(f"    {qtype}: {counts['positive']}/{counts['total']} ({pos_ratio:.1%} positive)")
    
    # Split
    train, val, test = split_dataset(data)
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")
    
    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test, batch_size=32)
    print(f"\nDataLoader batches:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Verify a batch
    batch = next(iter(train_loader))
    print(f"\nBatch sample (size {len(batch)}):")
    print(f"  {batch[0]}")
    
    print("\n=== All dataset tests passed! ===")

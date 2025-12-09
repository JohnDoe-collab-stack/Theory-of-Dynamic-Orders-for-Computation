"""
Dataset for Ω_trig++

Generates training/validation/test splits from the logical formula kernel.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from trig_pp_kernel import Angle, N_ANGLES, N_FORMULAS, get_formula, get_formula_desc
from dynamic_kernel import DynamicTrigPPKernel, HaltRank


# ============================================================================
# Data Sample
# ============================================================================

@dataclass
class TrigPPSample:
    """Single sample from Ω_trig++."""
    theta: int          # Angle index 0..359
    q: int              # Formula index 0..15
    y_star: int         # Ground truth {0, 1}
    t_first: int        # Stabilization time (None → -1)
    halt_rank: int      # 0=EARLY, 1=MID, 2=LATE, 3=NEVER
    
    def angle(self) -> Angle:
        return Angle(self.theta)


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_dataset(T_max: int = 10) -> List[TrigPPSample]:
    """Generate complete dataset."""
    kernel = DynamicTrigPPKernel(T_max=T_max)
    data = kernel.generate_all_data()
    
    samples = []
    for d in data:
        samples.append(TrigPPSample(
            theta=d["theta"],
            q=d["q"],
            y_star=d["y_star"],
            t_first=d["t_first"] if d["t_first"] is not None else -1,
            halt_rank=d["halt_rank"],
        ))
    
    return samples


def split_dataset(samples: List[TrigPPSample], 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List, List, List]:
    """Split into train/val/test by angle (not random sample)."""
    random.seed(seed)
    
    # Split by angle to test generalization
    angles = list(range(N_ANGLES))
    random.shuffle(angles)
    
    n_train = int(N_ANGLES * train_ratio)
    n_val = int(N_ANGLES * val_ratio)
    
    train_angles = set(angles[:n_train])
    val_angles = set(angles[n_train:n_train + n_val])
    test_angles = set(angles[n_train + n_val:])
    
    train = [s for s in samples if s.theta in train_angles]
    val = [s for s in samples if s.theta in val_angles]
    test = [s for s in samples if s.theta in test_angles]
    
    return train, val, test


# ============================================================================
# PyTorch Dataset
# ============================================================================

class TrigPPDataset(Dataset):
    """PyTorch Dataset wrapper."""
    
    def __init__(self, samples: List[TrigPPSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        angle = s.angle()
        
        # Features: sin, cos, formula embedding (one-hot or index)
        x = torch.tensor([
            angle.sin(),
            angle.cos(),
            float(s.q) / N_FORMULAS,  # Normalized formula index
        ], dtype=torch.float32)
        
        y = torch.tensor(s.y_star, dtype=torch.float32)
        halt = torch.tensor(s.halt_rank, dtype=torch.long)
        
        return x, y, halt, s.q


def create_dataloaders(train: List, val: List, test: List, 
                       batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders."""
    train_ds = TrigPPDataset(train)
    val_ds = TrigPPDataset(val)
    test_ds = TrigPPDataset(test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Statistics
# ============================================================================

def dataset_stats(samples: List[TrigPPSample]) -> Dict:
    """Compute dataset statistics."""
    n = len(samples)
    
    y_counts = {0: 0, 1: 0}
    halt_counts = {h.value: 0 for h in HaltRank}
    q_counts = {q: 0 for q in range(N_FORMULAS)}
    
    for s in samples:
        y_counts[s.y_star] += 1
        halt_counts[s.halt_rank] += 1
        q_counts[s.q] += 1
    
    return {
        "n_samples": n,
        "y_balance": y_counts[1] / n,
        "y_counts": y_counts,
        "halt_counts": {HaltRank(h).name: c for h, c in halt_counts.items()},
        "q_counts": q_counts,
    }


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== Ω_trig++ Dataset Test ===\n")
    
    # Generate
    print("Generating dataset...")
    samples = generate_dataset()
    print(f"  Total: {len(samples)} samples")
    
    # Split
    train, val, test = split_dataset(samples)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Stats
    print("\nDataset statistics:")
    stats = dataset_stats(samples)
    print(f"  y* balance: {stats['y_balance']:.1%}")
    print(f"  Halt distribution: {stats['halt_counts']}")
    
    # DataLoader test
    print("\nDataLoader test:")
    train_loader, _, _ = create_dataloaders(train, val, test)
    batch = next(iter(train_loader))
    x, y, halt, q = batch
    print(f"  Batch x shape: {x.shape}")
    print(f"  Batch y shape: {y.shape}")
    print(f"  Batch halt shape: {halt.shape}")
    
    print("\n=== Test Complete ===")

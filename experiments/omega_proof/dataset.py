"""
Dataset for Ω_proof

Generates training/validation/test splits with OOD depth generalization.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from proof_kernel import (
    Formula, Atom, Not, And, Or, Implies,
    random_formula, is_tautology, is_satisfiable,
    ProofKernel, HaltRank, halt_rank_of_tfirst, QuestionType
)


# ============================================================================
# Data Sample
# ============================================================================

@dataclass
class ProofSample:
    """Single sample from Ω_proof."""
    formula_str: str       # String representation
    formula_enc: List[int] # Encoded formula (prefix notation)
    n_atoms: int
    depth: int
    q_type: int            # 0=TAUT, 1=SAT
    y_star: int            # Ground truth {0, 1}
    t_first: int           # Stabilization time
    halt_rank: int         # 0=EARLY, 1=MID, 2=LATE


# ============================================================================
# Formula Encoding
# ============================================================================

# Token vocabulary for prefix notation
VOCAB = {
    "PAD": 0,
    "ATOM_START": 1,  # Atoms are 1 + atom_idx
    "NOT": 10,
    "AND": 11,
    "OR": 12,
    "IMPLIES": 13,
}

MAX_FORMULA_LEN = 50
MAX_ATOMS = 5


def encode_formula(formula: Formula) -> List[int]:
    """Encode formula in prefix notation."""
    tokens = []
    
    def encode(f: Formula):
        if isinstance(f, Atom):
            # Extract atom index from name "p0", "p1", etc.
            idx = int(f.name[1:])
            tokens.append(VOCAB["ATOM_START"] + idx)
        elif isinstance(f, Not):
            tokens.append(VOCAB["NOT"])
            encode(f.f)
        elif isinstance(f, And):
            tokens.append(VOCAB["AND"])
            encode(f.left)
            encode(f.right)
        elif isinstance(f, Or):
            tokens.append(VOCAB["OR"])
            encode(f.left)
            encode(f.right)
        elif isinstance(f, Implies):
            tokens.append(VOCAB["IMPLIES"])
            encode(f.left)
            encode(f.right)
    
    encode(formula)
    
    # Pad to MAX_FORMULA_LEN
    while len(tokens) < MAX_FORMULA_LEN:
        tokens.append(VOCAB["PAD"])
    
    return tokens[:MAX_FORMULA_LEN]


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_samples(n_samples: int, n_atoms_range: Tuple[int, int],
                     depth_range: Tuple[int, int], seed: int = 42) -> List[ProofSample]:
    """Generate samples with specified complexity ranges."""
    random.seed(seed)
    kernel = ProofKernel()
    samples = []
    
    for i in range(n_samples):
        n_atoms = random.randint(*n_atoms_range)
        max_depth = random.randint(*depth_range)
        
        formula = random_formula(n_atoms, max_depth, seed=seed + i)
        
        # Choose question type
        q_type = random.choice([QuestionType.TAUT, QuestionType.SAT])
        
        if q_type == QuestionType.TAUT:
            y_star = int(is_tautology(formula))
            _, t_first = kernel.compute_t_first_taut(formula)
        else:
            y_star = int(is_satisfiable(formula))
            _, t_first = kernel.compute_t_first_sat(formula)
        
        actual_n = len(formula.atoms())
        hr = halt_rank_of_tfirst(t_first, actual_n)
        
        samples.append(ProofSample(
            formula_str=str(formula),
            formula_enc=encode_formula(formula),
            n_atoms=actual_n,
            depth=formula.depth(),
            q_type=q_type.value,
            y_star=y_star,
            t_first=t_first,
            halt_rank=hr.value,
        ))
    
    return samples


def generate_dataset(seed: int = 42) -> Tuple[List[ProofSample], List[ProofSample], List[ProofSample]]:
    """
    Generate train/val/test with OOD depth split.
    
    Train/val: depth 1-3, 2-3 atoms (easy)
    Test OOD: depth 3-5, 3-4 atoms (harder)
    """
    random.seed(seed)
    
    # Train: easy formulas
    train_samples = generate_samples(3000, (2, 3), (1, 3), seed)
    
    # Val: same distribution
    val_samples = generate_samples(500, (2, 3), (1, 3), seed + 1000)
    
    # Test OOD: harder formulas
    test_samples = generate_samples(1000, (3, 4), (3, 5), seed + 2000)
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


# ============================================================================
# PyTorch Dataset
# ============================================================================

class ProofDataset(Dataset):
    """PyTorch Dataset wrapper."""
    
    def __init__(self, samples: List[ProofSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        formula = torch.tensor(s.formula_enc, dtype=torch.long)
        q_type = torch.tensor(s.q_type, dtype=torch.long)
        y = torch.tensor(s.y_star, dtype=torch.float32)
        halt = torch.tensor(s.halt_rank, dtype=torch.long)
        depth = torch.tensor(s.depth, dtype=torch.long)
        
        return formula, q_type, depth, y, halt


def create_dataloaders(train: List, val: List, test: List,
                       batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders."""
    train_ds = ProofDataset(train)
    val_ds = ProofDataset(val)
    test_ds = ProofDataset(test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Statistics
# ============================================================================

def dataset_stats(samples: List[ProofSample], name: str = ""):
    """Print dataset statistics."""
    n = len(samples)
    if n == 0:
        print(f"{name}: empty")
        return
    
    y_counts = {0: 0, 1: 0}
    halt_counts = {h.value: 0 for h in HaltRank}
    depth_counts = {}
    
    for s in samples:
        y_counts[s.y_star] += 1
        halt_counts[s.halt_rank] += 1
        depth_counts[s.depth] = depth_counts.get(s.depth, 0) + 1
    
    print(f"\n{name} ({n} samples):")
    print(f"  Y balance: {y_counts[1]/n:.1%}")
    print(f"  Halt: {', '.join(f'{HaltRank(h).name}:{c}' for h, c in halt_counts.items())}")
    print(f"  Depths: {depth_counts}")


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== Ω_proof Dataset Test ===")
    
    train, val, test = generate_dataset()
    
    dataset_stats(train, "Train")
    dataset_stats(val, "Val")
    dataset_stats(test, "Test (OOD)")
    
    # DataLoader test
    print("\nDataLoader test:")
    train_loader, _, _ = create_dataloaders(train, val, test)
    formula, q_type, depth, y, halt = next(iter(train_loader))
    print(f"  formula shape: {formula.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  halt shape: {halt.shape}")
    
    print("\n=== Test Complete ===")

"""
Dataset for Ω_arith

Generates training/validation/test splits with OOD length generalization.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from arith_kernel import (
    Number, Question, QuestionType, HaltRank,
    halt_rank_of_tfirst, generate_questions
)


# ============================================================================
# Data Sample
# ============================================================================

@dataclass
class ArithSample:
    """Single sample from Ω_arith."""
    a: int              # First operand
    b: int              # Second operand
    n_digits: int       # Number of digits
    q_type: int         # Question type enum value
    q_param1: int       # Question param1
    q_param2: int       # Question param2
    y_star: int         # Ground truth {0, 1}
    t_first: int        # Stabilization time
    halt_rank: int      # 0=EARLY, 1=MID, 2=LATE, 3=NEVER
    
    def to_numbers(self) -> Tuple[Number, Number]:
        return Number.from_int(self.a, self.n_digits), Number.from_int(self.b, self.n_digits)
    
    def get_question(self) -> Question:
        return Question(QuestionType(self.q_type), self.q_param1, self.q_param2)


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_samples_for_length(n_digits: int, n_samples: int, seed: int = 42) -> List[ArithSample]:
    """Generate samples for a specific number of digits."""
    random.seed(seed)
    samples = []
    
    # Value range for n-digit numbers
    min_val = 10 ** (n_digits - 1)
    max_val = 10 ** n_digits - 1
    
    # Questions for this length
    questions = generate_questions(n_digits)
    
    for _ in range(n_samples):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        
        q = random.choice(questions)
        a_num = Number.from_int(a, n_digits)
        b_num = Number.from_int(b, n_digits)
        
        y_star = int(q.answer(a_num, b_num))
        t_first = q.compute_t_first(a_num, b_num)
        hr = halt_rank_of_tfirst(t_first, n_digits)
        
        samples.append(ArithSample(
            a=a,
            b=b,
            n_digits=n_digits,
            q_type=q.q_type.value,
            q_param1=q.param1,
            q_param2=q.param2,
            y_star=y_star,
            t_first=t_first,
            halt_rank=hr.value,
        ))
    
    return samples


def generate_dataset(
    train_lengths: List[int] = [2, 3, 4],
    test_lengths: List[int] = [5, 6],
    samples_per_length: int = 2000,
    seed: int = 42
) -> Tuple[List[ArithSample], List[ArithSample], List[ArithSample]]:
    """
    Generate train/val/test with OOD length split.
    
    Train/val: short sequences (n=2,3,4)
    Test: longer sequences (n=5,6) for OOD generalization
    """
    random.seed(seed)
    
    # Train + val from short lengths
    train_val_samples = []
    for n in train_lengths:
        train_val_samples.extend(
            generate_samples_for_length(n, samples_per_length, seed + n)
        )
    
    # Shuffle and split
    random.shuffle(train_val_samples)
    n_train = int(len(train_val_samples) * 0.85)
    train = train_val_samples[:n_train]
    val = train_val_samples[n_train:]
    
    # Test from longer lengths (OOD)
    test = []
    for n in test_lengths:
        test.extend(
            generate_samples_for_length(n, samples_per_length // 2, seed + n + 100)
        )
    random.shuffle(test)
    
    return train, val, test


# ============================================================================
# PyTorch Dataset
# ============================================================================

MAX_DIGITS = 8  # Max sequence length

class ArithDataset(Dataset):
    """PyTorch Dataset wrapper."""
    
    def __init__(self, samples: List[ArithSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        a_num, b_num = s.to_numbers()
        
        # Pad to MAX_DIGITS
        a_digits = a_num.digits + [0] * (MAX_DIGITS - len(a_num.digits))
        b_digits = b_num.digits + [0] * (MAX_DIGITS - len(b_num.digits))
        
        # Features: digit embeddings will be learned
        a_tensor = torch.tensor(a_digits, dtype=torch.long)
        b_tensor = torch.tensor(b_digits, dtype=torch.long)
        
        # Question encoding
        q_type = torch.tensor(s.q_type, dtype=torch.long)
        q_param1 = torch.tensor(s.q_param1, dtype=torch.long)
        n_digits = torch.tensor(s.n_digits, dtype=torch.long)
        
        y = torch.tensor(s.y_star, dtype=torch.float32)
        halt = torch.tensor(s.halt_rank, dtype=torch.long)
        
        return a_tensor, b_tensor, q_type, q_param1, n_digits, y, halt


def create_dataloaders(train: List, val: List, test: List,
                       batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders."""
    train_ds = ArithDataset(train)
    val_ds = ArithDataset(val)
    test_ds = ArithDataset(test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Statistics
# ============================================================================

def dataset_stats(samples: List[ArithSample], name: str = ""):
    """Print dataset statistics."""
    n = len(samples)
    if n == 0:
        print(f"{name}: empty")
        return
    
    y_counts = {0: 0, 1: 0}
    halt_counts = {h.value: 0 for h in HaltRank}
    length_counts = {}
    qtype_counts = {qt.value: 0 for qt in QuestionType}
    
    for s in samples:
        y_counts[s.y_star] += 1
        halt_counts[s.halt_rank] += 1
        length_counts[s.n_digits] = length_counts.get(s.n_digits, 0) + 1
        qtype_counts[s.q_type] += 1
    
    print(f"\n{name} ({n} samples):")
    print(f"  Y balance: {y_counts[1]/n:.1%}")
    print(f"  Halt: {', '.join(f'{HaltRank(h).name}:{c}' for h, c in halt_counts.items())}")
    print(f"  Lengths: {length_counts}")
    print(f"  Q types: {', '.join(f'{QuestionType(q).name}:{c}' for q, c in qtype_counts.items())}")


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== Ω_arith Dataset Test ===")
    
    train, val, test = generate_dataset()
    
    dataset_stats(train, "Train")
    dataset_stats(val, "Val")
    dataset_stats(test, "Test (OOD)")
    
    # DataLoader test
    print("\nDataLoader test:")
    train_loader, _, _ = create_dataloaders(train, val, test)
    a, b, q_type, q_param1, n_digits, y, halt = next(iter(train_loader))
    print(f"  a shape: {a.shape}")
    print(f"  b shape: {b.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  halt shape: {halt.shape}")
    
    print("\n=== Test Complete ===")

"""
Multi-Task Training for T_θ

T_θ predicts BOTH:
- y*(σ): ground truth from Ω (main task)
- halt_rank_K(σ): difficulty class from K (auxiliary task)

Loss = BCE(y*, y_hat) + λ · CE(halt_rank_K, halt_logits)

This tests if K's structure provides useful learning signal.
"""

import json
import os
import sys
import logging
import math
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trig_kernel import AngleCode, TrigIndex, TrigIndexType
from dataset_trig import TrigQuestion, generate_trig_dataset, split_dataset
from pvec_trig import HaltRank, halt_rank_of_tfirst


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MultiTaskConfig:
    """Configuration for multi-task training."""
    epochs: int = 60
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 42
    
    # Model architecture
    angle_embed_dim: int = 16
    index_embed_dim: int = 8
    hidden_dim: int = 64
    
    # Multi-task
    lambda_halt: float = 0.5  # Weight for halt prediction loss
    n_halt_classes: int = 4  # EARLY, MID, LATE, NEVER
    
    # Ablation
    shuffle_K: bool = False  # Randomize halt_rank assignments
    
    # Checkpoints
    checkpoint_epochs: List[int] = None
    
    # Output
    output_dir: str = "checkpoints_multitask"
    t_first_K_path: str = "checkpoints_trig/t_first_K.json"
    
    def __post_init__(self):
        if self.checkpoint_epochs is None:
            self.checkpoint_epochs = [0, 1, 5, 10, 20, 50]


# ============================================================================
# Multi-Task Model
# ============================================================================

class MultiTaskTheoryModel(nn.Module):
    """
    Theory T_θ with two heads:
    - y_head: predicts y*(σ) ∈ {0,1}
    - halt_head: predicts halt_rank_K(σ) ∈ {EARLY, MID, LATE, NEVER}
    """
    
    def __init__(
        self,
        angle_embed_dim: int = 16,
        index_embed_dim: int = 8,
        hidden_dim: int = 64,
        n_halt_classes: int = 4,
    ):
        super().__init__()
        self.angle_embed_dim = angle_embed_dim
        self.index_embed_dim = index_embed_dim
        self.hidden_dim = hidden_dim
        self.n_halt_classes = n_halt_classes
        
        # Angle encoder (same as before)
        self.angle_proj = nn.Linear(2, angle_embed_dim)  # sin(θ), cos(θ) -> embed
        
        # Index encoder
        self.type_embedding = nn.Embedding(4, index_embed_dim // 2)  # 4 types
        self.threshold_proj = nn.Linear(1, index_embed_dim // 2)
        
        # Shared MLP for representation
        input_dim = angle_embed_dim + index_embed_dim
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Y head (binary classification)
        self.y_head = nn.Linear(hidden_dim, 1)
        
        # Halt head (4-class classification)
        self.halt_head = nn.Linear(hidden_dim, n_halt_classes)
    
    def encode_angle(self, x: AngleCode) -> torch.Tensor:
        import math
        theta = x.to_radians()
        sin_cos = torch.tensor([math.sin(theta), math.cos(theta)], dtype=torch.float32)
        return self.angle_proj(sin_cos)
    
    def encode_index(self, idx: TrigIndex) -> torch.Tensor:
        type_idx = {
            TrigIndexType.SIGN_SIN: 0,
            TrigIndexType.SIGN_COS: 1,
            TrigIndexType.SIN_GE: 2,
            TrigIndexType.COS_GE: 3,
        }[idx.kind]
        
        type_vec = self.type_embedding(torch.tensor(type_idx))
        r = idx.r if idx.r is not None else 0.0
        r_vec = self.threshold_proj(torch.tensor([r], dtype=torch.float32))
        
        return torch.cat([type_vec, r_vec])
    
    def forward_single(self, x: AngleCode, idx: TrigIndex):
        """Forward for single sample, returns (y_prob, halt_logits, embedding)."""
        angle_embed = self.encode_angle(x)
        index_embed = self.encode_index(idx)
        combined = torch.cat([angle_embed, index_embed])
        
        h = self.shared_mlp(combined)
        
        y_logit = self.y_head(h)
        y_prob = torch.sigmoid(y_logit).squeeze()
        
        halt_logits = self.halt_head(h)
        
        return y_prob, halt_logits, h
    
    def forward_batch(self, angles: List[AngleCode], indices: List[TrigIndex]):
        """Forward for batch, returns (y_probs, halt_logits, embeddings)."""
        batch_size = len(angles)
        
        # Encode batch
        angle_embeds = torch.stack([self.encode_angle(x) for x in angles])
        index_embeds = torch.stack([self.encode_index(idx) for idx in indices])
        combined = torch.cat([angle_embeds, index_embeds], dim=1)
        
        h = self.shared_mlp(combined)
        
        y_logits = self.y_head(h).squeeze(-1)
        y_probs = torch.sigmoid(y_logits)
        
        halt_logits = self.halt_head(h)
        
        return y_probs, halt_logits, h
    
    def forward(self, angles, indices):
        """Dispatch to single or batch."""
        if isinstance(angles, list):
            return self.forward_batch(angles, indices)
        else:
            return self.forward_single(angles, indices)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Data Loading
# ============================================================================

def load_t_first_K(path: str) -> Dict[Tuple, float]:
    """Load t_first^K mapping from JSON."""
    with open(path) as f:
        records = json.load(f)
    
    t_first_K = {}
    for r in records:
        key = (r["angle"], r["kind"], r["threshold"])
        t_first = r["t_first"]
        if t_first is None:
            t_first = float('inf')
        t_first_K[key] = t_first
    
    return t_first_K


def shuffle_t_first_K(t_first_K: Dict[Tuple, float], seed: int = 42) -> Dict[Tuple, float]:
    """Shuffle t_first^K values to create a random control."""
    random.seed(seed)
    keys = list(t_first_K.keys())
    vals = list(t_first_K.values())
    random.shuffle(vals)
    return {k: v for k, v in zip(keys, vals)}


def halt_rank_to_int(hr: HaltRank) -> int:
    """Convert HaltRank to integer class."""
    return {HaltRank.EARLY: 0, HaltRank.MID: 1, HaltRank.LATE: 2, HaltRank.NEVER: 3}[hr]


@dataclass
class MultiTaskSample:
    """Sample with both y* and halt_rank_K."""
    x: AngleCode
    idx: TrigIndex
    y_star: int
    halt_rank: int  # 0=EARLY, 1=MID, 2=LATE, 3=NEVER


def prepare_multitask_data(data: List[TrigQuestion], 
                           t_first_K: Dict[Tuple, float]) -> List[MultiTaskSample]:
    """Prepare dataset with halt labels."""
    samples = []
    for q in data:
        key = (q.x.k, q.idx.kind.name, q.idx.r)
        t_first = t_first_K.get(key, float('inf'))
        halt_rank = halt_rank_to_int(halt_rank_of_tfirst(t_first))
        samples.append(MultiTaskSample(q.x, q.idx, q.y_star, halt_rank))
    return samples


# ============================================================================
# Logging
# ============================================================================

def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"multitask_{timestamp}.log")
    
    logger = logging.getLogger("multitask")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return log_file


def log(msg: str):
    logger = logging.getLogger("multitask")
    logger.info(msg)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: MultiTaskTheoryModel,
                data: List[MultiTaskSample],
                optimizer: optim.Optimizer,
                config: MultiTaskConfig) -> Tuple[float, float, float]:
    """Train one epoch. Returns (total_loss, y_loss, halt_loss)."""
    model.train()
    total_loss = 0.0
    total_y_loss = 0.0
    total_halt_loss = 0.0
    n_samples = 0
    
    samples = data.copy()
    random.shuffle(samples)
    
    for i in range(0, len(samples), config.batch_size):
        batch = samples[i:i + config.batch_size]
        
        angles = [s.x for s in batch]
        indices = [s.idx for s in batch]
        y_labels = torch.tensor([s.y_star for s in batch], dtype=torch.float32)
        halt_labels = torch.tensor([s.halt_rank for s in batch], dtype=torch.long)
        
        optimizer.zero_grad()
        y_probs, halt_logits, _ = model(angles, indices)
        
        # Y loss (BCE)
        y_loss = F.binary_cross_entropy(y_probs, y_labels)
        
        # Halt loss (CE)
        halt_loss = F.cross_entropy(halt_logits, halt_labels)
        
        # Combined loss
        loss = y_loss + config.lambda_halt * halt_loss
        
        loss.backward()
        optimizer.step()
        
        batch_size = len(batch)
        total_loss += loss.item() * batch_size
        total_y_loss += y_loss.item() * batch_size
        total_halt_loss += halt_loss.item() * batch_size
        n_samples += batch_size
    
    return (total_loss / n_samples, 
            total_y_loss / n_samples, 
            total_halt_loss / n_samples)


def evaluate(model: MultiTaskTheoryModel,
             data: List[MultiTaskSample]) -> Tuple[float, float, List, List]:
    """Evaluate model. Returns (y_acc, halt_acc, E_val, correct_halts)."""
    model.eval()
    y_correct = 0
    halt_correct = 0
    E_val = []
    correct_halts = []
    
    with torch.no_grad():
        for s in data:
            y_prob, halt_logits, _ = model(s.x, s.idx)
            
            # Y prediction
            y_hat = 1 if y_prob.item() >= 0.5 else 0
            if y_hat == s.y_star:
                y_correct += 1
                E_val.append((s.x.k, s.idx.kind.name, s.idx.r))
            
            # Halt prediction
            halt_pred = halt_logits.argmax().item()
            if halt_pred == s.halt_rank:
                halt_correct += 1
                correct_halts.append((s.x.k, s.idx.kind.name, s.idx.r))
    
    y_acc = y_correct / len(data) if data else 0
    halt_acc = halt_correct / len(data) if data else 0
    
    return y_acc, halt_acc, E_val, correct_halts


# ============================================================================
# Main Training Loop
# ============================================================================

def train_multitask(config: MultiTaskConfig) -> Dict:
    """Full multi-task training."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    log_file = setup_logging(config.output_dir)
    log(f"=== Multi-Task Training (λ_halt={config.lambda_halt}) ===")
    log(f"Log: {log_file}")
    log(f"Config: {config}")
    log("")
    
    # Load t_first^K
    log(f"Loading t_first^K from {config.t_first_K_path}...")
    t_first_K = load_t_first_K(config.t_first_K_path)
    log(f"  Loaded {len(t_first_K)} difficulty values")
    
    if config.shuffle_K:
        log("  SHUFFLING t_first^K (random control)")
        t_first_K = shuffle_t_first_K(t_first_K, seed=config.seed + 1)
    
    # Generate dataset
    log("Generating dataset...")
    full_data = generate_trig_dataset()
    train_data, val_data, test_data = split_dataset(full_data, seed=config.seed)
    
    # Prepare multi-task samples
    train_samples = prepare_multitask_data(train_data, t_first_K)
    val_samples = prepare_multitask_data(val_data, t_first_K)
    test_samples = prepare_multitask_data(test_data, t_first_K)
    
    # Count halt distribution
    halt_counts = {}
    for s in train_samples:
        halt_counts[s.halt_rank] = halt_counts.get(s.halt_rank, 0) + 1
    log(f"  Train: {len(train_samples)}, halt dist: {halt_counts}")
    
    # Create model
    model = MultiTaskTheoryModel(
        angle_embed_dim=config.angle_embed_dim,
        index_embed_dim=config.index_embed_dim,
        hidden_dim=config.hidden_dim,
        n_halt_classes=config.n_halt_classes,
    )
    log(f"Model: {count_parameters(model):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    os.makedirs(config.output_dir, exist_ok=True)
    
    results = {"checkpoints": {}}
    max_epoch = max(config.checkpoint_epochs) + 1
    
    for epoch in range(max_epoch):
        if epoch > 0:
            loss, y_loss, halt_loss = train_epoch(model, train_samples, optimizer, config)
        else:
            loss = y_loss = halt_loss = float('inf')
        
        if epoch in config.checkpoint_epochs:
            y_acc_train, halt_acc_train, _, _ = evaluate(model, train_samples)
            y_acc_val, halt_acc_val, E_val, _ = evaluate(model, val_samples)
            
            log(f"Epoch {epoch:3d} | Loss: {loss:.4f} (y:{y_loss:.4f}, h:{halt_loss:.4f}) | "
                f"Y: {y_acc_val:.3f} | Halt: {halt_acc_val:.3f}")
            
            results["checkpoints"][epoch] = {
                "y_acc_train": y_acc_train,
                "y_acc_val": y_acc_val,
                "halt_acc_train": halt_acc_train,
                "halt_acc_val": halt_acc_val,
                "E_val_size": len(E_val),
            }
            
            # Save checkpoint
            torch.save(model.state_dict(), 
                       os.path.join(config.output_dir, f"epoch_{epoch:03d}.pt"))
            
            # Save JSON
            with open(os.path.join(config.output_dir, f"epoch_{epoch:03d}.json"), 'w') as f:
                json.dump(results["checkpoints"][epoch], f, indent=2)
    
    # Final test
    y_acc_test, halt_acc_test, E_test, _ = evaluate(model, test_samples)
    log(f"\nFinal Test: Y={y_acc_test:.3f}, Halt={halt_acc_test:.3f}")
    
    results["test_y_accuracy"] = y_acc_test
    results["test_halt_accuracy"] = halt_acc_test
    
    # Save results
    with open(os.path.join(config.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(config.output_dir, "final.pt"))
    
    log("")
    log("=== Multi-Task Training Complete ===")
    log(f"Checkpoints saved to: {config.output_dir}/")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-task training: y* + halt_rank")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="checkpoints_multitask")
    parser.add_argument("--t-first-K", type=str, default="checkpoints_trig/t_first_K.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-halt", type=float, default=0.5, 
                        help="Weight for halt prediction loss")
    parser.add_argument("--shuffle-K", action="store_true",
                        help="Shuffle halt labels (ablation)")
    
    args = parser.parse_args()
    
    config = MultiTaskConfig(
        epochs=args.epochs,
        output_dir=args.output_dir,
        t_first_K_path=args.t_first_K,
        seed=args.seed,
        lambda_halt=args.lambda_halt,
        shuffle_K=args.shuffle_K,
    )
    
    results = train_multitask(config)
    return results


if __name__ == "__main__":
    main()

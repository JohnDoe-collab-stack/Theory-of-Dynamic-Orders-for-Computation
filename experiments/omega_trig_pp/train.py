"""
Multi-Task Training for Ω_trig++

T_θ predicts both:
- y*(θ, q): ground truth formula evaluation
- halt_rank(θ, q): difficulty from K

Loss = BCE(y*) + λ · CE(halt_rank)
"""

import json
import os
import sys
import random
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from trig_pp_kernel import N_ANGLES, N_FORMULAS
from dataset import (
    generate_dataset, split_dataset, create_dataloaders,
    dataset_stats, TrigPPSample
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 42
    
    # Model
    hidden_dim: int = 128
    n_layers: int = 3
    formula_embed_dim: int = 16
    
    # Multi-task
    lambda_halt: float = 0.5
    n_halt_classes: int = 4
    
    # Ablation
    shuffle_K: bool = False
    
    # Output
    output_dir: str = "checkpoints"
    checkpoint_epochs: List[int] = None
    
    def __post_init__(self):
        if self.checkpoint_epochs is None:
            self.checkpoint_epochs = [0, 5, 10, 25, 50, 100]


# ============================================================================
# Model
# ============================================================================

class TrigPPModel(nn.Module):
    """
    Multi-task model for Ω_trig++.
    
    Input: (sin(θ), cos(θ), q)
    Output: y_logit, halt_logits
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        
        # Formula embedding
        self.formula_embed = nn.Embedding(N_FORMULAS, config.formula_embed_dim)
        
        # Shared encoder
        input_dim = 2 + config.formula_embed_dim  # sin, cos, formula_embed
        
        layers = []
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(config.n_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
        # Y head (binary)
        self.y_head = nn.Linear(config.hidden_dim, 1)
        
        # Halt head (4-class)
        self.halt_head = nn.Linear(config.hidden_dim, config.n_halt_classes)
    
    def forward(self, sin_cos: torch.Tensor, q: torch.Tensor):
        """
        Args:
            sin_cos: (batch, 2) - sin(θ), cos(θ)
            q: (batch,) - formula indices
        
        Returns:
            y_logits: (batch,)
            halt_logits: (batch, 4)
            embeddings: (batch, hidden_dim)
        """
        # Get formula embeddings
        q_embed = self.formula_embed(q)  # (batch, formula_embed_dim)
        
        # Concatenate
        x = torch.cat([sin_cos, q_embed], dim=1)  # (batch, 2 + embed_dim)
        
        # Encode
        h = self.encoder(x)  # (batch, hidden_dim)
        
        # Heads
        y_logits = self.y_head(h).squeeze(-1)  # (batch,)
        halt_logits = self.halt_head(h)  # (batch, 4)
        
        return y_logits, halt_logits, h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: TrigPPModel, 
                loader: DataLoader,
                optimizer: optim.Optimizer,
                config: TrainConfig) -> Tuple[float, float, float]:
    """Train one epoch. Returns (loss, y_loss, halt_loss)."""
    model.train()
    total_loss = 0
    total_y_loss = 0
    total_halt_loss = 0
    n_samples = 0
    
    for x, y, halt, q in loader:
        sin_cos = x[:, :2]
        
        optimizer.zero_grad()
        y_logits, halt_logits, _ = model(sin_cos, q)
        
        # Y loss
        y_loss = F.binary_cross_entropy_with_logits(y_logits, y)
        
        # Halt loss
        halt_loss = F.cross_entropy(halt_logits, halt)
        
        # Combined
        loss = y_loss + config.lambda_halt * halt_loss
        
        loss.backward()
        optimizer.step()
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_y_loss += y_loss.item() * batch_size
        total_halt_loss += halt_loss.item() * batch_size
        n_samples += batch_size
    
    return (total_loss / n_samples,
            total_y_loss / n_samples,
            total_halt_loss / n_samples)


def evaluate(model: TrigPPModel, loader: DataLoader) -> Tuple[float, float]:
    """Evaluate model. Returns (y_acc, halt_acc)."""
    model.eval()
    y_correct = 0
    halt_correct = 0
    n_samples = 0
    
    with torch.no_grad():
        for x, y, halt, q in loader:
            sin_cos = x[:, :2]
            y_logits, halt_logits, _ = model(sin_cos, q)
            
            # Y prediction
            y_pred = (torch.sigmoid(y_logits) >= 0.5).float()
            y_correct += (y_pred == y).sum().item()
            
            # Halt prediction
            halt_pred = halt_logits.argmax(dim=1)
            halt_correct += (halt_pred == halt).sum().item()
            
            n_samples += x.size(0)
    
    return y_correct / n_samples, halt_correct / n_samples


# ============================================================================
# Logging
# ============================================================================

def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    
    logger = logging.getLogger("train_pp")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return log_file


def log(msg: str):
    logger = logging.getLogger("train_pp")
    logger.info(msg)


# ============================================================================
# Shuffle K (Ablation)
# ============================================================================

def shuffle_halt_ranks(samples: List[TrigPPSample], seed: int) -> List[TrigPPSample]:
    """Shuffle halt_rank assignments for ablation."""
    random.seed(seed)
    halt_ranks = [s.halt_rank for s in samples]
    random.shuffle(halt_ranks)
    
    shuffled = []
    for s, hr in zip(samples, halt_ranks):
        shuffled.append(TrigPPSample(
            theta=s.theta,
            q=s.q,
            y_star=s.y_star,
            t_first=s.t_first,
            halt_rank=hr,
        ))
    
    return shuffled


# ============================================================================
# Main Training
# ============================================================================

def train(config: TrainConfig) -> Dict:
    """Full training loop."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    log_file = setup_logging(config.output_dir)
    log(f"=== Ω_trig++ Multi-Task Training (λ={config.lambda_halt}) ===")
    log(f"Config: {config}")
    log("")
    
    # Generate data
    log("Generating dataset...")
    samples = generate_dataset()
    
    if config.shuffle_K:
        log("SHUFFLING halt_rank assignments (ablation control)")
        samples = shuffle_halt_ranks(samples, config.seed + 1)
    
    train_samples, val_samples, test_samples = split_dataset(samples, seed=config.seed)
    
    log(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    stats = dataset_stats(train_samples)
    log(f"  Y balance: {stats['y_balance']:.1%}")
    log(f"  Halt dist: {stats['halt_counts']}")
    
    # Create loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, config.batch_size
    )
    
    # Model
    model = TrigPPModel(config)
    log(f"Model: {count_parameters(model):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    results = {"checkpoints": {}}
    
    for epoch in range(config.epochs + 1):
        if epoch > 0:
            loss, y_loss, h_loss = train_epoch(model, train_loader, optimizer, config)
        else:
            loss = y_loss = h_loss = float('inf')
        
        if epoch in config.checkpoint_epochs or epoch == config.epochs:
            y_acc_train, h_acc_train = evaluate(model, train_loader)
            y_acc_val, h_acc_val = evaluate(model, val_loader)
            
            log(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"Y: {y_acc_val:.3f} | Halt: {h_acc_val:.3f}")
            
            results["checkpoints"][epoch] = {
                "y_acc_train": y_acc_train,
                "y_acc_val": y_acc_val,
                "halt_acc_train": h_acc_train,
                "halt_acc_val": h_acc_val,
            }
            
            torch.save(model.state_dict(),
                       os.path.join(config.output_dir, f"epoch_{epoch:03d}.pt"))
    
    # Final test
    y_acc_test, h_acc_test = evaluate(model, test_loader)
    log(f"\nFinal Test: Y={y_acc_test:.3f}, Halt={h_acc_test:.3f}")
    
    results["test_y_accuracy"] = y_acc_test
    results["test_halt_accuracy"] = h_acc_test
    
    with open(os.path.join(config.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(config.output_dir, "final.pt"))
    
    log("\n=== Training Complete ===")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lambda-halt", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--shuffle-K", action="store_true")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        lambda_halt=args.lambda_halt,
        seed=args.seed,
        output_dir=args.output_dir,
        shuffle_K=args.shuffle_K,
    )
    
    train(config)


if __name__ == "__main__":
    main()

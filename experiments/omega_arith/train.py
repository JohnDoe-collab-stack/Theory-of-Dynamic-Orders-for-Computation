"""
Multi-Task Training for Ω_arith

T_θ predicts both:
- y*(a, b, q): ground truth answer
- halt_rank(a, b, q): difficulty from K

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

from dataset import (
    generate_dataset, create_dataloaders, dataset_stats,
    MAX_DIGITS, ArithSample
)
from arith_kernel import QuestionType


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
    digit_embed_dim: int = 16
    hidden_dim: int = 128
    n_layers: int = 3
    
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
            self.checkpoint_epochs = [0, 10, 25, 50, 100]


# ============================================================================
# Model
# ============================================================================

class ArithModel(nn.Module):
    """
    Multi-task model for Ω_arith.
    
    Input: (a_digits, b_digits, q_type, q_param, n_digits)
    Output: y_logit, halt_logits
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        
        # Digit embeddings (0-9)
        self.digit_embed = nn.Embedding(10, config.digit_embed_dim)
        
        # Question type embedding
        self.q_type_embed = nn.Embedding(len(QuestionType), config.digit_embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Embedding(MAX_DIGITS, config.digit_embed_dim)
        
        # Encoder for digit sequences
        # We'll aggregate digits with attention-like pooling
        input_dim = config.digit_embed_dim * 3  # a_embed + b_embed + pos
        self.digit_encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Aggregation over positions
        self.attention = nn.Linear(config.hidden_dim, 1)
        
        # Question encoder
        q_input_dim = config.digit_embed_dim + MAX_DIGITS  # q_type_embed + param one-hot
        self.q_encoder = nn.Linear(q_input_dim, config.hidden_dim)
        
        # Combined MLP
        combined_dim = config.hidden_dim * 2  # digits + question
        layers = []
        layers.append(nn.Linear(combined_dim, config.hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(config.n_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        # Heads
        self.y_head = nn.Linear(config.hidden_dim, 1)
        self.halt_head = nn.Linear(config.hidden_dim, config.n_halt_classes)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor, 
                q_type: torch.Tensor, q_param: torch.Tensor,
                n_digits: torch.Tensor):
        """
        Args:
            a: (batch, MAX_DIGITS) digit sequences
            b: (batch, MAX_DIGITS) digit sequences
            q_type: (batch,) question type indices
            q_param: (batch,) question param1
            n_digits: (batch,) actual length of numbers
        
        Returns:
            y_logits: (batch,)
            halt_logits: (batch, 4)
            embeddings: (batch, hidden_dim)
        """
        batch_size = a.size(0)
        
        # Position indices
        positions = torch.arange(MAX_DIGITS, device=a.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed digits
        a_embed = self.digit_embed(a)  # (batch, max_digits, embed_dim)
        b_embed = self.digit_embed(b)
        pos_embed = self.pos_embed(positions)
        
        # Combine per-position
        digit_features = torch.cat([a_embed, b_embed, pos_embed], dim=-1)  # (batch, max_digits, 3*embed)
        digit_encoded = self.digit_encoder(digit_features)  # (batch, max_digits, hidden)
        
        # Attention pooling
        attn_weights = F.softmax(self.attention(digit_encoded), dim=1)  # (batch, max_digits, 1)
        digit_pooled = (attn_weights * digit_encoded).sum(dim=1)  # (batch, hidden)
        
        # Question encoding
        q_type_embed = self.q_type_embed(q_type)  # (batch, embed_dim)
        
        # q_param as one-hot (position in [0, MAX_DIGITS))
        q_param_onehot = F.one_hot(q_param.clamp(0, MAX_DIGITS - 1), MAX_DIGITS).float()
        q_features = torch.cat([q_type_embed, q_param_onehot], dim=-1)
        q_encoded = self.q_encoder(q_features)  # (batch, hidden)
        
        # Combine and MLP
        combined = torch.cat([digit_pooled, q_encoded], dim=-1)
        h = self.mlp(combined)
        
        # Heads
        y_logits = self.y_head(h).squeeze(-1)
        halt_logits = self.halt_head(h)
        
        return y_logits, halt_logits, h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model: ArithModel,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                config: TrainConfig) -> Tuple[float, float, float]:
    """Train one epoch. Returns (loss, y_loss, halt_loss)."""
    model.train()
    total_loss = 0
    total_y_loss = 0
    total_halt_loss = 0
    n_samples = 0
    
    for a, b, q_type, q_param, n_digits, y, halt in loader:
        optimizer.zero_grad()
        y_logits, halt_logits, _ = model(a, b, q_type, q_param, n_digits)
        
        # Y loss
        y_loss = F.binary_cross_entropy_with_logits(y_logits, y)
        
        # Halt loss
        halt_loss = F.cross_entropy(halt_logits, halt)
        
        # Combined
        loss = y_loss + config.lambda_halt * halt_loss
        
        loss.backward()
        optimizer.step()
        
        batch_size = a.size(0)
        total_loss += loss.item() * batch_size
        total_y_loss += y_loss.item() * batch_size
        total_halt_loss += halt_loss.item() * batch_size
        n_samples += batch_size
    
    return (total_loss / n_samples,
            total_y_loss / n_samples,
            total_halt_loss / n_samples)


def evaluate(model: ArithModel, loader: DataLoader) -> Tuple[float, float]:
    """Evaluate model. Returns (y_acc, halt_acc)."""
    model.eval()
    y_correct = 0
    halt_correct = 0
    n_samples = 0
    
    with torch.no_grad():
        for a, b, q_type, q_param, n_digits, y, halt in loader:
            y_logits, halt_logits, _ = model(a, b, q_type, q_param, n_digits)
            
            # Y prediction
            y_pred = (torch.sigmoid(y_logits) >= 0.5).float()
            y_correct += (y_pred == y).sum().item()
            
            # Halt prediction
            halt_pred = halt_logits.argmax(dim=1)
            halt_correct += (halt_pred == halt).sum().item()
            
            n_samples += a.size(0)
    
    return y_correct / n_samples, halt_correct / n_samples


# ============================================================================
# Logging
# ============================================================================

def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    
    logger = logging.getLogger("train_arith")
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
    logger = logging.getLogger("train_arith")
    logger.info(msg)


# ============================================================================
# Shuffle K (Ablation)
# ============================================================================

def shuffle_halt_ranks(samples: List[ArithSample], seed: int) -> List[ArithSample]:
    """Shuffle halt_rank assignments for ablation."""
    random.seed(seed)
    halt_ranks = [s.halt_rank for s in samples]
    random.shuffle(halt_ranks)
    
    shuffled = []
    for s, hr in zip(samples, halt_ranks):
        shuffled.append(ArithSample(
            a=s.a, b=s.b, n_digits=s.n_digits,
            q_type=s.q_type, q_param1=s.q_param1, q_param2=s.q_param2,
            y_star=s.y_star, t_first=s.t_first,
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
    log(f"=== Ω_arith Multi-Task Training (λ={config.lambda_halt}) ===")
    log(f"Config: {config}")
    log("")
    
    # Generate data
    log("Generating dataset...")
    train_samples, val_samples, test_samples = generate_dataset(seed=config.seed)
    
    if config.shuffle_K:
        log("SHUFFLING halt_rank assignments (ablation control)")
        train_samples = shuffle_halt_ranks(train_samples, config.seed + 1)
        val_samples = shuffle_halt_ranks(val_samples, config.seed + 2)
        test_samples = shuffle_halt_ranks(test_samples, config.seed + 3)
    
    log(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test (OOD): {len(test_samples)}")
    
    # Create loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, config.batch_size
    )
    
    # Model
    model = ArithModel(config)
    log(f"Model: {count_parameters(model):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    results = {"checkpoints": {}}
    
    for epoch in range(config.epochs + 1):
        if epoch > 0:
            loss, y_loss, h_loss = train_epoch(model, train_loader, optimizer, config)
        else:
            loss = y_loss = h_loss = float('inf')
        
        if epoch in config.checkpoint_epochs or epoch == config.epochs:
            y_acc_val, h_acc_val = evaluate(model, val_loader)
            y_acc_test, h_acc_test = evaluate(model, test_loader)
            
            log(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"Val Y: {y_acc_val:.3f}, H: {h_acc_val:.3f} | "
                f"OOD Y: {y_acc_test:.3f}, H: {h_acc_test:.3f}")
            
            results["checkpoints"][epoch] = {
                "y_acc_val": y_acc_val,
                "halt_acc_val": h_acc_val,
                "y_acc_test": y_acc_test,
                "halt_acc_test": h_acc_test,
            }
            
            torch.save(model.state_dict(),
                       os.path.join(config.output_dir, f"epoch_{epoch:03d}.pt"))
    
    # Final
    y_acc_val, h_acc_val = evaluate(model, val_loader)
    y_acc_test, h_acc_test = evaluate(model, test_loader)
    
    log(f"\nFinal: Val Y={y_acc_val:.3f}, H={h_acc_val:.3f} | "
        f"OOD Y={y_acc_test:.3f}, H={h_acc_test:.3f}")
    
    results["val_y_accuracy"] = y_acc_val
    results["val_halt_accuracy"] = h_acc_val
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

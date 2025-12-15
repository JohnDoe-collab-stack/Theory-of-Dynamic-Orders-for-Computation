"""
Multi-Task Training for Ω_proof

T_θ predicts both:
- y*(φ, q): ground truth (tautology/sat)
- halt_rank(φ): difficulty from K

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
    MAX_FORMULA_LEN, VOCAB, ProofSample
)
from proof_kernel import HaltRank


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
    vocab_size: int = 20  # Enough for tokens
    embed_dim: int = 32
    hidden_dim: int = 128
    n_layers: int = 3
    
    # Multi-task
    lambda_halt: float = 0.5
    n_halt_classes: int = 3  # EARLY, MID, LATE (no NEVER for proof)
    
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

class ProofModel(nn.Module):
    """
    Multi-task model for Ω_proof.
    
    Input: encoded formula (prefix notation)
    Output: y_logit, halt_logits
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        
        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Embedding(MAX_FORMULA_LEN, config.embed_dim)
        
        # Question type embedding
        self.qtype_embed = nn.Embedding(2, config.embed_dim)
        
        # Transformer encoder (simple)
        self.attention = nn.MultiheadAttention(config.embed_dim, num_heads=4, batch_first=True)
        
        # Aggregation
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP
        mlp_input = config.embed_dim * 2  # pooled + qtype
        layers = []
        layers.append(nn.Linear(mlp_input, config.hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(config.n_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        # Heads
        self.y_head = nn.Linear(config.hidden_dim, 1)
        self.halt_head = nn.Linear(config.hidden_dim, config.n_halt_classes)
    
    def forward(self, formula: torch.Tensor, q_type: torch.Tensor,
                depth: torch.Tensor = None):
        """
        Args:
            formula: (batch, MAX_FORMULA_LEN) token indices
            q_type: (batch,) question type
            depth: (batch,) formula depth (optional, for analysis)
        
        Returns:
            y_logits: (batch,)
            halt_logits: (batch, 3)
            embeddings: (batch, hidden_dim)
        """
        batch_size = formula.size(0)
        
        # Token + position embeddings
        positions = torch.arange(MAX_FORMULA_LEN, device=formula.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        tok_embed = self.token_embed(formula)
        pos_embed = self.pos_embed(positions)
        x = tok_embed + pos_embed  # (batch, seq, embed)
        
        # Self-attention
        x, _ = self.attention(x, x, x)
        
        # Pool
        x = x.transpose(1, 2)  # (batch, embed, seq)
        x = self.pool(x).squeeze(-1)  # (batch, embed)
        
        # Question type
        q_embed = self.qtype_embed(q_type)  # (batch, embed)
        
        # Combine and MLP
        combined = torch.cat([x, q_embed], dim=-1)
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

def train_epoch(model: ProofModel,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                config: TrainConfig) -> Tuple[float, float, float]:
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_y_loss = 0
    total_halt_loss = 0
    n_samples = 0
    
    for formula, q_type, depth, y, halt in loader:
        optimizer.zero_grad()
        
        # Clamp halt to valid range for 3 classes
        halt = halt.clamp(0, config.n_halt_classes - 1)
        
        y_logits, halt_logits, _ = model(formula, q_type, depth)
        
        # Losses
        y_loss = F.binary_cross_entropy_with_logits(y_logits, y)
        halt_loss = F.cross_entropy(halt_logits, halt)
        
        loss = y_loss + config.lambda_halt * halt_loss
        
        loss.backward()
        optimizer.step()
        
        batch_size = formula.size(0)
        total_loss += loss.item() * batch_size
        total_y_loss += y_loss.item() * batch_size
        total_halt_loss += halt_loss.item() * batch_size
        n_samples += batch_size
    
    return (total_loss / n_samples,
            total_y_loss / n_samples,
            total_halt_loss / n_samples)


def evaluate(model: ProofModel, loader: DataLoader, n_halt_classes: int = 3) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    y_correct = 0
    halt_correct = 0
    n_samples = 0
    
    with torch.no_grad():
        for formula, q_type, depth, y, halt in loader:
            halt = halt.clamp(0, n_halt_classes - 1)
            
            y_logits, halt_logits, _ = model(formula, q_type, depth)
            
            y_pred = (torch.sigmoid(y_logits) >= 0.5).float()
            y_correct += (y_pred == y).sum().item()
            
            halt_pred = halt_logits.argmax(dim=1)
            halt_correct += (halt_pred == halt).sum().item()
            
            n_samples += formula.size(0)
    
    return y_correct / n_samples, halt_correct / n_samples


# ============================================================================
# Logging
# ============================================================================

def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    
    logger = logging.getLogger("train_proof")
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
    logger = logging.getLogger("train_proof")
    logger.info(msg)


# ============================================================================
# Shuffle K
# ============================================================================

def shuffle_halt_ranks(samples: List[ProofSample], seed: int) -> List[ProofSample]:
    """Shuffle halt_rank for ablation."""
    random.seed(seed)
    halt_ranks = [s.halt_rank for s in samples]
    random.shuffle(halt_ranks)
    
    shuffled = []
    for s, hr in zip(samples, halt_ranks):
        shuffled.append(ProofSample(
            formula_str=s.formula_str,
            formula_enc=s.formula_enc,
            n_atoms=s.n_atoms,
            depth=s.depth,
            q_type=s.q_type,
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
    log(f"=== Ω_proof Multi-Task Training (λ={config.lambda_halt}) ===")
    log(f"Config: {config}")
    log("")
    
    # Generate data
    log("Generating dataset...")
    train_samples, val_samples, test_samples = generate_dataset(seed=config.seed)
    
    if config.shuffle_K:
        log("SHUFFLING halt_rank (ablation)")
        train_samples = shuffle_halt_ranks(train_samples, config.seed + 1)
        val_samples = shuffle_halt_ranks(val_samples, config.seed + 2)
        test_samples = shuffle_halt_ranks(test_samples, config.seed + 3)
    
    log(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test (OOD): {len(test_samples)}")
    
    # Create loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, config.batch_size
    )
    
    # Model
    model = ProofModel(config)
    log(f"Model: {count_parameters(model):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    results = {"checkpoints": {}}
    
    for epoch in range(config.epochs + 1):
        if epoch > 0:
            loss, y_loss, h_loss = train_epoch(model, train_loader, optimizer, config)
        else:
            loss = y_loss = h_loss = float('inf')
        
        if epoch in config.checkpoint_epochs or epoch == config.epochs:
            y_acc_val, h_acc_val = evaluate(model, val_loader, config.n_halt_classes)
            y_acc_test, h_acc_test = evaluate(model, test_loader, config.n_halt_classes)
            
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
    y_acc_val, h_acc_val = evaluate(model, val_loader, config.n_halt_classes)
    y_acc_test, h_acc_test = evaluate(model, test_loader, config.n_halt_classes)
    
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



@torch.no_grad()
def evaluate_standalone(model, samples, device="cpu"):
    """Standalone evaluation for stress tests (bypassing DataLoader overhead logic if needed)."""
    from torch.utils.data import DataLoader
    from dataset import ProofDataset
    
    ds = ProofDataset(samples)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    
    correct_y = 0
    correct_h = 0
    total = 0
    
    model.eval()
    for formula, q_type, depth, y, halt in dl:
        formula = formula.to(device)
        q_type = q_type.to(device)
        depth = depth.to(device)
        y = y.to(device)
        halt = halt.to(device)
        
        y_logits, halt_logits, _ = model(formula, q_type, depth)
        
        y_pred = (torch.sigmoid(y_logits) >= 0.5).float()
        correct_y += (y_pred == y).sum().item()
        
        h_pred = halt_logits.argmax(dim=1)
        correct_h += (h_pred == halt).sum().item()
        
        total += len(y)
        
    return correct_y / total if total > 0 else 0, correct_h / total if total > 0 else 0

if __name__ == "__main__":
    main()


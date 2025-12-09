"""
Training for Theory T_θ

Training loop with:
- BCE loss on ground truth y* (from Ω-structure)
- Checkpoints at [0, 1, 5, 10, 20, 50] epochs
- θ snapshots and E(T_θ) saved at each checkpoint
- Automatic logging to timestamped files

Key principle: θ is treated as OUTPUT (theory), analyzed post-hoc.
"""

import json
import os
import sys
import copy
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trig_kernel import AngleCode, TrigIndex
from dataset_trig import TrigQuestion, generate_trig_dataset, split_dataset
from model_T import TrigTheoryModel, count_parameters


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: str) -> str:
    """Setup logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("train_T")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return log_file


def log(msg: str):
    """Log to both console and file."""
    logger = logging.getLogger("train_T")
    logger.info(msg)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 60
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 42
    
    # Model architecture
    angle_embed_dim: int = 16
    index_embed_dim: int = 8
    hidden_dim: int = 64
    
    # Checkpoints (epochs at which to snapshot θ and E(T_θ))
    checkpoint_epochs: List[int] = None
    
    # Output
    output_dir: str = "checkpoints_trig"
    
    def __post_init__(self):
        if self.checkpoint_epochs is None:
            self.checkpoint_epochs = [0, 1, 5, 10, 20, 50]


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model: TrigTheoryModel,
                train_data: List[TrigQuestion],
                optimizer: optim.Optimizer,
                batch_size: int) -> float:
    """
    Train for one epoch with BCE loss.
    
    Returns:
        Average loss over the epoch
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    # Shuffle data
    import random
    data = train_data.copy()
    random.shuffle(data)
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        # Extract angles, indices, labels
        angles = [q.x for q in batch]
        indices = [q.idx for q in batch]
        labels = torch.tensor([q.y_star for q in batch], dtype=torch.float32)
        
        # Forward
        optimizer.zero_grad()
        probs = model(angles, indices)
        
        # BCE loss
        loss = F.binary_cross_entropy(probs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch)
        n_samples += len(batch)
    
    return total_loss / n_samples


def evaluate(model: TrigTheoryModel,
             data: List[TrigQuestion],
             threshold: float = 0.5) -> Tuple[float, List[Tuple[AngleCode, TrigIndex]]]:
    """
    Evaluate model on dataset.
    
    Returns:
        (accuracy, E) where E = set of correctly answered questions
    """
    model.eval()
    correct = 0
    E = []  # Correctly answered questions
    
    with torch.no_grad():
        for q in data:
            prob = model(q.x, q.idx).item()
            y_hat = 1 if prob >= threshold else 0
            
            if y_hat == q.y_star:
                correct += 1
                E.append((q.x.k, q.idx.kind.name, q.idx.r))  # Serializable key
    
    accuracy = correct / len(data) if data else 0
    return accuracy, E


# ============================================================================
# Checkpoint and Analysis
# ============================================================================

@dataclass
class TheoryCheckpoint:
    """Snapshot of theory T_θ at a given epoch."""
    epoch: int
    theta: Dict  # State dict (CPU tensors)
    acc_train: float
    acc_val: float
    E_train: List  # Correctly answered on train
    E_val: List    # Correctly answered on val
    loss: float


def save_checkpoint(ckpt: TheoryCheckpoint, path: str):
    """Save checkpoint to file."""
    data = {
        "epoch": ckpt.epoch,
        "acc_train": ckpt.acc_train,
        "acc_val": ckpt.acc_val,
        "loss": ckpt.loss,
        "E_train_size": len(ckpt.E_train),
        "E_val_size": len(ckpt.E_val),
        "E_val": ckpt.E_val[:100],  # Sample for inspection
    }
    
    # Save metadata
    with open(path + ".json", 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save weights
    torch.save(ckpt.theta, path + ".pt")


# ============================================================================
# Main Training Loop
# ============================================================================

def train_with_checkpoints(
    config: TrainConfig
) -> Dict[int, TheoryCheckpoint]:
    """
    Full training with checkpoints.
    
    Returns:
        Dict mapping epoch → TheoryCheckpoint
    """
    torch.manual_seed(config.seed)
    
    # Generate and split dataset
    log("Generating dataset...")
    full_data = generate_trig_dataset()
    train_data, val_data, test_data = split_dataset(full_data, seed=config.seed)
    
    log(f"Dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Create model
    model = TrigTheoryModel(
        angle_embed_dim=config.angle_embed_dim,
        index_embed_dim=config.index_embed_dim,
        hidden_dim=config.hidden_dim
    )
    log(f"Model: {count_parameters(model):,} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Checkpoints storage
    checkpoints: Dict[int, TheoryCheckpoint] = {}
    
    # Training loop
    max_epoch = max(config.checkpoint_epochs) + 1
    
    for epoch in range(max_epoch):
        # Train
        if epoch > 0:
            loss = train_epoch(model, train_data, optimizer, config.batch_size)
        else:
            loss = float('inf')  # Before any training
        
        # Checkpoint?
        if epoch in config.checkpoint_epochs:
            # Evaluate
            acc_train, E_train = evaluate(model, train_data)
            acc_val, E_val = evaluate(model, val_data)
            
            # Snapshot weights
            theta = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Create checkpoint
            ckpt = TheoryCheckpoint(
                epoch=epoch,
                theta=theta,
                acc_train=acc_train,
                acc_val=acc_val,
                E_train=E_train,
                E_val=E_val,
                loss=loss
            )
            checkpoints[epoch] = ckpt
            
            # Save
            save_checkpoint(ckpt, os.path.join(config.output_dir, f"epoch_{epoch:03d}"))
            
            log(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"Train: {acc_train:.3f} ({len(E_train)}/{len(train_data)}) | "
                f"Val: {acc_val:.3f} ({len(E_val)}/{len(val_data)})")
    
    # Final test evaluation
    acc_test, E_test = evaluate(model, test_data)
    log(f"Final Test Accuracy: {acc_test:.3f} ({len(E_test)}/{len(test_data)})")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.output_dir, "final.pt"))
    
    # Save config
    with open(os.path.join(config.output_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    return checkpoints


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train theory T_θ")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints_trig")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    if args.quick:
        config.checkpoint_epochs = [0, 1, 5, 10]
    
    # Setup logging to file
    log_file = setup_logging(config.output_dir)
    
    log("=== Training Theory T_θ ===")
    log(f"Log file: {log_file}")
    log(f"Config: {config}")
    log("")
    
    checkpoints = train_with_checkpoints(config)
    
    log("")
    log("=== Training Complete ===")
    log(f"Checkpoints saved to: {config.output_dir}/")
    log(f"Log saved to: {log_file}")
    
    return checkpoints


if __name__ == "__main__":
    main()

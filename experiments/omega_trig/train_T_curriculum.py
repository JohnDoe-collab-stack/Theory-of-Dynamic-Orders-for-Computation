"""
Curriculum Training for T_θ guided by t_first^K

Two modes:
1. Weighted loss: weight each example by difficulty from K
2. Phased curriculum: train on EARLY first, then add MID, then all

Uses t_first_K.json exported from DynamicTrigKernel.
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
from model_T import TrigTheoryModel, count_parameters
from pvec_trig import HaltRank, halt_rank_of_tfirst


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CurriculumConfig:
    """Configuration for curriculum training."""
    epochs: int = 60
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 42
    
    # Model architecture
    angle_embed_dim: int = 16
    index_embed_dim: int = 8
    hidden_dim: int = 64
    
    # Curriculum mode: "weighted" or "phased"
    mode: str = "weighted"
    
    # For weighted mode: boost difficult examples
    weight_early: float = 1.0
    weight_mid: float = 2.0
    weight_late: float = 3.0
    weight_never: float = 4.0
    
    # For phased mode
    phase1_epochs: int = 10  # EARLY only
    phase2_epochs: int = 20  # EARLY + MID
    # Remaining epochs: all
    
    # Checkpoints
    checkpoint_epochs: List[int] = None
    
    # Output
    output_dir: str = "checkpoints_curriculum"
    t_first_K_path: str = "checkpoints_trig/t_first_K.json"
    
    def __post_init__(self):
        if self.checkpoint_epochs is None:
            self.checkpoint_epochs = [0, 1, 5, 10, 20, 50]


# ============================================================================
# Load t_first^K from JSON
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


def assign_difficulty(data: List[TrigQuestion], 
                      t_first_K: Dict[Tuple, float]) -> List[Tuple[TrigQuestion, HaltRank]]:
    """Assign difficulty (halt_rank) to each question based on t_first_K."""
    enriched = []
    for q in data:
        key = (q.x.k, q.idx.kind.name, q.idx.r)
        t_first = t_first_K.get(key, float('inf'))
        halt_rank = halt_rank_of_tfirst(t_first)
        enriched.append((q, halt_rank))
    return enriched


# ============================================================================
# Logging
# ============================================================================

def setup_logging(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"curriculum_{timestamp}.log")
    
    logger = logging.getLogger("curriculum")
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
    logger = logging.getLogger("curriculum")
    logger.info(msg)


# ============================================================================
# Weighted Training
# ============================================================================

def get_weight(halt_rank: HaltRank, config: CurriculumConfig) -> float:
    """Get loss weight for a given difficulty class."""
    if halt_rank == HaltRank.EARLY:
        return config.weight_early
    elif halt_rank == HaltRank.MID:
        return config.weight_mid
    elif halt_rank == HaltRank.LATE:
        return config.weight_late
    else:  # NEVER
        return config.weight_never


def train_epoch_weighted(model: TrigTheoryModel,
                         train_data: List[Tuple[TrigQuestion, HaltRank]],
                         optimizer: optim.Optimizer,
                         config: CurriculumConfig) -> float:
    """Train one epoch with weighted loss."""
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    
    data = train_data.copy()
    random.shuffle(data)
    
    for i in range(0, len(data), config.batch_size):
        batch = data[i:i + config.batch_size]
        
        angles = [q.x for q, _ in batch]
        indices = [q.idx for q, _ in batch]
        labels = torch.tensor([q.y_star for q, _ in batch], dtype=torch.float32)
        weights = torch.tensor([get_weight(hr, config) for _, hr in batch], dtype=torch.float32)
        
        optimizer.zero_grad()
        probs = model(angles, indices)
        
        # Element-wise BCE with weights
        loss_per_sample = F.binary_cross_entropy(probs, labels, reduction='none')
        weighted_loss = (loss_per_sample * weights).sum() / weights.sum()
        
        weighted_loss.backward()
        optimizer.step()
        
        total_loss += weighted_loss.item() * weights.sum().item()
        total_weight += weights.sum().item()
    
    return total_loss / total_weight if total_weight > 0 else 0


# ============================================================================
# Phased Training
# ============================================================================

def filter_by_phase(train_data: List[Tuple[TrigQuestion, HaltRank]], 
                    phase: int) -> List[Tuple[TrigQuestion, HaltRank]]:
    """Filter data based on curriculum phase."""
    if phase == 1:
        # EARLY only
        return [(q, hr) for q, hr in train_data if hr == HaltRank.EARLY]
    elif phase == 2:
        # EARLY + MID
        return [(q, hr) for q, hr in train_data if hr in (HaltRank.EARLY, HaltRank.MID)]
    else:
        # All
        return train_data


def train_epoch_phased(model: TrigTheoryModel,
                       train_data: List[Tuple[TrigQuestion, HaltRank]],
                       optimizer: optim.Optimizer,
                       epoch: int,
                       config: CurriculumConfig) -> float:
    """Train one epoch with phased curriculum."""
    # Determine phase
    if epoch <= config.phase1_epochs:
        phase = 1
    elif epoch <= config.phase1_epochs + config.phase2_epochs:
        phase = 2
    else:
        phase = 3
    
    phase_data = filter_by_phase(train_data, phase)
    
    if not phase_data:
        return 0.0
    
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    data = phase_data.copy()
    random.shuffle(data)
    
    for i in range(0, len(data), config.batch_size):
        batch = data[i:i + config.batch_size]
        
        angles = [q.x for q, _ in batch]
        indices = [q.idx for q, _ in batch]
        labels = torch.tensor([q.y_star for q, _ in batch], dtype=torch.float32)
        
        optimizer.zero_grad()
        probs = model(angles, indices)
        loss = F.binary_cross_entropy(probs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch)
        n_samples += len(batch)
    
    return total_loss / n_samples if n_samples > 0 else 0


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model: TrigTheoryModel,
             data: List[Tuple[TrigQuestion, HaltRank]]) -> Tuple[float, List]:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    E = []
    
    with torch.no_grad():
        for q, hr in data:
            prob = model(q.x, q.idx).item()
            y_hat = 1 if prob >= 0.5 else 0
            if y_hat == q.y_star:
                correct += 1
                E.append((q.x.k, q.idx.kind.name, q.idx.r))
    
    acc = correct / len(data) if data else 0
    return acc, E


# ============================================================================
# Main Training Loop
# ============================================================================

def train_with_curriculum(config: CurriculumConfig) -> Dict:
    """Full curriculum training."""
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    log_file = setup_logging(config.output_dir)
    log(f"=== Curriculum Training ({config.mode} mode) ===")
    log(f"Log: {log_file}")
    log(f"Config: {config}")
    log("")
    
    # Load t_first^K
    log(f"Loading t_first^K from {config.t_first_K_path}...")
    t_first_K = load_t_first_K(config.t_first_K_path)
    log(f"  Loaded {len(t_first_K)} difficulty values")
    
    # Generate dataset
    log("Generating dataset...")
    full_data = generate_trig_dataset()
    train_data, val_data, test_data = split_dataset(full_data, seed=config.seed)
    
    # Enrich with difficulty
    train_enriched = assign_difficulty(train_data, t_first_K)
    val_enriched = assign_difficulty(val_data, t_first_K)
    test_enriched = assign_difficulty(test_data, t_first_K)
    
    # Count by difficulty
    for split_name, data in [("Train", train_enriched), ("Val", val_enriched)]:
        counts = {}
        for _, hr in data:
            counts[hr.name] = counts.get(hr.name, 0) + 1
        log(f"  {split_name}: {len(data)} total, by difficulty: {counts}")
    
    # Create model
    model = TrigTheoryModel(
        angle_embed_dim=config.angle_embed_dim,
        index_embed_dim=config.index_embed_dim,
        hidden_dim=config.hidden_dim
    )
    log(f"Model: {count_parameters(model):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    os.makedirs(config.output_dir, exist_ok=True)
    
    results = {"checkpoints": {}}
    max_epoch = max(config.checkpoint_epochs) + 1
    
    for epoch in range(max_epoch):
        if epoch > 0:
            if config.mode == "weighted":
                loss = train_epoch_weighted(model, train_enriched, optimizer, config)
            else:  # phased
                loss = train_epoch_phased(model, train_enriched, optimizer, epoch, config)
        else:
            loss = float('inf')
        
        if epoch in config.checkpoint_epochs:
            acc_train, E_train = evaluate(model, train_enriched)
            acc_val, E_val = evaluate(model, val_enriched)
            
            log(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                f"Train: {acc_train:.3f} | Val: {acc_val:.3f}")
            
            results["checkpoints"][epoch] = {
                "acc_train": acc_train,
                "acc_val": acc_val,
                "E_val_size": len(E_val),
            }
            
            # Save checkpoint weights
            torch.save(model.state_dict(), 
                       os.path.join(config.output_dir, f"epoch_{epoch:03d}.pt"))
            
            # Save JSON metadata (compatible with analysis_T.py)
            ckpt_meta = {
                "epoch": epoch,
                "acc_train": acc_train,
                "acc_val": acc_val,
                "loss": loss if loss != float('inf') else None,
                "E_train_size": len(E_train),
                "E_val_size": len(E_val),
            }
            with open(os.path.join(config.output_dir, f"epoch_{epoch:03d}.json"), 'w') as f:
                json.dump(ckpt_meta, f, indent=2)
            
            # Save E_val separately (for inclusion analysis)
            with open(os.path.join(config.output_dir, f"epoch_{epoch:03d}_E_val.json"), 'w') as f:
                json.dump(E_val, f)
    
    # Final test
    acc_test, E_test = evaluate(model, test_enriched)
    log(f"\nFinal Test Accuracy: {acc_test:.3f}")
    results["test_accuracy"] = acc_test
    
    # Save config and results
    with open(os.path.join(config.output_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=2)
    with open(os.path.join(config.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(config.output_dir, "final.pt"))
    
    log("")
    log("=== Curriculum Training Complete ===")
    log(f"Checkpoints saved to: {config.output_dir}/")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="weighted", choices=["weighted", "phased"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="checkpoints_curriculum")
    parser.add_argument("--t-first-K", type=str, default="checkpoints_trig/t_first_K.json")
    args = parser.parse_args()
    
    config = CurriculumConfig(
        mode=args.mode,
        epochs=args.epochs,
        output_dir=args.output_dir,
        t_first_K_path=args.t_first_K,
    )
    
    results = train_with_curriculum(config)
    return results


if __name__ == "__main__":
    main()

"""
Visualization: Confusion Matrix and Lambda Barplot

1. Confusion matrix for Halt prediction (K-real model)
2. Barplot: Halt accuracy K-real vs Shuffle for λ ∈ {0.1, 0.5, 1.0}
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from trig_kernel import AngleCode, TrigIndex, TrigIndexType
from dataset_trig import TrigQuestion, generate_trig_dataset, split_dataset
from train_T_multitask import (
    MultiTaskTheoryModel, MultiTaskConfig,
    load_t_first_K, prepare_multitask_data
)
from pvec_trig import halt_rank_of_tfirst


def get_halt_predictions(model_path: str, t_first_K_path: str, 
                         shuffle_K: bool = False, seed: int = 42) -> Tuple[List, List]:
    """Load model and get halt predictions on test set."""
    import random
    
    # Load t_first_K
    t_first_K = load_t_first_K(t_first_K_path)
    
    if shuffle_K:
        random.seed(seed + 1)
        keys = list(t_first_K.keys())
        vals = list(t_first_K.values())
        random.shuffle(vals)
        t_first_K = {k: v for k, v in zip(keys, vals)}
    
    # Generate dataset
    full_data = generate_trig_dataset()
    _, _, test_data = split_dataset(full_data, seed=seed)
    test_samples = prepare_multitask_data(test_data, t_first_K)
    
    # Load model
    model = MultiTaskTheoryModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for s in test_samples:
            _, halt_logits, _ = model(s.x, s.idx)
            halt_pred = halt_logits.argmax().item()
            y_true.append(s.halt_rank)
            y_pred.append(halt_pred)
    
    return y_true, y_pred


def plot_confusion_matrix(y_true: List, y_pred: List, output_path: str):
    """Create and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    labels = ["EARLY", "MID", "LATE", "NEVER"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(4),
           yticks=np.arange(4),
           xticklabels=labels,
           yticklabels=labels,
           xlabel='Predicted',
           ylabel='True',
           title='Halt Prediction Confusion Matrix (K-real)')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(4):
        for j in range(4):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")
    
    # Print metrics
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(4))
    acc = correct / total
    print(f"Overall accuracy: {acc:.1%}")
    
    # Per-class accuracy
    for i, label in enumerate(labels):
        class_total = sum(cm[i])
        class_correct = cm[i][i]
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {label}: {class_correct}/{class_total} ({class_acc:.1%})")


def plot_lambda_barplot(output_path: str):
    """Create barplot of Halt accuracy vs λ for K-real and Shuffle."""
    # Load summary from lambda sweep
    summary_path = "mt_lambda_sweep/summary.json"
    
    if not os.path.exists(summary_path):
        print(f"Lambda sweep summary not found at {summary_path}")
        print("Running quick λ sweep...")
        # Run a quick sweep if not available
        return
    
    with open(summary_path) as f:
        data = json.load(f)
    
    lambdas = data["lambdas"]
    k_halt = [r["halt_acc"] * 100 for r in data["K_real"]]
    s_halt = [r["halt_acc"] * 100 for r in data["shuffle"]]
    
    # Filter λ > 0
    mask = [l > 0 for l in lambdas]
    lambdas = [l for l, m in zip(lambdas, mask) if m]
    k_halt = [h for h, m in zip(k_halt, mask) if m]
    s_halt = [h for h, m in zip(s_halt, mask) if m]
    
    # Create barplot
    x = np.arange(len(lambdas))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, k_halt, width, label='K-real', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, s_halt, width, label='Shuffle', color='#e74c3c', edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars1, k_halt):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, s_halt):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('λ_halt', fontsize=12)
    ax.set_ylabel('Halt Accuracy (%)', fontsize=12)
    ax.set_title('Halt Prediction Accuracy: K-real vs Shuffle\n(Multi-task Training)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'λ={l}' for l in lambdas])
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 110)
    
    # Add horizontal line at 50% (random chance)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='_nolegend_')
    ax.text(len(lambdas)-0.5, 52, 'Random chance', fontsize=10, color='gray')
    
    # Add delta annotations
    for i, (k, s) in enumerate(zip(k_halt, s_halt)):
        delta = k - s
        ax.annotate(f'+{delta:.0f}pp', xy=(i, 105), ha='center', fontsize=10,
                    color='#27ae60', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved barplot to {output_path}")


def main():
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Generating Visualizations ===\n")
    
    # 1. Confusion matrix for K-real
    print("1. Confusion Matrix (K-real model)...")
    model_path = "checkpoints_mt_K/final.pt"
    t_first_K_path = "checkpoints_trig/t_first_K.json"
    
    if os.path.exists(model_path):
        y_true, y_pred = get_halt_predictions(model_path, t_first_K_path, shuffle_K=False)
        plot_confusion_matrix(y_true, y_pred, f"{output_dir}/confusion_matrix_halt.png")
    else:
        print(f"  Model not found: {model_path}")
    
    # 2. Lambda barplot
    print("\n2. Lambda Barplot...")
    plot_lambda_barplot(f"{output_dir}/lambda_barplot.png")
    
    print("\n=== Done ===")
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()

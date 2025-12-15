import torch
import os
import json
import sys
import hashlib
import random
import numpy as np
from collections import Counter
from dataset import generate_samples, encode_formula
from train import ProofModel, TrainConfig, evaluate_standalone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from proof_kernel import ProofKernel, QuestionType, is_tautology, is_satisfiable, halt_rank_of_tfirst

# Ensure reproducible analysis stats
random.seed(42)
np.random.seed(42)

def compute_dataset_hash(samples):
    """Compute a hash of the INPUTS (formula, q_type, depth) to verify identity across variants."""
    hasher = hashlib.md5()
    for s in samples:
        content = f"{s.formula_str}|{s.q_type}|{s.n_atoms}|{s.depth}|{s.y_star}"
        hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()

def analyze_halt_distribution(samples):
    """Return distribution of halt ranks."""
    ranks = [s.halt_rank for s in samples]
    counts = Counter(ranks)
    total = len(ranks)
    dist = {k: v/total for k,v in counts.items()}
    return dist

def get_t_first_vector(samples):
    """Extract t_first vector from samples."""
    return [s.t_first for s in samples]

def stress_eval(checkpoint_path, num_samples=500, output_file="stress_results.json"):
    print(f"Loading model from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = TrainConfig() 
    model = ProofModel(config).to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    model.eval()

    variants = [
        ("lex", "sorted"),    
        ("shuffle", "sorted"), 
        ("lex", "shuffle"),    
        ("random", "sorted"), 
    ]
    
    results = {}
    baseline_hash = None
    baseline_t_first = None
    
    print(f"\nRunning GOLD STANDARD Stress Test (N={num_samples} per variant)...")
    print("-" * 120)
    print(f"{'Variant':<16} | {'InHash':<7} | {'BalAcc':>6} | {'MajBase':>7} | {'Pearson':>7} | {'Recall(E/M/L)':<15} | {'MaxStepHit'}")
    print("-" * 120)

    for v_order, a_order in variants:
        key = f"{v_order}_{a_order}"
        
        # Generator uses local RNG, ensuring strictly identical formulas
        samples = generate_samples(
            n_samples=num_samples, 
            n_atoms_range=(3, 4), 
            depth_range=(3, 5), 
            seed=42, 
            valuation_order=v_order,
            atom_order=a_order
        )
        
        # 1. Integrity Check
        current_hash = compute_dataset_hash(samples)
        hash_match = "FAIL"
        if baseline_hash is None:
            baseline_hash = current_hash
            hash_match = "REF"
        else:
            hash_match = "OK" if current_hash == baseline_hash else "FAIL"
            
        # 2. Distribution Analysis
        dist = analyze_halt_distribution(samples)
        majority_class_prob = max(dist.values()) if dist else 0
        
        # 3. Model Evaluation
        acc_y, acc_h = evaluate_standalone(model, samples, device)
        
        # 4. Detailed Metrics (Balanced Acc, Confusion Matrix)
        from torch.utils.data import DataLoader
        from dataset import ProofDataset
        ds = ProofDataset(samples)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        all_halt_preds, all_halt_true = [], []
        with torch.no_grad():
            for batch in dl:
                form, q, d, y, h = [t.to(device) for t in batch]
                _, h_logits, _ = model(form, q, d)
                all_halt_preds.extend(h_logits.argmax(dim=1).cpu().numpy())
                all_halt_true.extend(h.cpu().numpy())
        
        bal_acc_h = balanced_accuracy_score(all_halt_true, all_halt_preds)
        cm = confusion_matrix(all_halt_true, all_halt_preds, labels=[0, 1, 2])
        # Recall per class: TP / (TP+FN) -> diag / sum(axis=1)
        # Avoid division by zero
        class_counts = cm.sum(axis=1)
        recalls = [cm[i,i] / c if c > 0 else 0.0 for i, c in enumerate(class_counts)]
        recall_str = "/".join([f"{r:.2f}" for r in recalls])
        
        # 5. Correlation Analysis
        current_t_first = get_t_first_vector(samples)
        pearson_corr = 1.0
        if baseline_t_first is None:
            baseline_t_first = current_t_first
        else:
            # Clean correlation (t_first can be same length, just values differ)
            if np.std(baseline_t_first) > 0 and np.std(current_t_first) > 0:
                pearson_corr = np.corrcoef(baseline_t_first, current_t_first)[0, 1]
            else:
                pearson_corr = 0.0
        
        # 6. Max Steps Bounds Check
        max_steps_hit = sum(1 for s in samples if s.t_first >= 100) # kernel default 100
        # n=4 -> 16 < 100, strict bound ok. If randomized > 16 logic implies coverage ok?
        # ProofKernel random mode logic: limit = min(max_steps*2, 2**n) -> 16.
        # So t_first shouldn't exceed 16 for n=4. 
        # But let's check if any sample hit the hard cap of the kernel configuration
        
        results[key] = {
            "input_hash": current_hash,
            "y_accuracy": acc_y,
            "halt_accuracy": acc_h,
            "balanced_halt_accuracy": bal_acc_h,
            "majority_baseline": majority_class_prob,
            "recalls": recalls,
            "confusion_matrix": cm.tolist(),
            "pearson_correlation_with_baseline": pearson_corr,
            "max_steps_hit": max_steps_hit
        }
        
        print(f"{key:<16} | {hash_match:<7} | {bal_acc_h*100:>6.1f}% | {majority_class_prob*100:>7.1f}% | {pearson_corr:>7.2f} | {recall_str:<15} | {max_steps_hit}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("-" * 120)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    ckpt_path = "checkpoints_K/final.pt"
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    stress_eval(ckpt_path)

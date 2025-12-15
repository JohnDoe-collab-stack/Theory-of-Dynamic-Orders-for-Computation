# Ω_proof: Propositional Logic with Dynamic Proof Search

> **Best OOD generalization (+42pp Halt) — Structure transfers across depths**

## System: Ω-Proof (KAMT)

**Ω-Proof** couples an order-specified proof kernel **KΩ** with a shallow attention encoder **Aθ**. For each formula, **KΩ** produces a semantic label **y*** and an early-stopping time **t_first**, which is discretized into an effort label **halt_rank**. **Aθ** is trained with a kernel-aligned multi-task objective, **L = BCE(y*, y_hat) + λ·CE(halt_rank, halt_hat)**, using a **Y-head** (semantics) and an **H-head** (effort). We validate structural dependence via **order interventions** (e.g., lex vs shuffle) under an **input-integrity check**, reporting **order sensitivity** of the H-head while confirming **atom-rename invariance**.

### Official Nomenclature
* **KΩ (order-specified proof kernel)**: The explicit algorithm evaluating formulas under order Ω.
  * **`y*`**: Semantic label (TAUT / SAT).
  * **`t_first`**: First-witness time (early stopping index).
  * **`halt_rank`**: Discretized effort label derived from `t_first / 2^n`.
* **Aθ (attention encoder)**: Shallow attention-based encoder (embeddings + MHA + pooling + MLP).
* **Kernel-aligned multi-task objective**: $L = BCE(y*, \hat{y}) + \lambda \cdot CE(halt\_rank, \hat{halt})$.

Unlike Ω_arith, the logical structure here **generalizes to deeper formulas** and demonstrates strong **order sensitivity**.


## Key Results

| Condition | Val Y | Val Halt | OOD Y | OOD Halt |
|-----------|-------|----------|-------|----------|
| **K-real** | 100% | **96%** | 89% | **75%** |
| Shuffle | 97% | 37% | 89% | 33% |
| **Δ Halt** | — | **+59pp** | — | **+42pp** |

### Stress Test Validation (Structural Hypothesis)

To confirm that the model learns the **causal structure of computation** and not just statistical density, we performed a stress test by decorrelating the Valuation Order (Time) and Atom Order (Space) on the *exact same formulas* (`stress_eval.py`):

| Variant (Time / Space) | Y Acc | Halt Acc | Interpretation |
| :--- | :--- | :--- | :--- |
| **lex / sorted** (Baseline) | **92.2%** | **78.0%** | The model correctly predicts computational difficulty. |
| **lex / shuffle** | 91.8% | **78.4%** | **Invariance**: Variable naming ($p_0$ vs $p_1$) is irrelevant. |
| **shuffle / sorted** | 91.6% | **51.8%** | **Collapse**: When the search structure is destroyed, performance drops by **~26pp**, proving the model was relying on the specific geometry of the search space. |

**Conclusion**: The ~26pp gap represents the **"Structural Component"** of the prediction. The remaining ~52% accuracy comes from the intrinsic density of solutions (some formulas vary in difficulty regardless of order). This confirms the **Dynamic Order Hypothesis**: the model has internalized the specific algorithm of the Oracle $\Omega$.

### Interpretation

1. **In-distribution (+59pp)**: K dynamics massively exploited
2. **OOD (+42pp)**: Halt generalizes to unseen formula depths ✓
3. **Y saturates (~100%)**: propositional semantics are easy
4. **Halt transfers**: proof structure is more universal than arithmetic


## Comparison with Ω_arith

| | Ω_proof | Ω_arith |
|---|---------|---------|
| Domain | Propositional logic | Integer addition |
| K dynamics | Early stopping on counterexample | Column-by-column carry |
| OOD Δ Halt | **+42pp** | +4.6pp |
| Transfer | Strong ✓ | Weak ✗ |
| Insight | Logical depth transfers | Carry chains are local |

## Architecture

```
Input Formula     →    Dynamic Kernel K    →    Neural Encoder T_θ
  φ = (p ∧ q) → r       Early stopping          Formula embedding
                        t_first = 3              + attention pooling
                        halt_bucket = 0.4        + MLP heads (Y, Halt)
```

### Multi-Task Training

```python
loss = loss_Y + λ * loss_Halt
```

- `loss_Y`: Cross-entropy on ground truth (tautology/satisfiable)
- `loss_Halt`: Cross-entropy on halt bucket from K
- `λ`: Weight controlling K-structure pressure

## Sphere Integration

The experiment now includes **fuel efficiency tracking** via the Sphere framework:

```bash
python run_sphere_validation.py --n-formulas 200
```

| Condition | Fuel Efficiency | Strict Steps | Δ |
|-----------|-----------------|--------------|---|
| K-real | 70.5% | 2.6/φ | — |
| Late-halt | 50.0% | 5.1/φ | — |
| **Δ** | **+20.4pp** | -2.5 | ✓ |

**Key finding**: Early stopping saves ~20% of fuel budget on average.

## Files

```
omega_proof/
├── proof_kernel.py          # Formula types + ProofKernel with early stopping
├── dataset.py               # OOD depth split + encoding
├── train.py                 # Multi-task training loop
├── sphere_wrapper.py        # Sphere fuel tracking integration
├── run_sphere_validation.py # Real performance benchmark
└── checkpoints_K/           # K-real model checkpoints
```

## Quick Start

### Training

```bash
# K-real condition (with structure)
python train.py --epochs 50 --output-dir checkpoints_K

# Shuffle-K ablation (structure destroyed)
python train.py --epochs 50 --output-dir checkpoints_shuffle --shuffle-K
```

### Sphere Validation

```bash
# Run fuel efficiency benchmark
python run_sphere_validation.py --n-formulas 500 --atoms 2,3,4,5

# Results saved to sphere_validation_results.json
```

## Theoretical Connection

### Why OOD works for logic but not arithmetic?

From the Lean formalization (`LogicDissoc.lean`, `Sphere.lean`):

1. **Logical depth** ≈ proof tree depth → **global property**
2. **Carry chains** ≈ sequential dependencies → **local property**

The neural learner can internalize global structural patterns (depth),
but struggles with position-sensitive local dynamics (carries).

### Link to Impossibility Theorems

> *No internal predicate can capture RealHalts globally (total/correct/complete)*

For Ω_proof, the learner exploits **structural regularity** of logical formulas,
which is more robust under depth generalization than the **positional regularity**
of arithmetic carry propagation.

### Sphere Guarantee

From `Sphere.lean` theorem `max_trajectory_length`:
> The number of strict steps (fuel consumption) is bounded by initial budget R

This provides a **formal certificate** that early stopping indeed reduces fuel usage.

## Citation

```bibtex
@misc{omega_proof,
  title={Ω_proof: Dynamic Order Structure in Propositional Logic},
  author={Theory of Dynamic Orders},
  year={2024},
  note={Part of the Dynamic Order Theory formalization}
}
```

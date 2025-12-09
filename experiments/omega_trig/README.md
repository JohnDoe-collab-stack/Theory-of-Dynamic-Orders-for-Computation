# Ω-Trig Experiment

> **Validation of the Ω–K–E dissociation framework**

---

## What Makes This Different

In standard ML, three concepts are conflated:
- Where data comes from
- What labels mean
- How we measure success

This experiment **explicitly separates** them:

| Component | Role | In ω-Trig |
|-----------|------|-----------|
| **Ω (World)** | Source of instances | 360 angles × 8 question types |
| **K (Oracle)** | Source of truth (static + dynamic) | y* + halt_rank |
| **E (Evaluation)** | How we judge T_θ | Accuracy, sync, P_vec, ablations... |

**Key insight**: K provides signals; E judges what T_θ does with them. They are not the same thing.

---

## The Setup

```
┌─────────────────────────────────────────────────────────────┐
│                      Ω (World)                              │
│  Generates instances σ = (angle, question)                  │
│  360 × 8 = 2880 possible facts                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      K (Oracle)                             │
│  Provides truth for each σ:                                 │
│  - y*(σ): static truth (is sin(θ) ≥ r?)                     │
│  - halt_rank(σ): dynamic difficulty (EARLY/MID/LATE/NEVER)  │
│  For T_θ, K is a BLACK BOX external source of labels        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      T_θ (Learner)                          │
│  Neural approximator trained on K's signals                 │
│  Input: σ → Output: ŷ, halt_logits                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      E (Evaluation)                         │
│  Multiple independent metrics:                              │
│  - Accuracy on y* and halt_rank                             │
│  - Sync: correlation t_first^K ↔ t_first^T                  │
│  - P_vec: cut ⊥ bit in latent space                         │
│  - Theory gradient: E(T_e1) ⊆ E(T_e2)                       │
│  - Ablations: baseline / uniform / K-guided / shuffle       │
└─────────────────────────────────────────────────────────────┘
```

---

## What This Enables

Because Ω, K, and E are separated, we can ask questions that are usually impossible:

1. **Same Ω, same K, different E**: How does T_θ look under different evaluation lenses?
2. **Same Ω, different K**: What if the oracle changes (different dynamics)?
3. **K as black box**: T_θ doesn't know how K computes halt_rank - it just learns to match it

---

## Key Results

### Multi-Task Validation (3 seeds)

| Condition | Y Accuracy | Halt Accuracy | Δ Halt |
|-----------|------------|---------------|--------|
| **K-real** | 99.2% ± 0.3% | **98.7% ± 0.4%** | - |
| **Shuffle** | 98.9% ± 0.5% | 48.1% ± 2.6% | **-50.6pp** |

**Shuffle test**: When we destroy K's structure (permute halt_rank assignments), T_θ cannot learn halt anymore. This proves K encodes real, structured information.

### Confusion Matrix (K-real)

| True \ Pred | EARLY | MID | NEVER |
|-------------|-------|-----|-------|
| **EARLY** | 99 | 2 | 0 |
| **MID** | 2 | 135 | 0 |
| **NEVER** | 0 | 0 | 195 |

**99.1% overall** - T_θ learns all classes, not just majority.

### λ Sweep

| λ_halt | K-real Halt | Shuffle Halt | Δ |
|--------|-------------|--------------|---|
| 0.0 | 4% | 13% | -9pp |
| 0.1 | 98% | 50% | **+48pp** |
| 0.5 | 99% | 50% | **+49pp** |
| 1.0 | 99% | 50% | **+49pp** |

---

## What We Validated

### ✅ Framework Works

1. **Ω–K–E separation is implementable** and produces meaningful experiments
2. **T_θ can synchronize with external oracle K** (black box)
3. **K's structure matters**: shuffle destroys the halt signal (+50pp gap)
4. **Multiple evaluation lenses** (E) give consistent story

### ⚠️ Scope Limitations

1. **This is a proof-of-concept**, not a claim about "deep dynamics"
2. **ω-trig is a toy domain**: task saturates at ~99%
3. **Curriculum alone doesn't help**: weighting by K doesn't improve Y accuracy on this easy task
4. **K may be "flat"**: halt_rank could be a simple function of inputs (we don't prove otherwise)

### 🔮 What Would Strengthen the Claim

A future Ω where:
- Without K, T_θ fails or generalizes poorly
- With K, T_θ gains robustly
- And this gain is not trivial to explain

---

## Honest Summary

**What we showed**: T_θ can learn to match an external oracle K, and shuffle-control proves K is structured, not noise.

**What we did NOT show**: That K is "irréductiblement dynamique" or that K is indispensable for performance.

**The conceptual contribution**: Explicit separation of Ω (world) / K (oracle) / E (evaluation), which is rarely done in ML.

---

## File Structure

```
omega_trig/
├── trig_kernel.py            # Ω-syntax
├── dynamic_trig_kernel.py    # K: oracle with halt_rank
├── dataset_trig.py           # Data from Ω
├── model_T.py                # T_θ architecture
│
├── train_T.py                # Baseline (y* only)
├── train_T_multitask.py      # Multi-task (y* + halt_rank)
├── train_T_curriculum.py     # Curriculum (weighted by K)
│
├── analysis_T.py             # E: theory gradient
├── sync_K_T.py               # E: K ↔ T correlation
├── analysis_pvec_trig.py     # E: latent structure
│
├── run_mt_validation.py      # Multi-seed + λ sweep
├── visualize_results.py      # Confusion matrix + plots
└── figures/                  # Generated visualizations
```

---

## Quick Start

```bash
# 1. Run the main experiment
python run_mt_validation.py --seeds 3 --lambda-sweep

# 2. Generate figures
python visualize_results.py

# 3. Check sync K ↔ T
python sync_K_T.py
```

---

## Dependencies

```
torch>=2.0
numpy
scipy
matplotlib
scikit-learn
```

---

## Citation

Part of the **Theory of Dynamic Orders for Computation** project.

> *The key contribution is the explicit separation of Ω (world), K (oracle), and E (evaluation) —
> a modular architecture that enables experiments about T_θ's relationship to truth,
> not just its loss on labels.*

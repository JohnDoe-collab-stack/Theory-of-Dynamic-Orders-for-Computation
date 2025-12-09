# Ω-Trig Experiment

Experimental validation of the **Dynamic Order Theory** framework with a trigonometric kernel.

---

## Overview

This experiment tests if a parameterized theory **T_θ** can learn to observe a fixed **Ω-structure**, with:

- **Ω** = fixed syntax (angles, profiles, questions)
- **K** = dynamic kernel with monotone refinement (approx_t, val_t, E(K))
- **T_θ** = neural observer that approximates Ω
- **P_vec** = orthogonal structure (cut ⊥ bit)
- **Curriculum** = K-guided training (weighted loss by difficulty)

**Key principle**: θ is an **output** (analyzable object), not the source of structure.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Ω-Syntax (Fixed)                       │
│  AngleCode (360) │ PTrig (sin,cos) │ questionTrig → {0,1}   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Dynamic Kernel K (Monotone)                 │
│  approx_t(x) │ val_t(x,i) │ traces │ E(K) │ t_first^K      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Theory T_θ (Learner)                     │
│  angle_embed │ index_embed │ MLP → probability              │
│  [baseline] or [curriculum: weighted by halt_K]             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Analysis Layer                            │
│  E(T_θ) inclusions │ sync K↔T │ P_vec probes (cut⊥bit)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `trig_kernel.py` | Ω-syntax: AngleCode, PTrig, V_trig, questionTrig |
| `dataset_trig.py` | Dataset D = X_trig × I_trig with labels y* |
| `model_T.py` | Theory T_θ: neural model with return_embedding |
| `train_T.py` | Baseline training (BCE + checkpoints) |
| `train_T_curriculum.py` | K-guided training (weighted/phased curriculum) |
| `dynamic_trig_kernel.py` | DynamicTrigKernel: approx_t, val_t, traces, E(K) |
| `analysis_T.py` | E(T_θ) inclusion analysis, theory gradient |
| `pvec_trig.py` | P_vec structure: CutClass, BitClass, HaltRank |
| `analysis_pvec_trig.py` | Linear probes for cut/bit, orthogonality test |
| `sync_K_T.py` | K ↔ T_θ synchronization: correlation, confusion |

---

## Quick Start

```bash
# 1. Test kernels
python trig_kernel.py
python dynamic_trig_kernel.py

# 2. Export difficulty map from K
python -c "from dynamic_trig_kernel import *; k=DynamicTrigKernel(angles=list(range(360))); k.export_t_first_K()"

# 3. Train baseline T_θ
python train_T.py

# 4. Train curriculum T_θ (K-guided)
python train_T_curriculum.py --mode weighted

# 5. Analyze
python analysis_T.py --checkpoints-dir checkpoints_trig
python analysis_T.py --checkpoints-dir checkpoints_curriculum2
python sync_K_T.py --checkpoints-dir checkpoints_trig
python analysis_pvec_trig.py
```

---

## Results

### Test Accuracy

| Run | Test Acc |
|-----|----------|
| Baseline | 97.2% |
| Curriculum (K-guided) | **98.2%** (+1.0%) |

### Theory Gradient E(T_θ)

| Transition | Baseline | Curriculum |
|------------|----------|------------|
| 5 → 10 | 0.97 | 0.97 |
| 10 → 20 | 1.00 | 1.00 |
| 20 → 50 | 0.99 | 0.99 |
| **Verdict** | ✓ SUCCESS | ✓ SUCCESS |

### P_vec Decodability

| Axis | Accuracy | Random |
|------|----------|--------|
| cut (quadrant) | 99% | 25% |
| bit (q-type) | 100% | 25% |
| halt (t_first) | 70% | 64% |
| **cos(W_cut, W_bit)** | 0.058 | - |

### Sync K ↔ T_θ

| Run | Pearson | Spearman | Verdict |
|-----|---------|----------|---------|
| Baseline | 0.35 | 0.39 | ✓ DETECTED |
| Curriculum | 0.37 | 0.42 | ✓ DETECTED |

---

## Key Findings

1. **K pilote T_θ** via la loss (curriculum weighted)
2. **cut ⊥ bit** dans l'espace latent (cos ≈ 0.06)
3. **halt n'est pas linéairement séparable** (+6pp vs random)
4. **La dynamique est un objet second ordre** (sur trajectoires, pas feature statique)

---

## Theoretical Background

### Ω-Syntax
- Fixed structure, independent of learning
- `V_trig(x)` = ideal profile, `questionTrig(i, p)` = structural truth

### Dynamic Kernel K
- `approx_t(x)`: monotone refinement (intervals shrink)
- `val_t(x,i)`: once true, stays true
- `t_first^K(σ)`: structural difficulty measure

### P_vec
- **cut**: quadrant (depends only on angle)
- **bit**: question type (depends only on index)
- **halt**: learning difficulty (depends only on dynamics)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Angles | 360 |
| Questions | 8 types |
| Dataset | 2880 samples |
| Model params | 4,321 |
| Epochs | 60 |
| Curriculum weights | EARLY:1, MID:2, LATE:3, NEVER:4 |

---

## Dependencies

```
torch>=2.0
numpy
scipy
```

---

## License

Part of the Theory of Dynamic Orders for Computation project.

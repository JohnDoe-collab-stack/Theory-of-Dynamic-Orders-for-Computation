# Ω-Trig Experiment

Experimental validation of the **Dynamic Order Theory** framework with a trigonometric kernel.

---

## Overview

This experiment validates that a parameterized theory **T_θ** can learn to observe a fixed **Ω-structure**, with:

- **Ω** = fixed syntax (angles, profiles, questions)
- **K** = dynamic kernel with monotone refinement (approx_t, val_t, E(K))
- **T_θ** = neural observer that approximates Ω
- **P_vec** = orthogonal structure (cut ⊥ bit)

**Key finding**: K's temporal structure is **real, stable, and exploitable** by T_θ when used as an explicit learning objective.

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
│  approx_t(x) │ val_t(x,i) │ t_first^K │ halt_rank_K         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Theory T_θ (Learner)                     │
│  Multi-task: Y head (y*) + Halt head (halt_rank_K)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Results

### Multi-Task Validation (3 seeds)

| Condition | Y Accuracy | Halt Accuracy |
|-----------|------------|---------------|
| **K-real** | 99.2% ± 0.3% | **98.7% ± 0.4%** |
| **Shuffle** | 98.9% ± 0.5% | 48.1% ± 2.6% |
| **Δ Halt** | - | **+50.6pp** |

### λ_halt Sweep

| λ | K Halt | Shuffle Halt | Δ Halt |
|---|--------|--------------|--------|
| 0.0 | 3.9% | 13.4% | -9.5pp |
| 0.1 | 98.4% | 50.1% | **+48.3pp** |
| 0.5 | 99.1% | 50.1% | **+49.0pp** |
| 1.0 | 99.3% | 50.1% | **+49.2pp** |

### Sync K ↔ T_θ

- Pearson: ≈ 0.35
- Spearman: ≈ 0.39
- Verdict: ✓ DETECTED

### P_vec Structure

| Axis | Accuracy | Random |
|------|----------|--------|
| cut | 99% | 25% |
| bit | 100% | 25% |
| cos(W_cut, W_bit) | 0.06 | - |

---

## Interpretation

1. **K structure is REAL**: +50pp halt accuracy when K-real vs shuffle
2. **T_θ can learn K**: Multi-task achieves 99% halt accuracy
3. **Shuffle destroys signal**: Permuting t_first^K → random chance
4. **Curriculum alone doesn't help**: Task too easy for loss weighting
5. **P_vec is clean**: cut ⊥ bit in latent space

---

## Files

| File | Description |
|------|-------------|
| `trig_kernel.py` | Ω-syntax: AngleCode, PTrig, V_trig, questionTrig |
| `dataset_trig.py` | Dataset D = X_trig × I_trig with labels y* |
| `model_T.py` | Theory T_θ with embeddings |
| `train_T.py` | Baseline training |
| `train_T_curriculum.py` | K-guided curriculum training |
| `train_T_multitask.py` | Multi-task: y* + halt_rank_K |
| `dynamic_trig_kernel.py` | DynamicTrigKernel: monotone refinement |
| `analysis_T.py` | E(T_θ) inclusion analysis |
| `sync_K_T.py` | K ↔ T_θ synchronization |
| `analysis_pvec_trig.py` | P_vec linear probes |
| `run_ablation.py` | Curriculum ablation runner |
| `run_mt_validation.py` | Multi-task validation + λ sweep |

---

## Quick Start

```bash
# 1. Train baseline
python train_T.py

# 2. Export K structure
python -c "from dynamic_trig_kernel import *; DynamicTrigKernel(list(range(360))).export_t_first_K()"

# 3. Multi-task validation
python run_mt_validation.py --seeds 3 --lambda-sweep

# 4. Analyze
python analysis_T.py
python sync_K_T.py
python analysis_pvec_trig.py
```

---

## Conclusion

> "La structure temporelle K, définie au niveau des trajectoires d'un noyau dynamique, est réelle, stable, et effectivement exploitable par un réseau T dès qu'on en fait un objectif auxiliaire."

The "failure" of simple curriculum ≠ failure of method: it shows that on this saturated task, the dynamic order doesn't improve raw performance, but **K structure is there and learnable**.

---

## Dependencies

```
torch>=2.0
numpy
scipy
```

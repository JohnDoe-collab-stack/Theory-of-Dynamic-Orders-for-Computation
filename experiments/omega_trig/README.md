# Ω-Trig Experiment

Experimental validation of the **Dynamic Order Theory** framework with a trigonometric kernel.

---

## Overview

This experiment tests if a parameterized theory **T_θ** can learn to observe a fixed **Ω-structure** (trigonometry), with:

- **Ω** = fixed syntax (angles, profiles, questions)
- **K** = dynamic kernel with monotone refinement
- **T_θ** = neural observer that approximates Ω
- **P_vec** = orthogonal structure (cut ⊥ bit)

The key principle: **θ is an output**, not the source of structure.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Ω-Syntax (Fixed)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ AngleCode   │  │   PTrig     │  │   questionTrig    │   │
│  │ k ∈ {0..359}│  │ (sinI,cosI) │  │ (i, p) → {0,1}    │   │
│  └─────────────┘  └─────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Dynamic Kernel K (Monotone)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ approx_t(x) │  │  val_t(x,i) │  │     E(K)          │   │
│  │ → Profile   │  │ → {0,1}     │  │ stabilized asserts│   │
│  └─────────────┘  └─────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Theory T_θ (Learner)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ angle_embed │  │ index_embed │  │      MLP          │   │
│  │   (16-dim)  │  │   (8-dim)   │  │ → probability     │   │
│  └─────────────┘  └─────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   P_vec Analysis                            │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ cut (angle) │  │bit (q-type) │  │  halt (t_first)   │   │
│  │  quadrant   │  │ SIGN/GE_*   │  │ EARLY/MID/LATE    │   │
│  └─────────────┘  └─────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description | Lines |
|------|-------------|-------|
| `trig_kernel.py` | Ω-syntax: AngleCode, PTrig, V_trig, questionTrig | ~280 |
| `dataset_trig.py` | Dataset D = X_trig × I_trig with labels y* | ~180 |
| `model_T.py` | Theory T_θ: neural model with embeddings | ~240 |
| `train_T.py` | Training loop with BCE loss, checkpoints, logging | ~350 |
| `analysis_T.py` | E(T_θ) inclusion analysis, theory gradient | ~230 |
| `pvec_trig.py` | P_vec structure: CutClass, BitClass, HaltRank | ~220 |
| `analysis_pvec_trig.py` | Linear probes for cut/bit, orthogonality test | ~300 |
| `dynamic_trig_kernel.py` | DynamicTrigKernel: approx_t, val_t, traces, E(K) | ~320 |
| `sync_K_T.py` | K ↔ T_θ synchronization: correlation, confusion | ~280 |
| `__init__.py` | Package exports | ~50 |

---

## Quick Start

### 1. Run kernel self-tests

```bash
python trig_kernel.py
python dynamic_trig_kernel.py
python pvec_trig.py
```

### 2. Train theory T_θ

```bash
python train_T.py
```

Output:
- Checkpoints in `checkpoints_trig/`
- Logs with timestamps

### 3. Analyze theory gradient

```bash
python analysis_T.py
```

Expected:
```
VERDICT: ✓ SUCCESSFUL - Theory gradient detected
```

### 4. Analyze P_vec structure

```bash
python analysis_pvec_trig.py
```

Expected:
```
Cut: 99.1%
Bit: 100.0%
cos(W_cut, W_bit) ≈ 0.058
VERDICT: ✓ P_VEC STRUCTURE DETECTED
```

### 5. Synchronize K ↔ T_θ

```bash
python sync_K_T.py
```

Expected:
```
VERDICT: ✓ K-T SYNC DETECTED
```

---

## Configuration

### Dataset

| Parameter | Value |
|-----------|-------|
| N_FIXED | 360 angles |
| I_trig | 8 question types |
| Total samples | 2880 |
| Train / Val / Test | 2015 / 432 / 433 |

### Model T_θ

| Parameter | Value |
|-----------|-------|
| angle_embed_dim | 16 |
| index_embed_dim | 8 |
| hidden_dim | 64 |
| Total params | 4,321 |

### Training

| Parameter | Value |
|-----------|-------|
| Epochs | 60 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss | BCE |
| Checkpoints | [0, 1, 5, 10, 20, 50] |

### Dynamic Kernel K

| Parameter | Value |
|-----------|-------|
| T_max | 10 |
| Angles | 360 |
| Questions | 8 |

---

## Results

### T_θ Training

| Epoch | Val Acc | E(T_θ) |
|------:|--------:|-------:|
| 0 | 47.5% | 205 |
| 5 | 92.8% | 401 |
| 10 | 96.3% | 416 |
| 20 | 98.4% | 425 |
| 50 | 98.4% | 425 |

**Test Accuracy: 97.2%**

### Theory Gradient

| Transition | Inclusion |
|------------|-----------|
| 5 → 10 | 0.970 |
| 10 → 20 | 1.000 |
| 20 → 50 | 0.990 |

**Average: 0.980**

### P_vec Decodability

| Axis | Accuracy | Random |
|------|----------|--------|
| cut | 99.1% | 25% |
| bit | 100.0% | 25% |
| halt | 69.9% | 64% |

**cos(W_cut, W_bit) = 0.058** (quasi-orthogonal)

### Dynamic Kernel K

| Metric | Value |
|--------|-------|
| Monotonicity approx_t | ✓ |
| Monotonicity val_t | ✓ |
| E(K) | 53% stabilized |
| t_first avg | 6.2 |

---

## Theoretical Background

### Ω-Syntax

- **AngleCode**: x = k/360, k ∈ {0, ..., 359}
- **PTrig**: profile (sinI, cosI) with rational intervals
- **V_trig**: structural evaluation x → PTrig
- **questionTrig**: (i, p) → Bool, reads interval bounds

### Dynamic Kernel

- **approx_t(x)**: monotone refinement towards V_trig(x)
- **val_t(x, i)**: questionTrig(i, approx_t(x))
- **Monotonicity**: t ≤ t' ⇒ approx_t(x) ≤ approx_t'(x)
- **E(K)**: assertions that stabilize to true

### P_vec Structure

- **cut**: depends ONLY on angle (quadrant Q0-Q3)
- **bit**: depends ONLY on index type (SIGN, GE_*)
- **halt**: depends ONLY on dynamics (t_first)

By construction: **cut ⊥ bit** (orthogonal axes)

### Theory T_θ

- θ = 4,321 parameters (output, not source)
- Learns to approximate questionTrig via BCE
- E(T_θ) = correctly answered questions
- E(T_θ₁) ⊆ E(T_θ₂) for training epochs

---

## Success Criteria

| Criterion | Expected | Observed | Status |
|-----------|----------|----------|--------|
| Initial acc ~50% | 40-60% | 47.5% | ✓ |
| Final acc ≥90% | ≥90% | 97.2% | ✓ |
| E(T) monotonic | ≥95% | 96-100% | ✓ |
| Cut decodable | ≥85% | 99.1% | ✓ |
| Bit decodable | ≥85% | 100% | ✓ |
| cut ⊥ bit | \|cos\| ≤ 0.2 | 0.058 | ✓ |
| K monotone | ✓ | ✓ | ✓ |

---

## Dependencies

```
torch>=2.0
numpy
scipy (for Spearman correlation)
```

---

## License

Part of the Theory of Dynamic Orders for Computation project.

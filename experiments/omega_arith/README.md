# Ω_arith: Arithmetic with Carry Dynamics

> **Sequential K dynamics with non-local dependencies**

## Overview

This experiment validates the Ω–K–T–E framework on arithmetic addition where:
- **K** has truly **sequential dynamics** (column-by-column carry propagation)
- **t_first^K** depends on **non-local carry chains**
- **OOD test**: generalization to longer sequences (train n≤4, test n=5,6)

## Key Results

| Condition | Val Y | Val Halt | OOD Y | OOD Halt |
|-----------|-------|----------|-------|----------|
| K-real | 83.0% | **96.0%** | 84.5% | 44.1% |
| Shuffle | 84.4% | 43.0% | 81.1% | 39.5% |
| **Δ Halt** | - | **+53pp** | +3.4pp | **+4.6pp** |

### Interpretation

1. **In-distribution (+53pp)**: K structure is massively exploited
2. **OOD (+4.6pp)**: Halt doesn't generalize to new lengths
3. **Y generalizes** (~84%): additive semantics transfer
4. **Halt doesn't generalize**: dynamic order is local

## Architecture

```
Ω (Addition)         K (Carry Dynamics)      T_θ (Learner)
a + b = ?    →    Column-by-column    →    Digit embeddings
                  carry propagation        + attention pooling
                  t_first^K                + MLP heads
```

## Questions

- `SUM_GE(T)`: Is a + b ≥ T?
- `DIGIT_EQ(i, d)`: Is result digit at position i equal to d?
- `HAS_CARRY(i)`: Is there a carry out of column i?

## Quick Start

```bash
# Train K-real
python train.py --epochs 50 --output-dir checkpoints_K

# Train Shuffle (ablation)
python train.py --epochs 50 --output-dir checkpoints_shuffle --shuffle-K
```

## Files

```
omega_arith/
├── arith_kernel.py   # Addition + carry simulation
├── dataset.py        # OOD length split
└── train.py          # Multi-task training
```

## Theoretical Connection

This result aligns with the Lean impossibility theorems:
> No internal predicate can capture RealHalts globally (total/correct/complete)

The neural learner can approximate **local slices** of the P_vec profile,
but cannot internalize a globally correct rule for Halt across scales.

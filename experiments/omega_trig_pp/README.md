# Ω_trig++ : Logical Formulas Experiment

> Upgraded Ω with 16 logical formulas, 3-valued logic, and non-trivial halt dynamics.

## Architecture

```
Ω (World)              K (Oracle)              T_θ (Learner)
360 angles ×   →   3-valued logic    →   MLP (36K params)
16 formulas        interval refinement       Y head + Halt head
5760 samples       t_first^K                 BCE + λ·CE
```

## Key Results

| Condition | Y Test | Halt Test | Δ Halt |
|-----------|--------|-----------|--------|
| K-real | 98.9% | **88.6%** | - |
| Shuffle | 97.8% | 63.0% | **-25.6pp** |

**Signal**: K structure is exploited (+25pp halt accuracy).

## Formulas

16 logical combinations of atomic predicates:
- A1: sin≥0, A2: cos≥0, A3: |sin|≥|cos|
- A4: |sin|≥0.5, A5: |cos|≥0.5
- A7: |sin-cos|≤0.2 (diagonal proximity)

Example formulas:
- φ₀: (sin≥0 ∧ cos≥0)
- φ₇: ¬(sin≥0 ∧ cos≥0)
- φ₁₅: XOR(|sin|≥0.5, |cos|≥0.5)

## 3-Valued Logic

At each time t, predicates evaluate to {⊤, ⊥, ?}:
- ⊥ ∧ X = ⊥ (short-circuit)
- ⊤ ∨ X = ⊤ (short-circuit)
- ~? = ?

Formula decides when it leaves UNKNOWN state → t_first^K.

## Quick Start

```bash
# Test kernel
python dynamic_kernel.py

# Train K-real
python train.py --epochs 50 --output-dir checkpoints_K

# Train Shuffle (ablation)
python train.py --epochs 50 --output-dir checkpoints_shuffle --shuffle-K
```

## Files

```
omega_trig_pp/
├── trig_pp_kernel.py   # Atoms, formulas, 3-valued logic
├── dynamic_kernel.py   # Interval refinement, t_first
├── dataset.py          # Train/val/test splits
└── train.py            # Multi-task training
```

## Dependencies

```
torch>=2.0
```

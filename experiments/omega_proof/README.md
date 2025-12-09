# Ω_proof: Propositional Logic with Dynamic K

> **Best OOD generalization (+42pp Halt)**

## Overview

Dynamic kernel based on proof search:
- **Early stopping**: stops on first counterexample/witness
- **t_first**: number of valuations examined before decision
- **Halt ranks**: relative position in search space (t/2^n)

## Results

| Condition | Val Y | Val Halt | OOD Y | OOD Halt |
|-----------|-------|----------|-------|----------|
| K-real | 100% | **96%** | 89% | **75%** |
| Shuffle | 97% | 37% | 89% | 33% |
| **Δ Halt** | - | **+59pp** | - | **+42pp** |

## Key Insight

Unlike Ω_arith (OOD Δ +4.6pp), Ω_proof shows strong OOD generalization.
This suggests logical structure (proof depth) is more transferable than arithmetic structure (carry chains).

## Quick Start

```bash
python train.py --epochs 50 --output-dir checkpoints_K
python train.py --epochs 50 --output-dir checkpoints_shuffle --shuffle-K
```

# Theory of Dynamic Orders for Computation

> **A Lean 4 formalization of dynamic order theory with experimental validation**

[![Lean 4](https://img.shields.io/badge/Lean-4-blue)](https://lean-lang.org/)
[![Mathlib](https://img.shields.io/badge/Mathlib-latest-green)](https://github.com/leanprover-community/mathlib4)

## Overview

This project formalizes a complete theory of **dynamic orders** for computation, establishing a rigorous framework from temporal traces (`Rev`, `Halts`) to vector proofs, without going through ZFC or Turing-Gödel directly.

### Key Contributions

1. **Dynamic Layer**: Temporal traces, monotone closure (`up`), and the `Rev` operator for halting.
2. **Rev-CH Isomorphism**: Formal equivalence between reverse-halting and CH profiles.
3. **Effective Omega (Ω)**: Computable approximations of Chaitin's Ω with `Cut` and `Bit` programs.
4. **Vector Profiles (`P_vec`)**: Multi-dimensional characterization of logical phenomena via `(godel, omega, rank)` coordinates.
5. **Global Sphere Framework**: Termination bounds via fuel-based Lyapunov functions.
6. **Experimental Validation**: Neural network experiments demonstrating the K-real vs Shuffle-K gap.

## Project Structure

```
Theory-of-Dynamic-Orders-for-Computation/
├── LogicDissoc/
│   ├── LogicDissoc.lean    # Main formalization (1400+ lines)
│   ├── Sphere.lean         # Global sphere & termination bounds
│   └── FYI.lean            # AC internalisation impossibility
├── experiments/
│   ├── omega_proof/        # Propositional logic kernel (best OOD: +42pp)
│   └── omega_arith/        # Arithmetic kernel experiments
├── lakefile.lean           # Build configuration
└── README.md
```

## Formalization (Lean 4 + Mathlib)

### Core Modules

| File | Description |
|------|-------------|
| `LogicDissoc.lean` | Complete pipeline: Traces → Rev → Minsky Machine → Ω → P_vec → Order Algebra |
| `Sphere.lean` | Global sphere `B_R = {v | ∑ᵢ vᵢ ≤ R}`, `StrictStep`, `Fuel`, `Valley` |
| `FYI.lean` | Level 2 theorem: no internal AC_dyn predicate possible |

### Key Theorems

```lean
-- Rev equals Halts for monotone kits
theorem Rev_iff_Halts (K : RHKit) (DK : DetectsMonotone K) (T : Trace) :
    Rev K T ↔ Halts T

-- Strict steps decrease fuel
theorem strict_step_decreases_sum (x y : State) (h : StrictStep Step L x y) :
    Fuel L y < Fuel L x

-- Trajectory length bounded by initial fuel
theorem max_trajectory_length (R : Nat) (chain : Nat → State) (len : Nat)
    (h_start : GlobalProfile.InSphere R (L (chain 0)))
    (h_step : ∀ k, k < len → StrictStep Step L (chain k) (chain (k + 1))) :
    len ≤ R

-- Bit program halts iff OmegaBit matches
theorem bit_halts_iff (n : Nat) (a : Bool) :
    HaltsProg (Bit n a) ↔ OmegaBit n = a
```

## Experiments

### Ω_proof (Propositional Logic)

Dynamic kernel with early stopping on counterexamples.

| Condition | Val Halt | OOD Halt | Δ OOD |
|-----------|----------|----------|-------|
| K-real    | 96%      | **75%**  | —     |
| Shuffle-K | 37%      | 33%      | —     |
| **Δ**     | +59pp    | **+42pp**| ✓     |

**Key insight**: Logical structure (proof depth) is more transferable than arithmetic structure.

### Ω_arith (Arithmetic)

Halting-time prediction for addition/multiplication.

| Condition | Val Halt | OOD Halt |
|-----------|----------|----------|
| K-real    | 80%      | 47%      |
| Shuffle-K | 33%      | 42%      |

## Building

```bash
# Install dependencies
lake exe cache get

# Build all modules
lake build

# Check specific file
lake env lean LogicDissoc/Sphere.lean
```

## Requirements

- Lean 4.x
- Mathlib 4
- Python 3.8+ (for experiments)
- PyTorch (for training)

## Theoretical Background

The project establishes that:

1. **Dynamic choice `AC_dyn`** is the only axiomatic element (halting oracle).
2. **Vector profiles `P_vec`** characterize logical phenomena as order structures.
3. **CH and AC_dyn** properties separate on cut/bit axes via dependency requirements.
4. **Fuel-based termination** provides combinatorial bounds on execution length.

## License

MIT

## Citation

```bibtex
@misc{dynamic-orders-2024,
  title={Theory of Dynamic Orders for Computation},
  author={...},
  year={2024},
  howpublished={\url{https://github.com/.../Theory-of-Dynamic-Orders-for-Computation}}
}
```

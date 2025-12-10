# Theory of Dynamic Orders for Computation

> **A Lean 4 formalization of dynamic order theory with experimental validation**

[![Lean 4](https://img.shields.io/badge/Lean-4-blue)](https://lean-lang.org/)
[![Mathlib](https://img.shields.io/badge/Mathlib-latest-green)](https://github.com/leanprover-community/mathlib4)

## Overview

This project develops a **theory of dynamic orders for computation**, and formalizes it in Lean 4. The framework goes from temporal traces (`Rev`, `Halts`) up to vector profiles and order structures, without relying directly on ZFC or Turing–Gödel inside the core dynamic layer.

The theory is backed by **neural experiments** that probe how well models can learn halting-time structure (K-real) versus randomized halting signals (Shuffle-K).

### Main Contributions

1. **Dynamic Layer**  
   Temporal traces, monotone temporal closure (`up`), and a reverse-halting kernel `Rev` with:
   - `Rev ↔ Halts` for any kit that detects monotone families.
   - Robustness: many different reverse-halting kits collapse extensionally to the same `Halts`.

2. **Rev–CH Isomorphism**  
   An abstract equivalence between:
   - a `RevHalting` structure (program ↦ halts), and
   - a `CHProfile` (program ↦ CH-flag),
   giving a precise sense in which "Rev halting is isomorphic to CH" on a shared program space.

3. **Effective Omega (Ω)**  
   Computable approximations of Chaitin's Ω over a 2-counter Minsky machine:
   - `OmegaPartialScaled` (partial Ω),
   - `Cut` / `Bit` / `BitReal` programs,
   - a universal machine (`decodeProgram`, `encodeProgram`, `universalRun`).

4. **Vector Profiles (`P_vec`)**  
   A three-component profile for programs and logical phenomena:
   - `godel`   : Gödel-like / cut structure,
   - `omega`   : Ω-role (cut-like / bit-like / none),
   - `rank`    : obstruction level / abstract ordinal rank,
   with a lexicographic preorder and Boolean subalgebra structure on definable subsets.

5. **Global Sphere Framework**  
   A general "global sphere" formalism:
   - `GlobalProfile D := Fin D → Nat`,
   - `InSphere R v := ∑ᵢ vᵢ ≤ R`,
   - `StrictStep` and `Fuel := ∑ᵢ Lᵢ(x)`,
   - `Valley` as an absorbing, internally stable region.
   
   This yields a clean Lyapunov-style termination theorem:
   chains of strictly descending steps have length ≤ initial fuel ≤ `R`.

6. **AC_dyn & Non-Internalisation**  
   A concrete Rev–CH–AC system with:
   - an external dynamic choice operator `AC_dyn` on halting codes,
   - a Level 2 meta-theorem: no recursive consistent theory of ZFC strength can internalise this **specific** dynamic AC as a single total, correct and complete internal predicate for real halting (under a local reflection axiom for the meta-level halting profile).

7. **Experimental Validation**  
   Neural experiments (`Ω_proof`, `Ω_arith`) showing:
   - strong OOD generalisation for logical halting structure (K-real vs Shuffle-K),
   - weaker transfer for arithmetic carry chains.

---

## Project Structure

```text
Theory-of-Dynamic-Orders-for-Computation/
├── LogicDissoc/
│   ├── LogicDissoc.lean    # Main formalization: traces → Rev → Ω → P_vec → order algebra
│   ├── Sphere.lean         # Global sphere B_R, StrictStep, Fuel, Valley, Lyapunov-style bounds
│   └── FYI.lean            # Level 2: AC_dyn internalisation impossibility
├── experiments/
│   ├── omega_proof/        # Propositional logic halting kernel (best OOD: +42pp on Halt)
│   └── omega_arith/        # Arithmetic halting kernel experiments
├── lakefile.lean           # Build configuration
└── README.md
```

---

## Formalization (Lean 4 + Mathlib)

### Core Modules

| File | Description |
|------|-------------|
| `LogicDissoc.lean` | Full pipeline: traces → Rev → Minsky machine → Ω → `P_vec` → Boolean / order algebra |
| `Sphere.lean` | Global sphere `B_R = {v | ∑ᵢ vᵢ ≤ R}`, `StrictStep`, `Fuel`, `Valley`, Lyapunov-style bounds |
| `FYI.lean` | Level 2: no total internal predicate reproducing `AC_dyn` on real halting codes |

### Selected Theorems

```lean
-- Rev equals Halts for any kit detecting monotone families
theorem Rev_iff_Halts (K : RHKit) (DK : DetectsMonotone K) (T : Trace) :
    Rev K T ↔ Halts T

-- Strict steps decrease fuel
theorem strict_step_decreases_sum (x y : State) (h : StrictStep Step L x y) :
    Fuel L y < Fuel L x

-- Trajectory length is bounded by initial fuel ≤ R
theorem max_trajectory_length (R : Nat) (chain : Nat → State) (len : Nat)
    (h_start : GlobalProfile.InSphere R (L (chain 0)))
    (h_step : ∀ k, k < len → StrictStep Step L (chain k) (chain (k + 1))) :
    len ≤ R

-- Bit program halts iff its Ω-bit matches
theorem bit_halts_iff (n : Nat) (a : Bool) :
    HaltsProg (Bit n a) ↔ OmegaBit n = a
```

---

## Experiments

### Ω_proof (Propositional Logic Kernel)

Dynamic kernel with early stopping on the first counterexample/witness, and halting buckets defined by relative position in the `2^n` search space.

**Halting prediction (Halt head)**

| Condition | Val Halt | OOD Halt | Δ OOD |
|-----------|----------|----------|-------|
| K-real    | 96%      | **75%**  | —     |
| Shuffle-K | 37%      | 33%      | —     |
| **Gap**   | +59 pp   | **+42 pp** | ✓   |

**Observation:**
Logical structure (proof depth, search position) is highly transferable OOD when the model sees the true kernel (`K-real`), and collapses when halting labels are shuffled (`Shuffle-K`).

### Ω_arith (Arithmetic Kernel)

Halting-time prediction for arithmetic programs (e.g. addition / multiplication with carry structure).

| Condition | Val Halt | OOD Halt |
|-----------|----------|----------|
| K-real    | 80%      | 47%      |
| Shuffle-K | 33%      | 42%      |

**Observation:**
Arithmetic halting structure shows weaker OOD generalisation than propositional logic: carry chains are less easily transferred than proof-depth structure.

---

## Building

```bash
# Fetch Mathlib cache (recommended)
lake exe cache get

# Build all modules
lake build

# Check a specific file
lake env lean LogicDissoc/Sphere.lean
```

## Requirements

* Lean 4.x
* mathlib4
* Python 3.8+ (for experiments)
* PyTorch (for training kernels `Ω_proof` / `Ω_arith`)

---

## Theoretical Summary

The formalization establishes that:

1. The **dynamic halting layer** (`Rev`, `Halts`, `AC_dyn`) can be axiomatized with:
   * a minimal opaque halting oracle,
   * invariance `Rev ↔ Halts` for any kit detecting monotone families.

2. **Vector profiles** `P_vec = (godel, omega, rank)` provide an order-theoretic view of logical phenomena:
   * CH-like properties depend only on "cut" / Gödel components,
   * AC_dyn-like properties depend only on Ω-role / bit components,
     under suitable `LogicSpecs`.

3. The **global sphere framework** (`GlobalProfile`, `StrictStep`, `Fuel`, `Valley`) yields:
   * a general Lyapunov-style termination argument for any multi-dimensional monotone measure,
   * a uniform combinatorial bound on the length of strictly descending trajectories.

4. At Level 2, assuming a local reflection principle for the meta-level predicate built from `AC_dyn`, no recursive consistent theory of ZFC strength can internalise this specific Rev–CH–AC dynamic as a single total, correct and complete internal predicate for real halting.

---

## License

MIT

## Citation

```bibtex
@misc{dynamic-orders-2024,
  title        = {Theory of Dynamic Orders for Computation},
  author       = {...},
  year         = {2024},
  howpublished = {\url{https://github.com/.../Theory-of-Dynamic-Orders-for-Computation}}
}
```

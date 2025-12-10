import Mathlib.Data.Fin.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Ring

namespace LogicDissoc

open Finset
open scoped BigOperators

/-!
# Global Sphere and Multi-Cone Monotonicity

This module formalizes the "Global Sphere" framework for dynamic order theory.
It provides a foundation for reasoning about termination and stability in
systems with multi-dimensional resource profiles.

## Main Concepts

- **GlobalProfile**: A D-dimensional vector of natural numbers representing state resources
- **InSphere**: The global sphere B_R = {v | ∑ᵢ vᵢ ≤ R}
- **StrictStep**: A transition that strictly decreases at least one coordinate
- **Fuel**: The ℓ₁-norm (sum) of a profile, serving as a termination measure
- **Valley**: An absorbing set where no strict descent is possible
- **Mode**: Local transition with designated active coordinates

## Key Theorems

- `strict_step_decreases_sum`: Every strict step decreases fuel by ≥ 1
- `max_trajectory_length`: Chains of strict steps have length ≤ R
- `zero_fuel_stable`: States with zero fuel are stable
-/

-- ============================================================================
-- § 1. Global Profile & Sphere
-- ============================================================================

variable {State : Type}

/-- A global profile is a vector of `D` natural numbers. -/
def GlobalProfile (D : Nat) := Fin D → Nat

namespace GlobalProfile

variable {D : Nat}

/-- The sum of components (ℓ₁-norm) represents the total resource/rank. -/
@[simp]
def sum (v : GlobalProfile D) : Nat :=
  ∑ i : Fin D, v i

/-- The global sphere B_R: profiles with sum ≤ R. -/
@[simp]
def InSphere (R : Nat) (v : GlobalProfile D) : Prop :=
  v.sum ≤ R

/-- Being in the sphere is decidable. -/
instance (R : Nat) (v : GlobalProfile D) : Decidable (InSphere R v) :=
  inferInstanceAs (Decidable (v.sum ≤ R))

/-- The zero profile (all coordinates = 0). -/
def zero : GlobalProfile D := fun _ => 0

@[simp]
theorem sum_zero : (zero : GlobalProfile D).sum = 0 := by
  unfold sum zero
  simp only [Finset.sum_const_zero]

/-- Zero profile is in every sphere. -/
theorem zero_in_sphere (R : Nat) : InSphere R (zero : GlobalProfile D) := by
  unfold InSphere
  rw [sum_zero]
  exact Nat.zero_le R

/-- Pointwise order on profiles. -/
instance : LE (GlobalProfile D) where
  le v w := ∀ i, v i ≤ w i

theorem le_def (v w : GlobalProfile D) : v ≤ w ↔ ∀ i, v i ≤ w i := Iff.rfl

/-- Pointwise ≤ implies sum ≤. -/
theorem sum_le_sum_of_le {v w : GlobalProfile D} (h : v ≤ w) : v.sum ≤ w.sum := by
  simp only [sum]
  apply Finset.sum_le_sum
  intro i _
  exact h i

end GlobalProfile

-- ============================================================================
-- § 2. Dynamics & Monotonicity
-- ============================================================================

variable {D : Nat}
variable (Step : State → State → Prop)
variable (L : State → GlobalProfile D)

/-- Weak monotonicity: Step x y implies no coordinate increases. -/
def WeakMono (x y : State) : Prop :=
  Step x y → ∀ i : Fin D, L y i ≤ L x i

/-- Strict step: weakly monotone with at least one strictly decreasing coordinate. -/
def StrictStep (x y : State) : Prop :=
  WeakMono Step L x y ∧ Step x y ∧ ∃ i : Fin D, L y i < L x i

/-- Global strictness: every transition is a strict step. -/
def GloballyStrict : Prop :=
  ∀ x y, Step x y → StrictStep Step L x y

-- ============================================================================
-- § 3. Cones & Modes
-- ============================================================================

/-- A cone is a subset of the profile space (defined by constraints). -/
def Cone (D : Nat) := Set (GlobalProfile D)

/-- A mode defines a labeled transition relation with active coordinates. -/
structure Mode (State : Type) (D : Nat) where
  Label        : Type
  step_m       : Label → State → State → Prop
  activeCoords : Label → Set (Fin D)

/-- Mode-level monotonicity: active coords don't increase, at least one decreases. -/
def ModeMono (M : Mode State D) (L : State → GlobalProfile D)
    (label : M.Label) (x y : State) : Prop :=
  M.step_m label x y →
    (∀ i ∈ M.activeCoords label, L y i ≤ L x i) ∧
    (∃ i ∈ M.activeCoords label, L y i < L x i)

/-- Two modes have disjoint active coordinates for given labels. -/
def DisjointModes (M₁ M₂ : Mode State D) (l₁ : M₁.Label) (l₂ : M₂.Label) : Prop :=
  M₁.activeCoords l₁ ∩ M₂.activeCoords l₂ = ∅

-- ============================================================================
-- § 4. Stability & Valleys
-- ============================================================================

/-- A state is stable if it has no strictly descending successor.
    Plateau transitions (same profile) are allowed, but no strict descent. -/
def Stable (x : State) : Prop :=
  ∀ y, Step x y → ¬ StrictStep Step L x y

/-- A valley is an absorbing set with internal stability. -/
structure Valley (V : Set State) : Prop where
  absorb : ∀ {x y}, x ∈ V → Step x y → y ∈ V
  stable : ∀ {x}, x ∈ V → Stable Step L x

/-- The set of stable states. -/
def StableSet : Set State :=
  { x | Stable Step L x }

/-- Under GloballyStrict, stable means no successor at all. -/
theorem stable_iff_no_succ (h_strict : GloballyStrict Step L) (x : State) :
    Stable Step L x ↔ ∀ y, ¬ Step x y := by
  constructor
  · intro h_stable y h_step
    have := h_strict x y h_step
    exact h_stable y h_step this
  · intro h_no_succ y h_step _
    exact h_no_succ y h_step

-- ============================================================================
-- § 5. Fuel & Termination Bounds
-- ============================================================================

/-- The fuel of a state is the sum of its profile components. -/
@[simp]
def Fuel (x : State) : Nat :=
  (L x).sum

/-- A strict step strictly decreases fuel. -/
theorem strict_step_decreases_sum (x y : State) (h : StrictStep Step L x y) :
    Fuel L y < Fuel L x := by
  have h_mono : ∀ i : Fin D, L y i ≤ L x i := h.1 h.2.1
  rcases h.2.2 with ⟨k, hk_strict⟩
  simp only [Fuel, GlobalProfile.sum]
  apply Finset.sum_lt_sum
  · intro i _; exact h_mono i
  · exact ⟨k, Finset.mem_univ k, hk_strict⟩

/-- Strict step decreases fuel by at least 1. -/
theorem strict_step_fuel_succ (x y : State) (h : StrictStep Step L x y) :
    Fuel L x ≥ Fuel L y + 1 :=
  Nat.succ_le_of_lt (strict_step_decreases_sum Step L x y h)

/-- A state with zero fuel is stable under GloballyStrict. -/
theorem zero_fuel_stable (x : State) (h_fuel : Fuel L x = 0)
    (_ : GloballyStrict Step L) : Stable Step L x := by
  intro y h_step h_strict_step
  have decrease := strict_step_decreases_sum Step L x y h_strict_step
  simp only [Fuel, GlobalProfile.sum] at h_fuel decrease
  rw [h_fuel] at decrease
  exact Nat.not_lt_zero _ decrease

/-- Max trajectory length: strict chains from sphere have length ≤ R. -/
theorem max_trajectory_length (R : Nat) (chain : Nat → State) (len : Nat)
    (h_start : GlobalProfile.InSphere R (L (chain 0)))
    (h_step : ∀ k, k < len → StrictStep Step L (chain k) (chain (k + 1))) :
    len ≤ R := by
  -- fuel at step k
  let f : Nat → Nat := fun k => Fuel L (chain k)

  -- fuel strictly decreases along the chain
  have f_strict : ∀ k, k < len → f (k + 1) < f k := fun k hk =>
    strict_step_decreases_sum Step L (chain k) (chain (k + 1)) (h_step k hk)

  -- In a strictly decreasing sequence of Nats, f 0 ≥ f k + k
  have trajectory_fuel_bound : ∀ k, k ≤ len → f 0 ≥ f k + k := by
    intro k hk
    induction k with
    | zero => simp
    | succ i ih =>
        have hi_le : i ≤ len := Nat.le_of_succ_le hk
        have ih_apply := ih hi_le
        have h_step_i : i < len := Nat.lt_of_succ_le hk
        have decrease := f_strict i h_step_i
        have ineq : f i ≥ f (i + 1) + 1 := Nat.succ_le_of_lt decrease
        calc
          f 0 ≥ f i + i := ih_apply
          _   ≥ (f (i + 1) + 1) + i := Nat.add_le_add_right ineq i
          _   = f (i + 1) + (i + 1) := by ring

  have final_bound := trajectory_fuel_bound len (Nat.le_refl len)
  have f0_ge_len : f 0 ≥ len := calc
    f 0 ≥ f len + len := final_bound
    _   ≥ 0 + len := Nat.add_le_add_right (Nat.zero_le _) len
    _   = len := Nat.zero_add len

  exact Nat.le_trans f0_ge_len h_start

-- ============================================================================
-- § 6. Valley Characterization
-- ============================================================================

/-- The set of states with minimal fuel (= 0). -/
def MinimalFuelSet : Set State :=
  { x | Fuel L x = 0 }

/-- Under GloballyStrict, minimal fuel states form a valley. -/
theorem minimal_fuel_valley (h_strict : GloballyStrict Step L) :
    Valley Step L (MinimalFuelSet L) where
  absorb := by
    intro x y hx h_step
    simp only [MinimalFuelSet, Set.mem_setOf_eq, Fuel] at hx ⊢
    have stable_x := zero_fuel_stable Step L x hx h_strict
    have strict := h_strict x y h_step
    exact absurd strict (stable_x y h_step)
  stable := by
    intro x hx
    exact zero_fuel_stable Step L x hx h_strict

/-- Fuel is a Lyapunov function: strictly decreasing on transitions. -/
theorem fuel_lyapunov (h_strict : GloballyStrict Step L) (x y : State) (h_step : Step x y) :
    Fuel L y < Fuel L x :=
  strict_step_decreases_sum Step L x y (h_strict x y h_step)

-- ============================================================================
-- § 7. Mode Switching
-- ============================================================================

/-- A mode schedule assigns a mode label at each step. -/
structure ModeSchedule (M : Mode State D) where
  schedule : Nat → M.Label

/-- Execution follows a mode schedule if each step uses the scheduled mode. -/
def FollowsSchedule (M : Mode State D) (sched : ModeSchedule M)
    (chain : Nat → State) : Prop :=
  ∀ k, M.step_m (sched.schedule k) (chain k) (chain (k + 1))

-- ============================================================================
-- § 8. Helpers
-- ============================================================================

/-- A state is in the sphere if its profile sum is ≤ R. -/
@[simp]
def InSphereState (R : Nat) (x : State) : Prop :=
  GlobalProfile.InSphere R (L x)

/-- Equivalence for sphere membership. -/
theorem in_sphere_iff (R : Nat) (x : State) :
    InSphereState L R x ↔ Fuel L x ≤ R := by
  simp [InSphereState, GlobalProfile.InSphere, Fuel]

end LogicDissoc

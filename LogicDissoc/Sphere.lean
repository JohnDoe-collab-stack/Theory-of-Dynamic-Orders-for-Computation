import Mathlib.Data.Fin.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Ring

namespace LogicDissoc

/-!
# Global Sphere and Multi-Cone Monotonicity

This module formalizes the "Global Sphere" framework given in the specification.
It generalizes dynamic orders by considering a global D-dimensional profile
constrained within a bounded sphere, with local monotonicity cones.
-/

variable {State : Type} -- Abstract state space

-- ============================================================================
-- 1. GLOBAL PROFILE & SPHERE
-- ============================================================================

/-- A global profile is a vector of D natural numbers. -/
def GlobalProfile (D : Nat) := Fin D → Nat

namespace GlobalProfile

-- ============================================================================
-- 2. DYNAMICS & MONOTONICITY
-- ============================================================================


variable {D : Nat}

/-- The sum of components (L1-norm) represents the total resource/rank. -/
def sum (v : GlobalProfile D) : Nat :=
  ∑ i : Fin D, v i

/-- The Global Sphere B_R: set of profiles with sum ≤ R. -/
def InSphere (R : Nat) (v : GlobalProfile D) : Prop :=
  v.sum ≤ R

/-- Being in the sphere is a decidable property. -/
instance (R : Nat) (v : GlobalProfile D) : Decidable (InSphere R v) :=
  inferInstanceAs (Decidable (v.sum ≤ R))

end GlobalProfile

-- ============================================================================
-- 2. DYNAMICS & MONOTONICITY
-- ============================================================================

variable {D : Nat}

-- Abstract transition relation on states.
variable (Step : State → State → Prop)

-- A system is equipped with a global profile map L.
variable (L : State → GlobalProfile D)

-- 1. Weak Monotonicity: No coordinate increases.
def WeakMono (x y : State) : Prop :=
  Step x y → ∀ i : Fin D, L y i ≤ L x i

/-- 2. Strict Step: Weakly monotone AND at least one coordinate decreases. -/
def StrictStep (x y : State) : Prop :=
  WeakMono Step L x y ∧ Step x y ∧ ∃ i : Fin D, L y i < L x i

/-- Global Strictness: Every step in the system is a strict step. -/
def GloballyStrict : Prop :=
  ∀ x y, Step x y → StrictStep Step L x y

-- ============================================================================
-- 3. CONES & MODES
-- ============================================================================

/-- A Cone is simply a subset of the Global Profile space (usually defined by constraints). -/
def Cone (D : Nat) := Set (GlobalProfile D)

/-- A Mode m defines a specific transition relation and a set of active coordinates. -/
structure Mode (State : Type) (D : Nat) where
  Label : Type
  step_m : Label → State → State → Prop
  active_coords : Label → Set (Fin D)

/-- Multi-mode Monotonicity: For a mode m, active coords decrease strictly. -/
def ModeMono (M : Mode State D) (label : M.Label) (x y : State) : Prop :=
  M.step_m label x y →
    (∀ i ∈ M.active_coords label, L y i ≤ L x i) ∧
    (∃ i ∈ M.active_coords label, L y i < L x i)

-- ============================================================================
-- 4. STABILITY & VALLEYS
-- ============================================================================

/-- A state is Stable if no transition is possible (or strictly no descent). -/
def Stable (x : State) : Prop :=
  ∀ y, ¬ Step x y

/-- A Valley is a set of states that is Absorbing and Internally Stable. -/
structure Valley (V : Set State) : Prop where
  absorb : ∀ x y, x ∈ V → Step x y → y ∈ V
  stable : ∀ x, x ∈ V → Stable Step x

-- ============================================================================
-- 5. COMBINATORIAL BOUNDS (Trajectory Length)
-- ============================================================================

/--
The "Fuel" of a state is its profile sum.
If the system is GloballyStrict, every step reduces the Fuel by at least 1.
-/
def Fuel (x : State) : Nat := (L x).sum

lemma strict_step_decreases_sum (x y : State) (h : StrictStep Step L x y) :
    (L y).sum < (L x).sum := by
  -- Unpack strict step
  have h_mono : ∀ i, L y i ≤ L x i := h.1 h.2.1
  rcases h.2.2 with ⟨k, hk_strict⟩

  rw [GlobalProfile.sum, GlobalProfile.sum]
  apply Finset.sum_lt_sum
  · intro i _
    exact h_mono i
  · use k
    simp only [Finset.mem_univ, true_and]
    exact hk_strict

/--
Theorem: Max Trajectory Length.
Any chain of strict steps starting from a state with sum ≤ R has length at most R.
-/
theorem max_trajectory_length (R : Nat) (chain : Nat → State) (len : Nat)
    (h_start : GlobalProfile.InSphere R (L (chain 0)))
    (h_step : ∀ k, k < len → StrictStep Step L (chain k) (chain (k + 1))) :
    len ≤ R := by
  -- The fuel at step k is f_k. We know f_{k+1} < f_k.
  let f := fun k => (L (chain k)).sum
  have f_strict : ∀ k, k < len → f (k + 1) < f k := by
    intro k hk
    apply strict_step_decreases_sum Step L
    exact h_step k hk

  -- A strictly decreasing sequence of Nats of length L implies f(0) ≥ L
  -- (since f(L) ≥ 0 and each step adds at least 1 difference)
  -- Helper lemma: in a strictly decreasing sequence of nats, f(0) ≥ f(k) + k
  have trajectory_fuel_bound : ∀ k, k ≤ len → f 0 ≥ f k + k := by
    intro k hk
    induction k with
    | zero => simp
    | succ i ih =>
      have hi_le : i ≤ len := Nat.le_of_succ_le hk
      have ih_apply := ih hi_le
      have h_step_i : i < len := Nat.lt_of_succ_le hk
      have decrease := f_strict i h_step_i
      -- We have f(i) > f(i+1), so f(i) ≥ f(i+1) + 1
      have ineq : f i ≥ f (i + 1) + 1 := Nat.succ_le_of_lt decrease
      -- Combine: f(0) ≥ f(i) + i ≥ f(i+1) + 1 + i = f(i+1) + (i+1)
      calc
        f 0 ≥ f i + i := ih_apply
        _   ≥ (f (i + 1) + 1) + i := Nat.add_le_add_right ineq i
        _   = f (i + 1) + (i + 1) := by ring

  -- Apply to k = len
  have final_bound := trajectory_fuel_bound len (Nat.le_refl len)
  -- Since f(len) ≥ 0, we have f(0) ≥ len
  have f0_ge_len : f 0 ≥ len := calc
    f 0 ≥ f len + len := final_bound
    _   ≥ 0 + len := Nat.add_le_add_right (Nat.zero_le _) len
    _   = len := by simp

  -- Conclusion
  unfold GlobalProfile.InSphere at h_start
  exact Nat.le_trans f0_ge_len h_start

end LogicDissoc

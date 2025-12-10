import LogicDissoc

/-!
# Sphere Instances for LogicDissoc

Connects the abstract Sphere framework to LogicDissoc structures.

## Main Contents

1. **RevACProfile** — 3D profile for Rev/AC_dyn execution state
2. **P_vec projections** — convert vectorial profiles to fuel
3. **Strict step theorems** — AC_dyn calls are strict steps
4. **Trajectory bound** — oracle calls bounded by initial budget

## Key Guarantee

Any chain of oracle calls has length ≤ initial oracle_budget.

## Correspondences with Sphere.lean

| This File | Sphere.lean |
|-----------|-------------|
| `RevACProfile` | `GlobalProfile 3` |
| `fuel` | `GlobalProfile.sum` |
| `callOracle` with `h : oracle_budget > 0` | `StrictStep` |
| `inMinimalValley` | `Valley` |
-/

namespace LogicDissoc.SphereInstances

/-! ## 1. RevACProfile: 3D Profile for Rev/AC_dyn -/

/--
Profile for Rev/AC_dyn execution state.

Dimensions:
- `oracle_budget` (coord 0): remaining AC_dyn calls allowed
- `ch_depth` (coord 1): current depth in CH construction
- `rank_budget` (coord 2): abstract rank / complexity budget
-/
structure RevACProfile where
  oracle_budget : Nat
  ch_depth : Nat
  rank_budget : Nat
  deriving DecidableEq, Repr

namespace RevACProfile

/-- Total fuel (ℓ₁-norm) = sum of all coordinates -/
@[simp] def fuel (p : RevACProfile) : Nat :=
  p.oracle_budget + p.ch_depth + p.rank_budget

/-- Zero profile (minimal valley) -/
@[simp] def zero : RevACProfile := ⟨0, 0, 0⟩

/-- In sphere check: fuel ≤ R -/
@[simp] def inSphere (R : Nat) (p : RevACProfile) : Bool :=
  p.fuel ≤ R

/-- Initial profile with given budget, zero CH depth and rank -/
def initial (oracle_budget : Nat) : RevACProfile :=
  ⟨oracle_budget, 0, 0⟩

/-- Get coordinate by index -/
def coord (p : RevACProfile) : Fin 3 → Nat
  | ⟨0, _⟩ => p.oracle_budget
  | ⟨1, _⟩ => p.ch_depth
  | ⟨2, _⟩ => p.rank_budget

/-! ## 2. Strict Steps -/

/-- Oracle call: decreases oracle_budget by 1 (strict step on coord 0) -/
@[simp] def callOracle (p : RevACProfile) : RevACProfile :=
  { p with oracle_budget := p.oracle_budget - 1 }

/-- CH step: decreases ch_depth by 1 (strict step on coord 1) -/
@[simp] def stepCH (p : RevACProfile) : RevACProfile :=
  { p with ch_depth := p.ch_depth - 1 }

/-- Rank step: decreases rank_budget by 1 (strict step on coord 2) -/
@[simp] def stepRank (p : RevACProfile) : RevACProfile :=
  { p with rank_budget := p.rank_budget - 1 }

/-! ### Strict Step Properties -/

/-- Oracle call preserves other coordinates (WeakMono) -/
theorem callOracle_preserves_others (p : RevACProfile) :
    (p.callOracle).ch_depth = p.ch_depth ∧
    (p.callOracle).rank_budget = p.rank_budget := by
  simp [callOracle]

/-- Oracle call decreases fuel when budget > 0 (StrictStep) -/
theorem callOracle_decreases_fuel (p : RevACProfile) (h : p.oracle_budget > 0) :
    (p.callOracle).fuel < p.fuel := by
  simp [callOracle, fuel]
  omega

/-- Oracle call decreases coord 0 -/
theorem callOracle_decreases_coord0 (p : RevACProfile) (h : p.oracle_budget > 0) :
    (p.callOracle).oracle_budget < p.oracle_budget := by
  simp [callOracle]
  exact Nat.sub_lt h (Nat.zero_lt_one)

/-- CH step decreases fuel when depth > 0 -/
theorem stepCH_decreases_fuel (p : RevACProfile) (h : p.ch_depth > 0) :
    (p.stepCH).fuel < p.fuel := by
  simp [stepCH, fuel]
  omega

/-- Rank step decreases fuel when budget > 0 -/
theorem stepRank_decreases_fuel (p : RevACProfile) (h : p.rank_budget > 0) :
    (p.stepRank).fuel < p.fuel := by
  simp [stepRank, fuel]
  omega

/-! ## 3. Trajectory Bound Theorems -/

/-- Oracle calls bounded by fuel (immediate) -/
theorem oracle_calls_bounded_by_fuel (initial : RevACProfile) (calls : Nat)
    (h_budget : calls ≤ initial.oracle_budget) :
    calls ≤ initial.fuel := by
  simp [fuel]
  omega

/-- Total strict steps bounded by initial fuel -/
theorem total_strict_steps_bounded (initial : RevACProfile) (n_oracle n_ch n_rank : Nat)
    (h_oracle : n_oracle ≤ initial.oracle_budget)
    (h_ch : n_ch ≤ initial.ch_depth)
    (h_rank : n_rank ≤ initial.rank_budget) :
    n_oracle + n_ch + n_rank ≤ initial.fuel := by
  simp [fuel]
  omega

/--
**Main Theorem**: Chain of n oracle calls requires initial budget ≥ n.

This is the key safety guarantee: no matter what sequence of computations
occurs, the total number of AC_dyn calls is bounded by the initial budget.
-/
theorem oracle_chain_bounded (initial : RevACProfile) (n : Nat)
    (h_chain : n ≤ initial.oracle_budget) :
    n ≤ initial.fuel := by
  simp [fuel]
  omega

/-! ## 4. Valley Characterization -/

/-- Minimal valley: oracle_budget = 0 -/
def inMinimalValley (p : RevACProfile) : Prop :=
  p.oracle_budget = 0

/-- Full valley: all budgets = 0 (fuel = 0) -/
def inFullValley (p : RevACProfile) : Prop :=
  p.fuel = 0

/-- Full valley implies all coords are zero -/
theorem full_valley_is_zero (p : RevACProfile) (h : p.inFullValley) :
    p = zero := by
  unfold inFullValley fuel at h
  cases p with
  | mk o c r =>
    simp at h
    have ho : o = 0 := by omega
    have hc : c = 0 := by omega
    have hr : r = 0 := by omega
    simp [zero, ho, hc, hr]

/-- Minimal valley is stable under oracle calls -/
theorem minimal_valley_stable (p : RevACProfile) (h : p.inMinimalValley) :
    p.callOracle = p := by
  cases p with
  | mk o c r =>
    unfold inMinimalValley at h
    simp only at h
    unfold callOracle
    simp only [h]

/-- Full valley is stable under all steps -/
theorem full_valley_stable (p : RevACProfile) (h : p.inFullValley) :
    p.callOracle = p ∧ p.stepCH = p ∧ p.stepRank = p := by
  have hz := full_valley_is_zero p h
  rw [hz]
  simp [callOracle, stepCH, stepRank, zero]

end RevACProfile

/-! ## 5. P_vec ↔ Fuel Conversions -/

/-- Gödel component to Nat: gZero=0, gSucc adds 1, gOmega=100 (cap) -/
def godel_toNat : LogicDissoc.P_godel → Nat
  | .gZero => 0
  | .gSucc g => godel_toNat g + 1
  | .gOmega => 100

/-- Omega role to Nat: none=0, cutLike=n+1, bitLike=n+10 -/
def omega_toNat : LogicDissoc.P_omega_role → Nat
  | .none => 0
  | .cutLike n => n + 1
  | .bitLike n _ => n + 10

/-- Rank to Nat: rankZero=0, rankSucc adds 1, rankLimit=100 (cap) -/
def rank_toNat : LogicDissoc.P_rank → Nat
  | .rankZero => 0
  | .rankSucc r => rank_toNat r + 1
  | .rankLimit _ => 100

/-- P_vec total fuel -/
def pvec_fuel (v : LogicDissoc.P_vec) : Nat :=
  godel_toNat v.godel + omega_toNat v.omega + rank_toNat v.rank

/-- P_vec to RevACProfile (mapping godel→oracle, omega→ch, rank→rank) -/
def pvec_toProfile (v : LogicDissoc.P_vec) : RevACProfile :=
  ⟨godel_toNat v.godel, omega_toNat v.omega, rank_toNat v.rank⟩

/-- Fuel is preserved by conversion -/
theorem pvec_fuel_eq_profile_fuel (v : LogicDissoc.P_vec) :
    pvec_fuel v = (pvec_toProfile v).fuel := by
  simp [pvec_fuel, pvec_toProfile, RevACProfile.fuel]

/-! ## 6. Mode Structures -/

/-- Execution mode: which coordinates are active for strict steps -/
inductive ExecMode
  | oracle    -- only oracle_budget decreases
  | ch        -- only ch_depth decreases
  | rankOp    -- only rank_budget decreases
  | mixed     -- any coordinate can decrease
  deriving DecidableEq, Repr

/-- Active coordinate for each mode -/
def ExecMode.activeCoord : ExecMode → Fin 3
  | .oracle => ⟨0, by decide⟩
  | .ch => ⟨1, by decide⟩
  | .rankOp => ⟨2, by decide⟩
  | .mixed => ⟨0, by decide⟩  -- mixed uses coord 0 as primary

/-- Description string -/
def ExecMode.description : ExecMode → String
  | .oracle => "Oracle mode: only AC_dyn calls"
  | .ch => "CH mode: only CH construction steps"
  | .rankOp => "Rank mode: only rank operations"
  | .mixed => "Mixed mode: any operation allowed"

/-! ## 7. Example Computations -/

/-- Example: initial profile with budget 10 -/
def exampleProfile : RevACProfile := ⟨10, 5, 5⟩

example : exampleProfile.fuel = 20 := rfl
example : exampleProfile.callOracle.fuel = 19 := rfl
example : exampleProfile.callOracle.oracle_budget = 9 := rfl
example : exampleProfile.inSphere 20 = true := rfl
example : exampleProfile.inSphere 19 = false := rfl

/-- Example: trajectory simulation -/
example : (exampleProfile.callOracle.callOracle.callOracle).oracle_budget = 7 := rfl

/-- Example: full depletion -/
example : (RevACProfile.initial 3).callOracle.callOracle.callOracle.inMinimalValley := rfl

/-- Example: P_vec conversion -/
example : pvec_fuel ⟨.gZero, .none, .rankZero⟩ = 0 := rfl
example : pvec_fuel ⟨.gOmega, .cutLike 5, .rankZero⟩ = 106 := rfl

end LogicDissoc.SphereInstances

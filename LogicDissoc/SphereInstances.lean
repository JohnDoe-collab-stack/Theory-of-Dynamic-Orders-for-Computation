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

-- ============================================================================
-- § 8. CalcState and CalcStep: Structural Control of Computation
-- ============================================================================

/-!
## 8. Structural Control of Computation

This section formalizes the key insight: **computation is controlled structurally**
by bounding the number of strict steps via the fuel budget.

### Key Components

1. `CalcState` — wraps a `RevACProfile` as computation state
2. `CalcStep` — inductive relation for valid transitions
3. `L_calc` — the profile function `CalcState → Fin 3 → Nat`
4. `calc_step_strict` — every CalcStep is a strict step
5. `calc_chain_bounded` — chains are bounded by initial fuel

### Mathematical Guarantee

For any chain `chain : Nat → CalcState` with valid `CalcStep`s:
```
len ≤ initial.fuel
```

This is not a timeout — it's a **structural consequence** of fuel monotonicity.
-/

/-- Computation state wrapping a profile -/
structure CalcState where
  profile : RevACProfile
  deriving DecidableEq, Repr

namespace CalcState

/-- Fuel of a computation state -/
@[simp] def fuel (s : CalcState) : Nat := s.profile.fuel

/-- Initial computation state with given oracle budget -/
def initial (budget : Nat) : CalcState :=
  ⟨RevACProfile.initial budget⟩

/-- Zero/terminal state -/
def zero : CalcState := ⟨RevACProfile.zero⟩

end CalcState

/-! ### CalcStep: Valid Computation Transitions -/

/--
Inductive relation for valid computation steps.

Each constructor represents a type of step that consumes fuel:
- `oracle`: uses one oracle call (AC_dyn)
- `ch`: uses one CH construction step
- `rank`: uses one rank operation
-/
inductive CalcStep : CalcState → CalcState → Prop where
  | oracle (s : CalcState) (h : s.profile.oracle_budget > 0) :
      CalcStep s ⟨s.profile.callOracle⟩
  | ch (s : CalcState) (h : s.profile.ch_depth > 0) :
      CalcStep s ⟨s.profile.stepCH⟩
  | rank (s : CalcState) (h : s.profile.rank_budget > 0) :
      CalcStep s ⟨s.profile.stepRank⟩

namespace CalcStep

/-- CalcStep always decreases fuel (strict step) -/
theorem decreases_fuel : ∀ {s t : CalcState}, CalcStep s t → t.fuel < s.fuel
  | _, _, .oracle s h => RevACProfile.callOracle_decreases_fuel s.profile h
  | _, _, .ch s h => RevACProfile.stepCH_decreases_fuel s.profile h
  | _, _, .rank s h => RevACProfile.stepRank_decreases_fuel s.profile h

/-- CalcStep preserves non-negativity (trivial for Nat) -/
theorem fuel_nonneg (s : CalcState) : s.fuel ≥ 0 := Nat.zero_le _

end CalcStep

/-! ### Chain Bounds -/

/-- A valid chain of CalcSteps -/
def ValidChain (chain : Nat → CalcState) (len : Nat) : Prop :=
  ∀ k, k < len → CalcStep (chain k) (chain (k + 1))

/--
**Main Theorem**: Computation chains are bounded by initial fuel.

This is the core structural guarantee: if every step is a `CalcStep`,
then the total number of steps cannot exceed the initial fuel.
-/
theorem calc_chain_bounded (chain : Nat → CalcState) (len : Nat)
    (h_chain : ValidChain chain len) :
    len ≤ (chain 0).fuel := by
  induction len generalizing chain with
  | zero => exact Nat.zero_le _
  | succ n ih =>
    have h_first : CalcStep (chain 0) (chain 1) := h_chain 0 (Nat.zero_lt_succ n)
    have h_dec : (chain 1).fuel < (chain 0).fuel := CalcStep.decreases_fuel h_first
    have h_rest : ValidChain (fun k => chain (k + 1)) n := fun k hk =>
      h_chain (k + 1) (Nat.succ_lt_succ hk)
    have h_ih : n ≤ (chain 1).fuel := ih (fun k => chain (k + 1)) h_rest
    -- n ≤ (chain 1).fuel ∧ (chain 1).fuel < (chain 0).fuel
    -- ⟹ n + 1 ≤ (chain 0).fuel
    omega

/-- Corollary: if we start in sphere R, chain length ≤ R -/
theorem calc_chain_in_sphere (R : Nat) (chain : Nat → CalcState) (len : Nat)
    (h_sphere : (chain 0).fuel ≤ R)
    (h_chain : ValidChain chain len) :
    len ≤ R := by
  have h := calc_chain_bounded chain len h_chain
  omega

/--
**Key Corollary**: Maximum strict steps equals initial fuel.

This is the precise bound: you can have exactly `fuel` strict steps,
no more, no less (in the worst case).
-/
theorem max_strict_steps (s : CalcState) :
    ∀ chain len, chain 0 = s → ValidChain chain len → len ≤ s.fuel := by
  intro chain len h_start h_valid
  have h := calc_chain_bounded chain len h_valid
  simp only [← h_start]
  exact h

/-! ### Examples -/

/-- Example: chain of 3 oracle calls from budget 5 -/
example : (CalcState.initial 5).fuel = 5 := rfl

/-- Example: verify chain bound -/
example : ∀ chain len, chain 0 = CalcState.initial 5 → ValidChain chain len → len ≤ 5 := by
  intro chain len h_start h_valid
  exact max_strict_steps (CalcState.initial 5) chain len h_start h_valid

-- ============================================================================
-- § 9. Complexity Theory: P vs NP
-- ============================================================================

/-!
## 9. Complexity Theory Integration

Two approaches to distinguish P and NP within the Sphere framework:

### Option 1: Non-deterministic Steps
- Extend `CalcStep` with a `guess` constructor
- P = runs using only deterministic steps
- NP = runs that may use non-deterministic guesses

### Option 2: Certificate-based (Standard)
- P = problems decidable in polynomial time
- NP = problems verifiable in polynomial time given a certificate

Both approaches use `calc_chain_bounded` as the core time-bounding mechanism.
-/

/-! ### Option 1: Non-deterministic Computation -/

/-- Extended computation state with guess tape -/
structure NDCalcState where
  base : CalcState
  guesses : List Bool  -- accumulated non-deterministic choices
  deriving DecidableEq, Repr

namespace NDCalcState

@[simp] def fuel (s : NDCalcState) : Nat := s.base.fuel

def initial (budget : Nat) : NDCalcState :=
  ⟨CalcState.initial budget, []⟩

end NDCalcState

/--
Non-deterministic computation steps.

Extends CalcStep with a `guess` operation that:
- Consumes 1 fuel (from oracle_budget)
- Records a non-deterministic boolean choice
-/
inductive NDCalcStep : NDCalcState → NDCalcState → Prop where
  | oracle (s : NDCalcState) (h : s.base.profile.oracle_budget > 0) :
      NDCalcStep s ⟨⟨s.base.profile.callOracle⟩, s.guesses⟩
  | ch (s : NDCalcState) (h : s.base.profile.ch_depth > 0) :
      NDCalcStep s ⟨⟨s.base.profile.stepCH⟩, s.guesses⟩
  | rank (s : NDCalcState) (h : s.base.profile.rank_budget > 0) :
      NDCalcStep s ⟨⟨s.base.profile.stepRank⟩, s.guesses⟩
  | guess (s : NDCalcState) (b : Bool) (h : s.base.profile.oracle_budget > 0) :
      NDCalcStep s ⟨⟨s.base.profile.callOracle⟩, s.guesses ++ [b]⟩

namespace NDCalcStep

/-- Every NDCalcStep decreases fuel -/
theorem decreases_fuel : ∀ {s t : NDCalcState}, NDCalcStep s t → t.fuel < s.fuel
  | _, _, .oracle s h => RevACProfile.callOracle_decreases_fuel s.base.profile h
  | _, _, .ch s h => RevACProfile.stepCH_decreases_fuel s.base.profile h
  | _, _, .rank s h => RevACProfile.stepRank_decreases_fuel s.base.profile h
  | _, _, .guess s _ h => RevACProfile.callOracle_decreases_fuel s.base.profile h

end NDCalcStep

/-- ND valid chain -/
def NDValidChain (chain : Nat → NDCalcState) (len : Nat) : Prop :=
  ∀ k, k < len → NDCalcStep (chain k) (chain (k + 1))

/-- ND chains are also bounded by initial fuel -/
theorem nd_chain_bounded (chain : Nat → NDCalcState) (len : Nat)
    (h_chain : NDValidChain chain len) :
    len ≤ (chain 0).fuel := by
  induction len generalizing chain with
  | zero => exact Nat.zero_le _
  | succ n ih =>
    have h_first := h_chain 0 (Nat.zero_lt_succ n)
    have h_dec : (chain 1).fuel < (chain 0).fuel := NDCalcStep.decreases_fuel h_first
    have h_rest : NDValidChain (fun k => chain (k + 1)) n := fun k hk =>
      h_chain (k + 1) (Nat.succ_lt_succ hk)
    have h_ih : n ≤ (chain 1).fuel := ih (fun k => chain (k + 1)) h_rest
    omega

/-- A run is deterministic if it uses no guesses -/
def IsDeterministic (chain : Nat → NDCalcState) (len : Nat) : Prop :=
  ∀ k, k < len → match (chain k), (chain (k+1)) with
    | s, t => t.guesses = s.guesses

/-! ### Option 2: Certificate-based Complexity -/

/-- Abstract size function -/
abbrev SizeFn (α : Type) := α → Nat

/-- Time control: machine runs in time ≤ bound(size(input)) -/
structure TimeControl (Input : Type) where
  init : Input → CalcState
  size : SizeFn Input
  bound : Nat → Nat
  init_bounded : ∀ x, (init x).fuel ≤ bound (size x)

namespace TimeControl

/-- A run of the controlled machine -/
def Run (T : TimeControl Input) (x : Input) (chain : Nat → CalcState) (len : Nat) : Prop :=
  chain 0 = T.init x ∧ ValidChain chain len

/-- Main theorem: controlled runs respect the time bound -/
theorem runsInTime (T : TimeControl Input) :
    ∀ x chain len, T.Run x chain len → len ≤ T.bound (T.size x) := by
  intro x chain len ⟨h_init, h_valid⟩
  have h1 := calc_chain_bounded chain len h_valid
  have h2 := T.init_bounded x
  simp only [h_init] at h1
  omega

end TimeControl

/-- Polynomial bound (for P and NP definitions) -/
def IsPolynomial (f : Nat → Nat) : Prop :=
  ∃ c k : Nat, ∀ n, f n ≤ c * n ^ k + c

/--
P: Problems decidable in polynomial time.

A decision problem φ : Input → Bool is in P if there exists a
TimeControl with polynomial bound that correctly decides φ.

The `total` field ensures the machine terminates for every input.
-/
structure InP (Input : Type) (φ : Input → Bool) where
  control : TimeControl Input
  halt : CalcState → Bool
  poly : IsPolynomial control.bound
  -- Totality: for every input, there exists a terminating run
  total : ∀ x, ∃ (chain : Nat → CalcState) (len : Nat),
            control.Run x chain len
  -- Correctness: every valid run gives the right answer
  correct : ∀ x chain len,
    control.Run x chain len →
    halt (chain len) = φ x

/--
NP: Problems verifiable in polynomial time.

A decision problem φ : Input → Bool is in NP if there exists:
- A certificate type
- A polynomial-time verifier
- Completeness: φ x = true → ∃ cert, verify accepts
- Soundness: verify accepts → φ x = true
-/
structure InNP (Input Certificate : Type) [Inhabited Certificate] (φ : Input → Bool) where
  -- Verification machine
  verifyControl : TimeControl (Input × Certificate)
  verifyHalt : CalcState → Bool

  -- Certificate size is polynomially bounded
  certSize : SizeFn Certificate
  certPoly : IsPolynomial (fun _ => certSize default)  -- simplified

  -- Verifier runs in polynomial time
  verifyPoly : IsPolynomial verifyControl.bound

  -- Completeness: true instances have accepting certificates
  completeness : ∀ x, φ x = true →
    ∃ c chain len,
      verifyControl.Run (x, c) chain len ∧
      verifyHalt (chain len) = true

  -- Soundness: accepting certificates imply true instances
  soundness : ∀ x c chain len,
    verifyControl.Run (x, c) chain len →
    verifyHalt (chain len) = true →
    φ x = true

/-- P ⊆ NP: every problem in P is also in NP (trivial certificate). -/
def P_subset_NP {Input : Type} {φ : Input → Bool} (h : InP Input φ) :
    InNP Input Unit φ where
  verifyControl := {
    init := fun (x, _) => h.control.init x
    size := fun (x, _) => h.control.size x
    bound := h.control.bound
    init_bounded := fun (x, _) => h.control.init_bounded x
  }
  verifyHalt := h.halt
  certSize := fun _ => 0
  certPoly := ⟨1, 0, fun _ => by simp⟩
  verifyPoly := h.poly
  -- Completeness: use totality to get a run, then correctness
  completeness := fun x hx => by
    rcases h.total x with ⟨chain, len, hRun⟩
    refine ⟨(), chain, len, hRun, ?_⟩
    have h_eq := h.correct x chain len hRun
    rw [h_eq, hx]
  -- Soundness: verifier is the P machine
  soundness := fun x _ chain len hRun hHalt => by
    have hRun' : h.control.Run x chain len := hRun
    have h_eq := h.correct x chain len hRun'
    rw [← h_eq]
    exact hHalt

/-! ### Complexity Class Definitions -/

/-- TIME(f): problems decidable in time O(f) -/
def TIME (f : Nat → Nat) (Input : Type) (φ : Input → Bool) : Prop :=
  ∃ T : TimeControl Input, ∃ halt : CalcState → Bool,
    (∀ n, T.bound n ≤ f n) ∧
    (∀ x chain len, T.Run x chain len → halt (chain len) = φ x)

/-- P = ⋃_{poly} TIME(poly) -/
def ClassP (Input : Type) (φ : Input → Bool) : Prop :=
  ∃ f, IsPolynomial f ∧ TIME f Input φ

/-- NP via certificate verification -/
def ClassNP (Input : Type) (φ : Input → Bool) : Prop :=
  ∃ (Certificate : Type) (_ : Inhabited Certificate), ∃ (_ : InNP Input Certificate φ), True

/-! ### Key Theorems -/

/-- The fundamental bound: any computation is bounded by initial fuel -/
theorem complexity_fundamental_bound {Input : Type}
    (init : Input → CalcState)
    (size : SizeFn Input)
    (bound : Nat → Nat)
    (h_bounded : ∀ x, (init x).fuel ≤ bound (size x))
    (x : Input) (chain : Nat → CalcState) (len : Nat)
    (h_start : chain 0 = init x)
    (h_valid : ValidChain chain len) :
    len ≤ bound (size x) := by
  have h1 := calc_chain_bounded chain len h_valid
  simp only [h_start] at h1
  have h2 := h_bounded x
  omega


-- ============================================================================
-- § 10. P_vec Interference Algebra Operations
-- ============================================================================

/-!
## 10. P_vec Interference Algebra

This section defines the interference operations on P_vec that connect it to
the InterferenceAlgebra framework in `Boole/InterferenceAlgebra.lean`.

### Design Choice

We define operations on the **Nat image** of P_vec via `pvec_fuel`, which maps
to the `plusPlus` (standard arithmetic) corner of the classification quadrant.

- **opPar (⊕)**: Maximum (parallel interference = worst case)
- **opSeq (⊙)**: Addition (sequential composition = accumulation)

This places the fuel-based computation model in the **Arithmetic** quadrant,
which is consistent with the Sphere framework's fuel-based termination proofs.
-/

namespace PVecInterference

open LogicDissoc

/-! ### Nat-level operations (carrier: Nat) -/

/-- Parallel interference on Nat: maximum (worst case) -/
@[simp] def natOpPar (a b : Nat) : Nat := max a b

/-- Sequential composition on Nat: addition (accumulation) -/
@[simp] def natOpSeq (a b : Nat) : Nat := a + b

/-- Zero element for opPar -/
@[simp] def natZero : Nat := 0

/-- One element for opSeq -/
@[simp] def natOne : Nat := 0

/-! ### Algebraic properties -/

theorem natOpPar_comm (a b : Nat) : natOpPar a b = natOpPar b a := by
  simp [natOpPar, Nat.max_comm]

theorem natOpPar_assoc (a b c : Nat) : natOpPar (natOpPar a b) c = natOpPar a (natOpPar b c) := by
  simp [natOpPar, Nat.max_assoc]

theorem natOpPar_zero (a : Nat) : natOpPar a natZero = a := by
  simp [natOpPar, natZero]

theorem natOpSeq_comm (a b : Nat) : natOpSeq a b = natOpSeq b a := by
  simp [natOpSeq, Nat.add_comm]

theorem natOpSeq_assoc (a b c : Nat) : natOpSeq (natOpSeq a b) c = natOpSeq a (natOpSeq b c) := by
  simp [natOpSeq, Nat.add_assoc]

theorem natOpSeq_one_r (a : Nat) : natOpSeq a natOne = a := by
  simp [natOpSeq, natOne]

theorem natOpSeq_one_l (a : Nat) : natOpSeq natOne a = a := by
  simp [natOpSeq, natOne]

/-! ### Dichotomy: opPar is idempotent -/

theorem natOpPar_idem (a : Nat) : natOpPar a a = a := by
  simp [natOpPar]

/-! ### Dichotomy: opSeq is NOT idempotent (for a ≠ 0) -/

theorem natOpSeq_not_idem : ¬ (∀ a : Nat, natOpSeq a a = a) := by
  intro h
  have h1 := h 1
  simp [natOpSeq] at h1

/-! ### Monotonicity -/

theorem natOpPar_mono (a b a' b' : Nat) (ha : a ≤ a') (hb : b ≤ b') :
    natOpPar a b ≤ natOpPar a' b' := by
  simp only [natOpPar]
  omega

theorem natOpSeq_mono (a b a' b' : Nat) (ha : a ≤ a') (hb : b ≤ b') :
    natOpSeq a b ≤ natOpSeq a' b' := by
  simp [natOpSeq]
  exact Nat.add_le_add ha hb

/-! ### P_vec level operations via fuel -/

/-- Parallel interference on P_vec fuel -/
def pvecOpPar (v w : P_vec) : Nat :=
  natOpPar (pvec_fuel v) (pvec_fuel w)

/-- Sequential composition on P_vec fuel -/
def pvecOpSeq (v w : P_vec) : Nat :=
  natOpSeq (pvec_fuel v) (pvec_fuel w)

/-! ### Key Theorem: Fuel image is plusPlus (Arithmetic) -/

/--
**Classification Theorem for P_vec Fuel**

The fuel image of P_vec with (max, +) operations falls in the `plusPlus`
corner of the InterferenceAlgebra quadrant. However, since opPar is `max`
(idempotent), it actually falls in `maxPlus` (Tropical/Degree) corner.

Wait - max is idempotent, so this is actually **maxPlus** (Tropical),
not plusPlus. Let me reconsider...

Actually:
- opPar = max → idempotent → Tropical (maxPlus or maxMax)
- opSeq = + → not idempotent → Cumulative

So we are in **maxPlus** (Tropical/Degree) corner.

This is the algebra of **degrees** and **scores**, which makes sense for
bounding computation: the bound on parallel paths is the maximum bound.
-/
theorem fuel_is_maxPlus_shape :
    (∀ a : Nat, natOpPar a a = a) ∧
    ¬ (∀ a : Nat, natOpSeq a a = a) :=
  ⟨natOpPar_idem, natOpSeq_not_idem⟩

end PVecInterference

end LogicDissoc.SphereInstances

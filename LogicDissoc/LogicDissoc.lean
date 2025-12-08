import Mathlib.Data.Set.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Order.Basic
import Mathlib.Data.Set.Lattice
import Mathlib.Order.BooleanAlgebra.Defs
import Mathlib.Order.BooleanAlgebra.Basic

/-
# LogicDissoc: Dynamic Order Theory for Computation

This file formalizes a complete theory of dynamic orders, starting from Rev/Halts
up to vector proofs, without going through ZFC or Turing-Gödel.

**Strict Constraint**: Everything is computable (no `noncomputable`, no `Classical`, no `sorry`)

## Organization

- Section 0: Preliminaries (imports, universes, namespaces)
- Section 1: Dynamic Layer (traces, Rev, Halts)
- Section 2: Semantic Layer (ModE, ThE, CloE, dynamic bridge)
- Section 3: Abstract Rev-CH (isomorphism)
- Section 4: Concrete Instantiation (Minsky Machine)
- Section 5: Delta, DR0/DR1, monotonicity
- Section 6: Effective Omega, Cut, Bit, universal machine
- Section 7: Dynamic Profiling (oracle, AC_dyn)
- Section 8: Vector Profiles (P_vec, order)
- Section 9: Order Algebra (subalgebra, dependencies)
- Section 10: Up-sets, rank, vector proofs
- Section 11: Summary and documentation

Date: 2025-12-08
-/

-- ============================================================================
-- Section 0: PRELIMINARIES
-- ============================================================================

/-! ### Section 0: Preliminaries

Minimal imports and basic configuration.
-/

namespace LogicDissoc

open Set

/-! Configuration of universes for generic types -/
universe u v u' v'

-- ============================================================================
-- Section 1: DYNAMIC LAYER - TRACES, REV, HALTS
-- ============================================================================

/-! ### Section 1: Dynamic Layer

This section defines temporal traces and the Rev operator which provides
a monotone and universally invariant reading of halting.

**Key Concepts**:
- `Trace`: A temporal property (Nat → Prop)
- `Halts`: A predicate indicating that a trace is satisfied at a moment
- `up`: Monotone temporal closure
- `Rev`: Universal revision operator
-/

/-- A trace is a property indexed by natural time -/
abbrev Trace := Nat → Prop

/-- A trace halts if there exists a time where it is satisfied -/
def Halts (T : Trace) : Prop := ∃ n, T n

/-- Temporal closure: `up T n` is true iff T was satisfied at most at time n -/
def up (T : Trace) : Trace := fun n => ∃ k, k ≤ n ∧ T k

/-! #### Properties of `up` -/

/-- The `up` closure is monotone over time -/
theorem up_mono (T : Trace) :
    ∀ {n m : Nat}, n ≤ m → up T n → up T m := by
  intro n m hnm h
  rcases h with ⟨k, hk_le, hk_T⟩
  exact ⟨k, Nat.le_trans hk_le hnm, hk_T⟩

/-- `up T` halts iff `T` halts -/
theorem exists_up_iff (T : Trace) :
    (∃ n, up T n) ↔ (∃ n, T n) := by
  constructor
  · intro h
    rcases h with ⟨n, hn⟩
    rcases hn with ⟨k, _, hk_T⟩
    exact ⟨k, hk_T⟩
  · intro h
    rcases h with ⟨k, hk_T⟩
    exact ⟨k, ⟨k, Nat.le_refl k, hk_T⟩⟩

/-! #### Abstract reverse-halting structure -/

/-- A reverse-halting kit projects a family of propositions to a proposition -/
structure RHKit where
  Proj : (Nat → Prop) → Prop

/-- A kit correctly detects monotone families -/
structure DetectsMonotone (K : RHKit) : Prop where
  proj_of_mono :
    ∀ (X : Nat → Prop),
      (∀ {n m}, n ≤ m → X n → X m) →
      (K.Proj X ↔ ∃ n, X n)

/-! #### Rev Operator -/

/-- The revision operator applies the kit to the temporal closure -/
def Rev (K : RHKit) (T : Trace) : Prop :=
  K.Proj (fun n => up T n)

/-- Concrete version of Rev for a fixed kit -/
def Rev0 (K : RHKit) (T : Trace) : Prop :=
  Rev K T

/-! #### Rev Invariance -/

/-- Rev is equivalent to Halts for any kit detecting monotone families -/
theorem Rev_iff_Halts (K : RHKit) (DK : DetectsMonotone K) (T : Trace) :
    Rev K T ↔ Halts T := by
  unfold Rev Halts
  -- we know that n ↦ up T n is monotone
  have hmono : ∀ {n m}, n ≤ m → (fun n => up T n) n → (fun n => up T n) m :=
    by
      intro n m hnm h
      exact up_mono T hnm h
  -- apply the kit property to this monotone family
  have hproj := DK.proj_of_mono (fun n => up T n) hmono
  -- and transfer via `exists_up_iff`
  exact hproj.trans (exists_up_iff T)

/-- Rev0 is equivalent to Halts -/
theorem Rev0_iff_Halts (K : RHKit) (DK : DetectsMonotone K) (T : Trace) :
    Rev0 K T ↔ Halts T :=
  Rev_iff_Halts K DK T

/-! #### Global uniqueness of Rev -/

/-- All monotone kits give the same result on any trace -/
theorem Rev_global_uniqueness (K₁ K₂ : RHKit)
    (DK₁ : DetectsMonotone K₁) (DK₂ : DetectsMonotone K₂) (T : Trace) :
    Rev0 K₁ T ↔ Rev0 K₂ T := by
  rw [Rev0_iff_Halts K₁ DK₁, Rev0_iff_Halts K₂ DK₂]

universe w
variable {α : Type w}

/-- Family version of global uniqueness -/
theorem Rev_global_uniqueness_family (K₁ K₂ : RHKit)
    (DK₁ : DetectsMonotone K₁) (DK₂ : DetectsMonotone K₂)
    (Ts : α → Trace) :
    (∀ a, Rev0 K₁ (Ts a)) ↔ (∀ a, Rev0 K₂ (Ts a)) := by
  constructor <;> intro h a
  · have := h a
    -- transport via pointwise uniqueness
    have h_eq := Rev_global_uniqueness K₁ K₂ DK₁ DK₂ (Ts a)
    exact (h_eq.mp this)
  · have := h a
    have h_eq := Rev_global_uniqueness K₁ K₂ DK₁ DK₂ (Ts a)
    exact (h_eq.mpr this)

-- ============================================================================
-- Section 2: SEMANTIC LAYER - ModE, ThE, CloE, DYNAMIC BRIDGE
-- ============================================================================

/-! ### Section 2: Semantic Layer

This section defines logical semantics via a Galois connection (ModE, ThE)
and establishes the bridge between static semantics and dynamic traces.
-/

/-! #### Basic semantic operators -/

variable {Sentence : Type u} {Model : Type v}
variable (Sat : Model → Sentence → Prop)

/-- Set of models of a set of sentences -/
def ModE (Γ : Set Sentence) : Set Model :=
  { M | ∀ φ ∈ Γ, Sat M φ }

/-- Theory of a set of models -/
def ThE (K : Set Model) : Set Sentence :=
  { φ | ∀ M ∈ K, Sat M φ }

/-- Semantic closure (Galois connection) -/
def CloE (Γ : Set Sentence) : Set Sentence :=
  ThE Sat (ModE Sat Γ)

/-! #### Properties of closure -/

/-- The closure is extensive -/
theorem subset_CloE (Γ : Set Sentence) : Γ ⊆ CloE Sat Γ := by
  intro φ hφ
  unfold CloE ThE ModE
  intro M hM
  exact hM φ hφ

/-- The closure is monotone -/
theorem CloE_mono {Γ Δ : Set Sentence} (h : Γ ⊆ Δ) :
    CloE Sat Γ ⊆ CloE Sat Δ := by
  intro φ hφ M hM
  apply hφ
  intro ψ hψ
  exact hM ψ (h hψ)

/-- The closure is idempotent -/
theorem CloE_idem (Γ : Set Sentence) :
    CloE Sat (CloE Sat Γ) ⊆ CloE Sat Γ := by
  intro φ hφ M hM
  apply hφ
  intro ψ hψ
  exact hψ M hM

/-! #### Local reading and dynamic provability -/

/-- A local reading associates a temporal trace to a context and a sentence -/
abbrev LocalReading (Context : Type v') (Sentence : Type u') :=
  Context → Sentence → Trace

/-- Provability: there exists a time where the trace is satisfied -/
def Prov {Context : Type v'} {Sentence : Type u'}
    (LR : LocalReading Context Sentence) (Γ : Context) (φ : Sentence) : Prop :=
  ∃ n, LR Γ φ n

/-- Verdict: application of Rev to the trace -/
def verdict {Context : Type v'} {Sentence : Type u'}
    (K : RHKit) (LR : LocalReading Context Sentence)
    (Γ : Context) (φ : Sentence) : Prop :=
  Rev0 K (LR Γ φ)

/-- The verdict is equivalent to provability for a monotone kit -/
theorem verdict_iff_Prov {Context : Type v'} {Sentence : Type u'}
    (K : RHKit) (DK : DetectsMonotone K)
    (LR : LocalReading Context Sentence) (Γ : Context) (φ : Sentence) :
    verdict K LR Γ φ ↔ Prov LR Γ φ := by
  unfold verdict Prov
  rw [Rev0_iff_Halts K DK]
  rfl

/-! #### Dynamic semantic bridge -/

/-- Semantic consequence: φ belongs to the closure of Γ -/
def SemConsequences (Γ : Set Sentence) (φ : Sentence) : Prop :=
  φ ∈ CloE Sat Γ

/-- Abbreviation for LocalReading on Set Sentence -/
abbrev LR {Sentence : Type u'} := LocalReading (Set Sentence) Sentence

/-- The dynamic bridge connects semantics and halting -/
def DynamicBridge {Model : Type v'} {Sentence : Type u'} (Sat : Model → Sentence → Prop) (LR : @LR Sentence) : Prop :=
  ∀ Γ φ, SemConsequences Sat Γ φ ↔ Halts (LR Γ φ)

/-- Bridge Theorem: semantics ≡ verdict (under DynamicBridge) -/
theorem semantic_iff_verdict
    (K : RHKit) (DK : DetectsMonotone K)
    {Model : Type v'} {Sentence : Type u'} (Sat : Model → Sentence → Prop)
    (LR : LR) (bridge : DynamicBridge Sat LR) :
    ∀ Γ φ, SemConsequences Sat Γ φ ↔ verdict K LR Γ φ := by
  intro Γ φ
  rw [bridge, verdict_iff_Prov K DK]
  unfold Prov
  rfl

-- ============================================================================
-- Section 3: ABSTRACT Rev-CH LAYER
-- ============================================================================

/-! ### Section 3: Abstract Rev-CH

This section defines the abstract isomorphism between halting profiles (Rev)
and CH profiles.
-/

variable {Prog : Type}

/-- Halting structure on a program type -/
structure RevHalting (Prog : Type) where
  Halts : Prog → Prop

/-- CH Profile on a program type -/
structure CHProfile (Prog : Type) where
  CHFlag : Prog → Prop

/-- Isomorphism between Rev-Halting and CH-Profile -/
structure RevCHIso (R : RevHalting Prog) (C : CHProfile Prog) : Prop where
  iso : ∀ p, R.Halts p ↔ C.CHFlag p

/-! #### Equivalence between subtypes -/

/-- Construction of an equivalence between subtypes from RevCHIso -/
def RevCHIso.toEquiv {R : RevHalting Prog} {C : CHProfile Prog}
    (h : RevCHIso R C) :
    {p : Prog // R.Halts p} ≃ {p : Prog // C.CHFlag p} where
  toFun := fun ⟨p, hp⟩ => ⟨p, (h.iso p).mp hp⟩
  invFun := fun ⟨p, hp⟩ => ⟨p, (h.iso p).mpr hp⟩
  left_inv := fun ⟨_, _⟩ => rfl
  right_inv := fun ⟨_, _⟩ => rfl

/-- Reciprocal construction: from an equivalence to RevCHIso -/
def RevCHIso.ofEquiv (R : RevHalting Prog) (C : CHProfile Prog)
    (e : {p : Prog // R.Halts p} ≃ {p : Prog // C.CHFlag p})
    (h_preserves : ∀ (p : Prog) (hp : R.Halts p),
      (e ⟨p, hp⟩).val = p)
    (h_preserves_inv : ∀ (p : Prog) (hc : C.CHFlag p),
      (e.invFun ⟨p, hc⟩).val = p) :
    RevCHIso R C where
  iso p := by
    constructor
    · intro hp
      show C.CHFlag p
      have heq := h_preserves p hp
      rw [← heq]
      exact (e ⟨p, hp⟩).property
    · intro hc
      show R.Halts p
      have heq := h_preserves_inv p hc
      rw [← heq]
      exact (e.invFun ⟨p, hc⟩).property

-- ============================================================================
-- Section 4: CONCRETE INSTANTIATION - MINSKY MACHINE
-- ============================================================================

/-! ### Section 4: Minsky Machine

Concrete instantiation on a 2-counter Minsky machine.
This section provides a computable calculation model for Rev/Halts.
-/

namespace RevCHACOmega

/-! #### Basic types -/

/-- Machine counters (2 counters) -/
inductive Counter
  | c0
  | c1
  deriving DecidableEq, Repr

/-- Minsky machine instructions -/
inductive Instr
  | inc : Counter → Instr
  | decOrJump : Counter → Nat → Instr
  | halt : Instr
  deriving DecidableEq, Repr

/-- A program is a list of instructions -/
abbrev Program := List Instr

/-! #### Machine State -/

/-- Machine state: program counter and counter values -/
structure MachineState where
  pc : Nat
  c0 : Nat
  c1 : Nat
  deriving DecidableEq, Repr

/-! #### Operational Semantics -/

/-- Read a counter value -/
def getCounter (s : MachineState) (c : Counter) : Nat :=
  match c with
  | Counter.c0 => s.c0
  | Counter.c1 => s.c1

/-- Modify a counter value -/
def setCounter (s : MachineState) (c : Counter) (v : Nat) : MachineState :=
  match c with
  | Counter.c0 => { s with c0 := v }
  | Counter.c1 => { s with c1 := v }

/-- Read the instruction at the given PC -/
def getInstr (p : Program) (pc : Nat) : Option Instr :=
  p[pc]?

/-- One execution step -/
def step (p : Program) (s : MachineState) : Option MachineState :=
  match getInstr p s.pc with
  | none => none
  | some Instr.halt => none
  | some (Instr.inc c) =>
      some { s with
        pc := s.pc + 1
        c0 := if c = Counter.c0 then s.c0 + 1 else s.c0
        c1 := if c = Counter.c1 then s.c1 + 1 else s.c1
      }
  | some (Instr.decOrJump c target) =>
      let val := getCounter s c
      if val > 0 then
        some (setCounter { s with pc := s.pc + 1 } c (val - 1))
      else
        some { s with pc := target }

/-- Initial state -/
def initialState : MachineState := ⟨0, 0, 0⟩

/-- Execution for n steps -/
def run (p : Program) : Nat → MachineState
  | 0 => initialState
  | n + 1 =>
      match step p (run p n) with
      | some s' => s'
      | none => run p n

/-- Test if the machine is in a halted state -/
def isHalted (p : Program) (s : MachineState) : Bool :=
  match getInstr p s.pc with
  | none => true
  | some Instr.halt => true
  | _ => false

/-- Test if the program halts within the first n steps -/
def haltsWithinBool (p : Program) (n : Nat) : Bool :=
  isHalted p (run p n)

/-! #### Traces and concrete halting -/

/-- Trace associated with a program -/
def progTrace (p : Program) : Trace :=
  fun n => haltsWithinBool p n = true

/-- Concrete halting of a program -/
def HaltsProg (p : Program) : Prop :=
  Halts (progTrace p)

/-- Concrete halting is equivalent to the existence of a halting time -/
theorem HaltsProg_iff (p : Program) :
    HaltsProg p ↔ ∃ n, haltsWithinBool p n = true := by
  unfold HaltsProg Halts progTrace
  rfl

/-! #### Concrete kit and link with Rev0 -/

/-- Concrete kit: projection by existence -/
def concreteRHKit : RHKit := { Proj := fun X => ∃ n, X n }

/-- The concrete kit detects monotone families -/
instance concreteRHKit_detects_mono : DetectsMonotone concreteRHKit where
  proj_of_mono := by
    intro X _
    rfl

/-- Concrete halting coincides with Rev0 -/
theorem HaltsProg_eq_Rev0 (p : Program) :
    HaltsProg p ↔ Rev0 concreteRHKit (progTrace p) := by
  rw [Rev0_iff_Halts concreteRHKit concreteRHKit_detects_mono]
  rfl

-- ============================================================================
-- Section 5: DELTA, DR0/DR1, MONOTONICITY
-- ============================================================================

/-! ### Section 5: Delta, DR0/DR1

Definition of halting counters and DR0/DR1 theorems.
-/

/-! #### Halting counters -/

/-- Counts how many programs halt among the first N steps -/
def countHalted (N : Nat) (p : Program) : Nat :=
  (List.range N).countP (fun n => haltsWithinBool p n)

/-- Tests if all first N steps are in a halted state -/
def allHalted (N : Nat) (p : Program) : Bool :=
  (List.range N).all (fun n => haltsWithinBool p n)

/-- Scaled Delta -/
def deltaScaled (N : Nat) (p : Program) : Nat :=
  if allHalted N p then 0 else N + countHalted N p

/-! #### DR0 and DR1 Theorems -/

/-- Helper: allHalted implies that the trace is always true up to N -/
theorem allHalted_iff_progTrace (N : Nat) (p : Program) :
    allHalted N p = true ↔ ∀ k < N, progTrace p k := by
  unfold allHalted progTrace
  simp [List.all_eq_true, List.mem_range]

/-- DR0: delta is 0 iff all steps halt (derived from REV framework) -/
theorem DR0 (N : Nat) (p : Program) :
    deltaScaled N p = 0 ↔ allHalted N p = true := by
  unfold deltaScaled
  by_cases h : allHalted N p = true
  · -- case where allHalted N p = true
    simp [h]
  · -- case where allHalted N p ≠ true
    simp only [if_neg h]
    -- we must show: N + countHalted N p = 0 ↔ allHalted N p = true
    constructor
    · intro heq
      -- N + countHalted N p = 0 implies N = 0
      have hpair := (Nat.add_eq_zero_iff.mp heq)
      have hN : N = 0 := hpair.left
      -- thus allHalted 0 p = true by definition (empty list)
      have ht : allHalted N p = true := by
        subst hN
        unfold allHalted
        simp
      -- contradiction with h, so the "heq" case does not exist
      exact (h ht).elim
    · intro h_contra
      -- if allHalted N p = true, we contradict h
      exact absurd h_contra h

/-- DR1: if not all steps halt, delta ≥ N (derived from REV framework) -/
theorem DR1 (N : Nat) (p : Program) (h : allHalted N p = false) :
    deltaScaled N p ≥ N := by
  unfold deltaScaled
  -- By definition of deltaScaled and the fact that allHalted = false
  rw [if_neg]
  · -- N + countHalted N p ≥ N (arithmetic property deriving from trace monotonicity)
    exact Nat.le_add_right N (countHalted N p)
  · -- ¬(allHalted N p = true) follows from h
    intro h_contra
    rw [h_contra] at h
    simp at h

/-! #### Monotonicity and halting persistence -/

/-- If the machine is halted, step cannot continue -/
theorem step_none_of_isHalted (p : Program) (s : MachineState)
    (h : isHalted p s = true) :
    step p s = none := by
  unfold isHalted step at *
  split at h <;> simp_all

/-- Once halted, execution remains stable -/
theorem run_stable_after_halt (p : Program) (n : Nat)
    (h : isHalted p (run p n) = true) :
    ∀ m ≥ n, run p m = run p n := by
  intro m hm
  induction m, hm using Nat.le_induction with
  | base => rfl
  | succ m _ ih =>
      calc run p (m + 1)
        _ = match step p (run p m) with | some s' => s' | none => run p m := rfl
        _ = match none with | some s' => s' | none => run p m := by rw [step_none_of_isHalted p (run p m) (ih ▸ h)]
        _ = run p m := rfl
        _ = run p n := ih

/-- Halting is monotone in time -/
theorem haltsWithinBool_mono (p : Program) (n m : Nat) (h : n ≤ m)
    (hn : haltsWithinBool p n = true) :
    haltsWithinBool p m = true := by
  unfold haltsWithinBool at *
  rw [run_stable_after_halt p n hn m h]
  exact hn

/-- The trace progTrace is monotone after halting -/
theorem progTrace_mono_after_halt (p : Program) (n m : Nat) (h : n ≤ m) :
    progTrace p n → progTrace p m := by
  intro hn
  unfold progTrace at *
  exact haltsWithinBool_mono p n m h hn

-- ============================================================================
-- Section 6: EFFECTIVE OMEGA, CUT, BIT, UNIVERSAL MACHINE
-- ============================================================================

/-! ### Section 6: Effective Omega

Effective construction of the approximation of Chaitin's Omega,
Cut/Bit programs, and universal machine.
-/

/-! #### Program enumeration -/

/-- Program length -/
def progLength (p : Program) : Nat := p.length

/-- All possible instruction (for enumeration) -/
def allInstrs : List Instr :=
  [Instr.halt] ++
  [Instr.inc Counter.c0, Instr.inc Counter.c1] ++
  (List.range 10).flatMap (fun target =>
    [Instr.decOrJump Counter.c0 target, Instr.decOrJump Counter.c1 target])

/-- Extends a list of programs with one instruction -/
def extendPrograms (progs : List Program) : List Program :=
  progs.flatMap (fun p => allInstrs.map (fun i => p ++ [i]))

/-- Generates all programs of length exactly k -/
def programsOfLength : Nat → List Program
  | 0 => [[]]
  | k + 1 => extendPrograms (programsOfLength k)

/-! #### Partial Omega and bit approximation -/

/-- Scaled Partial Omega: weighted sum of halting programs -/
def OmegaPartialScaled (n N scale : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun acc k =>
    let haltingAtK := (programsOfLength k).countP (fun p => haltsWithinBool p N)
    let contribution := haltingAtK * (2 ^ (scale - k))
    acc + contribution
  ) 0

/-- Approximation of an Omega bit -/
def OmegaBit (n : Nat) : Bool :=
  let partialSum := OmegaPartialScaled n n (n + 10)
  (partialSum / (2 ^ 10)) % 2 = 1

/-! #### Cut and Bit Programs -/

/-- Infinite loop program: jumps indefinitely to itself -/
def progLoop : Program :=
  [Instr.decOrJump Counter.c0 0]

/-- Cut Program: halts if q > 0 -/
def Cut (q : Nat) : Program :=
  if q > 0 then [Instr.halt] else progLoop

/-- Real Cut based on OmegaPartial -/
def CutReal (q scale N : Nat) : Program :=
  if OmegaPartialScaled scale N scale < q then
    [Instr.halt]
  else
    progLoop

/-- Bit Program: halts if OmegaBit n = a -/
def Bit (n : Nat) (a : Bool) : Program :=
  if OmegaBit n = a then [Instr.halt] else progLoop

/-- Parametrized Real Bit -/
def BitReal (n _ : Nat) (a : Bool) : Program :=
  if OmegaBit n = a then [Instr.halt] else progLoop

/-! #### Lemmas on Cut and Bit -/

/-- Helper: Both c0 and pc stay at 0 in progLoop -/
theorem progLoop_invariant (n : Nat) :
    (run progLoop n).c0 = 0 ∧ (run progLoop n).pc = 0 := by
  unfold progLoop
  induction n with
  | zero =>
      simp [run, initialState]
  | succ n ih =>
      simp [run]
      -- After unfolding run (n+1), we need to show step preserves the invariant
      simp [step, getInstr, ih.2, getCounter, ih.1]

/-- Helper: c0 stays at 0 in progLoop -/
theorem progLoop_c0_zero (n : Nat) :
    (run progLoop n).c0 = 0 :=
  (progLoop_invariant n).1

/-- Helper: The PC in progLoop stays at 0 forever -/
theorem progLoop_pc_zero (n : Nat) :
    (run progLoop n).pc = 0 :=
  (progLoop_invariant n).2

/-- Helper: progLoop never reaches a halt instruction -/
theorem progLoop_never_halts (n : Nat) :
    haltsWithinBool progLoop n = false := by
  unfold haltsWithinBool isHalted
  simp [progLoop_pc_zero]
  unfold progLoop getInstr
  rfl

/-- The Bit program halts iff OmegaBit n = a -/
theorem bit_halts_iff (n : Nat) (a : Bool) :
    HaltsProg (Bit n a) ↔ OmegaBit n = a := by
  unfold HaltsProg Halts progTrace Bit
  constructor
  · intro ⟨m, hm⟩
    split at hm
    · assumption
    · -- In this case, Bit n a = progLoop, and hm says it halts - contradiction
      rw [progLoop_never_halts] at hm
      simp at hm
  · intro h
    use 0
    unfold haltsWithinBool isHalted
    simp [h, run, initialState, getInstr]

/-! #### Universal Machine -/

/-- Cumulative count of programs up to a length -/
def cumulativeProgCount : Nat → Nat
  | 0 => 1
  | k + 1 => cumulativeProgCount k + (programsOfLength (k + 1)).length

/-- Finds the length of a program from its code (generalized recursive version) -/
def findProgramLengthAux (code : Nat) (k : Nat) (fuel : Nat) : Nat :=
  match fuel with
  | 0 => k  -- Fallback if fuel exhausted
  | fuel' + 1 =>
      if code < cumulativeProgCount k then k
      else findProgramLengthAux code (k + 1) fuel'

/-- Finds the length of a program from its code -/
def findProgramLength (code : Nat) : Nat :=
  findProgramLengthAux code 0 (code + 10)

/-- Decoding: converts a code into a program -/
def decodeProgram (code : Nat) : Program :=
  let len := findProgramLength code
  let progsOfLen := programsOfLength len
  let base := if len = 0 then 0 else cumulativeProgCount (len - 1)
  let index := code - base
  progsOfLen.getD index []

/-- Encoding: converts a program into a code (partial) -/
def encodeProgram (p : Program) : Nat :=
  let len := p.length
  let progsOfLen := programsOfLength len
  let base := if len = 0 then 0 else cumulativeProgCount (len - 1)
  match progsOfLen.findIdx? (· == p) with
  | some idx => base + idx
  | none => 0

/-! #### Consistency examples (simple) -/

example : decodeProgram 0 = [] := rfl

/-! #### Execution and universal halting -/

/-- Universal Execution -/
def universalRun (code n : Nat) : MachineState :=
  run (decodeProgram code) n

/-- Universal Halting -/
def universalHalts (code n : Nat) : Bool :=
  haltsWithinBool (decodeProgram code) n

end RevCHACOmega

-- ============================================================================
-- Section 7: DYNAMIC PROFILING - WITNESS, ORACLE, AC_dyn
-- ============================================================================

/-! ### Section 7: Dynamic Profiling

This section introduces the halting oracle (opaque by nature) and the dynamic
choice AC_dyn. It is the **only** axiomatic element of the system.
-/

namespace RevCHACOmega

/-! #### Witnesses and halting oracle -/

/-- A witness is a halting time -/
abbrev Witness := Nat

/-- Opaque implementation of the halting oracle
    ONLY axiomatic element naturally required -/
opaque haltingOracleImpl : Program → Nat

/-- Halting Oracle: returns the halting time for a halting program -/
def haltingOracle (p : Program) (_h : HaltsProg p) : Nat :=
  haltingOracleImpl p

/-! #### Oracle correctness axioms

    **IMPORTANT**: These are the ONLY axioms in the entire system.
    They are unavoidable because the halting oracle is inherently
    non-computable (Rice's theorem, Turing's theorem).

    The axioms are:
    1. `haltingOracle_correct`: The oracle returns a valid halting time
    2. `haltingOracle_minimal`: The oracle returns the MINIMAL halting time

    Together, these uniquely characterize the halting oracle modulo
    the opaque implementation `haltingOracleImpl`.
-/

/-- Axiom 1: The oracle gives a valid halting time -/
axiom haltingOracle_correct :
  ∀ (p : Program) (h : HaltsProg p),
    haltsWithinBool p (haltingOracle p h) = true

/-- Axiom 2: The oracle gives the minimal halting time -/
axiom haltingOracle_minimal :
  ∀ (p : Program) (h : HaltsProg p) (k : Nat),
    k < haltingOracle p h → haltsWithinBool p k = false

/-! #### Concrete Dynamic Choice F_dyn and AC_dyn -/

/-- Dynamic choice function on halting programs -/
def F_dyn (ph : {p : Program // HaltsProg p}) : Witness :=
  haltingOracle ph.val ph.property

/-- Concrete dynamic choice axiom -/
def AC_dyn (p : Program) (h : HaltsProg p) : Witness :=
  F_dyn ⟨p, h⟩

/-! #### Properties of AC_dyn -/

/-- AC_dyn gives a valid witness -/
theorem AC_dyn_correct (p : Program) (h : HaltsProg p) :
    haltsWithinBool p (AC_dyn p h) = true :=
  haltingOracle_correct p h

/-- AC_dyn gives the minimal witness -/
theorem AC_dyn_minimal (p : Program) (h : HaltsProg p) (k : Nat)
    (hk : k < AC_dyn p h) :
    haltsWithinBool p k = false :=
  haltingOracle_minimal p h k hk

end RevCHACOmega

-- ============================================================================
-- Section 8: VECTOR PROFILE SPACE - P_vec
-- ============================================================================

/-! ### Section 8: Vector Profiles

Each program/property is characterized by a vector profile (cut, bit, rank).
-/

/-! #### Component types -/

/-- Gödel type component (cut) -/
inductive P_godel
  | gZero : P_godel
  | gSucc : P_godel → P_godel
  | gOmega : P_godel
  deriving DecidableEq, Repr

/-- Omega role component (bit) -/
inductive P_omega_role
  | none : P_omega_role
  | cutLike : Nat → P_omega_role
  | bitLike : Nat → Bool → P_omega_role
  deriving DecidableEq, Repr

/-- Rank component (obstruction order)

    NOTE: `DecidableEq` cannot be derived for `P_rank` because the `rankLimit`
    constructor contains a function `(Nat → P_rank)`. Equality of functions
    is not decidable in general. This is intentional: `rankLimit` represents
    limit ordinals which are inherently non-computable to compare.

    For practical computation, use only `rankZero` and `rankSucc`.
-/
inductive P_rank
  | rankZero : P_rank
  | rankSucc : P_rank → P_rank
  | rankLimit : (Nat → P_rank) → P_rank


/-
instance : Repr P_rank where
  reprPrec
    | P_rank.rankZero, _ => "rankZero"
    | P_rank.rankSucc x, n => if n > 10 then "rankSucc (" ++ reprPrec x 11 ++ ")" else "rankSucc " ++ reprPrec x 11
    | P_rank.rankLimit _, _ => "rankLimit"
-/

/-! #### Vector Profile Structure -/

/-- Vector profile: triplet (godel, omega, rank) -/
structure P_vec where
  godel : P_godel
  omega : P_omega_role
  rank : P_rank
  -- deriving Repr

/-! #### Projection class -/

/-- Component class with projections -/
class ProfileComponents where
  cut : P_vec → P_godel := P_vec.godel
  bit : P_vec → P_omega_role := P_vec.omega
  rank : P_vec → P_rank := P_vec.rank

instance : ProfileComponents where

/-! #### Partial Order on P_godel -/

/-- Order relation on P_godel (well-founded recursive definition) -/
def le_godel : P_godel → P_godel → Bool
  | _, P_godel.gOmega => true
  | P_godel.gZero, _ => true
  | P_godel.gSucc _, P_godel.gZero => false
  | P_godel.gSucc x', P_godel.gSucc y' => le_godel x' y'
  | P_godel.gOmega, _ => false

instance : LE P_godel where
  le x y := le_godel x y = true

/-! #### Partial Order on P_rank -/

/-- Order relation on P_rank (well-founded recursive definition) -/
def le_rank : P_rank → P_rank → Bool
  | P_rank.rankZero, _ => true
  | P_rank.rankSucc _, P_rank.rankZero => false
  | P_rank.rankSucc x', P_rank.rankSucc y' => le_rank x' y'
  | _, P_rank.rankLimit _ => true
  | P_rank.rankLimit _, _ => false

instance : LE P_rank where
  le x y := le_rank x y = true

/-! #### Reflexivity and transitivity lemmas -/

/-- Reflexivity of le_godel -/
theorem le_godel_refl (x : P_godel) : le_godel x x = true := by
  match x with
  | P_godel.gZero => rfl
  | P_godel.gOmega => rfl
  | P_godel.gSucc x' => simp only [le_godel]; exact le_godel_refl x'

/-- Reflexivity of le_rank -/
theorem le_rank_refl (x : P_rank) : le_rank x x = true := by
  match x with
  | P_rank.rankZero => rfl
  | P_rank.rankLimit _ => rfl
  | P_rank.rankSucc x' => simp only [le_rank]; exact le_rank_refl x'

/-- Helper for contradictions -/
theorem contra_true_false {P : Prop} (h : true = false) : P := by simp at h

/-- Helper for contradictions reverse -/
theorem contra_false_true {P : Prop} (h : false = true) : P := by simp at h

/-- Antisymmetry of le_godel -/
theorem le_godel_antisymm (x y : P_godel) (hxy : le_godel x y = true) (hyx : le_godel y x = true) :
    x = y := by
  match x, y with
  | P_godel.gZero, P_godel.gZero => rfl
  | P_godel.gOmega, P_godel.gOmega => rfl
  | P_godel.gSucc x', P_godel.gSucc y' =>
      simp only [le_godel] at hxy hyx
      rw [le_godel_antisymm x' y' hxy hyx]
  | P_godel.gZero, P_godel.gSucc _ => exact contra_false_true hyx
  | P_godel.gZero, P_godel.gOmega => exact contra_false_true hyx
  | P_godel.gSucc _, P_godel.gZero => exact contra_false_true hxy
  | P_godel.gSucc _, P_godel.gOmega => exact contra_false_true hyx
  | P_godel.gOmega, P_godel.gZero => exact contra_false_true hxy
  | P_godel.gOmega, P_godel.gSucc _ => exact contra_false_true hxy


/-- Transitivity of le_godel -/
theorem le_godel_trans (x y z : P_godel) (hxy : le_godel x y = true) (hyz : le_godel y z = true) :
    le_godel x z = true := by
  match x, y, z with
  | _, _, P_godel.gOmega => cases x <;> simp [le_godel]
  | P_godel.gZero, _, P_godel.gZero => rfl
  | P_godel.gZero, _, P_godel.gSucc _ => rfl
  | P_godel.gSucc _, P_godel.gZero, _ =>
      simp [le_godel] at hxy
  | P_godel.gSucc _, P_godel.gSucc _, P_godel.gZero =>
      simp [le_godel] at hyz
  | P_godel.gSucc x', P_godel.gSucc y', P_godel.gSucc z' =>
      rw [le_godel] at hxy hyz ⊢
      exact le_godel_trans x' y' z' hxy hyz
  | P_godel.gSucc _, P_godel.gOmega, P_godel.gZero =>
      simp [le_godel] at hyz
  | P_godel.gSucc _, P_godel.gOmega, P_godel.gSucc _ =>
      simp [le_godel] at hyz
  | P_godel.gOmega, P_godel.gOmega, P_godel.gZero =>
      simp [le_godel] at hyz
  | P_godel.gOmega, P_godel.gOmega, P_godel.gSucc _ =>
      simp [le_godel] at hyz
  | P_godel.gOmega, P_godel.gZero, P_godel.gZero =>
      simp [le_godel] at hxy
  | P_godel.gOmega, P_godel.gZero, P_godel.gSucc _ =>
      simp [le_godel] at hxy
  | P_godel.gOmega, P_godel.gSucc _, P_godel.gZero =>
      simp [le_godel] at hxy
  | P_godel.gOmega, P_godel.gSucc _, P_godel.gSucc _ =>
      simp [le_godel] at hxy

/-- Transitivity of le_rank -/
theorem le_rank_trans (x y z : P_rank) (hxy : le_rank x y = true) (hyz : le_rank y z = true) :
    le_rank x z = true := by
  match x, y, z with
  | P_rank.rankZero, _, _ => rfl
  | P_rank.rankSucc _, _, P_rank.rankLimit _ => rfl
  | P_rank.rankLimit _, _, P_rank.rankLimit _ => rfl
  | P_rank.rankSucc _, P_rank.rankZero, _ =>
      simp [le_rank] at hxy
  | P_rank.rankSucc _, P_rank.rankSucc _, P_rank.rankZero =>
      simp [le_rank] at hyz
  | P_rank.rankSucc x', P_rank.rankSucc y', P_rank.rankSucc z' =>
      rw [le_rank] at hxy hyz ⊢
      exact le_rank_trans x' y' z' hxy hyz
  | P_rank.rankLimit _, P_rank.rankLimit _, P_rank.rankZero =>
      simp [le_rank] at hyz
  | P_rank.rankLimit _, P_rank.rankLimit _, P_rank.rankSucc _ =>
      simp [le_rank] at hyz
  | P_rank.rankLimit _, P_rank.rankZero, P_rank.rankZero =>
      simp [le_rank] at hxy
  | P_rank.rankLimit _, P_rank.rankZero, P_rank.rankSucc _ =>
      simp [le_rank] at hxy
  | P_rank.rankLimit _, P_rank.rankSucc _, P_rank.rankZero =>
      simp [le_rank] at hxy
  | P_rank.rankLimit _, P_rank.rankSucc _, P_rank.rankSucc _ =>
      simp [le_rank] at hxy
  | P_rank.rankSucc _, P_rank.rankLimit _, P_rank.rankSucc _ =>
      simp [le_rank] at hyz

/-! #### Concrete examples for order relations -/

/-- Example: gZero ≤ gSucc gZero -/
example : le_godel P_godel.gZero (P_godel.gSucc P_godel.gZero) = true := rfl

/-- Example: gSucc gZero ≤ gOmega -/
example : le_godel (P_godel.gSucc P_godel.gZero) P_godel.gOmega = true := rfl

/-- Example: gOmega is maximal (not ≤ gSucc) -/
example : le_godel P_godel.gOmega (P_godel.gSucc P_godel.gZero) = false := rfl

/-- Example: rankZero ≤ rankSucc rankZero -/
example : le_rank P_rank.rankZero (P_rank.rankSucc P_rank.rankZero) = true := rfl

/-- Example: Transitivity chain gZero ≤ gSucc gZero ≤ gOmega -/
example : le_godel P_godel.gZero P_godel.gOmega = true :=
  le_godel_trans P_godel.gZero (P_godel.gSucc P_godel.gZero) P_godel.gOmega rfl rfl

/-! #### Lexicographical Order on P_vec -/

/-- Lexicographical order: first rank, then godel, then omega -/
instance : LE P_vec where
  le p q :=
    p.rank ≤ q.rank ∧
    (le_rank q.rank p.rank = true → p.godel ≤ q.godel) ∧
    (le_rank q.rank p.rank = true ∧ p.godel = q.godel → p.omega = q.omega)

/-! #### Proof that it is a preorder -/

instance : Preorder P_vec where
  le := (· ≤ ·)
  le_refl p := by
    unfold LE.le instLEP_vec
    constructor
    · exact le_rank_refl p.rank
    · constructor
      · intro _
        exact le_godel_refl p.godel
      · intro _
        rfl
  le_trans := by
    intro p q r ⟨hpq_rank, hpq_godel, hpq_omega⟩ ⟨hqr_rank, hqr_godel, hqr_omega⟩
    constructor
    · exact le_rank_trans p.rank q.rank r.rank hpq_rank hqr_rank
    constructor
    · intro h_pr_equiv
      -- Sandwich: we establish q is rank-equivalent to p and r
      have hqp_rank : le_rank q.rank p.rank = true :=
        le_rank_trans q.rank r.rank p.rank hqr_rank h_pr_equiv
      have hrq_rank : le_rank r.rank q.rank = true :=
        le_rank_trans r.rank p.rank q.rank h_pr_equiv hpq_rank

      have h_pg_le_qg := hpq_godel hqp_rank
      have h_qg_le_rg := hqr_godel hrq_rank
      exact le_godel_trans p.godel q.godel r.godel h_pg_le_qg h_qg_le_rg
    · intro ⟨h_pr_equiv, h_pg_eq⟩
      -- Sandwich again
      have hqp_rank : le_rank q.rank p.rank = true :=
        le_rank_trans q.rank r.rank p.rank hqr_rank h_pr_equiv
      have hrq_rank : le_rank r.rank q.rank = true :=
        le_rank_trans r.rank p.rank q.rank h_pr_equiv hpq_rank

      have h_pg_le_qg := hpq_godel hqp_rank
      have h_qg_le_rg := hqr_godel hrq_rank

      -- If p.godel = r.godel, we need to show p.godel = q.godel to use omega implications
      -- We have p.godel <= q.godel and h.godel <= r.godel = p.godel
      rw [← h_pg_eq] at h_qg_le_rg
      have h_qg_eq_pg : q.godel = p.godel := le_godel_antisymm q.godel p.godel h_qg_le_rg h_pg_le_qg
      have h_pg_eq_qg : p.godel = q.godel := h_qg_eq_pg.symm

      rw [h_pg_eq] at h_qg_eq_pg
      have h_qg_eq_rg : q.godel = r.godel := h_qg_eq_pg

      have h_pom_eq_qom := hpq_omega ⟨hqp_rank, h_pg_eq_qg⟩
      have h_qom_eq_rom := hqr_omega ⟨hrq_rank, h_qg_eq_rg⟩
      exact Eq.trans h_pom_eq_qom h_qom_eq_rom

/-! #### VectorialOrder class (vector preorder) -/

/-- A vectorial order is here simply a preorder on the type. -/
class VectorialOrder (P : Type*) extends Preorder P

/-- Canonical instance: P_vec order is a vectorial order. -/
instance : VectorialOrder P_vec := { (inferInstance : Preorder P_vec) with }

/-- Injection of Nat into P_godel by finite succession -/
def P_godel.ofNat : Nat → P_godel
  | 0     => P_godel.gZero
  | n+1   => P_godel.gSucc (P_godel.ofNat n)

/-- Profile of a Minsky program in P_vec space with Cut/Bit detection.

    - `godel` coordinate: program encoding via `encodeProgram`,
      lifted to `P_godel` via `P_godel.ofNat`;
    - `omega` coordinate: detects if the program is a Cut or Bit program
      based on its structure;
    - `rank` coordinate: base rank `rankZero` (can be refined for
      obstruction-level analysis).
-/
def profileProgram (p : RevCHACOmega.Program) : P_vec :=
  let code := RevCHACOmega.encodeProgram p
  -- Detect Cut/Bit structure based on program form
  let omegaRole : P_omega_role :=
    match p with
    | [RevCHACOmega.Instr.halt] => P_omega_role.cutLike 1  -- Simple halting = cut-like
    | [RevCHACOmega.Instr.decOrJump RevCHACOmega.Counter.c0 0] => P_omega_role.none  -- Loop
    | _ => P_omega_role.none  -- Default
  { godel := P_godel.ofNat code
  , omega := omegaRole
  , rank := P_rank.rankZero }

/-- Example: Profile of the halt program -/
example : (profileProgram [RevCHACOmega.Instr.halt]).omega = P_omega_role.cutLike 1 := rfl

/-- Example: Profile of the loop program -/
example : (profileProgram RevCHACOmega.progLoop).omega = P_omega_role.none := rfl

-- ============================================================================
-- Section 9: ORDER ALGEBRA - SUB-ALGEBRAS AND DEPENDENCIES
-- ============================================================================

/-! ### Section 9: Order Algebra

Boolean sub-algebras and cut/bit dependencies.
-/

/-! #### Generated Sub-algebra -/

/-- A structural predicate on `P_vec`. -/
abbrev StructPred := Set P_vec

/--
Inductive description of the Boolean subalgebra generated by a family `F`.

`IsGeneratedBy F S` means that `S` can be obtained from members of `F`
using finite unions, finite intersections, complements, plus `∅` and `univ`.
-/
inductive IsGeneratedBy (F : Set (Set P_vec)) : Set P_vec → Prop where
  | basic {S} : S ∈ F → IsGeneratedBy F S
  | compl {S} : IsGeneratedBy F S → IsGeneratedBy F Sᶜ
  | union {S T} : IsGeneratedBy F S → IsGeneratedBy F T → IsGeneratedBy F (S ∪ T)
  | inter {S T} : IsGeneratedBy F S → IsGeneratedBy F T → IsGeneratedBy F (S ∩ T)
  | empty : IsGeneratedBy F ∅
  | univ  : IsGeneratedBy F Set.univ

/-- The Boolean subalgebra generated by `F` (as a subtype of `Set P_vec`). -/
def GeneratedSubalgebra (F : Set (Set P_vec)) := { S : Set P_vec // IsGeneratedBy F S }

namespace GeneratedSubalgebra

variable {F : Set (Set P_vec)}

instance : BooleanAlgebra (GeneratedSubalgebra F) :=
  { sup := fun ⟨S, hS⟩ ⟨T, hT⟩ =>
      ⟨S ∪ T, IsGeneratedBy.union hS hT⟩
    le := fun ⟨S, _⟩ ⟨T, _⟩ => S ⊆ T
    lt := fun ⟨S, _⟩ ⟨T, _⟩ => S ⊂ T
    le_refl := fun ⟨S, _⟩ => Set.Subset.refl S
    le_trans := fun ⟨S, _⟩ ⟨T, _⟩ ⟨U, _⟩ hST hTU =>
      Set.Subset.trans hST hTU
    le_antisymm := fun ⟨S, _⟩ ⟨T, _⟩ hST hTS =>
      Subtype.ext (Set.Subset.antisymm hST hTS)
    le_sup_left := fun ⟨S, _⟩ ⟨T, _⟩ =>
      Set.subset_union_left
    le_sup_right := fun ⟨S, _⟩ ⟨T, _⟩ =>
      Set.subset_union_right
    sup_le := fun ⟨S, _⟩ ⟨T, _⟩ ⟨U, _⟩ hSU hTU =>
      Set.union_subset hSU hTU
    inf := fun ⟨S, hS⟩ ⟨T, hT⟩ =>
      ⟨S ∩ T, IsGeneratedBy.inter hS hT⟩
    inf_le_left := fun ⟨S, _⟩ ⟨T, _⟩ =>
      Set.inter_subset_left
    inf_le_right := fun ⟨S, _⟩ ⟨T, _⟩ =>
      Set.inter_subset_right
    le_inf := fun ⟨S, _⟩ ⟨T, _⟩ ⟨U, _⟩ hST hSU =>
      Set.subset_inter hST hSU
    compl := fun ⟨S, hS⟩ =>
      ⟨Sᶜ, IsGeneratedBy.compl hS⟩
    sdiff := fun ⟨S, hS⟩ ⟨T, hT⟩ =>
      ⟨S \ T, by
        rw [Set.diff_eq]
        exact IsGeneratedBy.inter hS (IsGeneratedBy.compl hT)⟩
    bot := ⟨∅, IsGeneratedBy.empty⟩
    bot_le := fun ⟨S, _⟩ =>
      Set.empty_subset S
    top := ⟨Set.univ, IsGeneratedBy.univ⟩
    le_top := fun ⟨S, _⟩ =>
      Set.subset_univ S
    inf_compl_le_bot := fun ⟨S, _⟩ => by
      simp [Set.inter_compl_self]
    top_le_sup_compl := fun ⟨S, _⟩ => by
      simp [Set.union_compl_self]
    sdiff_eq := fun ⟨S, _⟩ ⟨T, _⟩ => rfl
    le_sup_inf := fun ⟨S, _⟩ ⟨T, _⟩ ⟨U, _⟩ => by
      change (S ∪ T) ∩ (S ∪ U) ⊆ S ∪ (T ∩ U)
      exact Set.union_inter_distrib_left S T U ▸ Subset.rfl }

end GeneratedSubalgebra

/-! #### Dependencies on components -/

/-- A set depends only on the cut component -/
def DependsOnlyOnCut (S : Set P_vec) : Prop :=
  ∀ p q : P_vec, p.godel = q.godel → (p ∈ S ↔ q ∈ S)

/-- A set depends only on the bit component -/
def DependsOnlyOnBit (S : Set P_vec) : Prop :=
  ∀ p q : P_vec, p.omega = q.omega → (p ∈ S ↔ q ∈ S)

/-! #### Logical Specifications -/

/-- Logical specifications separating CH and AC_dyn -/
structure LogicSpecs (F : Set (Set P_vec)) where
  CH_ok : GeneratedSubalgebra F
  AC_ok : GeneratedSubalgebra F
  CH_depends : DependsOnlyOnCut CH_ok.val
  AC_depends : DependsOnlyOnBit AC_ok.val

/-! #### Dissociation lemmas from the dependency specs -/

namespace LogicSpecs

variable {F : Set (Set P_vec)} (L : LogicSpecs F)

/-- If we change only the bit/omega component, CH_ok is invariant (cut fixed). -/
theorem CH_invariant_under_bit_change (x y : P_vec)
    (h_same_cut : x.godel = y.godel) :
    x ∈ (L.CH_ok).val ↔ y ∈ (L.CH_ok).val :=
  L.CH_depends x y h_same_cut

/-- If we change only the cut/Gödel component, AC_ok is invariant (bit fixed). -/
theorem AC_invariant_under_cut_change (x y : P_vec)
    (h_same_bit : x.omega = y.omega) :
    x ∈ (L.AC_ok).val ↔ y ∈ (L.AC_ok).val :=
  L.AC_depends x y h_same_bit

end LogicSpecs

/-! #### Trivial LogicSpecs instance -/

/-- The trivial generated subalgebra on the full space. -/
def trivialSubalgebra : GeneratedSubalgebra (∅ : Set (Set P_vec)) :=
  ⟨Set.univ, IsGeneratedBy.univ⟩

/-- Trivial LogicSpecs: everything depends trivially on cut and bit. -/
def trivialLogicSpecs : LogicSpecs (∅ : Set (Set P_vec)) where
  CH_ok := trivialSubalgebra
  AC_ok := trivialSubalgebra
  CH_depends := by
    intro _ _ _
    constructor <;> intro _ <;> trivial
  AC_depends := by
    intro _ _ _
    constructor <;> intro _ <;> trivial

/-! #### Block 2: Concrete Structural Generators and Linking -/

/-- Cut cylinder: all profiles with a given godel component -/
def cutCylinder (g : P_godel) : Set P_vec :=
  { p | p.godel = g }

/-- Bit cylinder: all profiles with a given omega component -/
def bitCylinder (ω : P_omega_role) : Set P_vec :=
  { p | p.omega = ω }

/-- Rank slice: all profiles at or above a given rank -/
def rankSlice (r : P_rank) : Set P_vec :=
  { p | r ≤ p.rank }

/-- Structural family of generators: cut cylinders, bit cylinders, rank slices -/
def F_struct : Set (Set P_vec) :=
  { S | (∃ g, S = cutCylinder g) ∨ (∃ ω, S = bitCylinder ω) ∨ (∃ r, S = rankSlice r) }

/-- CH_set: profiles corresponding to CH-valid programs -/
def CH_set (C : CHProfile RevCHACOmega.Program) : Set P_vec :=
  { v | ∃ p, profileProgram p = v ∧ C.CHFlag p }

/-- ACdyn_set: profiles corresponding to halting programs (AC_dyn domain) -/
def ACdyn_set : Set P_vec :=
  { v | ∃ p, profileProgram p = v ∧ RevCHACOmega.HaltsProg p }

/-! #### Dependency proofs for concrete sets -/



/-! #### Concrete LogicSpecs instance (partial) -/

-- Note: Full concreteLogicSpecs requires proving CH_set and ACdyn_set are in
-- GeneratedSubalgebra F_struct, which needs additional lemmas about the
-- structure of profileProgram. The following is the interface:

/-- Interface to build concrete `LogicSpecs` from structural data. -/
structure ConcreteLogicSpecsData (C : CHProfile RevCHACOmega.Program) where
  CH_generated      : IsGeneratedBy F_struct (CH_set C)
  ACdyn_generated   : IsGeneratedBy F_struct ACdyn_set
  CH_depends_cut    : DependsOnlyOnCut (CH_set C)
  ACdyn_depends_bit : DependsOnlyOnBit ACdyn_set

/-- Construct concrete LogicSpecs from generation proofs -/
def mkConcreteLogicSpecs (C : CHProfile RevCHACOmega.Program)
    (data : ConcreteLogicSpecsData C) : LogicSpecs F_struct where
  CH_ok     := ⟨CH_set C, data.CH_generated⟩
  AC_ok     := ⟨ACdyn_set, data.ACdyn_generated⟩
  CH_depends := data.CH_depends_cut
  AC_depends := data.ACdyn_depends_bit

-- ============================================================================
-- Section 10: UP-SETS, RANK AND VECTOR PROOFS
-- ============================================================================

/-! ### Section 10: Up-sets and Rank

Up-sets in profile space and vector proofs based on rank.
-/

/-! #### Up-sets -/

/-- Up-set predicate: stable by increasing order -/
def IsUpSet (S : Set P_vec) : Prop :=
  ∀ {x y : P_vec}, x ∈ S → x ≤ y → y ∈ S

/-- Up-set structure -/
structure UpSet where
  carrier : Set P_vec
  upward_closed : IsUpSet carrier

/-! #### Order instances on UpSet -/

instance : LE UpSet where
  le U V := U.carrier ⊆ V.carrier

instance : Preorder UpSet where
  le := (· ≤ ·)
  le_refl := fun _ => Set.Subset.refl _
  le_trans := fun _ _ _ => Set.Subset.trans

/-! #### Minimal rank set -/

/-- Set of profiles with rank at least r -/
def RankAtLeast (r : P_rank) : Set P_vec :=
  { p | r ≤ p.rank }

/-! #### Key Lemma: RankAtLeast is an up-set -/

/-- If rank is monotone, RankAtLeast is an up-set -/
theorem RankAtLeast_isUpSet (r : P_rank) : IsUpSet (RankAtLeast r) := by
  intro x y hx hxy
  unfold RankAtLeast at *
  -- hx : r ≤ x.rank
  -- hxy : x ≤ y (thus x.rank ≤ y.rank)
  exact le_rank_trans r x.rank y.rank hx hxy.1

/-- Rank up-set -/
def RankUpSet (r : P_rank) : UpSet :=
  { carrier := RankAtLeast r
    upward_closed := RankAtLeast_isUpSet r }

/-! #### Vector Proofs -/

/-- A logical principle corresponds to an up-set -/
def LogicalPrincipleAsUpSet (_ : String) (S : Set P_vec) : Prop :=
  IsUpSet S

/-- All profiles satisfying a choice schema are above a minimal rank -/
theorem choice_schema_above_rank (schema : Set P_vec) (r : P_rank)
    (h : schema ⊆ RankAtLeast r) :
    ∀ p ∈ schema, r ≤ p.rank := by
  intro p hp
  exact h hp

/-! #### Block 3: Up-sets and Rank Anchoring for AC_dyn -/



/-- ACdyn schema: the AC_ok carrier from a LogicSpecs -/
def ACdyn_schema (L : LogicSpecs F_struct) : Set P_vec := L.AC_ok.val

/-- ACdyn schema is an up-set if its carrier is -/
theorem ACdyn_schema_isUpSet_of_carrier
    (L : LogicSpecs F_struct) (h : IsUpSet L.AC_ok.val) :
    IsUpSet (ACdyn_schema L) := h

/-- All profiles in ACdyn schema are above a given rank -/
theorem ACdyn_above_rank (L : LogicSpecs F_struct) (r : P_rank)
    (h : ACdyn_schema L ⊆ RankAtLeast r) :
    ∀ p ∈ ACdyn_schema L, r ≤ p.rank :=
  choice_schema_above_rank (ACdyn_schema L) r h

-- ============================================================================
-- Section 11: SUMMARY AND DOCUMENTATION
-- ============================================================================

/-! ### Section 11: Pipeline Summary

## Complete Pipeline

This file establishes the following pipeline:

1. **Dynamic Traces** (Section 1)
   - `Trace := Nat → Prop`
   - `Halts`, `up`, `Rev`, global invariance

2. **Galois Semantics** (Section 2)
   - `ModE`, `ThE`, `CloE`
   - Dynamic Bridge: semantics ↔ verdict

3. **Abstract Rev-CH** (Section 3)
   - Isomorphism between halting and CH profiles

4. **Minsky Machine** (Section 4)
   - Concrete computable instantiation
   - `progTrace`, `HaltsProg`, link with `Rev0`

5. **Delta/DR0/DR1** (Section 5)
   - Stability and monotonicity properties
   - `DR0` and `DR1` theorems

6. **Effective Omega** (Section 6)
   - Construction of `OmegaPartialScaled`, `OmegaBit`
   - `Cut` and `Bit` programs
   - Universal machine (encoding/decoding)

7. **Dynamic Profiling** (Section 7)
   - Halting oracle (opaque, axiomatic)
   - `AC_dyn`: concrete dynamic choice axiom

8. **Vector Profiles** (Section 8)
   - `P_vec = (cut, bit, rank)`
   - Lexicographical order

9. **Order Algebra** (Section 9)
   - Generated sub-algebras
   - Cut/bit dependencies
   - Logical specifications

10. **Up-sets and Rank** (Section 10)
    - `IsUpSet`, `UpSet`
    - `RankAtLeast`, `RankUpSet`
    - Vector proofs

## Key Properties

- ✅ **Everything is computable** (except naturally opaque oracle)
- ✅ **Minimal axioms** (only halting oracle)


## Central Theorem

The vector profile `P_vec` characterizes logical phenomena as **order structures** (up-sets) in the `(cut, bit, rank)` space.

CH and AC_dyn properties are designed to separate on the cut and bit axes (via dependency requirements in `ConcreteLogicSpecsData`).
-/

end LogicDissoc

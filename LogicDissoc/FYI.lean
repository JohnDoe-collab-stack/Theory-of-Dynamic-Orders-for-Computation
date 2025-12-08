import Mathlib.Data.Set.Basic
import Mathlib.Data.Nat.Basic

/-!
# BasicSemantics + Rev + Dynamic Bridge

This file shows, using only:

* a pure Galois-style semantic layer (`ModE`, `ThE`, `CloE`),
* a robust halting layer (`Rev`, `Rev0`, `Halts`),

that once we add a local reading and a bridge
`(Γ ⊨ φ) ↔ Halts (LR Γ φ)`, we obtain the equivalence

  `φ ∈ CloE Γ ↔ verdict K LR Γ φ`

for every admissible reverse–halting kit `K`.

Intuitively: the Galois/ZFC layer (`CloE`, `ModE`, `ThE`) captures
only the *static* picture; `Rev`, together with its invariance, adds
a canonical dynamic structure (the robust verdict) that the static
layer alone does not specify.

At the meta-level, this canonical dynamic can be organised into a
single concrete Rev–CH–AC system with an associated dynamic choice
operator `AC_dyn` on halting codes. Classical Gödel/Turing results
only forbid perfect internal predicates for `RealHalts`; here the
genuinely new object is the **term-level** dynamic choice operator
`AC_dyn`, and Level 2 is precisely a non-internalisation theorem
for this specific operator.

Level 2 of the file is entirely meta-theoretic. In a Turing–Gödel
context, and assuming a *local* reflection principle for the
meta-level halting predicate `H_from_Fint` attached to any
hypothetical internalisation of this Rev–CH–AC system (which is
then equivalent to `RealHalts` via `H_from_Fint_iff_RealHalts`),
it shows that no recursive consistent theory of ZFC strength can
internalise this specific operative AC as a single total internal
mechanism (predicate/function) that is correct and complete for
the real halting profile. Formally, the non-internalisation
statement is conditional on the axiom `reflect_for_this_H` stated
in §2.3 (Level 2: AC Operative Internalisation Impossibility).
-/

universe u v

/-
  STATIC LAYER: BasicSemantics
-/

variable {Sentence : Type u} {Model : Type v}
variable (Sat : Model → Sentence → Prop)

/-- Class of models of Γ. -/
def ModE (Γ : Set Sentence) : Set Model :=
  { M | ∀ φ ∈ Γ, Sat M φ }

/-- Theory true in all models of `K`. -/
def ThE (K : Set Model) : Set Sentence :=
  { φ | ∀ M ∈ K, Sat M φ }

/-- Semantic closure operator: `CloE(Γ) = ThE(ModE(Γ))`. -/
def CloE (Γ : Set Sentence) : Set Sentence :=
  ThE Sat (ModE Sat Γ)

namespace LogicDissoc

open Set

variable {Sat}

/-- Extensivity of semantic closure. -/
lemma subset_CloE (Γ : Set Sentence) :
  Γ ⊆ CloE Sat Γ := by
  intro φ hφ
  unfold CloE ThE ModE
  simp [Set.mem_setOf_eq] at *
  intro M hM
  exact hM φ hφ

/-- Monotonicity of semantic closure. -/
lemma CloE_mono {Γ Δ : Set Sentence} (h : Γ ⊆ Δ) :
  CloE Sat Γ ⊆ CloE Sat Δ := by
  intro φ hφ
  unfold CloE ThE ModE at *
  simp [Set.mem_setOf_eq] at *
  intro M hM
  apply hφ
  intro ψ hψ
  exact hM ψ (h hψ)

/-- Idempotence of semantic closure. -/
lemma CloE_idem (Γ : Set Sentence) :
  CloE Sat (CloE Sat Γ) ⊆ CloE Sat Γ := by
  -- By extensivity, we have CloE Γ ⊆ CloE (CloE Γ)
  -- We want to show CloE (CloE Γ) ⊆ CloE Γ
  -- By monotonicity: if Γ ⊆ Δ then CloE Γ ⊆ CloE Δ
  -- So CloE Γ ⊆ CloE (CloE Γ) by taking Δ = CloE Γ and using extensivity
  -- But we need the reverse direction
  -- Actually: every φ in CloE(CloE Γ) is in CloE Γ because CloE Γ already contains
  -- all its semantic consequences
  intro φ hφ
  unfold CloE ThE ModE at *
  simp [Set.mem_setOf_eq] at *
  intro M hM
  -- hφ says: for all M' satisfying CloE Γ, M' ⊨ φ
  -- We want: M ⊨ φ where M satisfies Γ
  -- Since M satisfies Γ, it satisfies CloE Γ
  apply hφ
  -- Show M satisfies CloE Γ
  intro ψ hψ
  exact hψ M hM

/-
  DYNAMIC LAYER: Rev (robust halting)

  The operator `Rev` is parametrised by an abstract kit `K`. Under the
  sole axiom `DetectsMonotone K`, its concrete incarnation `Rev0 K`
  is extensionally invariant: on traces, it coincides with `Halts`,
  even though this is not a definitional equality.
-/

/-- A temporal trace: at each time `n`, a proposition may hold. -/
abbrev Trace := ℕ → Prop

/-- Temporal closure: `(up T) n` means “there exists `k ≤ n` with `T k`”. -/
def up (T : Trace) : Trace :=
  fun n => ∃ k ≤ n, T k

lemma up_mono (T : Trace) :
  ∀ {n m : ℕ}, n ≤ m → up T n → up T m := by
  intro n m hnm h
  rcases h with ⟨k, hk_le_n, hk_T⟩
  exact ⟨k, Nat.le_trans hk_le_n hnm, hk_T⟩

lemma exists_up_iff (T : Trace) :
  (∃ n, up T n) ↔ ∃ n, T n := by
  constructor
  · rintro ⟨n, hn⟩
    rcases hn with ⟨k, _, hk_T⟩
    exact ⟨k, hk_T⟩
  · rintro ⟨k, hk_T⟩
    refine ⟨k, ?_⟩
    exact ⟨k, le_rfl, hk_T⟩

/-- Direct halting predicate on traces: `Halts T` iff `T` holds at some time. -/
def Halts (T : Trace) : Prop :=
  ∃ n : ℕ, T n

/--
A reverse-halting kit:
`Proj X` is the stabilized projection of a family `X : ℕ → Prop`.
-/
structure RHKit where
  Proj : (ℕ → Prop) → Prop

/--
`DetectsMonotone K` encodes that, for any monotone family `X`,
the projection `K.Proj X` coincides exactly with `∃ n, X n`.
-/
structure DetectsMonotone (K : RHKit) : Prop where
  proj_of_mono :
    ∀ X : ℕ → Prop,
      (∀ {n m}, n ≤ m → X n → X m) →
      (K.Proj X ↔ ∃ n, X n)

section

variable (K : RHKit)

/--
Revision operator obtained from a reverse-halting kit.

Its behaviour on traces is a priori parametrised by `K`, but under
`DetectsMonotone K` the induced predicate `Rev0 K` is extensionally
forced to agree with `Halts` on all traces (see `Rev0_eq_Halts`).
-/
def Rev (T : Trace) : Prop :=
  K.Proj (fun n => up T n)

/-- Concrete revision alias. -/
def Rev0 (T : Trace) : Prop :=
  Rev K T

/-- Core reverse-halting lemma: `Rev` detects “∃ n, T n` via monotonicity of `up T`. -/
lemma Rev_iff_exists (DK : DetectsMonotone K) (T : Trace) :
  Rev K T ↔ ∃ n, T n := by
  unfold Rev
  have hmono : ∀ {n m}, n ≤ m → up T n → up T m := up_mono T
  have hx : K.Proj (fun n => up T n) ↔ ∃ n, up T n :=
    DetectsMonotone.proj_of_mono DK (fun n => up T n) (by
      intro n m hnm h
      exact hmono hnm h)
  apply Iff.trans hx
  exact exists_up_iff T

/-- Reverse-halting equals direct halting on traces. -/
lemma Rev_iff_Halts (DK : DetectsMonotone K) (T : Trace) :
  Rev K T ↔ Halts T := by
  simpa [Halts] using Rev_iff_exists K DK T

/-- In particular, `Rev0` is also just `Halts`. -/
lemma Rev0_iff_Halts (DK : DetectsMonotone K) (T : Trace) :
  Rev0 K T ↔ Halts T := by
  simpa [Rev0] using Rev_iff_Halts K DK T

/--
Canonical invariance of `Rev0` with respect to `Halts`.

For any reverse–halting kit `K` satisfying `DetectsMonotone K`,
the verdict `Rev0 K` on traces coincides extensionally with the
plain halting predicate `Halts`. This is *not* a definitional
equality, but an invariance result: the observable behaviour of
`Rev` on traces is independent of the particular implementation
of `K.Proj`, as soon as it detects monotone families exactly as `∃ n`.

In other words, many different reverse–halting kits collapse to
the same robust halting predicate `Halts` when restricted to traces.
-/
lemma Rev0_eq_Halts (DK : DetectsMonotone K) :
    ∀ T, Rev0 K T ↔ Halts T := by
  intro T
  exact Rev0_iff_Halts K DK T

/--
Rev0 over a family of traces: the verdict is pointwise, independent of family size.
For any indexing type I (finite or infinite), we get Rev0 ↔ Halts for each trace.
-/
lemma Rev0_eq_Halts_family (DK : DetectsMonotone K)
    {I : Type*} (T : I → Trace) :
    ∀ i : I, Rev0 K (T i) ↔ Halts (T i) := by
  intro i
  exact Rev0_eq_Halts K DK (T i)

end  -- section K

/-!
### Global Uniqueness of Rev

Two different kits K₁ and K₂ (both satisfying DetectsMonotone) give the same
verdict on every trace. This shows Rev is robust: implementation details don't
matter as long as the kit property holds.
-/

/-- Two DetectsMonotone kits agree on all traces. -/
theorem Rev_global_uniqueness
    (K₁ K₂ : RHKit)
    (DK₁ : DetectsMonotone K₁)
    (DK₂ : DetectsMonotone K₂) :
    ∀ T : Trace, Rev0 K₁ T ↔ Rev0 K₂ T := by
  intro T
  have h₁ : Rev0 K₁ T ↔ Halts T := Rev0_iff_Halts K₁ DK₁ T
  have h₂ : Rev0 K₂ T ↔ Halts T := Rev0_iff_Halts K₂ DK₂ T
  exact h₁.trans h₂.symm

/-- Family version: two kits agree on every trace in any family. -/
theorem Rev_global_uniqueness_family
    (K₁ K₂ : RHKit)
    (DK₁ : DetectsMonotone K₁)
    (DK₂ : DetectsMonotone K₂)
    {I : Type*} (T : I → Trace) :
    ∀ i : I, Rev0 K₁ (T i) ↔ Rev0 K₂ (T i) := by
  intro i
  exact Rev_global_uniqueness K₁ K₂ DK₁ DK₂ (T i)

section
variable (K : RHKit)

/-!
  Local reading, provability and verdicts.
-/

universe u' v'

/-- Abstract local reading: to each pair `(Γ, φ)` it assigns a temporal trace. -/
abbrev LocalReading (Context : Type v') (Sentence : Type u') :=
  Context → Sentence → Trace

/-- Provability: there exists a time at which the reading of `φ` under `Γ` holds. -/
def Prov {Context : Type v'} {Sentence : Type u'}
    (LR : LocalReading Context Sentence)
    (Γ : Context) (φ : Sentence) : Prop :=
  ∃ n : ℕ, LR Γ φ n

/-- Stabilized verdict via `Rev0` coming from the kit `(K, DK)`. -/
def verdict {Context : Type v'} {Sentence : Type u'}
    (LR : LocalReading Context Sentence)
    (Γ : Context) (φ : Sentence) : Prop :=
  Rev0 K (LR Γ φ)

/-- Equivalence between stabilized verdict and provability. -/
lemma verdict_iff_Prov
    {Context : Type v'} {Sentence : Type u'}
    (DK : DetectsMonotone K)
    (LR : LocalReading Context Sentence)
    (Γ : Context) (φ : Sentence) :
  verdict K LR Γ φ ↔ Prov LR Γ φ := by
  unfold verdict Prov
  simpa using Rev0_iff_Halts K DK (LR Γ φ)

end  -- section K

/-- Direct halting for a local reading: the trace `LR Γ φ` ever holds. -/
def HaltsLR {Context : Type v'} {Sentence : Type u'}
    (LR : LocalReading Context Sentence)
    (Γ : Context) (φ : Sentence) : Prop :=
  ∃ n : ℕ, LR Γ φ n

lemma HaltsLR_iff_Prov
    {Context : Type v'} {Sentence : Type u'}
    (LR : LocalReading Context Sentence)
    (Γ : Context) (φ : Sentence) :
  HaltsLR LR Γ φ ↔ Prov LR Γ φ := Iff.rfl

/-
  BRIDGE: from semantic consequence to halting of LR
-/

universe uS vM

variable {Sentence : Type uS} {Model : Type vM}
variable (Sat : Model → Sentence → Prop)

/-- Semantic consequence via the Galois closure `CloE`. -/
def SemConsequences (Γ : Set Sentence) (φ : Sentence) : Prop :=
  φ ∈ CloE Sat Γ

/-- Local reading on contexts = sets of sentences. -/
abbrev LR := LocalReading (Set Sentence) Sentence

/--
Dynamic bridge: semantic consequence is realised as halting
of the trace produced by `LR`.
-/
def DynamicBridge (LR : LocalReading (Set Sentence) Sentence) : Prop :=
  ∀ Γ φ, SemConsequences Sat Γ φ ↔ Halts (LR Γ φ)

/--
Main equivalence:

Under `BasicSemantics` + `Rev` + `DynamicBridge`,
semantic consequence (φ ∈ CloE Γ) coincides with the robust
Rev–verdict on the trace `LR Γ φ`, for every admissible `K`.
-/
lemma semantic_iff_verdict
    {K : RHKit} (DK : DetectsMonotone K)
    (LR : LocalReading (Set Sentence) Sentence)
    (bridge : DynamicBridge Sat LR) :
  ∀ (Γ : Set Sentence) (φ : Sentence),
    SemConsequences Sat Γ φ ↔
      verdict K LR Γ φ := by
  intro Γ φ
  -- semantic consequence ↔ halting of LR Γ φ
  have h_sem : SemConsequences Sat Γ φ ↔ Halts (LR Γ φ) :=
    bridge Γ φ
  -- verdict ↔ Halts(LR Γ φ) via Rev0_iff_Halts
  have h_dyn : verdict K LR Γ φ ↔ Halts (LR Γ φ) := by
    -- `verdict_iff_Prov` + `HaltsLR_iff_Prov`
    have h1 : verdict K LR Γ φ ↔ Prov LR Γ φ :=
      verdict_iff_Prov K DK LR Γ φ
    have h2 : Halts (LR Γ φ) ↔ Prov LR Γ φ := by
      -- Halts = ∃ n, LR Γ φ n = HaltsLR = Prov
      unfold Halts
      exact HaltsLR_iff_Prov LR Γ φ
    -- combine
    exact h1.trans h2.symm
  -- combine semantic and dynamic sides
  exact h_sem.trans h_dyn.symm


/-
  REV–CH ISOMORPHISM LAYER

  Goal: give a precise notion of “Rev halting is isomorphic to CH”
  over a common type of programs/configurations.
-/

section RevCH

/-- Abstract halting structure attached to Rev. -/
structure RevHalting (Prog : Type) where
  Halts : Prog → Prop

/-- Abstract CH-profile structure on the same Prog. -/
structure CHProfile (Prog : Type) where
  CHFlag : Prog → Prop

/--
An isomorphism between Rev halting and a CH-profile:
they induce logically equivalent predicates on the same Prog.

This is the minimal formal content of the slogan
“Rev halting is isomorphic to CH”.
-/
structure RevCHIso {Prog : Type}
    (R : RevHalting Prog) (C : CHProfile Prog) : Prop where
  iso : ∀ p : Prog, R.Halts p ↔ C.CHFlag p

/--
From a Rev–CH isomorphism, we get a bijection between
the halting programs and the “CH-true” programs (as subtypes).
-/
def RevCHIso.toEquiv {Prog : Type}
    {R : RevHalting Prog} {C : CHProfile Prog}
    (h : RevCHIso R C) :
    {p : Prog // R.Halts p} ≃ {p : Prog // C.CHFlag p} :=
  { toFun := fun ⟨p, hp⟩ =>
      ⟨p, (h.iso p).1 hp⟩
    , invFun := fun ⟨p, hc⟩ =>
      ⟨p, (h.iso p).2 hc⟩
    , left_inv := by
        intro x
        cases x with
        | mk p hp =>
          -- the underlying element p is preserved
          rfl
    , right_inv := by
        intro x
        cases x with
        | mk p hc =>
          -- same here
          rfl }

/--
Conversely, a bijection between the halting programs and
the CH-true programs (with matching underlying Prog) induces
an isomorphism of predicates.
-/
def RevCHIso.ofEquiv {Prog : Type}
    (R : RevHalting Prog) (C : CHProfile Prog)
    (e : {p : Prog // R.Halts p} ≃ {p : Prog // C.CHFlag p})
    (h_preserves : ∀ p hp, (e ⟨p, hp⟩).val = p) :
    RevCHIso R C :=
  { iso := by
      intro p
      constructor
      · intro hp
        -- build an element on the CH side and use preservation
        have heq : (e ⟨p, hp⟩).val = p := h_preserves p hp
        have hprop := (e ⟨p, hp⟩).property
        rw [heq] at hprop
        exact hprop
      · intro hc
        -- use the inverse equivalence with preservation
        -- For the inverse, we need (e.symm ⟨p, hc⟩).val = p
        have heq : (e.symm ⟨p, hc⟩).val = p := by
          -- Let q = (e.symm ⟨p, hc⟩).val and hq = (e.symm ⟨p, hc⟩).property
          -- Then e ⟨q, hq⟩ = ⟨p, hc⟩ by Equiv.apply_symm_apply
          -- By h_preserves, (e ⟨q, hq⟩).val = q
          -- But (e ⟨q, hq⟩).val = p by the equality above
          -- So q = p
          let q := (e.symm ⟨p, hc⟩).val
          let hq := (e.symm ⟨p, hc⟩).property
          have h1 : e ⟨q, hq⟩ = ⟨p, hc⟩ := Equiv.apply_symm_apply e ⟨p, hc⟩
          have h2 : (e ⟨q, hq⟩).val = q := h_preserves q hq
          have h3 : (e ⟨q, hq⟩).val = p := congrArg (·.val) h1
          exact h2.symm.trans h3
        rw [← heq]
        exact (e.symm ⟨p, hc⟩).property }

end RevCH


/-
  Strategy formalisation (meta-level roadmap):

  - Real halting profile  : RealHalts : Code → Prop
  - Dynamic halting (Rev) : Halts_rev : Prog → Prop
  - CH-local profile      : CH_local : Prog → Prop
  - Dynamic AC            : F_dyn : {p // CH_local p} → Witness

  Hypotheses:
    • RealHalts(e)  ↔  Halts_rev (embed e)
    • Halts_rev(p)  ↔  CH_local p

  Meta-theorem (Turing–Gödel, abstracted in this file):
    Any internal predicate H(e) in a recursive consistent theory T
    that is total/correct/complete for RealHalts is impossible.

  Conclusion:
    Fix once and for all a concrete Rev–CH–AC system `S` as above and
    the induced dynamic choice `AC_dyn` on halting codes. Under these
    isomorphisms, `AC_dyn` is a single, well-defined operative form of
    Choice on the halting sector. Level 2 shows that no internal object
    in T can reproduce this specific Rev–CH–AC/`AC_dyn` dynamics as a
    single total predicate/function without contradicting Turing–Gödel.
-/



/-
  1. Turing–Gödel meta-context and impossibility of a perfect
     internal halting predicate.
-/

/-- Abstract Turing–Gödel context for a theory T. -/
structure TuringGodelContext' (Code PropT : Type) where
  /-- Real halting predicate in the meta-world. -/
  RealHalts : Code → Prop

  /-- Provability predicate for the theory T. -/
  Provable : PropT → Prop

  /-- Falsity inside T. -/
  FalseT   : PropT

  /-- Negation inside T. -/
  Not      : PropT → PropT

  /-- Consistency of T: it does not prove FalseT. -/
  consistent : ¬ Provable FalseT

  /-- If T proves P and ¬P, then T proves FalseT. -/
  absurd : ∀ {p}, Provable p → Provable (Not p) → Provable FalseT

  /-- Diagonalisation: for any formula H(x) there is a code e
      such that e halts iff T proves ¬H(e). -/
  diagonal_program :
    ∀ (H : Code → PropT), ∃ e : Code, RealHalts e ↔ Provable (Not (H e))



/-- A candidate internal halting predicate for T. -/
structure InternalHaltingPredicate {Code PropT : Type}
    (ctx : TuringGodelContext' Code PropT) where
  H : Code → PropT
  /-- Totality: T decides H(e) for every e. -/
  total : ∀ e, ctx.Provable (H e) ∨ ctx.Provable (ctx.Not (H e))
  /-- Correctness: if e really halts, T proves H(e). -/
  correct : ∀ e, ctx.RealHalts e → ctx.Provable (H e)
  /-- Completeness: if e does not halt, T proves ¬H(e). -/
  complete : ∀ e, ¬ ctx.RealHalts e → ctx.Provable (ctx.Not (H e))


/--
Turing–Gödel meta-theorem (abstract form):

No internal halting predicate can be total, correct, and complete
for the real halting predicate RealHalts, if T is consistent.
-/
theorem no_internal_halting_predicate'
    {Code PropT : Type}
    (ctx : TuringGodelContext' Code PropT) :
    ¬ ∃ _ : InternalHaltingPredicate ctx, True := by
  intro h
  rcases h with ⟨I, _⟩
  -- Use diagonal program with the internal candidate I.H
  obtain ⟨e, he⟩ := ctx.diagonal_program I.H

  by_cases hReal : ctx.RealHalts e
  · -- Case 1: e halts in reality
    have hH  : ctx.Provable (I.H e) := I.correct e hReal
    have hNotH : ctx.Provable (ctx.Not (I.H e)) := he.mp hReal
    have hFalse : ctx.Provable ctx.FalseT := ctx.absurd hH hNotH
    exact ctx.consistent hFalse
  · -- Case 2: e does not halt in reality
    have hNotH : ctx.Provable (ctx.Not (I.H e)) := I.complete e hReal
    have hHalt : ctx.RealHalts e := he.mpr hNotH
    exact hReal hHalt


/-
  2. Rev–CH–AC system: dynamic halting, CH_local, and dynamic AC.
-/

/-- A Rev–CH–AC system sitting over a given Turing–Gödel context. -/
structure RevCHACSystem
    {Code PropT : Type}
    (ctx : TuringGodelContext' Code PropT) where

  /-- Type of dynamic programs/configurations. -/
  Prog : Type

  /-- Type of witnesses produced by the dynamic AC. -/
  Witness : Type

  /-- Embedding of codes into dynamic programs. -/
  embed : Code → Prog

  /-- Halting predicate coming from Rev on programs. -/
  Halts_rev : Prog → Prop

  /-- CH-local profile on programs. -/
  CH_local : Prog → Prop

  /-- Dynamic choice function restricted to CH_local. -/
  F_dyn : {p : Prog // CH_local p} → Witness

  /-- Iso 1: real halting ≃ Rev halting via embed. -/
  iso_real_rev : ∀ e : Code, ctx.RealHalts e ↔ Halts_rev (embed e)

  /-- Iso 2: Rev halting ≃ CH_local on Prog. -/
  iso_rev_CH : ∀ p : Prog, Halts_rev p ↔ CH_local p


namespace RevCHACSystem

variable {Code PropT : Type}
variable (ctx : TuringGodelContext' Code PropT)
variable (S   : RevCHACSystem ctx)

/-- Forbidden halting profile on codes (real halting). -/
def H_forbidden (ctx : TuringGodelContext' Code PropT) : Code → Prop :=
  ctx.RealHalts

/-- Transport of CH_local back to codes via embed. -/
def CH_on_code {ctx : TuringGodelContext' Code PropT} (S : RevCHACSystem ctx) : Code → Prop :=
  fun e => S.CH_local (S.embed e)

/-- Isomorphism between the forbidden profile and CH_local ∘ embed. -/
theorem H_forbidden_iff_CH_on_code (e : Code) :
    H_forbidden ctx e ↔ CH_on_code S e := by
  unfold H_forbidden CH_on_code
  have h1 := S.iso_real_rev e
  -- h1 : ctx.RealHalts e ↔ S.Halts_rev (S.embed e)
  have h2 := S.iso_rev_CH (S.embed e)
  -- h2 : S.Halts_rev (S.embed e) ↔ S.CH_local (S.embed e)
  exact h1.trans h2

/-- Dynamic AC on codes, defined whenever the code really halts. -/
def AC_dyn (e : Code) (h : ctx.RealHalts e) : S.Witness :=
  let p  := S.embed e
  have h_rev : S.Halts_rev p :=
    (S.iso_real_rev e).mp h
  have h_CH : S.CH_local p :=
    (S.iso_rev_CH p).mp h_rev
  S.F_dyn ⟨p, h_CH⟩


/-
  3. Internalisation attempt: what it would mean for a theory T
     to internalise perfectly this Rev–CH–AC dynamics.
-/

/-- A candidate internalisation of the dynamic halting profile. -/
structure InternalisationCandidate where
  /-- Internal predicate H(e) in T. -/
  H : Code → PropT
  /-- Totality: T decides H(e) for every e. -/
  total : ∀ e, ctx.Provable (H e) ∨ ctx.Provable (ctx.Not (H e))
  /-- Correctness w.r.t. the real halting profile. -/
  correct : ∀ e, ctx.RealHalts e → ctx.Provable (H e)
  /-- Completeness w.r.t. the real halting profile. -/
  complete : ∀ e, ¬ ctx.RealHalts e → ctx.Provable (ctx.Not (H e))


/--
Main theorem (schematic form):

Under the Turing–Gödel hypotheses and the Rev–CH–AC isomorphisms,
there is no internal predicate H(e) in T that is total, correct and
complete for the real halting profile. In particular, T cannot
fully internalise the Rev–CH–AC dynamics.
-/
theorem no_full_internalisation :
    ¬ ∃ _ : InternalisationCandidate ctx, True := by
  -- This is just a rephrasing of no_internal_halting_predicate,
  -- since InternalisationCandidate has the same shape as
  -- InternalHaltingPredicate.
  intro h
  rcases h with ⟨I, _⟩
  let J : InternalHaltingPredicate ctx :=
    { H := I.H
      , total := I.total
      , correct := I.correct
      , complete := I.complete }
  have : ¬ ∃ J' : InternalHaltingPredicate ctx, True :=
    no_internal_halting_predicate' ctx
  exact this ⟨J, trivial⟩

/-
  LEVEL 2: AC OPERATIVE INTERNALISATION IMPOSSIBILITY

  This section extends Level 1 (no_full_internalisation) to show that
  even with an internal choice function F_int that "respects" AC_dyn
  via a Decode function, we cannot internalize the full dynamic.

  Key insight: if ZFC could internalize a perfect choice function aligned
  with RealHalts via AC_dyn, we could construct H(e) that is total/correct/complete,
  violating Level 1.
-/

/--
Hypothetical internalisation with AC operative alignment.

This structure captures what it would mean for ZFC (or a similar
recursive consistent theory T) to *clone* the specific external
dynamic choice `AC_dyn` into a single internal mechanism:

* an internal choice function `F_int` (total on all codes),
* a decoding `Decode : W_int → Witness` linking internal to external,
* perfect alignment: on each code `e` that really halts, `F_int e`
  decodes exactly to the external `AC_dyn ctx S e h`.

If such a structure existed for this fixed Rev–CH–AC system `S`, it
would allow constructing a forbidden internal halting predicate.
-/
structure InternalisationWithAC where
  /-- Type of internal witnesses -/
  W_int : Type

  /-- Internal choice function (total in ZFC) -/
  F_int : Code → W_int

  /-- Decoding from internal to external witnesses -/
  Decode : W_int → S.Witness

  /-- Correction axiom: on halting codes, F_int decodes to AC_dyn -/
  correct_on_halting :
    ∀ (e : Code) (h : ctx.RealHalts e),
      Decode (F_int e) = AC_dyn ctx S e h

/--
Halting predicate constructed from internal choice function.

H(e) says: "e halts AND the witness F_int gives matches AC_dyn"

This directly asserts halting, making it total/correct/complete.
-/
def H_from_Fint (I : InternalisationWithAC ctx S) (e : Code) : Prop :=
  ∃ h : ctx.RealHalts e, I.Decode (I.F_int e) = AC_dyn ctx S e h

/--
`H_from_Fint` decides exactly `RealHalts` at the meta-level:
for every code `e`, `H_from_Fint I e` is equivalent to `RealHalts e`.
-/
lemma H_from_Fint_iff_RealHalts
    (I : InternalisationWithAC ctx S) (e : Code) :
    H_from_Fint ctx S I e ↔ ctx.RealHalts e := by
  constructor
  · intro h
    rcases h with ⟨h_real, _⟩
    exact h_real
  · intro h_real
    exact ⟨h_real, I.correct_on_halting e h_real⟩

/--
Local reflection principle, specific to the halting predicate
`H_from_Fint ctx S I` constructed from an internalisation `I`.

Formally, for any Turing–Gödel context `ctx`, any Rev–CH–AC system `S`
over `ctx`, and any hypothetical internalisation
`I : InternalisationWithAC ctx S`, there exists an internal predicate
`H_enc : Code → PropT` such that:

* whenever `H_from_Fint ctx S I e` holds at the meta-level,
  the theory proves `H_enc e`;
* whenever `H_from_Fint ctx S I e` fails at the meta-level,
  the theory proves `Not (H_enc e)`;
* and for every `e`, the theory proves either `H_enc e` or `Not (H_enc e)`.

In particular, given `I`, this makes `H_enc` a total, correct and complete
internal encoding of the meta-level predicate `H_from_Fint ctx S I`
(which, by `H_from_Fint_iff_RealHalts`, is equivalent to `RealHalts`).
-/
axiom reflect_for_this_H :
  ∀ {Code PropT : Type}
    (ctx : TuringGodelContext' Code PropT)
    (S   : RevCHACSystem ctx)
    (I   : InternalisationWithAC ctx S),
    ∃ H_enc : Code → PropT,
      (∀ e, H_from_Fint ctx S I e → ctx.Provable (H_enc e)) ∧
      (∀ e, ¬ H_from_Fint ctx S I e → ctx.Provable (ctx.Not (H_enc e))) ∧
      (∀ e, ctx.Provable (H_enc e) ∨ ctx.Provable (ctx.Not (H_enc e)))

/--
Main theorem (Level 2): AC operative internalisation impossibility.

If a theory `T` satisfying the Turing–Gödel hypotheses could internalise
the Rev–CH–AC system `S` via a kit `I : InternalisationWithAC ctx S`,
then the induced halting predicate `H_from_Fint ctx S I` could be
reflected internally, yielding a total/correct/complete internal
predicate for `RealHalts`, contradicting Level 1.
-/
theorem no_AC_operative_internalisation :
    ¬ ∃ _ : InternalisationWithAC ctx S, True := by
  intro ⟨I, _⟩

  -- 1. Local reflection for this specific H_from_Fint ctx S I
  obtain ⟨H_enc, h_enc_pos, h_enc_neg, h_enc_total⟩ :=
    reflect_for_this_H ctx S I

  -- 2. Level 1 obstruction: no total/correct/complete internal predicate
  have h_no_candidate : ¬ ∃ C : InternalisationCandidate ctx, True :=
    no_full_internalisation ctx

  -- 3. Build such a candidate C from H_enc and the meta-equivalence
  --    H_from_Fint ↔ RealHalts
  let C : InternalisationCandidate ctx :=
    { H := H_enc
      , total := by
          intro e
          exact h_enc_total e
      , correct := by
          intro e hReal
          -- RealHalts e ⇒ H_from_Fint e via the equivalence
          have hH : H_from_Fint ctx S I e :=
            (H_from_Fint_iff_RealHalts ctx S I e).2 hReal
          exact h_enc_pos e hH
      , complete := by
          intro e hNotReal
          -- ¬RealHalts e ⇒ ¬H_from_Fint e via the equivalence
          have hNotH : ¬ H_from_Fint ctx S I e := by
            intro hH
            have hReal : ctx.RealHalts e :=
              (H_from_Fint_iff_RealHalts ctx S I e).1 hH
            exact hNotReal hReal
          exact h_enc_neg e hNotH }

  -- 4. Contradiction with Level 1
  exact h_no_candidate ⟨C, trivial⟩

end RevCHACSystem


end LogicDissoc

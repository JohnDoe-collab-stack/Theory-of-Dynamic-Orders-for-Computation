import Boole.AbstractKernel

namespace LogicDissoc
namespace Boole
namespace OmegaInstance

open LogicDissoc.Boole.AbstractKernel

/-!
# Omega Instance of the Abstract Kernel

This module instantiates the abstract kernel with the concrete types from `LogicDissoc`.

## Design Constraints

- **Fully computable**: No `noncomputable` annotations allowed.
- **No rationals**: We use `Nat` pairs instead of `ℚ` for cut indices.
-/

section Definitions

universe u v w

variable {Model : Type u} {Sentence : Type v} {CodeOmega : Type w}

/-! ## 1. Index Type -/

/-- Cut index as a pair of naturals (numerator, denominator) for computability. -/
abbrev CutIndex := Nat × Nat

/-- Bit index as a pair (position, value). -/
abbrev BitIndex := Nat × Nat

/-- Omega index: either a cut index or a bit index. -/
inductive OmegaIndex
  | cutIdx : CutIndex → OmegaIndex
  | bitIdx : BitIndex → OmegaIndex
  deriving DecidableEq, Repr

/-! ## 2. Valuation -/

/-- Valuation function derived from the satisfaction relation. -/
def valOmega
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : CutIndex → CodeOmega → Sentence)
    (Bit : BitIndex → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (M : Model) (i : OmegaIndex) : Bool :=
  match i with
  | OmegaIndex.cutIdx q => decide (Sat M (Cut q omega0))
  | OmegaIndex.bitIdx b => decide (Sat M (Bit b omega0))

/-! ## 3. Instantiated Kernel -/

/-- The concrete `BitAt` predicate for Omega. -/
abbrev BitAtOmega
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : CutIndex → CodeOmega → Sentence)
    (Bit : BitIndex → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (i : OmegaIndex) : Set Model :=
  BitAt OmegaIndex Model (valOmega Sat Cut Bit omega0) i

/-- The concrete `BitIsZero` predicate for Omega. -/
abbrev BitIsZeroOmega
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : CutIndex → CodeOmega → Sentence)
    (Bit : BitIndex → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (i : OmegaIndex) : Set Model :=
  BitIsZero OmegaIndex Model (valOmega Sat Cut Bit omega0) i

/-! ## 4. Structural Family and Subalgebra -/

/--
The structural family `F_state` for Omega.
It includes all `BitAtOmega` predicates.
-/
def F_state_Omega
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : CutIndex → CodeOmega → Sentence)
    (Bit : BitIndex → CodeOmega → Sentence)
    (omega0 : CodeOmega) : Set (Set Model) :=
  { S | ∃ i : OmegaIndex, S = BitAtOmega Sat Cut Bit omega0 i }

/-! ## 5. Model Equivalence Relations -/

/-- The set of cut indices in Omega. -/
def J_cut : Set OmegaIndex := { i | ∃ (q : CutIndex), i = OmegaIndex.cutIdx q }

/-- The set of bit indices in Omega. -/
def J_bit : Set OmegaIndex := { i | ∃ (b : BitIndex), i = OmegaIndex.bitIdx b }

/-- Cut-equivalence: two models are cut-equivalent if they agree on all cut indices. -/
def CutEquivalent
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : CutIndex → CodeOmega → Sentence)
    (Bit : BitIndex → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (M₁ M₂ : Model) : Prop :=
  JEquivalent (valOmega Sat Cut Bit omega0) J_cut M₁ M₂

/-- Bit-equivalence: two models are bit-equivalent if they agree on all bit indices. -/
def BitEquivalent
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : CutIndex → CodeOmega → Sentence)
    (Bit : BitIndex → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (M₁ M₂ : Model) : Prop :=
  JEquivalent (valOmega Sat Cut Bit omega0) J_bit M₁ M₂

/-! ## 6. Dependency Theorems -/

/-- Sets in F_state depend only on the Omega valuation. -/
theorem F_state_depends_on_val
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : CutIndex → CodeOmega → Sentence)
    (Bit : BitIndex → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (S : Set Model)
    (hS : S ∈ F_state_Omega Sat Cut Bit omega0) :
    DependsOnlyOn (valOmega Sat Cut Bit omega0) Set.univ S := by
  rcases hS with ⟨i, rfl⟩
  intro M₁ M₂ h
  simp only [BitAt, Set.mem_setOf_eq]
  constructor
  · intro h₁
    rw [← h i (Set.mem_univ i)]
    exact h₁
  · intro h₂
    rw [h i (Set.mem_univ i)]
    exact h₂

end Definitions

end OmegaInstance
end Boole
end LogicDissoc

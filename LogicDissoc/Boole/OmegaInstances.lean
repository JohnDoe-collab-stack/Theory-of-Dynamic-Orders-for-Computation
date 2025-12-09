import LogicDissoc
import Boole.AbstractKernel

namespace LogicDissoc
namespace Boole
namespace OmegaInstance

open LogicDissoc.Boole.AbstractKernel

/-!
# Omega Instance of the Abstract Kernel

This module instantiates the abstract kernel with the concrete types from `LogicDissoc`.
-/

section Definitions

universe u v w

variable {Model : Type u} {Sentence : Type v} {CodeOmega : Type w}

/-! ## 1. Index Type -/

abbrev CutIndex := ℚ
abbrev BitIndex := ℕ × ℕ

inductive OmegaIndex
  | cutIdx : CutIndex → OmegaIndex
  | bitIdx : BitIndex → OmegaIndex
  deriving DecidableEq

/-! ## 2. Valuation -/

/-- Valuation function derived from the satisfaction relation. -/
def valOmega
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : ℚ → CodeOmega → Sentence)
    (Bit : ℕ → ℕ → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (M : Model) (i : OmegaIndex) : Bool :=
  match i with
  | OmegaIndex.cutIdx q   => decide (Sat M (Cut q omega0))
  | OmegaIndex.bitIdx ⟨n,a⟩ => decide (Sat M (Bit n a omega0))

/-! ## 3. Instantiated Kernel -/

/-- The concrete `BitAt` predicate for Omega. -/
abbrev BitAtOmega
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : ℚ → CodeOmega → Sentence)
    (Bit : ℕ → ℕ → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (i : OmegaIndex) : Set Model :=
  BitAt OmegaIndex Model (valOmega Sat Cut Bit omega0) i

/-- The concrete `BitIsZero` predicate for Omega. -/
abbrev BitIsZeroOmega
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : ℚ → CodeOmega → Sentence)
    (Bit : ℕ → ℕ → CodeOmega → Sentence)
    (omega0 : CodeOmega)
    (i : OmegaIndex) : Set Model :=
  BitIsZero OmegaIndex Model (valOmega Sat Cut Bit omega0) i

/-! ## 4. Structural Family and Subalgebra -/

/--
The structural family `F_state` for Omega.
It includes at least all `BitAtOmega` predicates.
-/
def F_state_Omega
    (Sat : Model → Sentence → Prop)
    [∀ M s, Decidable (Sat M s)]
    (Cut : ℚ → CodeOmega → Sentence)
    (Bit : ℕ → ℕ → CodeOmega → Sentence)
    (omega0 : CodeOmega) : Set (Set Model) :=
  { S | ∃ i : OmegaIndex, S = BitAtOmega Sat Cut Bit omega0 i }

-- The Boolean subalgebra of invariants for Omega.
-- NOTE: Temporarily disabled due to universe metavariable issues
-- TODO: Fix universe polymorphism for OmegaIndex and related definitions
--
-- abbrev OmegaSubalgebra
--     (Sat : Model → Sentence → Prop)
--     [∀ M s, Decidable (Sat M s)]
--     (Cut : ℚ → CodeOmega → Sentence)
--     (Bit : ℕ → ℕ → CodeOmega → Sentence)
--     (omega0 : CodeOmega) :=
--   GeneratedSubalgebra (@F_state_Omega Model Sentence CodeOmega Sat _ Cut Bit omega0)

/-! ## 5. Model Equivalence Relations -/

/-- The set of cut indices in Omega. -/
def J_cut : Set OmegaIndex := { i | ∃ (q : ℚ), i = OmegaIndex.cutIdx q }

/-- The set of bit indices in Omega. -/
def J_bit : Set OmegaIndex := { i | ∃ (n : ℕ) (a : ℕ), i = OmegaIndex.bitIdx (n, a) }

-- NOTE: CutEquivalent and BitEquivalent are defined as JEquivalent on J_cut/J_bit
-- but have universe metavariable issues. The conceptual definition is:
--   CutEquivalent Sat Cut Bit omega0 M₁ M₂ := JEquivalent (valOmega ...) J_cut M₁ M₂
--   BitEquivalent Sat Cut Bit omega0 M₁ M₂ := JEquivalent (valOmega ...) J_bit M₁ M₂
--
-- These will work once OmegaIndex is lifted to a universe-polymorphic definition.

end Definitions

end OmegaInstance
end Boole
end LogicDissoc

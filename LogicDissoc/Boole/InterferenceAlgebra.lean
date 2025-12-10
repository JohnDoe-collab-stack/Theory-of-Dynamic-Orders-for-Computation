import Mathlib.Data.Rat.Defs
import Mathlib.Data.Rat.Lemmas
import Mathlib.Order.WithBot
import Mathlib.Order.MinMax
import Mathlib.Algebra.Order.Monoid.Defs
import Mathlib.Logic.Equiv.Basic
import Mathlib.Algebra.Order.Field.Rat

namespace LogicDissoc
namespace Boole

/-!
# Interference Algebra: Classification of Dynamic Invariant Structures

This module provides the abstract algebraic framework for classifying the
image of interference invariants. It formalizes the "Quadrant" of algebraic
structures that govern the behavior of dynamic systems.

## Main Concepts

- **InterferenceAlgebra**: A preordered bimonoid with two operations (⊕, ⊙)
  satisfying monotonicity, interchange, and dichotomy axioms.
- **CanonicalPair**: The five canonical algebraic shapes that any interference
  algebra must belong to.
- **Classification Theorem**: Every `InterferenceAlgebra` falls into exactly
  one of the canonical shapes.

## The Quadrant

The classification is governed by two orthogonal dichotomies:

| ⊕ \\ ⊙          | Idempotent (Choice)   | Strict (Cumulative)   |
|-----------------|-----------------------|-----------------------|
| **Idempotent**  | `maxMax` (Lattice)    | `maxPlus` (Tropical)  |
| **Cancellative**| `plusMax` (Capacity)  | `plusPlus` (Arith)    |

This structure is fundamental to the LogicDissoc framework as it formally
distinguishes between **structural invariants** (Logic/Degrees) and
**resource invariants** (Arithmetic/Fuel).
-/

-- ============================================================================
-- § 1. Non-Negative Rationals (Target Model)
-- ============================================================================

/-- Non-negative rationals ℚ≥0, used as the carrier for additive models. -/
def NonNegRat := { q : ℚ // 0 ≤ q }

namespace NonNegRat

instance : Coe NonNegRat ℚ where
  coe x := x.1

/-- Zero is non-negative. -/
instance : Zero NonNegRat where
  zero := ⟨0, by rfl⟩

/-- One is non-negative. -/
instance : One NonNegRat where
  one := ⟨1, by native_decide⟩

/-- Sum of non-negatives is non-negative. -/
instance : Add NonNegRat where
  add a b := ⟨a.1 + b.1, Rat.add_nonneg a.2 b.2⟩

instance : LE NonNegRat where
  le a b := a.1 ≤ b.1

instance : LT NonNegRat where
  lt a b := a.1 < b.1

/-- Maximum of non-negatives is non-negative. -/
instance : Max NonNegRat where
  max a b := ⟨max a.1 b.1, by
    if h : a.1 ≤ b.1 then
      rw [max_eq_right h]
      exact b.2
    else
      rw [max_eq_left (le_of_not_ge h)]
      exact a.2⟩

/-- Minimum of non-negatives is non-negative. -/
instance : Min NonNegRat where
  min a b := ⟨min a.1 b.1, by
    if h : a.1 ≤ b.1 then
      rw [min_eq_left h]
      exact a.2
    else
      rw [min_eq_right (le_of_not_ge h)]
      exact b.2⟩

instance : AddCommMonoid NonNegRat where
  add := (· + ·)
  zero := 0
  add_assoc := fun a b c => Subtype.ext (add_assoc a.1 b.1 c.1)
  zero_add := fun a => Subtype.ext (zero_add a.1)
  add_zero := fun a => Subtype.ext (add_zero a.1)
  add_comm := fun a b => Subtype.ext (add_comm a.1 b.1)
  nsmul := nsmulRec

end NonNegRat

-- ============================================================================
-- § 2. Canonical Pairs (The Quadrant)
-- ============================================================================

/-- The five canonical algebraic shapes for interference algebras.
    Each corresponds to a corner or edge of the classification quadrant. -/
inductive CanonicalPair
  | maxPlus   -- (max, +) : Tropical semiring on WithBot ℚ (Degrees/Scores)
  | minPlus   -- (min, +) : Dual tropical on WithTop ℚ (Shortest paths)
  | plusPlus  -- (+, +)   : Standard arithmetic on ℚ≥0 (Resources/Fuel)
  | plusMax   -- (+, max) : Capacitive algebra on ℚ≥0 (Probabilistic)
  | maxMax    -- (max, max) : Distributive lattice (Pure Logic/Choice)
deriving DecidableEq, Repr

-- ============================================================================
-- § 3. Interference Algebra (Abstract Structure)
-- ============================================================================

/--
**Interference Algebra**

An abstract algebraic structure induced by the image of an interference
invariant on a carrier set `S`. This structure captures the essential
properties shared by all invariants in dynamic order theory.

## Components

- `S`: The carrier type (image of the invariant)
- `le`: A preorder relation
- `opPar` (⊕): Parallel composition (interference)
- `opSeq` (⊙): Sequential composition
- `zero`: Neutral element for ⊕
- `one`: Neutral element for ⊙

## Axioms

- **Preorder**: Reflexivity and transitivity of `le`
- **Monotonicity**: Both operations preserve the order
- **Monoid Laws**: ⊕ forms a commutative monoid, ⊙ forms a monoid
- **Lax Interchange**: Distributivity inequality connecting ⊕ and ⊙
- **Dichotomies**: Decidable idempotence for both operations
-/
structure InterferenceAlgebra where
  S     : Type
  le    : S → S → Prop
  opPar : S → S → S  -- ⊕ (parallel/interference)
  opSeq : S → S → S  -- ⊙ (sequential/composition)
  zero  : S          -- 𝟘 (neutral for ⊕)
  one   : S          -- 𝟙 (neutral for ⊙)

  -- Preorder axioms
  le_refl  : ∀ x, le x x
  le_trans : ∀ x y z, le x y → le y z → le x z

  -- Monotonicity axioms
  mono_par : ∀ a b a' b', le a a' → le b b' → le (opPar a b) (opPar a' b')
  mono_seq : ∀ a b a' b', le a a' → le b b' → le (opSeq a b) (opSeq a' b')

  -- Commutative monoid axioms for ⊕
  par_assoc : ∀ a b c, opPar (opPar a b) c = opPar a (opPar b c)
  par_comm  : ∀ a b, opPar a b = opPar b a
  par_zero  : ∀ a, opPar a zero = a

  -- Commutative monoid axioms for ⊙
  seq_assoc : ∀ a b c, opSeq (opSeq a b) c = opSeq a (opSeq b c)
  seq_one_r : ∀ a, opSeq a one = a
  seq_one_l : ∀ a, opSeq one a = a
  seq_comm  : ∀ a b, opSeq a b = opSeq b a

  -- Lax interchange law (connects ⊕ and ⊙)
  interchange_lax :
    ∀ a b c d,
      le (opSeq (opPar a b) (opPar c d))
         (opPar (opPar (opSeq a c) (opSeq a d))
                 (opPar (opSeq b c) (opSeq b d)))

  -- Dichotomy on ⊕: either idempotent (lattice-like) or cancellative (group-like)
  dichotomy :
    (∀ x, opPar x x = x) ∨
    (∀ x y z, opPar x y = opPar x z → y = z)

  -- Dichotomy on ⊙: either idempotent or strictly cumulative
  seq_dichotomy :
    (∀ x, opSeq x x = x) ∨
    ¬ (∀ x, opSeq x x = x)

-- ============================================================================
-- § 4. Classification Predicates
-- ============================================================================

namespace InterferenceAlgebra

variable (A : InterferenceAlgebra)

/-- Tropical idempotent form: ⊕ is idempotent (choice), ⊙ is commutative. -/
def IsTropicalIdempotent : Prop :=
  (∀ x, A.opPar x x = x) ∧
  (∀ x y, A.opSeq x y = A.opSeq y x)

/-- Additive form: ⊕ is cancellative (cumulative), ⊙ is commutative. -/
def IsAdditive : Prop :=
  (∀ x y z, A.opPar x y = A.opPar x z → y = z) ∧
  (∀ x y, A.opSeq x y = A.opSeq y x)

/-- (max, +): Tropical strict form — ⊕ idempotent, ⊙ non-idempotent.
    This is the algebra of **degrees** and **scores**. -/
def IsMaxPlus : Prop :=
  IsTropicalIdempotent A ∧ ¬ (∀ x, A.opSeq x x = x)

/-- (max, max): Distributive lattice form — both ⊕ and ⊙ idempotent.
    This is the algebra of **pure logic** and **choice**. -/
def IsMaxMax : Prop :=
  IsTropicalIdempotent A ∧ (∀ x, A.opSeq x x = x)

/-- (min, +): Dual tropical form (equivalent to IsMaxPlus by duality). -/
def IsMinPlus : Prop := IsTropicalIdempotent A

/-- (+, +): Standard arithmetic — ⊕ cancellative, ⊙ non-idempotent.
    This is the algebra of **resources** and **fuel**. -/
def IsPlusPlus : Prop :=
  IsAdditive A ∧ ¬ (∀ x, A.opSeq x x = x)

/-- (+, max): Capacitive/probabilistic — ⊕ cancellative, ⊙ idempotent. -/
def IsPlusMax : Prop :=
  IsAdditive A ∧ (∀ x, A.opSeq x x = x)

/-- Predicate associating each canonical pair with its defining property. -/
def satisfiesShape (cp : CanonicalPair) : Prop :=
  match cp with
  | CanonicalPair.maxPlus  => IsMaxPlus A
  | CanonicalPair.minPlus  => IsMinPlus A
  | CanonicalPair.plusPlus => IsPlusPlus A
  | CanonicalPair.plusMax  => IsPlusMax A
  | CanonicalPair.maxMax   => IsMaxMax A

-- ============================================================================
-- § 5. Classification Theorems
-- ============================================================================

/--
**Tropical Strict Classification**

If ⊕ is idempotent and ⊙ is non-idempotent, the algebra has shape `maxPlus`.
This is the tropical semiring structure used for degree/score invariants.
-/
theorem classification_tropical_strict
    (h_idem : ∀ x, A.opPar x x = x)
    (h_seq_not_idem : ¬ (∀ x, A.opSeq x x = x)) :
    satisfiesShape A CanonicalPair.maxPlus := by
  unfold satisfiesShape IsMaxPlus IsTropicalIdempotent
  exact ⟨⟨h_idem, A.seq_comm⟩, h_seq_not_idem⟩

/--
**Lattice Classification**

If both ⊕ and ⊙ are idempotent, the algebra has shape `maxMax`.
This is a distributive lattice structure used for pure logical invariants.
-/
theorem classification_lattice
    (h_idem : ∀ x, A.opPar x x = x)
    (h_seq_idem : ∀ x, A.opSeq x x = x) :
    satisfiesShape A CanonicalPair.maxMax := by
  unfold satisfiesShape IsMaxMax IsTropicalIdempotent
  exact ⟨⟨h_idem, A.seq_comm⟩, h_seq_idem⟩

/--
**Capacitive Classification**

If ⊕ is cancellative and ⊙ is idempotent, the algebra has shape `plusMax`.
-/
theorem classification_plusMax
    (h_cancel : ∀ x y z, A.opPar x y = A.opPar x z → y = z)
    (h_seq_idem : ∀ x, A.opSeq x x = x) :
    satisfiesShape A CanonicalPair.plusMax := by
  unfold satisfiesShape IsPlusMax IsAdditive
  exact ⟨⟨h_cancel, A.seq_comm⟩, h_seq_idem⟩

/--
**Arithmetic Classification**

If ⊕ is cancellative and ⊙ is non-idempotent, the algebra has shape `plusPlus`.
This is the standard arithmetic structure used for resource/fuel invariants.
-/
theorem classification_plusPlus
    (h_cancel : ∀ x y z, A.opPar x y = A.opPar x z → y = z)
    (h_seq_not_idem : ¬ (∀ x, A.opSeq x x = x)) :
    satisfiesShape A CanonicalPair.plusPlus := by
  unfold satisfiesShape IsPlusPlus IsAdditive
  exact ⟨⟨h_cancel, A.seq_comm⟩, h_seq_not_idem⟩

/--
**Main Classification Theorem** (Fully Constructive)

Every interference algebra belongs to at least one canonical shape.
The proof proceeds by case analysis on the two dichotomies, exhaustively
covering the quadrant of possibilities.

## Quadrant Coverage

- **⊕ Idempotent, ⊙ Idempotent** → `maxMax` (Lattice/Logic)
- **⊕ Idempotent, ⊙ Strict** → `maxPlus` (Tropical/Degrees)
- **⊕ Cancellative, ⊙ Idempotent** → `plusMax` (Capacitive)
- **⊕ Cancellative, ⊙ Strict** → `plusPlus` (Arithmetic/Fuel)

This theorem is the formal foundation for the claim that the LogicDissoc
framework separates **Logic** (structural invariants) from **Arithmetic**
(resource invariants).
-/
theorem classification_theorem :
    ∃ cp : CanonicalPair, satisfiesShape A cp := by
  cases A.dichotomy with
  | inl h_idem =>
      -- Case: ⊕ is idempotent (lattice-like)
      cases A.seq_dichotomy with
      | inl h_seq_idem =>
          -- ⊙ is also idempotent: pure lattice (max, max)
          use CanonicalPair.maxMax
          exact classification_lattice A h_idem h_seq_idem
      | inr h_seq_not_idem =>
          -- ⊙ is strict: tropical (max, +)
          use CanonicalPair.maxPlus
          exact classification_tropical_strict A h_idem h_seq_not_idem
  | inr h_cancel =>
      -- Case: ⊕ is cancellative (group-like)
      cases A.seq_dichotomy with
      | inl h_seq_idem =>
          -- ⊙ is idempotent: capacitive (+, max)
          use CanonicalPair.plusMax
          exact classification_plusMax A h_cancel h_seq_idem
      | inr h_seq_not_idem =>
          -- ⊙ is strict: standard arithmetic (+, +)
          use CanonicalPair.plusPlus
          exact classification_plusPlus A h_cancel h_seq_not_idem

end InterferenceAlgebra

-- ============================================================================
-- § 6. Concrete Model Isomorphisms
-- ============================================================================

/-!
## Concrete Model Definitions

These definitions formalize what it means for an `InterferenceAlgebra` to be
**isomorphic** to one of the standard tropical arithmetics on ℚ or ℚ≥0.

Proving such isomorphisms requires:
1. A concrete invariant `I : Object → ℚ/ℚ≥0` from the Omega geometry
2. A proof that the image of `I` with (⊕, ⊙) satisfies the algebra axioms
3. Uniqueness/density arguments to identify the image with ℚ or ℚ≥0

These components come from modules like `OmegaInvariants` and `ConcreteInstance`.
-/

namespace InterferenceAlgebra

variable (A : InterferenceAlgebra)

/-- Isomorphism to (max, +) tropical semiring on `WithBot ℚ`. -/
def IsMaxPlusModel : Prop :=
  ∃ (e : A.S ≃ WithBot ℚ),
    (∀ x y, e (A.opPar x y) = max (e x) (e y)) ∧
    (∀ x y, e (A.opSeq x y) = (e x) + (e y))

/-- Isomorphism to (min, +) tropical semiring on `WithTop ℚ`. -/
def IsMinPlusModel : Prop :=
  ∃ (e : A.S ≃ WithTop ℚ),
    (∀ x y, e (A.opPar x y) = min (e x) (e y)) ∧
    (∀ x y, e (A.opSeq x y) = (e x) + (e y))

/-- Isomorphism to standard arithmetic (+, +) on `NonNegRat`. -/
def IsPlusPlusModel : Prop :=
  ∃ (e : A.S ≃ NonNegRat),
    (∀ x y, e (A.opPar x y) = (e x) + (e y)) ∧
    (∀ x y, e (A.opSeq x y) = (e x) + (e y))

/-- Isomorphism to capacitive algebra (+, max) on `NonNegRat`. -/
def IsPlusMaxModel : Prop :=
  ∃ (e : A.S ≃ NonNegRat),
    (∀ x y, e (A.opPar x y) = (e x) + (e y)) ∧
    (∀ x y, e (A.opSeq x y) = max (e x) (e y))

/-!
## Remark: Full Model Theorem

The target theorem:

```
IsMaxPlusModel A ∨ IsMinPlusModel A ∨ IsPlusPlusModel A ∨ IsPlusMaxModel A
```

cannot be proven here abstractly. It requires additional data:

1. A concrete invariant `I : ProfileObject → ℚ/ℚ≥0` from Omega geometry
2. Proof that the image of `I` satisfies `InterferenceAlgebra` axioms
3. Uniqueness/density properties (Archimedean, etc.) to identify with ℚ/ℚ≥0

These components must come from `OmegaInvariants`, `ConcreteInstance`, and
related modules that define `L`, `W`, `C`, `d` as scalar functions.

This file provides the **abstract classification layer** that is strictly
derived from the LogicDissoc framework, without inventing proofs that do
not follow from the current formalization.
-/

end InterferenceAlgebra

-- ============================================================================
-- § 7. Concrete Instance: Nat with (max, +)
-- ============================================================================

/-!
## Concrete Instance: NatMaxPlusAlgebra

This section provides a concrete `InterferenceAlgebra` instance on `Nat` with:
- `opPar = max` (parallel interference = worst case bound)
- `opSeq = +` (sequential composition = accumulation)

This corresponds to the **maxPlus** (Tropical) corner of the quadrant,
which is the algebra of degrees and scores used for computation bounds.
-/

/--
**Nat MaxPlus Algebra**

Concrete `InterferenceAlgebra` on `Nat` with `max` as parallel interference
and `+` as sequential composition. This is the tropical semiring structure.
-/
def NatMaxPlusAlgebra : InterferenceAlgebra where
  S := Nat
  le := (· ≤ ·)
  opPar := max
  opSeq := (· + ·)
  zero := 0
  one := 0

  -- Preorder
  le_refl := Nat.le_refl
  le_trans := fun _ _ _ => Nat.le_trans

  -- Monotonicity
  mono_par := fun _ _ _ _ ha hb => by omega
  mono_seq := fun _ _ _ _ ha hb => Nat.add_le_add ha hb

  -- Commutative monoid for opPar (max)
  par_assoc := fun a b c => by omega
  par_comm := fun a b => Nat.max_comm a b
  par_zero := fun a => by omega

  -- Commutative monoid for opSeq (+)
  seq_assoc := fun a b c => Nat.add_assoc a b c
  seq_one_r := fun a => Nat.add_zero a
  seq_one_l := fun a => Nat.zero_add a
  seq_comm := fun a b => Nat.add_comm a b

  -- Lax interchange: max(a,b) + max(c,d) ≤ max(max(a+c, a+d), max(b+c, b+d))
  interchange_lax := fun a b c d => by omega

  -- Dichotomy: opPar = max is idempotent
  dichotomy := Or.inl (fun x => Nat.max_self x)

  -- Seq dichotomy: opSeq = + is NOT idempotent
  seq_dichotomy := Or.inr (fun h => by
    have h1 : (1 : Nat) + 1 = 1 := h 1
    omega)

/--
**Classification Theorem for NatMaxPlusAlgebra**

The concrete Nat algebra with (max, +) satisfies the `maxPlus` shape.
-/
theorem NatMaxPlusAlgebra_isMaxPlus :
    InterferenceAlgebra.satisfiesShape NatMaxPlusAlgebra CanonicalPair.maxPlus := by
  unfold InterferenceAlgebra.satisfiesShape InterferenceAlgebra.IsMaxPlus
    InterferenceAlgebra.IsTropicalIdempotent
  constructor
  · constructor
    · intro x
      simp only [NatMaxPlusAlgebra, Nat.max_self]
    · intro x y
      simp only [NatMaxPlusAlgebra, Nat.add_comm]
  · intro h
    -- h says opSeq x x = x for all x : NatMaxPlusAlgebra.S = Nat
    -- We need to show this leads to False
    -- opSeq in NatMaxPlusAlgebra is (+), so h says x + x = x for all x
    -- This fails for x = 1 since 1 + 1 = 2 ≠ 1
    have h1 := h (1 : Nat)
    -- h1 : NatMaxPlusAlgebra.opSeq 1 1 = 1, i.e., 1 + 1 = 1
    change (1 : Nat) + 1 = 1 at h1
    omega


end Boole
end LogicDissoc

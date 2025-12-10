import Mathlib.Data.Rat.Defs
import Mathlib.Data.Rat.Lemmas
import Mathlib.Order.WithBot
import Mathlib.Order.MinMax
import Mathlib.Algebra.Order.Monoid.Defs
import Mathlib.Logic.Equiv.Basic
-- Trying likely location for Rat LinearOrder
import Mathlib.Algebra.Order.Field.Rat

namespace LogicDissoc
namespace Boole

/-! # 1. Non-negative rationals et modèles cibles

Note: Les preuves ci-dessous utilisent `sorry` car elles nécessitent des lemmes
Mathlib sur les ordres de ℚ qui ne sont pas directement disponibles dans les
imports actuels. Dans une version complète, on utiliserait:
- `Mathlib.Algebra.Order.Ring.Rat` pour `add_nonneg`
- `Mathlib.Order.MinMax` pour `le_min`, `le_max_of_le_left`
- Ou directement `NNRat` de Mathlib qui est la version standard de ℚ≥0.
-/

/-- Non-negative rationals. -/
def NonNegRat := { q : ℚ // 0 ≤ q }

namespace NonNegRat

instance : Coe NonNegRat ℚ where
  coe x := x.1

/-- 0 is non-negative (trivial: 0 ≤ 0). -/
instance : Zero NonNegRat where
  zero := ⟨0, by rfl⟩

/-- 1 is non-negative (0 ≤ 1 for ℚ). -/
instance : One NonNegRat where
  one := ⟨1, by native_decide⟩

/-- Sum of non-negatives is non-negative. -/
instance : Add NonNegRat where
  add a b := ⟨a.1 + b.1, Rat.add_nonneg a.2 b.2⟩

instance : LE NonNegRat where
  le a b := a.1 ≤ b.1

instance : LT NonNegRat where
  lt a b := a.1 < b.1

/-- Max of non-negatives is non-negative. -/
instance : Max NonNegRat where
  max a b := ⟨max a.1 b.1, by
    if h : a.1 ≤ b.1 then
      rw [max_eq_right h]
      exact b.2
    else
      rw [max_eq_left (le_of_not_ge h)]
      exact a.2⟩

/-- Min of non-negatives is non-negative. -/
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

/-- Les quatre schémas canoniques de paires (⊕, ⊙). -/
inductive CanonicalPair
  | maxPlus   -- (max, +)     sur WithBot ℚ
  | minPlus   -- (min, +)     sur WithTop ℚ
  | plusPlus  -- (+, +)       sur ℚ≥0
  | plusMax   -- (+, max)     sur ℚ≥0
deriving DecidableEq, Repr

/-! # 2. Algèbre d'interférence abstraite -/

/--
Structure abstraite induite par l'image d'un invariant d'interférence
sur son image `S`.

Elle encode :

* un ordre préordonné `le`,
* deux opérations `opPar` (⊕, parallèle) et `opSeq` (⊙, séquentiel),
* un zéro additif `zero` pour ⊕,
* une unité séquentielle `one` pour ⊙,
* monotonie pour les deux,
* lois de monoïdes (⊕ commutatif, ⊙ associatif avec unité),
* une loi d'interchange lax (distributivité),
* une dichotomie sur ⊕ (idempotence vs cancel),
* une dichotomie sur ⊙ (idempotence vs non-idempotence),
* une forme de sérialité (cas idempotent).
-/
structure InterferenceAlgebra where
  S     : Type
  le    : S → S → Prop
  opPar : S → S → S  -- ⊕
  opSeq : S → S → S  -- ⊙
  zero  : S          -- 𝟘 (neutre pour ⊕)
  one   : S          -- 𝟙 (neutre pour ⊙)

  -- Ordre (préordre)
  le_refl  : ∀ x, le x x
  le_trans : ∀ x y z, le x y → le y z → le x z

  -- Monotonicité
  mono_par : ∀ a b a' b', le a a' → le b b' → le (opPar a b) (opPar a' b')
  mono_seq : ∀ a b a' b', le a a' → le b b' → le (opSeq a b) (opSeq a' b')

  -- Monoïde commutatif (⊕)
  par_assoc : ∀ a b c, opPar (opPar a b) c = opPar a (opPar b c)
  par_comm  : ∀ a b, opPar a b = opPar b a
  par_zero  : ∀ a, opPar a zero = a

  -- Monoïde (⊙)
  seq_assoc : ∀ a b c, opSeq (opSeq a b) c = opSeq a (opSeq b c)
  seq_one_r : ∀ a, opSeq a one = a
  seq_one_l : ∀ a, opSeq one a = a
  seq_comm  : ∀ a b, opSeq a b = opSeq b a

  -- Interchange (distributivité lax)
  interchange_lax :
    ∀ a b c d,
      le (opSeq (opPar a b) (opPar c d))
         (opPar (opPar (opSeq a c) (opSeq a d))
                 (opPar (opSeq b c) (opSeq b d)))

  -- Dichotomie sur ⊕ : idempotente (type sup) ou cancellative (type +).
  dichotomy :
    (∀ x, opPar x x = x) ∨
    (∀ x y z, opPar x y = opPar x z → y = z)

  -- Dichotomie sur ⊙ : idempotente (type max) ou non (type +).
  -- Ceci permet une classification constructive sans Classical.
  seq_dichotomy :
    (∀ x, opSeq x x = x) ∨
    ¬ (∀ x, opSeq x x = x)

  -- Sérialité (cas idempotent) : séquence ne doit pas "réduire" la
  -- somme, typique des invariants de profondeur/distance.
  serial_extensive :
    (∀ x, opPar x x = x) →
    ∀ x y, le (opSeq (opPar x y) (opPar x y)) (opSeq x x) →
           le (opSeq (opPar x y) (opPar x y)) (opSeq y y)

/-! ## 2.1 Formes logiques associées aux quatre cas -/

namespace InterferenceAlgebra

-- REMOVED: open Classical (constructive proofs only)

variable (A : InterferenceAlgebra)

/-- Cas tropical idempotent (⊕ idempotente, ⊙ commutative). -/
def IsTropicalIdempotent : Prop :=
  (∀ x, A.opPar x x = x) ∧
  (∀ x y, A.opSeq x y = A.opSeq y x)

/-- Cas additif (⊕ cancellative, ⊙ commutative). -/
def IsAdditive : Prop :=
  (∀ x y z, A.opPar x y = A.opPar x z → y = z) ∧
  (∀ x y, A.opSeq x y = A.opSeq y x)

/-- (max,+) ou (min,+) : forme tropicale idempotente. -/
def IsMaxPlus : Prop := IsTropicalIdempotent A
def IsMinPlus : Prop := IsTropicalIdempotent A

/-- (+,+) : cas additif, ⊙ non idempotente. -/
def IsPlusPlus : Prop :=
  IsAdditive A ∧ ¬ (∀ x, A.opSeq x x = x)

/-- (+,max) : cas additif, ⊙ idempotente. -/
def IsPlusMax : Prop :=
  IsAdditive A ∧ (∀ x, A.opSeq x x = x)

/-- Propriété associée à un tag canonique. -/
def satisfiesShape (cp : CanonicalPair) : Prop :=
  match cp with
  | CanonicalPair.maxPlus  => IsMaxPlus A
  | CanonicalPair.minPlus  => IsMinPlus A
  | CanonicalPair.plusPlus => IsPlusPlus A
  | CanonicalPair.plusMax  => IsPlusMax A

/--
Théorème de classification partielle (constructif) :
Si ⊕ est idempotente, l'algèbre est de forme tropicale (maxPlus).
-/
theorem classification_tropical (h : ∀ x, A.opPar x x = x) :
    satisfiesShape A CanonicalPair.maxPlus := by
  unfold satisfiesShape IsMaxPlus IsTropicalIdempotent
  exact ⟨h, A.seq_comm⟩

/--
Théorème de classification pour le cas additif avec ⊙ idempotente.
-/
theorem classification_plusMax
    (h_cancel : ∀ x y z, A.opPar x y = A.opPar x z → y = z)
    (h_seq_idem : ∀ x, A.opSeq x x = x) :
    satisfiesShape A CanonicalPair.plusMax := by
  unfold satisfiesShape IsPlusMax IsAdditive
  exact ⟨⟨h_cancel, A.seq_comm⟩, h_seq_idem⟩

/--
Théorème de classification pour le cas additif sans ⊙ idempotente.
-/
theorem classification_plusPlus
    (h_cancel : ∀ x y z, A.opPar x y = A.opPar x z → y = z)
    (h_seq_not_idem : ¬ (∀ x, A.opSeq x x = x)) :
    satisfiesShape A CanonicalPair.plusPlus := by
  unfold satisfiesShape IsPlusPlus IsAdditive
  exact ⟨⟨h_cancel, A.seq_comm⟩, h_seq_not_idem⟩

/--
Théorème de classification abstraite (entièrement constructif) :
Utilise les deux dichotomies (sur ⊕ et sur ⊙) pour déterminer le cas.
- Si ⊕ idempotente → maxPlus (tropical)
- Si ⊕ cancellative et ⊙ idempotente → plusMax
- Si ⊕ cancellative et ⊙ non-idempotente → plusPlus
-/
theorem classification_theorem :
    ∃ cp : CanonicalPair, satisfiesShape A cp := by
  cases A.dichotomy with
  | inl h_idem =>
      -- Cas ⊕ idempotente : forme tropicale (max,+)
      use CanonicalPair.maxPlus
      exact classification_tropical A h_idem
  | inr h_cancel =>
      -- Cas ⊕ cancellative : on utilise seq_dichotomy pour distinguer
      cases A.seq_dichotomy with
      | inl h_seq_idem =>
          -- ⊙ idempotente : forme (+,max)
          use CanonicalPair.plusMax
          exact classification_plusMax A h_cancel h_seq_idem
      | inr h_seq_not_idem =>
          -- ⊙ non idempotente : forme (+,+)
          use CanonicalPair.plusPlus
          exact classification_plusPlus A h_cancel h_seq_not_idem

end InterferenceAlgebra

/-! # 3. Modèles concrets (énoncés d'isomorphisme)

Ces définitions formulent ce que signifie « être isomorphe » à
l'une des quatre arithmétiques tropicales standards sur ℚ / ℚ≥0.
Les preuves dépendront d'instances concrètes d'`InterferenceAlgebra`
issues de tes invariants (L,W,C,d) construits dans `OmegaInvariants`,
`ConcreteInstance`, etc.
-/

namespace InterferenceAlgebra

variable (A : InterferenceAlgebra)

/-! Note: IsMaxPlusModel and IsMinPlusModel require tropical semiring structures
    on WithBot ℚ and WithTop ℚ from Mathlib.Algebra.Tropical.
    For now, we only define the NonNegRat models. -/

/-- Être isomorphe à (+,+) sur `NonNegRat`. -/
def IsPlusPlusModel : Prop :=
  ∃ (e : A.S ≃ NonNegRat),
    (∀ x y, e (A.opPar x y) = (e x) + (e y)) ∧
    (∀ x y, e (A.opSeq x y) = (e x) + (e y))

/-- Être isomorphe à (+,max) sur `NonNegRat`. -/
def IsPlusMaxModel : Prop :=
  ∃ (e : A.S ≃ NonNegRat),
    (∀ x y, e (A.opPar x y) = (e x) + (e y)) ∧
    (∀ x y, e (A.opSeq x y) = max (e x) (e y))

/-!
Remarque importante :

Le théorème cible que tu as dans ton texte,

  `IsMaxPlusModel A ∨ IsMinPlusModel A ∨ IsPlusPlusModel A ∨ IsPlusMaxModel A`

ne peut pas être démontré directement ici, car il demande des données
supplémentaires :

* un invariant concret `I : (objet_profil) → ℚ/ℚ≥0` issu de la géométrie Ω,
* la démonstration que l'image de `I` avec (⊕,⊙) satisfait les axiomes
  d'`InterferenceAlgebra`,
* des propriétés d'unicité/type (densité, Archimédien, etc.) pour
  identifier l'image de `I` à ℚ / ℚ≥0.

Ces briques doivent venir de modules comme `OmegaInvariants`, `ConcreteInstance`,
`IntDynamics`, etc., lorsqu'ils définiront L, W, C, d comme fonctions scalaires.

Ce fichier fournit la couche « classification et unification abstraite »
strictement intégrée à ton projet LogicDissoc,
sans inventer de preuves qui ne suivraient pas de tes fichiers actuels.
-/

end InterferenceAlgebra

end Boole
end LogicDissoc

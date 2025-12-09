"""
Ω_trig++ Kernel: Logical Formulas over Trigonometric Predicates

This module defines:
- Atomic predicates A1..A7 on angles
- Logical formulas φ with AND/OR/NOT
- 3-valued logic (TRUE/FALSE/UNKNOWN)
- Ground truth evaluation
"""

import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Callable, Union, Tuple
from abc import ABC, abstractmethod


# ============================================================================
# 1. Angle Representation
# ============================================================================

N_ANGLES = 360

@dataclass(frozen=True)
class Angle:
    """Angle represented as k/N * 2π."""
    k: int
    N: int = N_ANGLES
    
    def radians(self) -> float:
        return 2 * math.pi * self.k / self.N
    
    def sin(self) -> float:
        return math.sin(self.radians())
    
    def cos(self) -> float:
        return math.cos(self.radians())
    
    def __repr__(self):
        return f"θ={self.k}°"


# ============================================================================
# 2. Three-Valued Logic
# ============================================================================

class TVal(Enum):
    """Three-valued logic: True, False, Unknown."""
    TRUE = 1
    FALSE = 0
    UNKNOWN = -1
    
    def __and__(self, other: 'TVal') -> 'TVal':
        """AND: FALSE dominates, then UNKNOWN, then TRUE."""
        if self == TVal.FALSE or other == TVal.FALSE:
            return TVal.FALSE
        if self == TVal.UNKNOWN or other == TVal.UNKNOWN:
            return TVal.UNKNOWN
        return TVal.TRUE
    
    def __or__(self, other: 'TVal') -> 'TVal':
        """OR: TRUE dominates, then UNKNOWN, then FALSE."""
        if self == TVal.TRUE or other == TVal.TRUE:
            return TVal.TRUE
        if self == TVal.UNKNOWN or other == TVal.UNKNOWN:
            return TVal.UNKNOWN
        return TVal.FALSE
    
    def __invert__(self) -> 'TVal':
        """NOT: swap TRUE/FALSE, keep UNKNOWN."""
        if self == TVal.TRUE:
            return TVal.FALSE
        if self == TVal.FALSE:
            return TVal.TRUE
        return TVal.UNKNOWN
    
    def is_decided(self) -> bool:
        return self != TVal.UNKNOWN
    
    def to_bool(self) -> bool:
        """Convert to bool (only if decided)."""
        if self == TVal.UNKNOWN:
            raise ValueError("Cannot convert UNKNOWN to bool")
        return self == TVal.TRUE


# ============================================================================
# 3. Interval Arithmetic
# ============================================================================

@dataclass
class Interval:
    """Closed interval [lo, hi]."""
    lo: float
    hi: float
    
    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi
    
    def width(self) -> float:
        return self.hi - self.lo
    
    def all_ge(self, threshold: float) -> TVal:
        """Check if ALL values in interval are >= threshold."""
        if self.lo >= threshold:
            return TVal.TRUE
        if self.hi < threshold:
            return TVal.FALSE
        return TVal.UNKNOWN
    
    def all_lt(self, threshold: float) -> TVal:
        """Check if ALL values in interval are < threshold."""
        if self.hi < threshold:
            return TVal.TRUE
        if self.lo >= threshold:
            return TVal.FALSE
        return TVal.UNKNOWN
    
    def abs_all_ge(self, threshold: float) -> TVal:
        """Check if |x| >= threshold for all x in interval."""
        # min(|x|) over interval
        if self.lo >= 0:
            min_abs = self.lo
        elif self.hi <= 0:
            min_abs = -self.hi
        else:  # interval contains 0
            min_abs = 0
        
        # max(|x|) over interval
        max_abs = max(abs(self.lo), abs(self.hi))
        
        if min_abs >= threshold:
            return TVal.TRUE
        if max_abs < threshold:
            return TVal.FALSE
        return TVal.UNKNOWN


# ============================================================================
# 4. Atomic Predicates
# ============================================================================

class AtomicPredicate(ABC):
    """Base class for atomic predicates on angles."""
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def eval_exact(self, angle: Angle) -> bool:
        """Exact evaluation with true sin/cos."""
        pass
    
    @abstractmethod
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        """3-valued evaluation with interval approximations."""
        pass


class SinGe0(AtomicPredicate):
    """A1: sin(θ) >= 0"""
    def name(self) -> str:
        return "sin≥0"
    
    def eval_exact(self, angle: Angle) -> bool:
        return angle.sin() >= 0
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        return sin_I.all_ge(0)


class CosGe0(AtomicPredicate):
    """A2: cos(θ) >= 0"""
    def name(self) -> str:
        return "cos≥0"
    
    def eval_exact(self, angle: Angle) -> bool:
        return angle.cos() >= 0
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        return cos_I.all_ge(0)


class AbsSinGeAbsCos(AtomicPredicate):
    """A3: |sin(θ)| >= |cos(θ)|"""
    def name(self) -> str:
        return "|sin|≥|cos|"
    
    def eval_exact(self, angle: Angle) -> bool:
        return abs(angle.sin()) >= abs(angle.cos())
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        # This is more complex - approximate conservatively
        min_sin = min(abs(sin_I.lo), abs(sin_I.hi)) if sin_I.lo * sin_I.hi >= 0 else 0
        max_sin = max(abs(sin_I.lo), abs(sin_I.hi))
        min_cos = min(abs(cos_I.lo), abs(cos_I.hi)) if cos_I.lo * cos_I.hi >= 0 else 0
        max_cos = max(abs(cos_I.lo), abs(cos_I.hi))
        
        if min_sin >= max_cos:
            return TVal.TRUE
        if max_sin < min_cos:
            return TVal.FALSE
        return TVal.UNKNOWN


class AbsSinGeThreshold(AtomicPredicate):
    """A4: |sin(θ)| >= α (default α=0.5)"""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def name(self) -> str:
        return f"|sin|≥{self.alpha}"
    
    def eval_exact(self, angle: Angle) -> bool:
        return abs(angle.sin()) >= self.alpha
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        return sin_I.abs_all_ge(self.alpha)


class AbsCosGeThreshold(AtomicPredicate):
    """A5: |cos(θ)| >= β (default β=0.5)"""
    def __init__(self, beta: float = 0.5):
        self.beta = beta
    
    def name(self) -> str:
        return f"|cos|≥{self.beta}"
    
    def eval_exact(self, angle: Angle) -> bool:
        return abs(angle.cos()) >= self.beta
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        return cos_I.abs_all_ge(self.beta)


class SinGeThreshold(AtomicPredicate):
    """A6: sin(θ) >= γ"""
    def __init__(self, gamma: float):
        self.gamma = gamma
    
    def name(self) -> str:
        return f"sin≥{self.gamma}"
    
    def eval_exact(self, angle: Angle) -> bool:
        return angle.sin() >= self.gamma
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        return sin_I.all_ge(self.gamma)


class DiagProximity(AtomicPredicate):
    """A7: |sin(θ) - cos(θ)| <= δ (near diagonal)"""
    def __init__(self, delta: float = 0.2):
        self.delta = delta
    
    def name(self) -> str:
        return f"|sin-cos|≤{self.delta}"
    
    def eval_exact(self, angle: Angle) -> bool:
        return abs(angle.sin() - angle.cos()) <= self.delta
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        # diff in [sin.lo - cos.hi, sin.hi - cos.lo]
        diff_lo = sin_I.lo - cos_I.hi
        diff_hi = sin_I.hi - cos_I.lo
        
        # |diff| <= delta for all in interval?
        if diff_hi < 0:
            max_abs_diff = -diff_lo
            min_abs_diff = -diff_hi
        elif diff_lo > 0:
            max_abs_diff = diff_hi
            min_abs_diff = diff_lo
        else:  # interval contains 0
            max_abs_diff = max(-diff_lo, diff_hi)
            min_abs_diff = 0
        
        if max_abs_diff <= self.delta:
            return TVal.TRUE
        if min_abs_diff > self.delta:
            return TVal.FALSE
        return TVal.UNKNOWN


# ============================================================================
# 5. Standard Atoms
# ============================================================================

# Standard set of atoms
ATOMS = {
    "A1": SinGe0(),
    "A2": CosGe0(),
    "A3": AbsSinGeAbsCos(),
    "A4": AbsSinGeThreshold(0.5),
    "A5": AbsCosGeThreshold(0.5),
    "A6_0": SinGeThreshold(0.0),
    "A6_05": SinGeThreshold(0.5),
    "A6_08": SinGeThreshold(0.8),
    "A7": DiagProximity(0.2),
}


# ============================================================================
# 6. Logical Formulas
# ============================================================================

class Formula(ABC):
    """Abstract base for logical formulas."""
    
    @abstractmethod
    def eval_exact(self, angle: Angle) -> bool:
        pass
    
    @abstractmethod
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


class Atom(Formula):
    """Atomic formula wrapping a predicate."""
    def __init__(self, pred: AtomicPredicate):
        self.pred = pred
    
    def eval_exact(self, angle: Angle) -> bool:
        return self.pred.eval_exact(angle)
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        return self.pred.eval_interval(sin_I, cos_I)
    
    def __repr__(self):
        return self.pred.name()


class Not(Formula):
    """Negation."""
    def __init__(self, f: Formula):
        self.f = f
    
    def eval_exact(self, angle: Angle) -> bool:
        return not self.f.eval_exact(angle)
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        return ~self.f.eval_interval(sin_I, cos_I)
    
    def __repr__(self):
        return f"¬({self.f})"


class And(Formula):
    """Conjunction."""
    def __init__(self, *formulas: Formula):
        self.formulas = formulas
    
    def eval_exact(self, angle: Angle) -> bool:
        return all(f.eval_exact(angle) for f in self.formulas)
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        result = TVal.TRUE
        for f in self.formulas:
            result = result & f.eval_interval(sin_I, cos_I)
            if result == TVal.FALSE:
                return TVal.FALSE  # Short-circuit
        return result
    
    def __repr__(self):
        return "(" + " ∧ ".join(str(f) for f in self.formulas) + ")"


class Or(Formula):
    """Disjunction."""
    def __init__(self, *formulas: Formula):
        self.formulas = formulas
    
    def eval_exact(self, angle: Angle) -> bool:
        return any(f.eval_exact(angle) for f in self.formulas)
    
    def eval_interval(self, sin_I: Interval, cos_I: Interval) -> TVal:
        result = TVal.FALSE
        for f in self.formulas:
            result = result | f.eval_interval(sin_I, cos_I)
            if result == TVal.TRUE:
                return TVal.TRUE  # Short-circuit
        return result
    
    def __repr__(self):
        return "(" + " ∨ ".join(str(f) for f in self.formulas) + ")"


# ============================================================================
# 7. Formula Catalog (16 formulas)
# ============================================================================

# Shorthand atoms
A1 = Atom(ATOMS["A1"])  # sin >= 0
A2 = Atom(ATOMS["A2"])  # cos >= 0
A3 = Atom(ATOMS["A3"])  # |sin| >= |cos|
A4 = Atom(ATOMS["A4"])  # |sin| >= 0.5
A5 = Atom(ATOMS["A5"])  # |cos| >= 0.5
A6 = Atom(ATOMS["A6_05"])  # sin >= 0.5
A7 = Atom(ATOMS["A7"])  # |sin - cos| <= 0.2

FORMULAS: List[Tuple[int, Formula, str]] = [
    (0,  And(A1, A2),                    "Q1: sin≥0 ∧ cos≥0"),
    (1,  And(A1, Not(A2)),               "Q2: sin≥0 ∧ cos<0"),
    (2,  And(Not(A1), A2),               "Q3: sin<0 ∧ cos≥0"),
    (3,  And(Not(A1), Not(A2)),          "Q4: sin<0 ∧ cos<0"),
    (4,  A3,                             "A3: |sin|≥|cos|"),
    (5,  And(A1, A3),                    "sin≥0 ∧ |sin|≥|cos|"),
    (6,  And(Or(A1, A2), A4),            "(sin≥0 ∨ cos≥0) ∧ |sin|≥0.5"),
    (7,  Not(And(A1, A2)),               "¬(sin≥0 ∧ cos≥0)"),
    (8,  And(A4, A5),                    "|sin|≥0.5 ∧ |cos|≥0.5"),
    (9,  Or(A4, A5),                     "|sin|≥0.5 ∨ |cos|≥0.5"),
    (10, Or(And(A1, A4), And(A2, A5)),   "(sin≥0∧|sin|≥0.5) ∨ (cos≥0∧|cos|≥0.5)"),
    (11, And(Not(A3), Or(A1, A2)),       "|sin|<|cos| ∧ (sin≥0 ∨ cos≥0)"),
    (12, And(A1, A2, A3),                "sin≥0 ∧ cos≥0 ∧ |sin|≥|cos|"),
    (13, Or(A1, A2, A4),                 "sin≥0 ∨ cos≥0 ∨ |sin|≥0.5"),
    (14, Or(And(Not(A1), Not(A2)), A7),  "(sin<0∧cos<0) ∨ near diagonal"),
    (15, Or(And(A4, Not(A5)), And(Not(A4), A5)), "XOR: |sin|≥0.5 ⊕ |cos|≥0.5"),
]

N_FORMULAS = len(FORMULAS)


def get_formula(q: int) -> Formula:
    """Get formula by index."""
    return FORMULAS[q][1]


def get_formula_desc(q: int) -> str:
    """Get formula description."""
    return FORMULAS[q][2]


# ============================================================================
# 8. Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== Ω_trig++ Kernel Test ===\n")
    
    # Test 3-valued logic
    print("3-valued logic:")
    print(f"  TRUE & FALSE = {TVal.TRUE & TVal.FALSE}")
    print(f"  TRUE | UNKNOWN = {TVal.TRUE | TVal.UNKNOWN}")
    print(f"  ~UNKNOWN = {~TVal.UNKNOWN}")
    
    # Test interval
    print("\nInterval:")
    I = Interval(-0.3, 0.7)
    print(f"  {I}.all_ge(0) = {I.all_ge(0)}")  # UNKNOWN
    print(f"  {I}.all_ge(-0.5) = {I.all_ge(-0.5)}")  # TRUE
    
    # Test formulas
    print(f"\n{N_FORMULAS} formulas defined:")
    for q, formula, desc in FORMULAS:
        print(f"  φ_{q}: {desc}")
    
    # Test evaluation on sample angles
    print("\nSample evaluations:")
    for k in [0, 30, 45, 90, 135, 180, 270]:
        angle = Angle(k)
        print(f"\n  θ = {k}° (sin={angle.sin():.3f}, cos={angle.cos():.3f}):")
        for q in [0, 4, 8, 15]:
            formula = get_formula(q)
            result = formula.eval_exact(angle)
            print(f"    φ_{q} = {result}")
    
    print("\n=== Test Complete ===")

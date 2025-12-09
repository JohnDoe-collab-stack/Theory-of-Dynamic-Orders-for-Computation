"""
Ω-Trig Kernel: Syntax Layer

This module defines the FIXED structural components:
- AngleCode (X_trig): angle representation as k/N
- IntervalQ, PTrig: rational interval profiles
- V_trig: structural evaluation X_trig → PTrig  
- TrigIndex: question types
- questionTrig: structural reading I_trig × P_trig → Bool

Everything here is PURE SYNTAX - no learning, no parameters.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# ============================================================================
# 1.1. AngleCode (X_trig)
# ============================================================================

N_FIXED = 360  # Fixed discretization of the circle

@dataclass(frozen=True)
class AngleCode:
    """
    Represents an angle 2π * (k / N).
    Domain: X_trig = {AngleCode(k, N_FIXED) for k in range(N_FIXED)}
    """
    k: int
    N: int = N_FIXED
    
    def to_radians(self) -> float:
        """Convert to radians (for computation only)."""
        return 2 * math.pi * (self.k / self.N)
    
    def to_degrees(self) -> float:
        """Convert to degrees (for display)."""
        return 360.0 * (self.k / self.N)
    
    def __repr__(self) -> str:
        return f"Angle({self.k}/{self.N})"


def generate_X_trig(N: int = N_FIXED) -> List[AngleCode]:
    """Generate the full domain X_trig."""
    return [AngleCode(k, N) for k in range(N)]


# ============================================================================
# 1.2. Interval Rationnel and PTrig Profile
# ============================================================================

@dataclass
class IntervalQ:
    """
    Rational interval approximation [a, b].
    In practice we use floats to approximate ℚ.
    Invariant: a <= b
    """
    a: float
    b: float
    
    def __post_init__(self):
        if self.a > self.b:
            raise ValueError(f"Invalid interval: [{self.a}, {self.b}]")
    
    def contains(self, x: float) -> bool:
        """Check if x ∈ [a, b]."""
        return self.a <= x <= self.b
    
    def width(self) -> float:
        """Interval width."""
        return self.b - self.a
    
    def __repr__(self) -> str:
        return f"[{self.a:.4f}, {self.b:.4f}]"


@dataclass
class PTrig:
    """
    Trigonometric profile: intervals for sin and cos.
    This is the STRUCTURAL type from the Lean formalization.
    """
    sinI: IntervalQ
    cosI: IntervalQ
    
    @staticmethod
    def bot() -> 'PTrig':
        """Bottom element: maximally imprecise."""
        return PTrig(
            sinI=IntervalQ(-2.0, 2.0),
            cosI=IntervalQ(-2.0, 2.0)
        )
    
    def __repr__(self) -> str:
        return f"PTrig(sin={self.sinI}, cos={self.cosI})"


def le_p_trig(p1: PTrig, p2: PTrig) -> bool:
    """
    Order on PTrig: p2 is MORE PRECISE than p1.
    (Refinement order, inverted from interval containment)
    
    p1 ≤ p2 iff p2's intervals are contained in p1's intervals.
    """
    return (
        p2.sinI.a >= p1.sinI.a and
        p2.sinI.b <= p1.sinI.b and
        p2.cosI.a >= p1.cosI.a and
        p2.cosI.b <= p1.cosI.b
    )


# ============================================================================
# 1.3. V_trig: Structural Evaluation
# ============================================================================

def V_trig(x: AngleCode, eps: float = 1e-6) -> PTrig:
    """
    IDEAL structural profile for angle x.
    
    Conceptually: V_trig is the TRUE Ω-structure.
    We compute sin/cos to approximate it, but by definition
    V_trig IS the syntactic object (ideal profile).
    
    Args:
        x: AngleCode
        eps: Width of interval around true value
        
    Returns:
        PTrig with tight intervals around sin(θ), cos(θ)
    """
    theta = x.to_radians()
    s = math.sin(theta)
    c = math.cos(theta)
    
    return PTrig(
        sinI=IntervalQ(s - eps, s + eps),
        cosI=IntervalQ(c - eps, c + eps)
    )


# ============================================================================
# 1.4. TrigIndex: Question Types
# ============================================================================

class TrigIndexType(Enum):
    """Types of questions on trigonometric profiles."""
    SIGN_SIN = 0   # Is sin ≥ 0?
    SIGN_COS = 1   # Is cos ≥ 0?
    SIN_GE = 2     # Is sin ≥ r?
    COS_GE = 3     # Is cos ≥ r?


@dataclass(frozen=True)
class TrigIndex:
    """
    A question index for trigonometric structure.
    
    - SIGN_SIN/SIGN_COS: binary sign questions (r=None)
    - SIN_GE/COS_GE: threshold questions (r=threshold value)
    """
    kind: TrigIndexType
    r: Optional[float] = None
    
    def __repr__(self) -> str:
        if self.r is not None:
            return f"{self.kind.name}(r={self.r})"
        return self.kind.name


# Standard thresholds for GE questions
R_THRESH = [-0.5, 0.0, 0.5]

def generate_I_trig() -> List[TrigIndex]:
    """
    Generate the standard set of question indices.
    
    Returns:
        I_trig = {SIGN_SIN, SIGN_COS} ∪ {SIN_GE(r), COS_GE(r) | r ∈ R_THRESH}
    """
    indices = [
        TrigIndex(TrigIndexType.SIGN_SIN),
        TrigIndex(TrigIndexType.SIGN_COS),
    ]
    for r in R_THRESH:
        indices.append(TrigIndex(TrigIndexType.SIN_GE, r))
        indices.append(TrigIndex(TrigIndexType.COS_GE, r))
    
    return indices


# ============================================================================
# 1.5. questionTrig: Structural Reading
# ============================================================================

def question_trig(i: TrigIndex, p: PTrig) -> bool:
    """
    Answer a structural question on a profile.
    
    This is PURE STRUCTURAL READING - equivalent to valProfil_trig
    in the Lean formalization.
    
    Args:
        i: TrigIndex (the question)
        p: PTrig (the profile to query)
        
    Returns:
        bool: The answer according to the structure
    """
    if i.kind == TrigIndexType.SIGN_SIN:
        # sin ≥ 0 iff lower bound of sinI ≥ 0
        return p.sinI.a >= 0.0
    
    elif i.kind == TrigIndexType.SIGN_COS:
        # cos ≥ 0 iff lower bound of cosI ≥ 0
        return p.cosI.a >= 0.0
    
    elif i.kind == TrigIndexType.SIN_GE:
        # sin ≥ r iff lower bound of sinI ≥ r
        return p.sinI.a >= i.r
    
    elif i.kind == TrigIndexType.COS_GE:
        # cos ≥ r iff lower bound of cosI ≥ r
        return p.cosI.a >= i.r
    
    else:
        raise ValueError(f"Unknown TrigIndex kind: {i.kind}")


# ============================================================================
# Convenience: Direct evaluation
# ============================================================================

def evaluate_question(x: AngleCode, i: TrigIndex) -> bool:
    """
    Evaluate question i on angle x.
    
    This is the composition: questionTrig(i, V_trig(x))
    """
    p = V_trig(x)
    return question_trig(i, p)


# ============================================================================
# Main: Self-test
# ============================================================================

if __name__ == "__main__":
    print("=== Trig Kernel Tests ===\n")
    
    # Test X_trig
    X = generate_X_trig()
    print(f"X_trig size: {len(X)}")
    print(f"Sample angles: {X[0]}, {X[90]}, {X[180]}, {X[270]}")
    
    # Test V_trig
    print("\nV_trig samples:")
    for k in [0, 45, 90, 135, 180, 270]:
        x = AngleCode(k)
        p = V_trig(x)
        print(f"  {k}°: {p}")
    
    # Test I_trig
    I = generate_I_trig()
    print(f"\nI_trig size: {len(I)}")
    print(f"Question types: {I}")
    
    # Test questionTrig
    print("\nquestionTrig samples:")
    test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    sign_sin = TrigIndex(TrigIndexType.SIGN_SIN)
    sign_cos = TrigIndex(TrigIndexType.SIGN_COS)
    
    for k in test_angles:
        x = AngleCode(k)
        p = V_trig(x)
        ss = question_trig(sign_sin, p)
        sc = question_trig(sign_cos, p)
        print(f"  {k:3d}°: sin≥0={ss}, cos≥0={sc}")
    
    # Test le_p_trig
    print("\nle_p_trig order test:")
    p_precise = PTrig(IntervalQ(0.49, 0.51), IntervalQ(0.85, 0.87))
    p_imprecise = PTrig(IntervalQ(0.4, 0.6), IntervalQ(0.8, 0.9))
    p_bot = PTrig.bot()
    
    print(f"  precise ≤ imprecise: {le_p_trig(p_precise, p_imprecise)}")  # False
    print(f"  imprecise ≤ precise: {le_p_trig(p_imprecise, p_precise)}")  # True
    print(f"  bot ≤ precise: {le_p_trig(p_bot, p_precise)}")  # True
    
    print("\n=== All kernel tests passed! ===")

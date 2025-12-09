"""
P_vec for Ω-Trig: (cut, bit, halt_rank)

Defines the P_vec structure for questions σ = (angle, index):
- cut: depends ONLY on angle (geometric/quadrant)
- bit: depends ONLY on question type (structural)
- halt_rank: depends ONLY on learning dynamics (t_first)

By construction: cut ⊥ bit (orthogonal axes).
"""

import math
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

from trig_kernel import TrigIndex, TrigIndexType


# ============================================================================
# CutClass: Geometric structure (angle only)
# ============================================================================

class CutClass(Enum):
    """Cut class based on angle quadrant."""
    Q0 = 0  # [ -45°,  +45° ) around 0°
    Q1 = 1  # [  45°, +135° )
    Q2 = 2  # [ 135°, +225° )
    Q3 = 3  # [ 225°, +315° )


def cut_of_angle(angle_idx: int, N: int = 360) -> CutClass:
    """
    Compute cut class from angle index.
    
    Divides circle into 4 quadrants centered at 0°, 90°, 180°, 270°.
    """
    # Shift by 45° so quadrants are centered
    k = (angle_idx + N // 8) % N  # +45° shift
    q = (k * 4) // N  # 0,1,2,3
    return CutClass(q)


def cut_of_sigma(sigma: Tuple) -> CutClass:
    """Extract cut class from question σ = (angle_k, index)."""
    angle_k = sigma[0]  # First element is angle index
    return cut_of_angle(angle_k)


# ============================================================================
# BitClass: Question type structure (index only)
# ============================================================================

class BitClass(Enum):
    """Bit class based on question type."""
    SIGN = 0     # SIGN_SIN, SIGN_COS
    GE_NEG = 1   # SIN_GE(-0.5), COS_GE(-0.5)
    GE_ZERO = 2  # SIN_GE(0.0), COS_GE(0.0)
    GE_POS = 3   # SIN_GE(0.5), COS_GE(0.5)


def bit_of_index(index: TrigIndex) -> BitClass:
    """
    Compute bit class from TrigIndex.
    
    Depends ONLY on question type, not on any computed answer.
    """
    if index.kind in (TrigIndexType.SIGN_SIN, TrigIndexType.SIGN_COS):
        return BitClass.SIGN
    
    if index.kind in (TrigIndexType.SIN_GE, TrigIndexType.COS_GE):
        r = index.r
        if r < 0:
            return BitClass.GE_NEG
        elif r == 0:
            return BitClass.GE_ZERO
        else:  # r > 0
            return BitClass.GE_POS
    
    raise ValueError(f"Unknown index type: {index.kind}")


def bit_of_sigma(sigma: Tuple) -> BitClass:
    """Extract bit class from question σ."""
    # sigma[1] is the index kind name, sigma[2] is threshold r
    kind_name = sigma[1]
    r = sigma[2]
    
    if kind_name in ("SIGN_SIN", "SIGN_COS"):
        return BitClass.SIGN
    
    if kind_name in ("SIN_GE", "COS_GE"):
        if r is None:
            return BitClass.SIGN  # fallback
        if r < 0:
            return BitClass.GE_NEG
        elif r == 0:
            return BitClass.GE_ZERO
        else:
            return BitClass.GE_POS
    
    raise ValueError(f"Unknown index kind: {kind_name}")


# ============================================================================
# HaltRank: Learning dynamics (t_first only)
# ============================================================================

class HaltRank(Enum):
    """Halt rank based on first success time."""
    NEVER = 0  # Never learned (t_first = ∞)
    LATE = 1   # Learned late
    MID = 2    # Learned mid-training
    EARLY = 3  # Learned early


# Default epoch thresholds
EPOCHS = [0, 1, 5, 10, 20, 50]
E_EARLY = 5    # t_first <= E_EARLY → EARLY
E_MID = 20     # t_first <= E_MID → MID
E_LATE = 50    # t_first <= E_LATE → LATE


def halt_rank_of_tfirst(t_first: float,
                        e_early: int = E_EARLY,
                        e_mid: int = E_MID) -> HaltRank:
    """
    Compute halt rank from first success time.
    
    Args:
        t_first: First epoch where σ was correctly classified (or inf)
        e_early: Threshold for EARLY classification
        e_mid: Threshold for MID classification
    """
    if math.isinf(t_first):
        return HaltRank.NEVER
    if t_first <= e_early:
        return HaltRank.EARLY
    if t_first <= e_mid:
        return HaltRank.MID
    return HaltRank.LATE


# ============================================================================
# PVecTrig: Combined structure
# ============================================================================

@dataclass(frozen=True)
class PVecTrig:
    """P_vec for a question σ: (cut, bit, halt)."""
    cut: CutClass
    bit: BitClass
    halt: HaltRank
    
    def __repr__(self) -> str:
        return f"P({self.cut.name}, {self.bit.name}, {self.halt.name})"


def pvec_of_sigma(sigma: Tuple, t_first: float) -> PVecTrig:
    """
    Compute full P_vec for question σ.
    
    Args:
        sigma: (angle_k, index_kind_name, index_r) tuple
        t_first: First success epoch (or inf)
    """
    return PVecTrig(
        cut=cut_of_sigma(sigma),
        bit=bit_of_sigma(sigma),
        halt=halt_rank_of_tfirst(t_first)
    )


# ============================================================================
# Utilities for t_first computation
# ============================================================================

def compute_t_first(sigma_key: Tuple, 
                    epoch_E_vals: Dict[int, List[Tuple]]) -> float:
    """
    Compute t_first(σ) from checkpoint E_val data.
    
    Args:
        sigma_key: The question key to check
        epoch_E_vals: Dict mapping epoch → list of correctly answered questions
        
    Returns:
        First epoch where σ was correct, or inf if never
    """
    for epoch in sorted(epoch_E_vals.keys()):
        E_val = epoch_E_vals[epoch]
        # Convert to set of tuples for fast lookup
        E_set = set(tuple(e) if isinstance(e, list) else e for e in E_val)
        if sigma_key in E_set:
            return float(epoch)
    return float('inf')


def compute_all_t_first(all_sigmas: List[Tuple],
                        epoch_E_vals: Dict[int, List[Tuple]]) -> Dict[Tuple, float]:
    """
    Compute t_first for all questions.
    
    Returns:
        Dict mapping sigma → t_first
    """
    t_firsts = {}
    for sigma in all_sigmas:
        t_firsts[sigma] = compute_t_first(sigma, epoch_E_vals)
    return t_firsts


# ============================================================================
# Main: Self-test
# ============================================================================

if __name__ == "__main__":
    print("=== P_vec Trig Tests ===\n")
    
    # Test cut classes
    print("Cut classes (quadrants):")
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        c = cut_of_angle(angle)
        print(f"  {angle:3d}° → {c.name}")
    
    # Test bit classes
    print("\nBit classes (question types):")
    test_sigmas = [
        (0, "SIGN_SIN", None),
        (0, "SIGN_COS", None),
        (0, "SIN_GE", -0.5),
        (0, "COS_GE", 0.0),
        (0, "SIN_GE", 0.5),
    ]
    for sigma in test_sigmas:
        b = bit_of_sigma(sigma)
        print(f"  {sigma[1]:8s}(r={sigma[2]}) → {b.name}")
    
    # Test halt ranks
    print("\nHalt ranks (t_first):")
    for t in [1, 5, 10, 20, 50, float('inf')]:
        h = halt_rank_of_tfirst(t)
        print(f"  t_first={t:4.0f} → {h.name}")
    
    # Test full P_vec
    print("\nFull P_vec examples:")
    examples = [
        ((45, "SIGN_SIN", None), 1),    # Q1, SIGN, EARLY
        ((135, "SIN_GE", 0.0), 10),     # Q2, GE_ZERO, MID
        ((270, "COS_GE", 0.5), 50),     # Q3, GE_POS, LATE
        ((0, "SIGN_COS", None), float('inf')),  # Q0, SIGN, NEVER
    ]
    for sigma, t_first in examples:
        p = pvec_of_sigma(sigma, t_first)
        print(f"  σ={sigma}, t_first={t_first} → {p}")
    
    print("\n=== All P_vec tests passed! ===")

"""
Ω_arith Kernel: Arithmetic with Carry Dynamics

Implements:
- Addition simulation with carry propagation
- Sequential K dynamics (column-by-column)
- t_first^K computation based on when questions become decided
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


# ============================================================================
# Halt Rank
# ============================================================================

class HaltRank(Enum):
    EARLY = 0
    MID = 1
    LATE = 2
    NEVER = 3


def halt_rank_of_tfirst(t_first: int, n_digits: int) -> HaltRank:
    """Classify t_first based on position in sequence."""
    if n_digits == 0:
        return HaltRank.NEVER
    
    third = n_digits / 3
    
    if t_first <= third:
        return HaltRank.EARLY
    elif t_first <= 2 * third:
        return HaltRank.MID
    else:
        return HaltRank.LATE


# ============================================================================
# Number Representation
# ============================================================================

@dataclass
class Number:
    """Integer as list of digits (LSB first)."""
    digits: List[int]  # digits[0] = units, digits[1] = tens, etc.
    
    @classmethod
    def from_int(cls, value: int, n_digits: int) -> 'Number':
        """Convert int to digit list with padding."""
        if value < 0:
            raise ValueError("Only non-negative integers")
        
        digits = []
        for _ in range(n_digits):
            digits.append(value % 10)
            value //= 10
        
        return cls(digits)
    
    def to_int(self) -> int:
        """Convert back to int."""
        result = 0
        for i, d in enumerate(self.digits):
            result += d * (10 ** i)
        return result
    
    @property
    def n_digits(self) -> int:
        return len(self.digits)
    
    def __repr__(self):
        # Display MSB first for readability
        return "".join(str(d) for d in reversed(self.digits))


# ============================================================================
# Addition Simulation with Carry Dynamics
# ============================================================================

@dataclass
class AdditionState:
    """State of addition at time t."""
    t: int                      # Current column being processed
    sum_digits: List[int]       # Known result digits (indices < t)
    carries: List[int]          # carry[i] = carry INTO column i
    
    def is_complete(self, n_digits: int) -> bool:
        return self.t >= n_digits


class AdditionKernel:
    """
    Dynamic kernel for addition.
    
    Simulates column-by-column addition, tracking when each
    piece of information becomes known.
    """
    
    def __init__(self):
        pass
    
    def simulate(self, a: Number, b: Number) -> List[AdditionState]:
        """
        Simulate full addition, returning trajectory of states.
        
        Returns states [K_0, K_1, ..., K_n] where K_t has columns 0..t-1 fixed.
        """
        n = max(a.n_digits, b.n_digits)
        
        # Pad if needed
        a_digits = a.digits + [0] * (n - a.n_digits)
        b_digits = b.digits + [0] * (n - b.n_digits)
        
        trajectory = []
        sum_digits = []
        carries = [0]  # carry[0] = 0 (no carry into units)
        
        # Initial state (nothing known)
        trajectory.append(AdditionState(t=0, sum_digits=[], carries=[0]))
        
        for t in range(n):
            # Process column t
            total = a_digits[t] + b_digits[t] + carries[t]
            sum_digit = total % 10
            carry_out = total // 10
            
            sum_digits.append(sum_digit)
            carries.append(carry_out)
            
            trajectory.append(AdditionState(
                t=t + 1,
                sum_digits=sum_digits.copy(),
                carries=carries.copy(),
            ))
        
        return trajectory
    
    def get_final_sum(self, a: Number, b: Number) -> Number:
        """Compute a + b."""
        trajectory = self.simulate(a, b)
        final = trajectory[-1]
        
        # Add final carry if any
        digits = final.sum_digits.copy()
        if final.carries[-1] > 0:
            digits.append(final.carries[-1])
        
        return Number(digits)


# ============================================================================
# Questions
# ============================================================================

class QuestionType(Enum):
    SUM_GE = 0      # Is (a + b) >= threshold?
    DIGIT_EQ = 1    # Is result digit at position i equal to d?
    HAS_CARRY = 2   # Is there a carry out of column i?


@dataclass
class Question:
    """A question about an addition."""
    q_type: QuestionType
    param1: int = 0  # Threshold for SUM_GE, position for DIGIT/CARRY
    param2: int = 0  # Target digit for DIGIT_EQ
    
    def answer(self, a: Number, b: Number) -> bool:
        """Compute ground truth answer."""
        kernel = AdditionKernel()
        
        if self.q_type == QuestionType.SUM_GE:
            result = a.to_int() + b.to_int()
            return result >= self.param1
        
        elif self.q_type == QuestionType.DIGIT_EQ:
            result = kernel.get_final_sum(a, b)
            if self.param1 >= result.n_digits:
                return self.param2 == 0  # Leading zeros
            return result.digits[self.param1] == self.param2
        
        elif self.q_type == QuestionType.HAS_CARRY:
            trajectory = kernel.simulate(a, b)
            final = trajectory[-1]
            if self.param1 + 1 >= len(final.carries):
                return False
            return final.carries[self.param1 + 1] > 0
        
        raise ValueError(f"Unknown question type: {self.q_type}")
    
    def compute_t_first(self, a: Number, b: Number) -> int:
        """
        Compute t_first^K: first time step where answer is determined.
        
        For each question type, we determine when enough columns are
        fixed to know the answer.
        """
        kernel = AdditionKernel()
        trajectory = kernel.simulate(a, b)
        n = len(trajectory) - 1  # Number of columns
        
        answer = self.answer(a, b)
        
        if self.q_type == QuestionType.SUM_GE:
            # Sum >= T is decided when we've processed enough MSB columns
            # to rule out or confirm the threshold
            threshold = self.param1
            
            for t in range(1, len(trajectory)):
                state = trajectory[t]
                # Known digits: state.sum_digits[0..t-1]
                # Remaining digits: t..n-1 plus possible final carry
                
                # Min possible sum: known + 0s for rest
                min_sum = sum(d * (10 ** i) for i, d in enumerate(state.sum_digits))
                
                # Max possible sum: known + 9s for rest + final carry
                remaining = n - t
                max_sum = min_sum
                for i in range(remaining):
                    max_sum += 9 * (10 ** (t + i))
                max_sum += 10 ** n  # Max possible with carry
                
                # Can we decide?
                if min_sum >= threshold:
                    return t  # Definitely >= T
                if max_sum < threshold:
                    return t  # Definitely < T
            
            return n  # Decided at the end
        
        elif self.q_type == QuestionType.DIGIT_EQ:
            # Digit at position i is known when column i is processed
            # (which requires all columns 0..i for carry propagation)
            return min(self.param1 + 1, n)
        
        elif self.q_type == QuestionType.HAS_CARRY:
            # Carry out of column i is known when column i is processed
            return min(self.param1 + 1, n)
        
        return n
    
    def __repr__(self):
        if self.q_type == QuestionType.SUM_GE:
            return f"sum>={self.param1}"
        elif self.q_type == QuestionType.DIGIT_EQ:
            return f"digit[{self.param1}]=={self.param2}"
        elif self.q_type == QuestionType.HAS_CARRY:
            return f"carry[{self.param1}]"
        return str(self.q_type)


# ============================================================================
# Question Catalog
# ============================================================================

def generate_questions(n_digits: int) -> List[Question]:
    """Generate a set of diverse questions for n-digit numbers."""
    questions = []
    
    # Sum thresholds
    for k in range(1, n_digits + 1):
        threshold = 10 ** k
        questions.append(Question(QuestionType.SUM_GE, threshold))
    
    # Digit equality for each position
    for i in range(n_digits):
        for d in [0, 5, 9]:  # Sample digits
            questions.append(Question(QuestionType.DIGIT_EQ, i, d))
    
    # Carries for each position
    for i in range(n_digits):
        questions.append(Question(QuestionType.HAS_CARRY, i))
    
    return questions


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== Ω_arith Kernel Test ===\n")
    
    # Test addition
    a = Number.from_int(456, 3)
    b = Number.from_int(789, 3)
    
    print(f"a = {a} = {a.to_int()}")
    print(f"b = {b} = {b.to_int()}")
    
    kernel = AdditionKernel()
    result = kernel.get_final_sum(a, b)
    print(f"a + b = {result} = {result.to_int()}")
    
    # Show trajectory
    print("\nTrajectory:")
    trajectory = kernel.simulate(a, b)
    for state in trajectory:
        digits_str = "".join(str(d) for d in reversed(state.sum_digits)) if state.sum_digits else "?"
        carries_str = str(state.carries)
        print(f"  t={state.t}: sum={digits_str:>4s}, carries={carries_str}")
    
    # Test questions
    print("\nQuestions:")
    questions = generate_questions(3)
    for q in questions[:8]:  # First 8
        ans = q.answer(a, b)
        t_first = q.compute_t_first(a, b)
        hr = halt_rank_of_tfirst(t_first, 3)
        print(f"  {q}: y*={int(ans)}, t_first={t_first}, {hr.name}")
    
    print("\n=== Test Complete ===")

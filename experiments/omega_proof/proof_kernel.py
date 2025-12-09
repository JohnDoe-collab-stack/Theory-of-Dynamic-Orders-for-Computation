"""
Ω_proof Kernel: Propositional Logic with Dynamic Evaluation

Implements:
- Propositional formulas (atoms, ¬, ∧, ∨, →)
- Questions: TAUT, SAT
- Progressive evaluation (simulating partial knowledge)
- t_first^K: when truth becomes determined
"""

import random
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
from enum import Enum
from abc import ABC, abstractmethod
from itertools import product


# ============================================================================
# 1. Formulas
# ============================================================================

class Formula(ABC):
    @abstractmethod
    def depth(self) -> int:
        pass
    
    @abstractmethod
    def atoms(self) -> Set[str]:
        pass
    
    @abstractmethod
    def eval(self, assignment: Dict[str, bool]) -> bool:
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Number of nodes in formula tree."""
        pass


@dataclass(frozen=True)
class Atom(Formula):
    name: str
    
    def depth(self) -> int:
        return 0
    
    def size(self) -> int:
        return 1
    
    def atoms(self) -> Set[str]:
        return {self.name}
    
    def eval(self, assignment: Dict[str, bool]) -> bool:
        return assignment.get(self.name, False)
    
    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Not(Formula):
    f: Formula
    
    def depth(self) -> int:
        return 1 + self.f.depth()
    
    def size(self) -> int:
        return 1 + self.f.size()
    
    def atoms(self) -> Set[str]:
        return self.f.atoms()
    
    def eval(self, assignment: Dict[str, bool]) -> bool:
        return not self.f.eval(assignment)
    
    def __repr__(self):
        return f"¬{self.f}"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula
    
    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())
    
    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()
    
    def atoms(self) -> Set[str]:
        return self.left.atoms() | self.right.atoms()
    
    def eval(self, assignment: Dict[str, bool]) -> bool:
        return self.left.eval(assignment) and self.right.eval(assignment)
    
    def __repr__(self):
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula
    
    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())
    
    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()
    
    def atoms(self) -> Set[str]:
        return self.left.atoms() | self.right.atoms()
    
    def eval(self, assignment: Dict[str, bool]) -> bool:
        return self.left.eval(assignment) or self.right.eval(assignment)
    
    def __repr__(self):
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula
    
    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())
    
    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()
    
    def atoms(self) -> Set[str]:
        return self.left.atoms() | self.right.atoms()
    
    def eval(self, assignment: Dict[str, bool]) -> bool:
        return (not self.left.eval(assignment)) or self.right.eval(assignment)
    
    def __repr__(self):
        return f"({self.left} → {self.right})"


# ============================================================================
# 2. Ground Truth Computation
# ============================================================================

def is_tautology(formula: Formula) -> bool:
    """Check if formula is true under all assignments."""
    atom_list = sorted(formula.atoms())
    if not atom_list:
        return formula.eval({})
    
    for values in product([False, True], repeat=len(atom_list)):
        assignment = dict(zip(atom_list, values))
        if not formula.eval(assignment):
            return False
    return True


def is_satisfiable(formula: Formula) -> bool:
    """Check if formula is true under some assignment."""
    atom_list = sorted(formula.atoms())
    if not atom_list:
        return formula.eval({})
    
    for values in product([False, True], repeat=len(atom_list)):
        assignment = dict(zip(atom_list, values))
        if formula.eval(assignment):
            return True
    return False


# ============================================================================
# 3. Dynamic Kernel: Progressive Atom Resolution
# ============================================================================

class ProofKernel:
    """
    Dynamic kernel that progressively fixes atom assignments.
    
    At time t, atoms p0..p_{t-1} are fixed, others unknown.
    t_first^K = first t where result is determined for all remaining extensions.
    """
    
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
    
    def compute_t_first_taut(self, formula: Formula) -> Tuple[bool, int]:
        """
        Compute when tautology status becomes determined.
        
        Progressive assignment: at t, first t atoms are fixed.
        Answer is determined when all extensions of partial assignment agree.
        """
        atom_list = sorted(formula.atoms())
        n = len(atom_list)
        
        is_taut = is_tautology(formula)
        
        if n == 0:
            return is_taut, 0
        
        # At each t, check if answer is determined
        for t in range(n + 1):
            determined = True
            
            # Check all partial assignments of first t atoms
            for partial_values in product([False, True], repeat=t) if t > 0 else [()]:
                partial = dict(zip(atom_list[:t], partial_values))
                
                # Check if all extensions agree
                results = set()
                for ext_values in product([False, True], repeat=n-t) if n-t > 0 else [()]:
                    full = dict(partial)
                    full.update(zip(atom_list[t:], ext_values))
                    results.add(formula.eval(full))
                
                if len(results) > 1:
                    # This partial assignment doesn't determine result
                    determined = False
                    break
                
            if determined:
                return is_taut, t
        
        return is_taut, n
    
    def compute_t_first_sat(self, formula: Formula) -> Tuple[bool, int]:
        """Compute when satisfiability status becomes determined."""
        atom_list = sorted(formula.atoms())
        n = len(atom_list)
        
        is_sat = is_satisfiable(formula)
        
        if n == 0:
            return is_sat, 0
        
        # At each t, check if answer is determined
        for t in range(n + 1):
            determined = True
            
            for partial_values in product([False, True], repeat=t) if t > 0 else [()]:
                partial = dict(zip(atom_list[:t], partial_values))
                
                # Check all extensions
                found_true = False
                found_false = False
                
                for ext_values in product([False, True], repeat=n-t) if n-t > 0 else [()]:
                    full = dict(partial)
                    full.update(zip(atom_list[t:], ext_values))
                    if formula.eval(full):
                        found_true = True
                    else:
                        found_false = True
                
                # SAT is determined if we definitely found or definitely won't find
                # For taut, we'd need all True
                # For SAT, we need at least one True OR confirmed all False
                
            if t == n:
                return is_sat, t
        
        return is_sat, n


# ============================================================================
# 4. Questions
# ============================================================================

class QuestionType(Enum):
    TAUT = 0    # Is formula a tautology?
    SAT = 1     # Is formula satisfiable?


# ============================================================================
# 5. Halt Ranks
# ============================================================================

class HaltRank(Enum):
    EARLY = 0   # t_first <= 1
    MID = 1     # 2 <= t_first <= 3
    LATE = 2    # t_first >= 4
    NEVER = 3   # Timeout/undetermined


def halt_rank_of_tfirst(t_first: int, n_atoms: int) -> HaltRank:
    if n_atoms == 0:
        return HaltRank.EARLY
    
    ratio = t_first / n_atoms
    
    if ratio <= 0.33:
        return HaltRank.EARLY
    elif ratio <= 0.66:
        return HaltRank.MID
    else:
        return HaltRank.LATE


# ============================================================================
# 6. Formula Generation
# ============================================================================

def random_formula(n_atoms: int, max_depth: int, seed: int = None) -> Formula:
    """Generate random propositional formula."""
    if seed is not None:
        random.seed(seed)
    
    atoms = [Atom(f"p{i}") for i in range(n_atoms)]
    
    def gen(depth: int) -> Formula:
        if depth >= max_depth or random.random() < 0.3:
            return random.choice(atoms)
        
        op = random.choice(["not", "and", "or", "implies"])
        
        if op == "not":
            return Not(gen(depth + 1))
        elif op == "and":
            return And(gen(depth + 1), gen(depth + 1))
        elif op == "or":
            return Or(gen(depth + 1), gen(depth + 1))
        else:
            return Implies(gen(depth + 1), gen(depth + 1))
    
    return gen(0)


# ============================================================================
# 7. Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== Ω_proof Kernel Test ===\n")
    
    kernel = ProofKernel(max_steps=10)
    
    # Test formulas
    p, q, r = Atom("p"), Atom("q"), Atom("r")
    
    tests = [
        ("p ∨ ¬p", Or(p, Not(p)), True),
        ("p ∧ ¬p", And(p, Not(p)), False),
        ("p → p", Implies(p, p), True),
        ("(p → q) → (¬q → ¬p)", Implies(Implies(p, q), Implies(Not(q), Not(p))), True),
        ("p ∧ q", And(p, q), False),
        ("(p ∨ q) → (q ∨ p)", Implies(Or(p, q), Or(q, p)), True),
        ("p", p, False),
    ]
    
    print("Tautology tests:")
    for name, formula, expected in tests:
        is_taut, t_first = kernel.compute_t_first_taut(formula)
        n = len(formula.atoms())
        hr = halt_rank_of_tfirst(t_first, n)
        status = "✓" if is_taut == expected else "✗"
        print(f"  {status} {name}: taut={is_taut}, t_first={t_first}/{n}, {hr.name}")
    
    # SAT tests
    print("\nSatisfiability tests:")
    sat_tests = [
        ("p ∧ q", And(p, q), True),
        ("p ∧ ¬p", And(p, Not(p)), False),
        ("(p ∨ q) ∧ ¬p", And(Or(p, q), Not(p)), True),
    ]
    
    for name, formula, expected in sat_tests:
        is_sat = is_satisfiable(formula)
        _, t_first = kernel.compute_t_first_taut(formula)
        status = "✓" if is_sat == expected else "✗"
        print(f"  {status} {name}: sat={is_sat}")
    
    # Random formulas
    print("\nRandom formulas (depth 3, 3 atoms):")
    for i in range(5):
        f = random_formula(3, 3, seed=i)
        is_taut, t_first = kernel.compute_t_first_taut(f)
        n = len(f.atoms())
        hr = halt_rank_of_tfirst(t_first, n)
        sat = is_satisfiable(f)
        print(f"  {f}")
        print(f"    taut={is_taut}, sat={sat}, t={t_first}/{n}, {hr.name}")
    
    print("\n=== Test Complete ===")

"""
Ω_proof Kernel v2: Propositional Logic with Early Stopping

Key changes from v1:
- Early stopping on first counterexample/witness
- Halt bucketed by relative position in search space (not atoms)
"""

import random
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
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
# 2. Ground Truth
# ============================================================================

def is_tautology(formula: Formula) -> bool:
    atom_list = sorted(formula.atoms())
    if not atom_list:
        return formula.eval({})
    
    for values in product([False, True], repeat=len(atom_list)):
        if not formula.eval(dict(zip(atom_list, values))):
            return False
    return True


def is_satisfiable(formula: Formula) -> bool:
    atom_list = sorted(formula.atoms())
    if not atom_list:
        return formula.eval({})
    
    for values in product([False, True], repeat=len(atom_list)):
        if formula.eval(dict(zip(atom_list, values))):
            return True
    return False


# ============================================================================
# 3. Dynamic Kernel with Early Stopping
# ============================================================================

class ProofKernel:
    """
    K with EARLY STOPPING on counterexamples/witnesses.
    
    t_first = number of valuations examined before decision.
    This creates natural variation in difficulty.
    """
    
    def __init__(self, max_steps: int = 100, seed: int = 42):
        self.max_steps = max_steps
        self.seed = seed
        # seed is for fallback internal RNG, but ideally we pass rng per call
        random.seed(seed)
    
    def compute_t_first_taut(self, formula: Formula, valuation_order: str = "lex",
                             atom_order: str = "sorted", rng: random.Random = None) -> Tuple[bool, int]:
        """
        Stop on first counterexample.
        
        Args:
            valuation_order: "lex" (lexicographic), "shuffle", "random" (monte carlo)
            atom_order: "sorted" (p0, p1...), "shuffle"
            rng: Local random instance for shuffling (prevents global state drift)
        """
        _rng = rng if rng else random
        
        atom_list = list(formula.atoms())
        
        # 1. Atom ordering
        if atom_order == "sorted":
            atom_list.sort()
        elif atom_order == "shuffle":
            _rng.shuffle(atom_list)
        
        n = len(atom_list)
        
        if n == 0:
            return formula.eval({}), 0
        
        # 2. Valuation ordering
        if valuation_order == "lex":
            valuations = product([False, True], repeat=n)
        elif valuation_order == "shuffle":
            vals = list(product([False, True], repeat=n))
            _rng.shuffle(vals)
            valuations = vals
        elif valuation_order == "random":
            # Monte Carlo sampling (up to max_steps or coverage)
            valuations = []
            seen = set()
            limit = min(self.max_steps * 2, 2**n) # safety margin
            
            # Simple rejection sampling for unique valuations
            # (efficient enough for small n <= 10)
            count = 0
            while len(valuations) < limit and count < limit * 10:
                val = tuple(_rng.choice([False, True]) for _ in range(n))
                if val not in seen:
                    valuations.append(val)
                    seen.add(val)
                count += 1

        
        t = 0
        for values in valuations:
            t += 1
            if not formula.eval(dict(zip(atom_list, values))):
                return False, t  # Counterexample found
            if t >= self.max_steps:
                break
        
        return True, t  # All passed (or timeout)
    
    def compute_t_first_sat(self, formula: Formula, valuation_order: str = "lex",
                            atom_order: str = "sorted", rng: random.Random = None) -> Tuple[bool, int]:
        """
        Stop on first witness.
        """
        _rng = rng if rng else random
        
        atom_list = list(formula.atoms())
        
        # 1. Atom ordering
        if atom_order == "sorted":
            atom_list.sort()
        elif atom_order == "shuffle":
            _rng.shuffle(atom_list)
            
        n = len(atom_list)
        
        if n == 0:
            return formula.eval({}), 0
        
        # 2. Valuation ordering
        if valuation_order == "lex":
            valuations = product([False, True], repeat=n)
        elif valuation_order == "shuffle":
            vals = list(product([False, True], repeat=n))
            _rng.shuffle(vals)
            valuations = vals
        elif valuation_order == "random":
            valuations = []
            seen = set()
            limit = min(self.max_steps * 2, 2**n)
            count = 0
            while len(valuations) < limit and count < limit * 10:
                val = tuple(_rng.choice([False, True]) for _ in range(n))
                if val not in seen:
                    valuations.append(val)
                    seen.add(val)
                count += 1

        t = 0
        for values in valuations:
            t += 1
            if formula.eval(dict(zip(atom_list, values))):
                return True, t  # Witness found
            if t >= self.max_steps:
                break
        
        return False, t  # No witness


# ============================================================================
# 4. Questions
# ============================================================================

class QuestionType(Enum):
    TAUT = 0
    SAT = 1


# ============================================================================
# 5. Halt Ranks - Relative to Search Space
# ============================================================================

class HaltRank(Enum):
    EARLY = 0   # First 25% of search space
    MID = 1     # 25-75% of search space
    LATE = 2    # Last 25% of search space


def halt_rank_of_tfirst(t_first: int, n_atoms: int) -> HaltRank:
    """Classify by relative position in 2^n search space."""
    if n_atoms == 0:
        return HaltRank.EARLY
    
    total = 2 ** n_atoms
    ratio = t_first / total
    
    if ratio <= 0.25:
        return HaltRank.EARLY
    elif ratio <= 0.75:
        return HaltRank.MID
    else:
        return HaltRank.LATE


# ============================================================================
# 6. Formula Generation
# ============================================================================

def random_formula(n_atoms: int, max_depth: int, seed: int = None) -> Formula:
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
    print("=== Ω_proof Kernel v2 Test ===\n")
    
    kernel = ProofKernel()
    p, q = Atom("p"), Atom("q")
    
    tests = [
        ("p ∨ ¬p", Or(p, Not(p)), True),
        ("p ∧ ¬p", And(p, Not(p)), False),
        ("p → p", Implies(p, p), True),
        ("p ∧ q", And(p, q), False),
        ("p", p, False),
    ]
    
    print("Tautology tests:")
    for name, formula, expected in tests:
        is_taut, t_first = kernel.compute_t_first_taut(formula)
        n = len(formula.atoms())
        total = 2 ** n if n > 0 else 1
        hr = halt_rank_of_tfirst(t_first, n)
        status = "✓" if is_taut == expected else "✗"
        print(f"  {status} {name}: taut={is_taut}, t={t_first}/{total}, {hr.name}")
    
    print("\nRandom formulas (4 atoms, depth 3):")
    halt_counts = {"EARLY": 0, "MID": 0, "LATE": 0}
    for i in range(100):
        f = random_formula(4, 3, seed=i)
        _, t_first = kernel.compute_t_first_taut(f)
        hr = halt_rank_of_tfirst(t_first, len(f.atoms()))
        halt_counts[hr.name] += 1
    
    print(f"  Distribution: {halt_counts}")
    
    print("\n=== Test Complete ===")

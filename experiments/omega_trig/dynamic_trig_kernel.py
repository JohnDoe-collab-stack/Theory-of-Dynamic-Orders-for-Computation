"""
Dynamic Trig Kernel - Aligned with Lean DynamicKernel

Implements explicit dynamic objects:
- approx_t(x): monotone profil approximation
- val_t(x,i): dynamic valuation
- traces: t ↦ val_t(x,i)
- E(K): stabilized assertions

Monotonicity: t ≤ t' ⇒ approx_t(x) ≤ approx_t'(x)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, NamedTuple, Optional


# ==========================
# 1. Domaine, profil, ordre
# ==========================

@dataclass(frozen=True)
class Interval:
    """Intervalle réel [a, b] avec a ≤ b."""
    a: float
    b: float

    def width(self) -> float:
        return self.b - self.a

    @staticmethod
    def full() -> "Interval":
        # Large assez pour contenir sin/cos
        return Interval(a=-2.0, b=2.0)

    @staticmethod
    def point(x: float) -> "Interval":
        return Interval(a=x, b=x)


@dataclass(frozen=True)
class Profile:
    """Profil trigonométrique: intervalles pour sin et cos."""
    sin: Interval
    cos: Interval


def le_profile(p1: Profile, p2: Profile) -> bool:
    """
    Ordre partiel sur Profile:
    p1 ≤ p2  ⇔  p2 est plus précis (intervalles plus étroits).
    """
    return (
        p2.sin.a >= p1.sin.a and
        p2.sin.b <= p1.sin.b and
        p2.cos.a >= p1.cos.a and
        p2.cos.b <= p1.cos.b
    )


# ==============
# 2. Ω-syntaxe
# ==============

class QuestionKind:
    SIGN_SIN = "SIGN_SIN"
    SIGN_COS = "SIGN_COS"
    SIN_GE = "SIN_GE"
    COS_GE = "COS_GE"


class Question(NamedTuple):
    """Index i ∈ I_trig : type + seuil éventuel."""
    kind: str
    threshold: Optional[float] = None  # pour SIN_GE / COS_GE


def V_trig(angle_deg: int) -> Profile:
    """
    Profil 'vrai' (idéal) : point interval autour de sin/cos(angle).
    angle_deg ∈ {0, …, 359}.
    """
    theta = math.radians(angle_deg)
    s = math.sin(theta)
    c = math.cos(theta)
    return Profile(
        sin=Interval.point(s),
        cos=Interval.point(c),
    )


def question_trig(q: Question, p: Profile) -> bool:
    """
    q(i, p): question structurelle sur un profil p.
    On lit les bornes inférieures (sin.a, cos.a).
    """
    if q.kind == QuestionKind.SIGN_SIN:
        return p.sin.a >= 0.0
    elif q.kind == QuestionKind.SIGN_COS:
        return p.cos.a >= 0.0
    elif q.kind == QuestionKind.SIN_GE:
        assert q.threshold is not None
        return p.sin.a >= q.threshold
    elif q.kind == QuestionKind.COS_GE:
        assert q.threshold is not None
        return p.cos.a >= q.threshold
    else:
        raise ValueError(f"Unknown question kind: {q.kind}")


# ==========================
# 3. Noyau dynamique jouet
# ==========================

@dataclass
class DynState:
    """État dynamique: temps t + angle x."""
    t: int
    angle_deg: int  # x ∈ X


class DynamicTrigKernel:
    """
    Jouet qui implémente explicitement:
      - approx_t(x)
      - val_t(x, i)
      - traces
      - E(K)
    avec une dynamique monotone sur les profils.
    """

    def __init__(
        self,
        T_max: int = 10,
        angles: List[int] = None,
        questions: List[Question] = None,
    ):
        self.T_max = T_max

        # Domaine X : sous-ensemble de {0..359}
        if angles is None:
            # Par défaut: 8 angles de base
            self.angles = [0, 45, 90, 135, 180, 225, 270, 315]
        else:
            self.angles = angles

        # Indices I_trig
        if questions is None:
            self.questions = [
                Question(QuestionKind.SIGN_SIN),
                Question(QuestionKind.SIGN_COS),
                Question(QuestionKind.SIN_GE, -0.5),
                Question(QuestionKind.COS_GE, -0.5),
                Question(QuestionKind.SIN_GE, 0.0),
                Question(QuestionKind.COS_GE, 0.0),
                Question(QuestionKind.SIN_GE, 0.5),
                Question(QuestionKind.COS_GE, 0.5),
            ]
        else:
            self.questions = questions

        # Pré-calcul des profils vrais
        self.V_cache: Dict[int, Profile] = {
            a: V_trig(a) for a in self.angles
        }

    # --------- approx_t(x) ---------

    def approx(self, t: int, angle_deg: int) -> Profile:
        """
        approx_t(x): profil approché à temps t.
        t=0 : bot (intervalle très large).
        t≥1 : intervalle centré sur V(x), largeur décroissante en t.
        Monotonie: approx_t(x) ≤ approx_{t+1}(x).
        """
        # Profil exact
        v = self.V_cache[angle_deg]

        if t <= 0:
            # bot: incertitude max
            return Profile(sin=Interval.full(), cos=Interval.full())

        # On choisit une largeur décroissante linéairement jusqu'à 0 à T_max
        # Largeur max ≈ 1.5 (on couvre [-1-ε, +1+ε])
        t_clamped = min(t, self.T_max)
        alpha = t_clamped / self.T_max  # 0 → 1
        # largeur = w_max * (1 - alpha)
        w_max = 1.5
        w = w_max * (1.0 - alpha)

        def shrink_interval(true_val: float) -> Interval:
            a = max(-1.0, true_val - w)
            b = min(1.0, true_val + w)
            if a > b:  # sécurité numérique
                a = b = true_val
            return Interval(a=a, b=b)

        sin_int = shrink_interval(v.sin.a)
        cos_int = shrink_interval(v.cos.a)

        return Profile(sin=sin_int, cos=cos_int)

    # --------- val_t(x,i) ---------

    def val(self, t: int, angle_deg: int, q: Question) -> int:
        """
        val_t(x,i): lecture booléenne sur approx_t(x).
        Renvoie 0 ou 1 (int) pour coller à {0,1}.
        """
        p = self.approx(t, angle_deg)
        return int(question_trig(q, p))

    # --------- traces & E(K) ---------

    def compute_traces(self) -> Dict[Tuple[int, Question], List[int]]:
        """
        Calcule toutes les traces (x,i) ↦ [val_t]_t pour t=0..T_max.
        """
        traces: Dict[Tuple[int, Question], List[int]] = {}
        for angle in self.angles:
            for q in self.questions:
                key = (angle, q)
                trace = [self.val(t, angle, q) for t in range(self.T_max + 1)]
                traces[key] = trace
        return traces

    def compute_EK(self) -> Dict[Tuple[int, Question], bool]:
        """
        Calcule E(K) ≃ assertions (x,i) telles que:
          - question vraie sur V(x)
          - et val_t(x,i) devient 1 à partir d'un certain t₀ puis reste 1.
        Renvoie un dict (angle, q) ↦ bool (dans E(K) ou pas).
        """
        traces = self.compute_traces()
        EK: Dict[Tuple[int, Question], bool] = {}

        for (angle, q), trace in traces.items():
            # vérité structurelle
            v = self.V_cache[angle]
            true_ans = question_trig(q, v)

            if not true_ans:
                EK[(angle, q)] = False
                continue

            # Chercher t₀: premier indice où val_t = 1
            t0 = None
            for t, v_t in enumerate(trace):
                if v_t == 1:
                    t0 = t
                    break

            if t0 is None:
                EK[(angle, q)] = False
                continue

            # Vérifier que pour tout t ≥ t₀, val_t = 1
            stable = all(v_t == 1 for v_t in trace[t0:])
            EK[(angle, q)] = stable

        return EK

    # --------- petite vérification de monotonie ---------

    def check_monotonicity(self) -> Tuple[bool, bool]:
        """
        Vérifie:
          - approx_t(x) ≤ approx_{t+1}(x)
          - val_t(x,i) = 1 ⇒ val_{t+1}(x,i) = 1
        Retourne (ok_approx, ok_val).
        """
        ok_approx = True
        ok_val = True

        for angle in self.angles:
            for t in range(self.T_max):
                p_t = self.approx(t, angle)
                p_tp1 = self.approx(t + 1, angle)

                if not le_profile(p_t, p_tp1):
                    ok_approx = False

                for q in self.questions:
                    v_t = self.val(t, angle, q)
                    v_tp1 = self.val(t + 1, angle, q)
                    if v_t == 1 and v_tp1 == 0:
                        ok_val = False

        return ok_approx, ok_val

    # --------- statistiques ---------

    def stats(self) -> Dict:
        """Compute statistics about the kernel."""
        traces = self.compute_traces()
        EK = self.compute_EK()
        
        n_total = len(self.angles) * len(self.questions)
        n_in_EK = sum(1 for v in EK.values() if v)
        
        # t_first distribution
        t_firsts = []
        for (angle, q), trace in traces.items():
            for t, v_t in enumerate(trace):
                if v_t == 1:
                    t_firsts.append(t)
                    break
            else:
                t_firsts.append(float('inf'))
        
        finite_t = [t for t in t_firsts if t != float('inf')]
        
        return {
            "n_angles": len(self.angles),
            "n_questions": len(self.questions),
            "n_total": n_total,
            "n_in_EK": n_in_EK,
            "EK_ratio": n_in_EK / n_total if n_total > 0 else 0,
            "t_first_avg": sum(finite_t) / len(finite_t) if finite_t else float('inf'),
            "t_first_min": min(finite_t) if finite_t else float('inf'),
            "t_first_max": max(finite_t) if finite_t else float('inf'),
        }

    def export_t_first_K(self, output_path: str = "checkpoints_trig/t_first_K.json"):
        """
        Export t_first^K(σ) for all σ = (angle, question) to JSON.
        
        Format: List of {"angle": int, "kind": str, "threshold": float|null, "t_first": float}
        """
        import json
        import os
        
        traces = self.compute_traces()
        records = []
        
        for (angle, q), trace in traces.items():
            # Find t_first
            t_first = float('inf')
            for t, v_t in enumerate(trace):
                if v_t == 1:
                    if all(v == 1 for v in trace[t:]):
                        t_first = float(t)
                        break
            
            records.append({
                "angle": angle,
                "kind": q.kind,
                "threshold": q.threshold,
                "t_first": t_first if not math.isinf(t_first) else None
            })
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2)
        
        print(f"Exported {len(records)} t_first^K values to {output_path}")
        return output_path


# ==========================
# 4. Démo minimale
# ==========================

if __name__ == "__main__":
    kernel = DynamicTrigKernel(T_max=10)

    print("=== Dynamic Trig Kernel Demo ===\n")

    # Vérifier la monotonie
    ok_approx, ok_val = kernel.check_monotonicity()
    print(f"Monotonicité approx_t: {'✓ OK' if ok_approx else '✗ FAIL'}")
    print(f"Monotonicité val_t:    {'✓ OK' if ok_val else '✗ FAIL'}")
    print()

    # Montrer quelques traces
    traces = kernel.compute_traces()
    print("Traces val_t(x, SIGN_SIN):")
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        q = Question(QuestionKind.SIGN_SIN)
        trace = traces[(angle, q)]
        print(f"  {angle:3d}°: {trace}")

    print()

    # Calculer E(K)
    EK = kernel.compute_EK()
    print("E(K): assertions stabilisées")
    in_EK = [(a, q) for (a, q), v in EK.items() if v]
    print(f"  Total: {len(in_EK)} / {len(EK)}")
    
    print("\n  Sample assertions in E(K):")
    for angle, q in sorted(in_EK, key=lambda x: (x[0], x[1].kind))[:10]:
        if q.threshold is None:
            descr = q.kind
        else:
            descr = f"{q.kind}(r={q.threshold})"
        print(f"    angle={angle:3d}°, {descr}")

    # Stats
    print("\nKernel Statistics:")
    stats = kernel.stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Dynamic Kernel Demo Complete ===")

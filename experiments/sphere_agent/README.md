# Sphere Agent Framework

> **Bounded-budget safety guarantees for AI agents**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)

## Overview

This is a Python implementation of the **Global Sphere Framework** from the Lean 4 formalization (`LogicDissoc/Sphere.lean`). It provides tools to enforce budget constraints on AI agents, with **mathematical guarantees** on the maximum number of "structurally significant" actions.

## Key Guarantee

From the Lean theorem `max_trajectory_length`:

> **If an agent starts with fuel ≤ R and each "risky" action consumes ≥ 1 fuel, then at most R risky actions are possible.**

This is an absolute bound, regardless of what the agent does between risky actions.

## Installation

```bash
cd experiments/sphere_agent
pip install numpy
```

## Quick Start

```python
from sphere_agent import (
    GlobalProfile,
    SphereConstraint,
    evaluate_step,
    SphereMonitor,
    create_safety_constraint,
)

# Create a safety constraint with 20 total budget
constraint = create_safety_constraint(
    risk_budget=10,
    deviation_budget=5,
    env_modification_budget=5,
)

# Initial profile
profile = constraint.initial_profile()
print(f"Budget: {profile}")  # Profile(risk=10, deviation=5, env_mod=5, fuel=20)

# After a risky action
new_profile = GlobalProfile([8, 5, 5], names=constraint.names)
result = evaluate_step(profile, new_profile)
print(f"Type: {result.step_type.name}")  # STRICT
print(f"Consumed: {result.fuel_consumed}")  # 2

# Monitor for alerts
monitor = SphereMonitor(constraint)
alerts = monitor.check(result, new_profile)
```

## Core Concepts

| Concept | Description | Lean Equivalent |
|---------|-------------|-----------------|
| `GlobalProfile` | D-dimensional resource vector | `GlobalProfile D` |
| `fuel` | Sum of all dimensions (ℓ₁-norm) | `GlobalProfile.sum` |
| `InSphere(R)` | fuel ≤ R | `InSphere R v` |
| `StrictStep` | Action consuming ≥1 fuel | `StrictStep Step L x y` |
| `Valley` | Stable region (fuel = 0) | `Valley Step L V` |
| `Mode` | Active coordinates for action type | `Mode State D` |

## Components

### 1. `sphere.py` - Core Abstractions

```python
# Profile with named dimensions
profile = GlobalProfile([10, 5, 5], names=["risk", "deviation", "env"])

# Check if in sphere
profile.in_sphere(R=20)  # True

# Evaluate a transition
result = evaluate_step(before, after)
result.is_strict      # True if consumed fuel
result.fuel_consumed  # Amount consumed
```

### 2. `monitors.py` - Alerts & Auditing

```python
monitor = SphereMonitor(constraint)
alerts = monitor.check(step_result, new_profile)

for alert in alerts:
    if alert.level == AlertLevel.CRITICAL:
        print(f"🚨 {alert.message}")
```

### 3. `agent_wrapper.py` - Guarded Execution

```python
guarded = SphereGuardedAgent(
    agent=my_agent,
    env=my_env,
    extractor=profile_extractor,
    constraint=constraint,
    veto_on_violation=True,
)

# Run with guarantees
results = guarded.run(max_steps=1000)
print(f"Strict steps: {guarded.history.strict_steps}")  # Guaranteed ≤ R
```

## Use Cases

### 1. RL Agent Control
```python
constraint = SphereConstraint(
    R=100,
    D=3,
    names=["exploration", "exploitation", "reset"],
)
# Guarantees: ≤100 "significant" exploration steps
```

### 2. LLM Tool Calling
```python
constraint = SphereConstraint(
    R=50,
    D=4,
    names=["dangerous_tools", "unverified_claims", "external_effects", "autonomy"],
)
# Guarantees: bounded danger regardless of conversation length
```

### 3. Training/Fine-tuning
```python
constraint = create_training_constraint(
    distribution_shift_budget=5,
    regularization_debt=10,
    safety_constraint_budget=5,
)
# Guarantees: ≤20 "structural" training changes
```

## Running the Demo

```bash
cd experiments/sphere_agent
python demo.py
```

Output shows:
- Sphere mechanics
- Monitoring and alerts
- Mode-based control
- Trajectory bound verification
- AI safety scenario

## Mathematical Foundation

The framework is based on the Lean 4 theorems:

```lean
-- Every strict step decreases fuel
theorem strict_step_decreases_sum (x y : State) (h : StrictStep Step L x y) :
    Fuel L y < Fuel L x

-- Trajectory length is bounded
theorem max_trajectory_length (R : Nat) (chain : Nat → State) (len : Nat)
    (h_start : GlobalProfile.InSphere R (L (chain 0)))
    (h_step : ∀ k, k < len → StrictStep Step L (chain k) (chain (k + 1))) :
    len ≤ R
```

These are proved in `LogicDissoc/Sphere.lean` and form the **verified foundation** for this implementation.

## API Reference

### SphereConstraint
- `R: int` - Maximum fuel (sphere radius)
- `D: int` - Number of dimensions
- `names: List[str]` - Coordinate names
- `validate(profile)` - Check if in sphere
- `max_strict_steps(profile)` - Upper bound on remaining strict steps

### SphereMonitor
- `check(result, profile)` - Generate alerts
- `should_veto(result, profile)` - Veto recommendation
- `summary()` - Monitoring statistics

### SphereGuardedAgent
- `step(state)` - Execute one guarded step
- `run(max_steps)` - Run until done/veto/exhausted
- `remaining_budget()` - Remaining strict step budget

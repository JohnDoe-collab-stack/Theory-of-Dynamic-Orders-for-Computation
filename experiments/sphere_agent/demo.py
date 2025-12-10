#!/usr/bin/env python3
"""
Demo: Sphere-Guarded Agent
==========================

This demo shows how to use the Sphere framework to control an agent
with guaranteed bounds on "risky" actions.

Scenario:
- An agent explores a grid world
- Each "risky" move (entering new territory) consumes fuel
- The sphere constraint guarantees at most R risky moves

Run from project root:
    python -m experiments.sphere_agent.demo
    
Or from this directory:
    python demo.py
"""

import sys
from pathlib import Path

# Add parent to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import from local modules
from sphere import (
    GlobalProfile,
    SphereConstraint,
    Mode,
    evaluate_step,
    StepType,
    ExecutionHistory,
)
from monitors import SphereMonitor, Alert, AlertLevel


# =============================================================================
# Demo 1: Simple State Tracking
# =============================================================================

def demo_simple():
    """
    Simple demonstration of sphere mechanics.
    """
    print("\n" + "="*60)
    print("Demo 1: Simple Sphere Mechanics")
    print("="*60)
    
    # Create a constraint with 3 dimensions
    constraint = SphereConstraint(
        R=20,
        D=3,
        names=["risk", "deviation", "resources"],
    )
    
    # Create initial profile
    profile = constraint.initial_profile([10, 5, 5])  # Total = 20 = R
    print(f"\nInitial: {profile}")
    print(f"In sphere: {constraint.validate(profile)}")
    print(f"Max strict steps possible: {constraint.max_strict_steps(profile)}")
    
    # Simulate some transitions
    print("\n--- Simulating transitions ---\n")
    
    # Transition 1: Strict (consume risk budget)
    profile2 = GlobalProfile([8, 5, 5], names=constraint.names.copy())
    result = evaluate_step(profile, profile2)
    print(f"Step 1: {result.step_type.name}")
    print(f"  Fuel: {result.fuel_before} → {result.fuel_after}")
    print(f"  Consumed: {result.fuel_consumed}")
    
    # Transition 2: Plateau (no change)
    profile3 = GlobalProfile([8, 5, 5], names=constraint.names.copy())
    result2 = evaluate_step(profile2, profile3)
    print(f"\nStep 2: {result2.step_type.name}")
    print(f"  No fuel consumed (allowed, doesn't count toward limit)")
    
    # Transition 3: Another strict step
    profile4 = GlobalProfile([5, 4, 5], names=constraint.names.copy())
    result3 = evaluate_step(profile3, profile4)
    print(f"\nStep 3: {result3.step_type.name}")
    print(f"  Fuel: {result3.fuel_before} → {result3.fuel_after}")
    print(f"  Decreased coords: {[constraint.names[i] for i in result3.decreased_coords]}")
    
    # Try a violation
    print("\n--- Testing violation detection ---\n")
    profile_bad = GlobalProfile([6, 4, 5], names=constraint.names.copy())
    result_bad = evaluate_step(profile4, profile_bad)
    print(f"Attempted increase: {result_bad.step_type.name}")
    print(f"  Increased coords: {[constraint.names[i] for i in result_bad.increased_coords]}")
    print("  This would be VETOED by the monitor!")
    
    print("\n✓ Demo 1 complete")


# =============================================================================
# Demo 2: Monitoring and Alerts
# =============================================================================

def demo_monitoring():
    """
    Demonstrate the monitoring and alert system.
    """
    print("\n" + "="*60)
    print("Demo 2: Monitoring and Alerts")
    print("="*60)
    
    constraint = SphereConstraint(
        R=100,
        D=4,
        names=["risk", "deviation", "env_mod", "queries"],
    )
    
    monitor = SphereMonitor(
        constraint,
        warning_threshold=0.3,
        critical_threshold=0.1,
        high_consumption_threshold=0.15,
    )
    
    # Starting profile
    profile = constraint.initial_profile([40, 30, 20, 10])
    print(f"\nInitial: {profile}")
    
    # Simulate a series of steps with decreasing fuel
    profiles = [
        [35, 28, 18, 10],  # -9
        [30, 25, 15, 8],   # -13
        [25, 20, 10, 5],   # -20 (high consumption)
        [15, 12, 5, 3],    # -25 (warning zone)
        [8, 5, 2, 1],      # -19 (critical zone)
    ]
    
    print("\n--- Monitoring execution ---\n")
    
    prev_profile = profile
    for i, values in enumerate(profiles):
        new_profile = GlobalProfile(np.array(values), constraint.names.copy())
        result = evaluate_step(prev_profile, new_profile)
        alerts = monitor.check(result, new_profile)
        
        print(f"Step {i+1}: fuel {result.fuel_before} → {result.fuel_after}")
        for alert in alerts:
            print(f"  {alert}")
        
        prev_profile = new_profile
    
    print(f"\n--- Monitor Summary ---")
    summary = monitor.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    print("\n✓ Demo 2 complete")


# =============================================================================
# Demo 3: Mode-Based Control
# =============================================================================

def demo_modes():
    """
    Demonstrate mode-based control with different active coordinates.
    """
    print("\n" + "="*60)
    print("Demo 3: Mode-Based Control")
    print("="*60)
    
    constraint = SphereConstraint(
        R=30,
        D=3,
        names=["exploration", "exploitation", "safety"],
        modes=[
            Mode("explore", {0}, "Only exploration fuel consumed"),
            Mode("exploit", {1}, "Only exploitation fuel consumed"),
            Mode("safe", {2}, "Only safety budget consumed"),
            Mode("mixed", {0, 1}, "Exploration + exploitation"),
        ],
    )
    
    print(f"\nModes defined:")
    for mode in constraint.modes:
        print(f"  {mode.name}: active on {[constraint.names[i] for i in mode.active_coords]}")
    
    # Initial profile
    profile = constraint.initial_profile([10, 12, 8])
    print(f"\nInitial: {profile}")
    
    # Mode: explore (only coord 0 should decrease)
    explore_mode = constraint.modes[0]
    profile2 = GlobalProfile([7, 12, 8], constraint.names.copy())
    result = evaluate_step(profile, profile2)
    valid = explore_mode.validate_step(result)
    print(f"\n[explore mode] Decrease exploration: valid={valid}")
    
    # Invalid: decrease exploitation in explore mode
    profile3 = GlobalProfile([7, 10, 8], constraint.names.copy())
    result2 = evaluate_step(profile2, profile3)
    valid2 = explore_mode.validate_step(result2)
    print(f"[explore mode] Decrease exploitation: valid={valid2} ← VIOLATED!")
    
    # Mixed mode: can decrease both
    mixed_mode = constraint.modes[3]
    valid3 = mixed_mode.validate_step(result2)
    print(f"[mixed mode] Same step: valid={valid3} ← OK in mixed mode")
    
    print("\n✓ Demo 3 complete")


# =============================================================================
# Demo 4: Trajectory Bound Guarantee
# =============================================================================

def demo_trajectory_bound():
    """
    Demonstrate the max_trajectory_length theorem in action.
    
    Theorem: chains of strict steps have length ≤ initial fuel ≤ R
    """
    print("\n" + "="*60)
    print("Demo 4: Trajectory Bound Guarantee")
    print("="*60)
    
    R = 15
    constraint = SphereConstraint(R=R, D=2, names=["a", "b"])
    
    # Start with fuel = 10
    initial = constraint.initial_profile([6, 4])
    print(f"\nInitial: {initial}")
    print(f"Max strict steps by theorem: ≤ {initial.fuel}")
    
    # Simulate max strict steps (each consumes 1 fuel)
    profile = initial
    strict_count = 0
    
    print("\n--- Simulating strict steps ---")
    
    while profile.fuel > 0:
        # Decrease coord 0 by 1 (or coord 1 if 0 is empty)
        values = profile.values.copy()
        if values[0] > 0:
            values[0] -= 1
        else:
            values[1] -= 1
        
        new_profile = GlobalProfile(values, constraint.names.copy())
        result = evaluate_step(profile, new_profile)
        
        if result.is_strict:
            strict_count += 1
            print(f"  Strict step {strict_count}: {profile.fuel} → {new_profile.fuel}")
        
        profile = new_profile
    
    print(f"\n--- Result ---")
    print(f"Total strict steps: {strict_count}")
    print(f"Bound from theorem: {initial.fuel}")
    print(f"Guarantee satisfied: {strict_count <= initial.fuel} ✓")
    
    print("\n✓ Demo 4 complete")


# =============================================================================
# Demo 5: AI Safety Scenario
# =============================================================================

def demo_ai_safety():
    """
    Realistic AI safety scenario: tool-calling agent with budget.
    """
    print("\n" + "="*60)
    print("Demo 5: AI Safety - Tool-Calling Agent")
    print("="*60)
    
    # Define constraint for a tool-calling AI
    constraint = SphereConstraint(
        R=50,
        D=4,
        names=[
            "dangerous_tools",    # Budget for dangerous tool calls
            "unverified_claims",  # Budget for claims made without verification
            "external_effects",   # Budget for actions with external effects
            "autonomy_escalation", # Budget for self-modification attempts
        ],
        critical_threshold=0.15,
    )
    
    print(f"Safety constraint for AI agent:")
    print(f"  Total budget: R = {constraint.R}")
    print(f"  Dimensions: {constraint.names}")
    print(f"  Critical threshold: {constraint.critical_threshold * 100}%")
    
    # Scenario: agent executing a task
    actions = [
        ("search_web", [0, 0, 1, 0]),           # External effect: -1
        ("make_claim", [0, 1, 0, 0]),           # Unverified: -1
        ("call_api", [0, 0, 2, 0]),             # External: -2
        ("execute_code", [3, 0, 1, 0]),         # Dangerous + external: -4
        ("modify_prompt", [0, 0, 0, 2]),        # Autonomy escalation: -2
        ("make_claim", [0, 1, 0, 0]),           # Unverified: -1
        ("execute_code", [2, 0, 1, 0]),         # Another risky action: -3
    ]
    
    monitor = SphereMonitor(constraint)
    profile = constraint.initial_profile()
    
    print(f"\nInitial budget: {profile}")
    print("\n--- Executing actions ---\n")
    
    for action_name, consumption in actions:
        # Calculate new profile
        new_values = profile.values - np.array(consumption)
        
        if np.any(new_values < 0):
            idx = np.where(new_values < 0)[0][0]
            print(f"  BLOCKED: {action_name} would exhaust {constraint.names[idx]}")
            continue
        
        new_profile = GlobalProfile(new_values, constraint.names.copy())
        result = evaluate_step(profile, new_profile)
        alerts = monitor.check(result, new_profile)
        
        print(f"  {action_name}: consumed {result.fuel_consumed} fuel")
        for alert in alerts:
            if alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                print(f"    ⚠️  {alert.message}")
        
        profile = new_profile
    
    print(f"\n--- Final State ---")
    print(f"Remaining budget: {profile}")
    summary = monitor.summary()
    print(f"Summary: {summary}")
    
    # Key guarantee
    print(f"\n🔒 GUARANTEE: At most {constraint.R} 'risky' actions total, no matter what!")
    
    print("\n✓ Demo 5 complete")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("SPHERE AGENT FRAMEWORK - DEMONSTRATIONS")
    print("Based on Lean 4 formalization of Global Sphere termination")
    print("="*60)
    
    demo_simple()
    demo_monitoring()
    demo_modes()
    demo_trajectory_bound()
    demo_ai_safety()
    
    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)

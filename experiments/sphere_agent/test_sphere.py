#!/usr/bin/env python3
"""
Unit Tests for Sphere Agent Framework
======================================

Run with: pytest test_sphere.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from sphere import (
    GlobalProfile,
    SphereConstraint,
    StepType,
    StepResult,
    evaluate_step,
    Mode,
    disjoint_modes,
    Valley,
    ExecutionHistory,
    StepRecord,
)
from monitors import SphereMonitor, Alert, AlertLevel


# =============================================================================
# § 1. GlobalProfile Tests
# =============================================================================

class TestGlobalProfile:
    
    def test_creation(self):
        """Test profile creation with values."""
        p = GlobalProfile([10, 5, 5])
        assert p.D == 3
        assert p.fuel == 20
        assert list(p.values) == [10, 5, 5]
    
    def test_creation_with_names(self):
        """Test profile with custom names."""
        p = GlobalProfile([10, 5], names=["risk", "deviation"])
        assert p.names == ["risk", "deviation"]
    
    def test_fuel_is_sum(self):
        """Fuel should be ℓ₁-norm (sum of components)."""
        p = GlobalProfile([3, 7, 2, 8])
        assert p.fuel == 20
    
    def test_in_sphere_true(self):
        """Profile with fuel ≤ R should be in sphere."""
        p = GlobalProfile([5, 5, 5])  # fuel = 15
        assert p.in_sphere(15) == True
        assert p.in_sphere(20) == True
    
    def test_in_sphere_false(self):
        """Profile with fuel > R should not be in sphere."""
        p = GlobalProfile([10, 10, 10])  # fuel = 30
        assert p.in_sphere(20) == False
    
    def test_zero_profile(self):
        """Zero profile should have fuel = 0."""
        p = GlobalProfile.zero(3)
        assert p.fuel == 0
        assert list(p.values) == [0, 0, 0]
    
    def test_zero_in_any_sphere(self):
        """Zero profile should be in any sphere."""
        p = GlobalProfile.zero(3)
        assert p.in_sphere(0) == True
        assert p.in_sphere(100) == True
    
    def test_negative_values_rejected(self):
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError):
            GlobalProfile([-1, 5, 5])
    
    def test_indexing(self):
        """Profile should support indexing."""
        p = GlobalProfile([10, 20, 30])
        assert p[0] == 10
        assert p[1] == 20
        assert p[2] == 30


# =============================================================================
# § 2. StepResult Tests
# =============================================================================

class TestEvaluateStep:
    
    def test_strict_step(self):
        """Transition with decrease should be STRICT."""
        before = GlobalProfile([10, 5, 5])
        after = GlobalProfile([8, 5, 5])
        
        result = evaluate_step(before, after)
        
        assert result.step_type == StepType.STRICT
        assert result.is_strict == True
        assert result.fuel_consumed == 2
        assert 0 in result.decreased_coords
    
    def test_plateau_step(self):
        """Transition with no change should be PLATEAU."""
        before = GlobalProfile([10, 5, 5])
        after = GlobalProfile([10, 5, 5])
        
        result = evaluate_step(before, after)
        
        assert result.step_type == StepType.PLATEAU
        assert result.is_plateau == True
        assert result.fuel_consumed == 0
    
    def test_violation_step(self):
        """Transition with increase should be VIOLATION."""
        before = GlobalProfile([10, 5, 5])
        after = GlobalProfile([12, 5, 5])
        
        result = evaluate_step(before, after)
        
        assert result.step_type == StepType.VIOLATION
        assert result.is_violation == True
        assert 0 in result.increased_coords
    
    def test_multi_decrease(self):
        """Multiple coords decreasing should be STRICT."""
        before = GlobalProfile([10, 10, 10])
        after = GlobalProfile([8, 8, 8])
        
        result = evaluate_step(before, after)
        
        assert result.is_strict == True
        assert result.fuel_consumed == 6
        assert len(result.decreased_coords) == 3
    
    def test_dimension_mismatch_raises(self):
        """Different dimensions should raise ValueError."""
        before = GlobalProfile([10, 5])
        after = GlobalProfile([10, 5, 5])
        
        with pytest.raises(ValueError):
            evaluate_step(before, after)
    
    def test_fuel_values(self):
        """StepResult should have correct fuel values."""
        before = GlobalProfile([10, 5, 5])  # fuel = 20
        after = GlobalProfile([7, 4, 5])    # fuel = 16
        
        result = evaluate_step(before, after)
        
        assert result.fuel_before == 20
        assert result.fuel_after == 16
        assert result.fuel_consumed == 4


# =============================================================================
# § 3. SphereConstraint Tests
# =============================================================================

class TestSphereConstraint:
    
    def test_creation(self):
        """Test constraint creation."""
        c = SphereConstraint(R=20, D=3, names=["a", "b", "c"])
        assert c.R == 20
        assert c.D == 3
    
    def test_validation_pass(self):
        """Profile in sphere should validate."""
        c = SphereConstraint(R=20, D=3, names=["a", "b", "c"])
        p = GlobalProfile([5, 5, 5])
        assert c.validate(p) == True
    
    def test_validation_fail(self):
        """Profile outside sphere should not validate."""
        c = SphereConstraint(R=10, D=3, names=["a", "b", "c"])
        p = GlobalProfile([5, 5, 5])  # fuel = 15 > R = 10
        assert c.validate(p) == False
    
    def test_max_strict_steps(self):
        """Max strict steps should equal fuel."""
        c = SphereConstraint(R=20, D=3, names=["a", "b", "c"])
        p = GlobalProfile([5, 5, 5])  # fuel = 15
        assert c.max_strict_steps(p) == 15
    
    def test_initial_profile(self):
        """Initial profile should sum to ≤ R."""
        c = SphereConstraint(R=20, D=3, names=["a", "b", "c"])
        p = c.initial_profile()
        assert p.fuel <= c.R
    
    def test_initial_profile_with_values(self):
        """Initial profile with custom values."""
        c = SphereConstraint(R=20, D=3, names=["a", "b", "c"])
        p = c.initial_profile([10, 5, 5])
        assert list(p.values) == [10, 5, 5]
    
    def test_critical_threshold(self):
        """Critical threshold check."""
        c = SphereConstraint(R=100, D=2, names=["a", "b"], critical_threshold=0.1)
        
        p_ok = GlobalProfile([20, 30])  # fuel = 50 > 10%
        p_critical = GlobalProfile([5, 3])  # fuel = 8 < 10%
        
        assert c.is_critical(p_ok) == False
        assert c.is_critical(p_critical) == True


# =============================================================================
# § 4. Mode Tests
# =============================================================================

class TestMode:
    
    def test_mode_creation(self):
        """Test mode creation."""
        m = Mode("explore", {0, 1}, "Exploration mode")
        assert m.name == "explore"
        assert m.active_coords == {0, 1}
    
    def test_is_active(self):
        """Test active coordinate check."""
        m = Mode("explore", {0, 1})
        assert m.is_active(0) == True
        assert m.is_active(1) == True
        assert m.is_active(2) == False
    
    def test_validate_step_valid(self):
        """Valid mode step (only active coords decreased)."""
        m = Mode("explore", {0})
        
        before = GlobalProfile([10, 5, 5])
        after = GlobalProfile([8, 5, 5])  # only coord 0 decreased
        result = evaluate_step(before, after)
        
        assert m.validate_step(result) == True
    
    def test_validate_step_invalid(self):
        """Invalid mode step (non-active coord decreased)."""
        m = Mode("explore", {0})
        
        before = GlobalProfile([10, 5, 5])
        after = GlobalProfile([10, 3, 5])  # coord 1 decreased, not in active
        result = evaluate_step(before, after)
        
        assert m.validate_step(result) == False
    
    def test_disjoint_modes(self):
        """Test disjoint modes check."""
        m1 = Mode("a", {0, 1})
        m2 = Mode("b", {2, 3})
        m3 = Mode("c", {1, 2})
        
        assert disjoint_modes(m1, m2) == True
        assert disjoint_modes(m1, m3) == False


# =============================================================================
# § 5. Trajectory Bound Theorem Test
# =============================================================================

class TestTrajectoryBoundTheorem:
    """
    Test the key theorem: max_trajectory_length ≤ initial_fuel ≤ R
    """
    
    def test_strict_steps_bounded_by_fuel(self):
        """Number of strict steps should be ≤ initial fuel."""
        initial = GlobalProfile([5, 5, 0])  # fuel = 10
        initial_fuel = initial.fuel
        
        # Simulate strict steps
        strict_count = 0
        profile = initial
        
        while profile.fuel > 0:
            # Each step decreases by 1
            values = profile.values.copy()
            if values[0] > 0:
                values[0] -= 1
            elif values[1] > 0:
                values[1] -= 1
            
            new_profile = GlobalProfile(values)
            result = evaluate_step(profile, new_profile)
            
            if result.is_strict:
                strict_count += 1
            
            profile = new_profile
        
        # KEY ASSERTION: strict steps ≤ initial fuel
        assert strict_count <= initial_fuel
        # In this case, exactly equal since each step consumes 1
        assert strict_count == initial_fuel
    
    def test_bound_with_multi_unit_steps(self):
        """Bound holds even with multi-unit consumption."""
        initial = GlobalProfile([10, 10, 10])  # fuel = 30
        initial_fuel = initial.fuel
        
        # Simulate: each step consumes 3 (one from each coord)
        strict_count = 0
        profile = initial
        
        while min(profile.values) > 0:
            new_values = profile.values - 1  # decrease all by 1
            new_profile = GlobalProfile(new_values)
            result = evaluate_step(profile, new_profile)
            
            if result.is_strict:
                strict_count += 1
            
            profile = new_profile
        
        # strict_count should be 10 (each step consumes 3)
        # Total consumed = 30, fuel = 30, so strict_count ≤ initial_fuel
        assert strict_count <= initial_fuel
    
    def test_bound_within_sphere(self):
        """Trajectory from sphere stays bounded."""
        R = 20
        constraint = SphereConstraint(R=R, D=3, names=["a", "b", "c"])
        
        initial = constraint.initial_profile([8, 7, 5])  # fuel = 20 = R
        assert constraint.validate(initial)
        
        # Simulate strict steps
        strict_count = 0
        profile = initial
        
        for _ in range(100):  # way more than possible
            if profile.fuel == 0:
                break
            
            values = profile.values.copy()
            for i in range(len(values)):
                if values[i] > 0:
                    values[i] -= 1
                    break
            
            new_profile = GlobalProfile(values, constraint.names.copy())
            result = evaluate_step(profile, new_profile)
            
            if result.is_strict:
                strict_count += 1
            
            profile = new_profile
        
        # KEY ASSERTION: strict steps ≤ R
        assert strict_count <= R


# =============================================================================
# § 6. Monitor Tests
# =============================================================================

class TestSphereMonitor:
    
    def test_monitor_creation(self):
        """Test monitor creation."""
        c = SphereConstraint(R=100, D=3, names=["a", "b", "c"])
        m = SphereMonitor(c)
        assert m.constraint == c
    
    def test_violation_alert(self):
        """Violation should generate VIOLATION alert."""
        c = SphereConstraint(R=100, D=3, names=["a", "b", "c"])
        m = SphereMonitor(c)
        
        before = GlobalProfile([10, 5, 5])
        after = GlobalProfile([12, 5, 5])  # increase = violation
        result = evaluate_step(before, after)
        
        alerts = m.check(result, after)
        
        assert any(a.level == AlertLevel.VIOLATION for a in alerts)
    
    def test_critical_alert(self):
        """Low fuel should generate CRITICAL alert."""
        c = SphereConstraint(R=100, D=3, names=["a", "b", "c"])
        m = SphereMonitor(c, critical_threshold=0.1)
        
        before = GlobalProfile([10, 5, 5])
        after = GlobalProfile([3, 2, 2])  # fuel = 7 < 10% of 100
        result = evaluate_step(before, after)
        
        alerts = m.check(result, after)
        
        assert any(a.level == AlertLevel.CRITICAL for a in alerts)
    
    def test_warning_alert(self):
        """Medium-low fuel should generate WARNING alert."""
        c = SphereConstraint(R=100, D=3, names=["a", "b", "c"])
        m = SphereMonitor(c, warning_threshold=0.3, critical_threshold=0.1)
        
        before = GlobalProfile([50, 30, 20])
        after = GlobalProfile([10, 8, 7])  # fuel = 25 < 30%
        result = evaluate_step(before, after)
        
        alerts = m.check(result, after)
        
        assert any(a.level == AlertLevel.WARNING for a in alerts)
    
    def test_veto_on_violation(self):
        """Monitor should recommend veto on violation."""
        c = SphereConstraint(R=100, D=3, names=["a", "b", "c"])
        m = SphereMonitor(c)
        
        before = GlobalProfile([10, 5, 5])
        after = GlobalProfile([12, 5, 5])
        result = evaluate_step(before, after)
        
        veto, reason = m.should_veto(result, after)
        
        assert veto == True
        assert "violated" in reason.lower()
    
    def test_summary(self):
        """Monitor should provide summary."""
        c = SphereConstraint(R=100, D=3, names=["a", "b", "c"])
        m = SphereMonitor(c)
        
        # Simulate some steps
        before = GlobalProfile([50, 30, 20])
        after = GlobalProfile([48, 30, 20])
        result = evaluate_step(before, after)
        m.check(result, after)
        
        summary = m.summary()
        
        assert 'total_steps' in summary
        assert 'strict_steps' in summary
        assert summary['total_steps'] == 1
        assert summary['strict_steps'] == 1


# =============================================================================
# § 7. Valley Tests
# =============================================================================

class TestValley:
    
    def test_minimal_valley(self):
        """Minimal valley should contain fuel=0 profiles."""
        v = Valley.minimal()
        
        p_zero = GlobalProfile.zero(3)
        p_nonzero = GlobalProfile([1, 0, 0])
        
        assert v.contains(p_zero) == True
        assert v.contains(p_nonzero) == False
    
    def test_custom_valley(self):
        """Custom valley with threshold."""
        v = Valley("low_fuel", fuel_threshold=5)
        
        p_in = GlobalProfile([2, 2, 1])    # fuel = 5
        p_out = GlobalProfile([3, 2, 1])   # fuel = 6
        
        assert v.contains(p_in) == True
        assert v.contains(p_out) == False


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Sphere Agent Framework
======================

A Python implementation of the Global Sphere framework for AI safety.

This package provides tools to enforce budget constraints on agents,
guaranteeing that the number of "structurally significant" actions
is bounded.

Based on the Lean 4 formalization in LogicDissoc/Sphere.lean.
"""

from .sphere import (
    GlobalProfile,
    StepType,
    StepResult,
    evaluate_step,
    Mode,
    disjoint_modes,
    Valley,
    SphereConstraint,
    StepRecord,
    ExecutionHistory,
    create_safety_constraint,
    create_training_constraint,
)

from .monitors import (
    AlertLevel,
    Alert,
    SphereMonitor,
    AuditLogger,
    CompositeMonitor,
)

from .agent_wrapper import (
    Agent,
    Environment,
    ProfileExtractor,
    SimpleProfileExtractor,
    VetoError,
    SphereViolationError,
    GuardedStepResult,
    SphereGuardedAgent,
    create_guarded_agent,
)

__all__ = [
    # Core
    'GlobalProfile',
    'StepType',
    'StepResult',
    'evaluate_step',
    'Mode',
    'disjoint_modes',
    'Valley',
    'SphereConstraint',
    'StepRecord',
    'ExecutionHistory',
    
    # Convenience
    'create_safety_constraint',
    'create_training_constraint',
    
    # Monitors
    'AlertLevel',
    'Alert',
    'SphereMonitor',
    'AuditLogger',
    'CompositeMonitor',
    
    # Agent wrapper
    'Agent',
    'Environment',
    'ProfileExtractor',
    'SimpleProfileExtractor',
    'VetoError',
    'SphereViolationError',
    'GuardedStepResult',
    'SphereGuardedAgent',
    'create_guarded_agent',
]

__version__ = '0.1.0'

"""
Ω-Trig Package

Clean architecture:
- trig_kernel: Ω-syntax (fixed structure)
- dataset_trig: D = X_trig × I_trig with ground truth
- model_T: Theory T_θ (learner)
- train_T: Training with checkpoints
- analysis_T: E(T_θ) inclusion analysis
"""

from .trig_kernel import (
    AngleCode, IntervalQ, PTrig,
    TrigIndexType, TrigIndex,
    N_FIXED,
    generate_X_trig, generate_I_trig,
    V_trig, question_trig, le_p_trig,
    evaluate_question,
)

from .dataset_trig import (
    TrigQuestion,
    generate_trig_dataset, split_dataset,
    TrigDataset, create_dataloaders,
    dataset_stats,
)

from .model_T import (
    TrigTheoryModel,
    count_parameters,
)

__all__ = [
    # Kernel
    'AngleCode', 'IntervalQ', 'PTrig',
    'TrigIndexType', 'TrigIndex',
    'N_FIXED',
    'generate_X_trig', 'generate_I_trig',
    'V_trig', 'question_trig', 'le_p_trig',
    'evaluate_question',
    
    # Dataset
    'TrigQuestion',
    'generate_trig_dataset', 'split_dataset',
    'TrigDataset', 'create_dataloaders',
    'dataset_stats',
    
    # Model
    'TrigTheoryModel',
    'count_parameters',
]

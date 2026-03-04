"""
Ensemble Layer: Model weighting, uncertainty-aware aggregation, and stacking.

- aggregator: Inverse-MAE / stability weighting for regression forecasts.
- stacking: Temporal stacking for trend classification (meta-model on base model proba).
"""

from src.ensemble.aggregator import (
    EnsembleResult,
    aggregate_forecasts,
    compute_stability_weights,
)
from src.ensemble.stacking import (
    TemporalStackingEnsemble,
    compare_models,
    print_comparison,
    stack_predictions,
)

__all__ = [
    "EnsembleResult",
    "aggregate_forecasts",
    "compute_stability_weights",
    "TemporalStackingEnsemble",
    "compare_models",
    "print_comparison",
    "stack_predictions",
]

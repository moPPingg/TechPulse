"""
Ensemble Layer: Model weighting and uncertainty-aware aggregation.

What goes in:  Per-model forecasts (mean, optional std) + historical performance metrics.
What comes out: Ensemble mean, ensemble std, per-model weights.

Why this exists: Single models overfit; weighting by stability reduces variance.
Penalize unstable (high variance in backtest) and overfit (val >> test) models.
"""

from src.ensemble.aggregator import (
    EnsembleResult,
    aggregate_forecasts,
    compute_stability_weights,
)

__all__ = [
    "EnsembleResult",
    "aggregate_forecasts",
    "compute_stability_weights",
]

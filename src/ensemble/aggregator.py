"""
Ensemble aggregator: Weight models by historical stability, aggregate with uncertainty.

TEACHING:
- Inverse-MAE weighting: models with lower MAE get higher weight.
- Stability penalty: models with high variance in rolling backtest get penalized.
- Uncertainty propagation: If models provide std, we use sqrt(sum(w² * std²)).
  If not, we use empirical residual std from validation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ModelForecast:
    """Single model output: mean and optional uncertainty (std)."""
    name: str
    mean: float
    std: Optional[float] = None  # If None, will use empirical or ensemble-level estimate


@dataclass
class EnsembleResult:
    """Aggregated forecast from multiple models."""
    mean: float
    std: float
    weights: dict  # model_name -> weight
    n_models: int


def compute_stability_weights(
    mae_per_model: dict,
    rolling_mae_std_per_model: Optional[dict] = None,
    stability_penalty: float = 1.0,
) -> dict:
    """
    Compute weights: inverse MAE, penalized by rolling MAE variance.

    Args:
        mae_per_model: {model_name: MAE on test/val}
        rolling_mae_std_per_model: {model_name: std of MAE across rolling folds}. Optional.
        stability_penalty: How much to penalize unstable models (higher = more penalty).

    Returns:
        {model_name: weight} summing to 1.0

    Why: Models with low MAE and low variance are most trustworthy.
    """
    names = list(mae_per_model.keys())
    inv_mae = np.array([1.0 / max(mae_per_model[n], 1e-8) for n in names])

    if rolling_mae_std_per_model and stability_penalty > 0:
        # Penalize high variance: weight *= 1 / (1 + penalty * std)
        stds = np.array([
            rolling_mae_std_per_model.get(n, 0) for n in names
        ])
        inv_mae = inv_mae / (1.0 + stability_penalty * np.maximum(stds, 0))

    w = inv_mae / inv_mae.sum()
    return dict(zip(names, w))


def aggregate_forecasts(
    forecasts: List[ModelForecast],
    weights: Optional[dict] = None,
    mae_per_model: Optional[dict] = None,
    rolling_mae_std_per_model: Optional[dict] = None,
) -> EnsembleResult:
    """
    Aggregate model forecasts into single mean and std.

    Args:
        forecasts: List of ModelForecast (name, mean, optional std)
        weights: Optional explicit weights. If None, use inverse-MAE from mae_per_model.
        mae_per_model: Used to compute weights when weights is None.
        rolling_mae_std_per_model: Used for stability penalty when weights is None.

    Returns:
        EnsembleResult with mean, std, weights.

    Uncertainty math:
        - Weighted mean: mu = sum(w_i * mean_i)
        - Weighted variance (assuming independence): sigma² = sum(w_i² * sigma_i²)
        - If sigma_i not provided, use max of provided stds or 0.5% as default.
    """
    if not forecasts:
        return EnsembleResult(mean=0.0, std=0.5, weights={}, n_models=0)

    names = [f.name for f in forecasts]
    means = np.array([f.mean for f in forecasts])

    if weights is None:
        if mae_per_model:
            weights = compute_stability_weights(
                {n: mae_per_model.get(n, 1.0) for n in names},
                rolling_mae_std_per_model,
            )
        else:
            # Equal weights
            w_arr = np.ones(len(forecasts)) / len(forecasts)
            weights = dict(zip(names, w_arr))

    w_arr = np.array([weights.get(n, 0) for n in names])
    w_arr = w_arr / w_arr.sum()

    ensemble_mean = float(np.dot(w_arr, means))

    # Std: use model std if available, else default
    default_std = 0.5  # 0.5% daily return as conservative default
    stds = np.array([
        f.std if f.std is not None and f.std > 0 else default_std
        for f in forecasts
    ])
    # Weighted variance: sigma² = sum(w_i² * sigma_i²)
    ensemble_var = np.sum(w_arr ** 2 * stds ** 2)
    ensemble_std = float(np.sqrt(max(ensemble_var, 1e-10)))

    return EnsembleResult(
        mean=ensemble_mean,
        std=ensemble_std,
        weights={n: float(w_arr[i]) for i, n in enumerate(names)},
        n_models=len(forecasts),
    )

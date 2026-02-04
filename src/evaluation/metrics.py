"""
Unified metrics for forecasting. Same metrics for all models.
"""

from typing import Dict, Union
import numpy as np

METRIC_NAMES = ["mae", "rmse", "mape", "r2", "direction_acc"]


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """MAPE; avoid div by zero."""
    denom = np.abs(y_true)
    denom = np.where(denom < epsilon, epsilon, denom)
    return np.abs((y_true - y_pred) / denom)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: list = None,
) -> Dict[str, float]:
    """
    Compute regression and directional metrics. Same for every model.

    Args:
        y_true: Ground truth (1d).
        y_pred: Predictions (1d), same length as y_true.
        metric_names: Which metrics to compute (default: all).

    Returns:
        Dict of metric name -> float.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    if metric_names is None:
        metric_names = METRIC_NAMES

    out: Dict[str, float] = {}
    n = len(y_true)

    if "mae" in metric_names:
        out["mae"] = float(np.mean(np.abs(y_true - y_pred)))

    if "rmse" in metric_names:
        out["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    if "mape" in metric_names:
        mape_arr = _safe_mape(y_true, y_pred)
        out["mape"] = float(np.mean(mape_arr) * 100.0)  # as percentage

    if "r2" in metric_names:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        out["r2"] = float(1.0 - (ss_res / (ss_tot + 1e-10)))

    if "direction_acc" in metric_names:
        dir_true = (y_true > 0).astype(np.float64)
        dir_pred = (y_pred > 0).astype(np.float64)
        out["direction_acc"] = float(np.mean(dir_true == dir_pred) * 100.0)

    return out

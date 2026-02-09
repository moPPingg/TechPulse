"""
Unified evaluation framework for forecasting models.

- Same train/val/test splits (time-based)
- Same metrics (MAE, RMSE, MAPE, R2, directional accuracy)
- Same rolling-window backtesting

Use: prepare_data() -> get_splits() -> metrics + backtest for each model.
"""

from src.evaluation.splits import get_train_val_test_splits, get_purged_train_val_test_splits
from src.evaluation.metrics import compute_metrics, METRIC_NAMES
from src.evaluation.data import prepare_tabular, prepare_sequential, ForecastDataset, get_rolling_fold_tabular
from src.evaluation.backtest import rolling_backtest, simple_backtest

__all__ = [
    "get_train_val_test_splits",
    "get_purged_train_val_test_splits",
    "compute_metrics",
    "METRIC_NAMES",
    "prepare_tabular",
    "prepare_sequential",
    "ForecastDataset",
    "get_rolling_fold_tabular",
    "rolling_backtest",
    "simple_backtest",
]

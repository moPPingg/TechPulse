"""
Rolling-window backtesting. Same procedure for all models.
Expanding training window: at each test step t, train on [0, t], predict at t+1.
"""

from typing import Dict, List, Optional, Any, Callable
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, METRIC_NAMES


def rolling_backtest(
    model_factory: Callable[[], Any],
    df: pd.DataFrame,
    get_fold: Callable[[pd.DataFrame, int], Optional[tuple]],
    test_start: int,
    test_end: Optional[int] = None,
    date_col: str = "date",
    metric_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Expanding-window backtest: for each t in [test_start, test_end), get_fold(df, t)
    returns (X_train, y_train, X_test_1, y_true_1). Refit model, predict, collect.

    get_fold(df, t) must return (X_train, y_train, X_test, y_test) where X_test has 1 row,
    or None to skip. Train data should use rows 0..t (so we have target at t+1 for last row).
    """
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if test_end is None:
        test_end = n - 1  # need t+1 for y_true
    all_y_true: List[float] = []
    all_y_pred: List[float] = []

    for t in range(test_start, min(test_end, n - 1)):
        fold = get_fold(df, t)
        if fold is None:
            continue
        X_train, y_train, X_test, y_test = fold
        if len(X_test) == 0 or len(y_test) == 0:
            continue
        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.asarray(pred).ravel()
        all_y_true.append(float(y_test.ravel()[0]))
        all_y_pred.append(float(pred[0]))

    if not all_y_true:
        return {m: float("nan") for m in (metric_names or METRIC_NAMES)}
    return compute_metrics(
        np.array(all_y_true),
        np.array(all_y_pred),
        metric_names=metric_names or METRIC_NAMES,
    )


def simple_backtest(
    model: Any,
    data: Any,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    One-shot: fit on train, predict on test. Same metrics for all models.
    """
    model.fit(data.X_train, data.y_train)
    pred = model.predict(data.X_test)
    pred = np.asarray(pred).ravel()
    return compute_metrics(
        data.y_test,
        pred,
        metric_names=metric_names or METRIC_NAMES,
    )

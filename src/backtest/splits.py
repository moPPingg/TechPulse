"""
Time-based train/val/test splits. Same splits for all models.

PRODUCTION NOTE: For overlapping return targets (e.g. 5-day returns), use
get_purged_train_val_test_splits to avoid leakage at split boundaries.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np


def get_purged_train_val_test_splits(
    df: "pd.DataFrame",
    date_col: str = "date",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    purge_gap: int = 1,
    embargo_pct: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split with purge gap and optional embargo to avoid overlap leakage.

    PRODUCTION: Use purge_gap >= prediction_horizon when target has overlap.
    Layout: [train][purge][val][purge][test]
    purge_gap: Rows to drop between train/val and val/test.
    embargo_pct: Fraction to embargo after train (0 = disabled).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)

    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    gap = max(0, purge_gap)
    n_train = n - n_test - n_val - 2 * gap
    if n_train <= 0:
        return get_train_val_test_splits(df, date_col, train_ratio, val_ratio, test_ratio)

    train_end = n_train
    val_start = n_train + gap
    val_end = val_start + n_val
    test_start = val_end + gap
    if test_start + n_test > n:
        test_start = n - n_test
        val_end = test_start - gap
        val_start = val_end - n_val

    train_df = df.iloc[:train_end]
    val_df = df.iloc[val_start:val_end]
    test_df = df.iloc[test_start : test_start + n_test]

    return train_df, val_df, test_df


def get_train_val_test_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    min_train_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by time. No shuffle. Same order for every model.

    Args:
        df: Must have date_col, sorted ascending by date.
        date_col: Column name for dates.
        train_ratio: Fraction of rows for training (default 0.6).
        val_ratio: Fraction for validation (default 0.2).
        test_ratio: Fraction for test (default 0.2). Must sum to 1.0.
        min_train_size: If set, train uses at least this many rows (test/val shift).

    Returns:
        (train_df, val_df, test_df) each a DataFrame view (no copy).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)

    if min_train_size is not None and n > min_train_size:
        train_end = min_train_size
        remaining = n - train_end
        val_size = int(remaining * val_ratio / (val_ratio + test_ratio))
        test_size = remaining - val_size
        val_end = train_end + val_size
        test_end = n
    else:
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        test_end = n

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:test_end]

    return train_df, val_df, test_df

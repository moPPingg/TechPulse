"""
Prepare tabular and sequential datasets from feature DataFrame.
Same feature/target definition for all models; different shapes for tabular vs sequence.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


# Default target for 1-step ahead forecasting
DEFAULT_TARGET = "return_1d"

# Columns we never use as features (identifiers and raw price/volume that may leak)
EXCLUDE_FROM_FEATURES = {"date", "open", "high", "low", "close", "volume", "ticker"}


def get_feature_columns(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> List[str]:
    """List of column names to use as features (numeric, excluding id/target/raw)."""
    exclude = EXCLUDE_FROM_FEATURES | {target}
    cand = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cand


@dataclass
class ForecastDataset:
    """Unified container for train/val/test X and y."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_name: str
    dates_train: Optional[np.ndarray] = None
    dates_val: Optional[np.ndarray] = None
    dates_test: Optional[np.ndarray] = None


def prepare_tabular(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    feature_cols: Optional[List[str]] = None,
    date_col: str = "date",
) -> ForecastDataset:
    """
    Build tabular (X, y) for Linear/XGBoost: each row = one time step.
    Target at row i = next-period return (so we predict forward return).
    """

    def _align_X_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Use row i features to predict target at row i (current period return as target for that row)
        # So we don't use future: X[i] = features at i, y[i] = target at i (return that just realized)
        # For "predict next day return": X[i] = features at i, y[i] = target at i+1
        # So we need to shift target backward: y[i] = target[i+1] -> last row has no y
        if feature_cols is None:
            feats = get_feature_columns(df, target)
        else:
            feats = [c for c in feature_cols if c in df.columns]
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame")
        X = df[feats].values
        # Predict next step: y[i] = target at i+1
        y = df[target].shift(-1).values
        dates = df[date_col].values if date_col in df.columns else None
        # Drop last row (no y)
        n = len(X) - 1
        if n <= 0:
            raise ValueError("Not enough rows for tabular (need at least 2)")
        X = X[:n]
        y = y[:n]
        if dates is not None:
            dates = dates[:n]
        return X.astype(np.float64), y.astype(np.float64), dates

    X_train, y_train, d_train = _align_X_y(train_df)
    X_val, y_val, d_val = _align_X_y(val_df)
    X_test, y_test, d_test = _align_X_y(test_df)

    feats = get_feature_columns(train_df, target) if feature_cols is None else feature_cols
    return ForecastDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feats,
        target_name=target,
        dates_train=d_train,
        dates_val=d_val,
        dates_test=d_test,
    )


def prepare_sequential(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int,
    target: str = DEFAULT_TARGET,
    feature_cols: Optional[List[str]] = None,
    date_col: str = "date",
) -> ForecastDataset:
    """
    Build sequential (samples, seq_len, n_features) for LSTM/PatchTST/Transformer.
    Each sample: X[i] = window of length seq_len ending at row i (exclusive of target time),
    y[i] = target at time i (next step after window).
    """

    def _build_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if feature_cols is None:
            feats = get_feature_columns(df, target)
        else:
            feats = [c for c in feature_cols if c in df.columns]
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame")
        M = df[feats].values.astype(np.float64)
        y = df[target].values.astype(np.float64)
        dates = df[date_col].values if date_col in df.columns else None
        n = len(M)
        if n <= seq_len:
            raise ValueError(f"Need len > seq_len (got {n}, seq_len={seq_len})")
        X_list = []
        y_list = []
        date_list = []
        for i in range(seq_len, n):
            X_list.append(M[i - seq_len : i])
            y_list.append(y[i])  # predict current step (or use i+1 for true next; here we predict same step as in tabular for consistency we could do y[i] = next return; so y[i] = df[target].iloc[i] is "return at i". For "predict next" we want y[i] = df[target].iloc[i] which is return at i - so window ends at i-1, target is i. So window [i-seq_len : i] and y = value at i. Good.
            if dates is not None:
                date_list.append(dates[i])
        X = np.stack(X_list, axis=0)
        y_out = np.array(y_list, dtype=np.float64)
        d_out = np.array(date_list) if date_list else None
        return X, y_out, d_out

    # For train we need to allow using last row of train for first val sample; so we pass full df and slice inside by indices. Actually we're passing already split train/val/test. So train has no access to future; val has no access to test. For val, the first val sample needs seq_len points before it - those can be from train. So we need to build sequences with "warm" history. For simplicity we build sequences only from each split's own data; then the first val sample has history from last seq_len of train. So we need to pass context. Easiest: concatenate train+val+test (in order), build sequences for whole thing, then split by index. So:
    # Full = concat(train, val, test). Build seq: for i in range(seq_len, len(Full)), X[i] = Full[i-seq_len:i], y[i] = Full[target][i]. Then train indices: 0 to len(train)-1 (but we need to map: train is 0:len_train, so samples that end at index in [seq_len, len_train) are train. Val: samples ending in [len_train, len_train+len_val). Test: samples ending in [len_train+len_val, len_train+len_val+len_test).
    full = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    if feature_cols is None:
        feats = get_feature_columns(full, target)
    else:
        feats = [c for c in feature_cols if c in full.columns]
    if target not in full.columns:
        raise ValueError(f"Target column '{target}' not in DataFrame")
    M = full[feats].values.astype(np.float64)
    y_all = full[target].values.astype(np.float64)
    dates_all = full[date_col].values if date_col in full.columns else None
    n_full = len(full)
    n_train, n_val = len(train_df), len(val_df)
    n_test = len(test_df)

    X_list, y_list, d_list = [], [], []
    for i in range(seq_len, n_full):
        X_list.append(M[i - seq_len : i])
        y_list.append(y_all[i])
        if dates_all is not None:
            d_list.append(dates_all[i])
    X_full = np.stack(X_list, axis=0)
    y_full = np.array(y_list, dtype=np.float64)
    d_full = np.array(d_list) if d_list else None

    # Split: train ends at index n_train-1, so sample that ends at n_train-1 has start n_train-1-seq_len. So first train sample index in X_full is 0 when i=seq_len in full corresponds to full index seq_len. So full index i (0-based) maps to X_full index (i - seq_len). Train: full indices [seq_len, n_train) -> X_full indices [0, n_train - seq_len]. Val: full indices [n_train, n_train + n_val) -> X_full indices [n_train - seq_len, n_train - seq_len + n_val]. Test: full indices [n_train + n_val, n_full) -> X_full indices [n_train - seq_len + n_val, n_full - seq_len].
    def _slice(start: int, end: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        x = X_full[start:end]
        y = y_full[start:end]
        d = d_full[start:end] if d_full is not None else None
        return x, y, d

    # start/end in full indices for the *end* of each sample
    train_end_full = n_train
    val_end_full = n_train + n_val
    # X_full index = full index - seq_len
    start_train = seq_len  # first sample ends at full index seq_len
    end_train = n_train
    start_val = n_train
    end_val = n_train + n_val
    start_test = n_train + n_val
    end_test = n_full

    start_train_ix = start_train - seq_len  # 0
    end_train_ix = end_train - seq_len
    start_val_ix = end_train_ix
    end_val_ix = start_val_ix + n_val
    start_test_ix = end_val_ix
    end_test_ix = start_test_ix + n_test

    # Fix: end_train_ix = n_train - seq_len, so we have (n_train - seq_len) train samples. end_val_ix = end_train_ix + n_val = n_train - seq_len + n_val. end_test_ix = end_val_ix + n_test = n_train - seq_len + n_val + n_test = n_full - seq_len. Good.
    X_train, y_train, d_train = _slice(start_train_ix, end_train_ix)
    X_val, y_val, d_val = _slice(start_val_ix, end_val_ix)
    X_test, y_test, d_test = _slice(start_test_ix, end_test_ix)

    return ForecastDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feats,
        target_name=target,
        dates_train=d_train,
        dates_val=d_val,
        dates_test=d_test,
    )


def get_rolling_fold_tabular(
    df: pd.DataFrame,
    t: int,
    target: str = DEFAULT_TARGET,
    feature_cols: Optional[List[str]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    For rolling backtest: at step t we have rows 0..t (inclusive). Build train on 0..t (y = next-step target),
    test on row t (predict target at t+1). Returns (X_train, y_train, X_test_1row, y_test_1row) or None.
    """
    if feature_cols is None:
        feats = get_feature_columns(df, target)
    else:
        feats = [c for c in feature_cols if c in df.columns]
    if target not in df.columns or t + 1 >= len(df):
        return None
    sub = df.iloc[: t + 2]
    X = sub[feats].values.astype(np.float64)
    y = sub[target].shift(-1).values.astype(np.float64)
    X_train = X[: t + 1]
    y_train = y[: t + 1]
    X_test = X[t : t + 1]
    y_test = y[t : t + 1]
    if np.any(np.isnan(y_train)) or np.any(np.isnan(y_test)):
        return None
    return (X_train, y_train, X_test, y_test)
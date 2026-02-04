"""
Data validation for the pipeline: clean/ and features/ stages.

Purpose:
    Validate data *before* modeling so that train/val/test splits and models
    see consistent, leakage-free, time-ordered data.

What to check after clean/:
    - Required columns (date, OHLCV) exist and types are correct.
    - No negatives in price/volume; OHLC logic (high >= open,close,low; low <= ...).
    - No duplicate dates; dates sorted; no unreasonably large gaps.
    - Why it matters: garbage in → garbage features → broken backtests and metrics.

What to check after features/:
    - Feature matrix has no NaNs (or acceptable count), numeric only.
    - Target column exists and has no (or few) NaNs.
    - Time order preserved; sufficient rows for train/val/test.
    - Label leakage: no feature that uses future information (e.g. next-day close).
    - Why it matters: leakage inflates metrics; NaNs break sklearn/torch; wrong order breaks time series.

Usage:
    from src.utils.data_validation import validate_clean_data, validate_feature_data

    report_clean = validate_clean_data(df_clean)
    report_feat  = validate_feature_data(df_features, target="return_1d")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured result of a validation run: passed, warnings, failed, and details."""

    passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "failed": self.failed,
            "details": self.details,
        }

    @property
    def ok(self) -> bool:
        """True if no failed checks."""
        return len(self.failed) == 0

    def raise_if_failed(self, message: str = "Data validation failed") -> None:
        """Raise ValueError with failed checks if any."""
        if self.failed:
            raise ValueError(f"{message}: {self.failed}")


# ---------------------------------------------------------------------------
# Clean-stage validation (after clean/)
# ---------------------------------------------------------------------------

# Expected columns from clean_price.py
CLEAN_REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
CLEAN_PRICE_COLUMNS = ["open", "high", "low", "close"]


def validate_clean_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    max_date_gap_days: int = 30,
    raise_on_fail: bool = False,
) -> ValidationReport:
    """
    Validate DataFrame after the clean/ stage (OHLCV stock price data).

    Checks performed:
        1. Required columns exist (date, open, high, low, close, volume).
        2. date is datetime and sorted ascending.
        3. No duplicate dates.
        4. No negative prices or volume.
        5. OHLC logic: high >= open, close, low; low <= open, close, high.
        6. No zero prices (suspicious).
        7. Date gaps: no gap larger than max_date_gap_days (e.g. missing data).
        8. Basic sanity: at least 2 rows for any time series use.

    Args:
        df: DataFrame produced by clean/ (e.g. clean_price).
        required_columns: Override default CLEAN_REQUIRED_COLUMNS.
        max_date_gap_days: Flag if max gap between consecutive dates exceeds this.
        raise_on_fail: If True, raise ValueError when any check fails.

    Returns:
        ValidationReport with passed/warnings/failed and details.

    Example (time series - stock prices):
        >>> df = pd.read_csv("data/clean/VCB.csv")
        >>> df["date"] = pd.to_datetime(df["date"])
        >>> r = validate_clean_data(df)
        >>> if not r.ok:
        ...     print("Failed:", r.failed)
        >>> print(r.details.get("rows", 0))
    """

    report = ValidationReport()
    required = required_columns or CLEAN_REQUIRED_COLUMNS

    # --- 1. Required columns ---
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.failed.append(f"Missing required columns: {missing}")
        report.details["missing_columns"] = missing
        if raise_on_fail:
            report.raise_if_failed("validate_clean_data")
        return report
    report.passed.append("All required columns present")

    # --- 2. Non-empty, minimum rows ---
    n = len(df)
    report.details["rows"] = n
    if n == 0:
        report.failed.append("DataFrame is empty")
        if raise_on_fail:
            report.raise_if_failed("validate_clean_data")
        return report
    if n < 2:
        report.warnings.append("Fewer than 2 rows; time series and splits will be invalid")
    else:
        report.passed.append("Sufficient rows for time series")

    # --- 3. date: type and sort ---
    if "date" not in df.columns:
        if raise_on_fail:
            report.raise_if_failed("validate_clean_data")
        return report
    try:
        dates = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        report.failed.append(f"date column not parseable: {e}")
        if raise_on_fail:
            report.raise_if_failed("validate_clean_data")
        return report
    null_dates = dates.isna().sum()
    if null_dates > 0:
        report.failed.append(f"date has {null_dates} null/invalid values")
        report.details["null_dates"] = int(null_dates)
    else:
        report.passed.append("date column valid (no nulls)")
    df_sorted = df.sort_values("date").reset_index(drop=True)
    if not df_sorted["date"].equals(df["date"].reset_index(drop=True)):
        report.warnings.append("date was not sorted ascending; use sort_values('date') before modeling")
    report.details["date_min"] = str(dates.min())
    report.details["date_max"] = str(dates.max())

    # --- 4. Duplicate dates ---
    dup = df["date"].duplicated().sum()
    if dup > 0:
        report.failed.append(f"Found {dup} duplicate dates (one row per date expected)")
        report.details["duplicate_dates"] = int(dup)
    else:
        report.passed.append("No duplicate dates")

    # --- 5. Negative prices and volume ---
    for col in CLEAN_PRICE_COLUMNS:
        if col not in df.columns:
            continue
        neg = (df[col] < 0).sum()
        if neg > 0:
            report.failed.append(f"Column '{col}' has {neg} negative values")
            report.details[f"negative_{col}"] = int(neg)
    if "volume" in df.columns:
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol > 0:
            report.failed.append(f"volume has {neg_vol} negative values")
            report.details["negative_volume"] = int(neg_vol)
    if not any("negative" in f for f in report.failed):
        report.passed.append("No negative price or volume")

    # --- 6. OHLC logic ---
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        high_ok = (df["high"] >= df["open"]) & (df["high"] >= df["close"]) & (df["high"] >= df["low"])
        low_ok = (df["low"] <= df["open"]) & (df["low"] <= df["close"]) & (df["low"] <= df["high"])
        high_viol = (~high_ok).sum()
        low_viol = (~low_ok).sum()
        if high_viol > 0 or low_viol > 0:
            report.failed.append(
                f"OHLC logic violated: high not max in {high_viol} rows, low not min in {low_viol} rows"
            )
            report.details["ohlc_high_violations"] = int(high_viol)
            report.details["ohlc_low_violations"] = int(low_viol)
        else:
            report.passed.append("OHLC logic consistent")
    # --- 7. Zero prices (suspicious) ---
    for col in CLEAN_PRICE_COLUMNS:
        if col not in df.columns:
            continue
        z = (df[col] == 0).sum()
        if z > 0:
            report.warnings.append(f"Column '{col}' has {z} zero values (suspicious for price)")
            report.details[f"zero_{col}"] = int(z)

    # --- 8. Date gaps ---
    if "date" in df.columns and n > 1:
        d = pd.to_datetime(df_sorted["date"])
        gap = d.diff().dt.days
        max_gap = gap.max()
        if pd.isna(max_gap):
            pass
        elif max_gap > max_date_gap_days:
            report.warnings.append(
                f"Maximum date gap is {int(max_gap)} days (>{max_date_gap_days}); possible missing data"
            )
            report.details["max_date_gap_days"] = int(max_gap)
        else:
            report.passed.append("Date gaps within acceptable range")

    # Emit Python warnings for failures so callers see them even if they don't check report
    for msg in report.failed:
        warnings.warn(f"[validate_clean_data] {msg}", UserWarning, stacklevel=2)
    for msg in report.warnings:
        warnings.warn(f"[validate_clean_data] {msg}", UserWarning, stacklevel=2)

    if raise_on_fail and report.failed:
        report.raise_if_failed("validate_clean_data")
    return report


# ---------------------------------------------------------------------------
# Feature-stage validation (after features/)
# ---------------------------------------------------------------------------

# Columns we do not use as features (identifiers / raw that can leak or are not predictors)
EXCLUDE_FROM_FEATURES = {"date", "open", "high", "low", "close", "volume", "ticker"}

# Substrings in column names that suggest future / target (potential leakage)
LEAKAGE_NAME_PATTERNS = [
    "next", "future", "target", "tomorrow", "forward",
]


def _get_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    """Numeric columns that are not id/raw/target."""
    out = [
        c for c in df.columns
        if c != target and c not in EXCLUDE_FROM_FEATURES and pd.api.types.is_numeric_dtype(df[c])
    ]
    return out


def validate_feature_data(
    df: pd.DataFrame,
    target: str = "return_1d",
    date_col: str = "date",
    min_rows: int = 100,
    max_nan_frac: float = 0.0,
    leakage_corr_threshold: float = 0.99,
    raise_on_fail: bool = False,
) -> ValidationReport:
    """
    Validate DataFrame after the features/ stage (tabular/ML and time series).

    Checks performed:
        1. Target column exists and is numeric.
        2. At least one feature column (numeric, not id/raw/target).
        3. No (or acceptable) NaN in feature columns and target.
        4. Rows sufficient for train/val/test (e.g. >= min_rows).
        5. If date_col present: sorted ascending (time order).
        6. Label leakage: no feature with suspicious name; no feature almost perfectly correlated with target.

    Why leakage matters:
        If a feature uses future information (e.g. next day return), model will look good in backtest
        but fail in production. We check: (a) names that look like "next"/"future", (b) correlation
        with target near ±1 (feature might be target in disguise or shifted target).

    Args:
        df: DataFrame produced by features/ (e.g. build_features).
        target: Target column name (default return_1d for 1-day ahead return).
        date_col: Name of date column for time-order check.
        min_rows: Minimum number of rows to consider data valid for splitting.
        max_nan_frac: Max allowed fraction of NaNs in feature columns (0 = no NaNs).
        leakage_corr_threshold: If abs(correlation(feature, target)) >= this, flag as possible leakage.
        raise_on_fail: If True, raise ValueError when any check fails.

    Returns:
        ValidationReport with passed/warnings/failed and details.

    Example (tabular ML):
        >>> df = pd.read_csv("data/features/vn30/VCB.csv")
        >>> df["date"] = pd.to_datetime(df["date"])
        >>> r = validate_feature_data(df, target="return_1d")
        >>> print(r.to_dict())

    Example (label leakage detection):
        >>> # If you accidentally added 'return_1d_next' as feature:
        >>> r = validate_feature_data(df, target="return_1d")
        >>> # r.warnings or r.failed will mention leakage
    """

    report = ValidationReport()
    report.details["target"] = target
    report.details["rows"] = len(df)
    report.details["columns"] = list(df.columns)

    # --- 1. Target exists and numeric ---
    if target not in df.columns:
        report.failed.append(f"Target column '{target}' not found")
        if raise_on_fail:
            report.raise_if_failed("validate_feature_data")
        return report
    if not pd.api.types.is_numeric_dtype(df[target]):
        report.failed.append(f"Target '{target}' is not numeric")
        if raise_on_fail:
            report.raise_if_failed("validate_feature_data")
        return report
    report.passed.append("Target column present and numeric")

    # --- 2. Feature columns ---
    feature_cols = _get_feature_columns(df, target)
    report.details["feature_columns"] = feature_cols
    report.details["n_features"] = len(feature_cols)
    if len(feature_cols) == 0:
        report.failed.append("No feature columns (numeric, excluding id/raw/target)")
        if raise_on_fail:
            report.raise_if_failed("validate_feature_data")
        return report
    report.passed.append(f"Found {len(feature_cols)} feature columns")

    # --- 3. NaN in features and target ---
    nan_feat = df[feature_cols].isna().sum().sum()
    n_cells = len(df) * len(feature_cols)
    nan_frac = nan_feat / n_cells if n_cells else 0
    report.details["nan_count_features"] = int(nan_feat)
    report.details["nan_frac_features"] = round(nan_frac, 4)
    if nan_frac > max_nan_frac:
        report.failed.append(
            f"Feature matrix has {nan_frac:.2%} NaNs (max allowed {max_nan_frac:.2%})"
        )
    else:
        report.passed.append("Feature NaN fraction within limit")
    target_nan = df[target].isna().sum()
    if target_nan > 0:
        report.warnings.append(f"Target has {target_nan} NaNs (will drop or impute before training)")
        report.details["target_nan_count"] = int(target_nan)
    else:
        report.passed.append("Target has no NaNs")

    # --- 4. Minimum rows ---
    n = len(df)
    if n < min_rows:
        report.warnings.append(
            f"Only {n} rows (recommend >= {min_rows} for train/val/test splits)"
        )
        report.details["min_rows_required"] = min_rows
    else:
        report.passed.append(f"Row count >= {min_rows}")

    # --- 5. Time order (if date present) ---
    if date_col in df.columns:
        try:
            d = pd.to_datetime(df[date_col], errors="coerce")
            sorted_d = d.sort_values()
            if not d.reset_index(drop=True).equals(sorted_d.reset_index(drop=True)):
                report.warnings.append(
                    f"'{date_col}' not sorted ascending; time-based splits assume ascending order"
                )
            else:
                report.passed.append("Date column sorted ascending")
        except Exception as e:
            report.warnings.append(f"Could not check time order: {e}")
    else:
        report.warnings.append("No date column; time order not checked")

    # --- 6. Label leakage: name patterns ---
    suspicious = [
        c for c in feature_cols
        if any(p in c.lower() for p in LEAKAGE_NAME_PATTERNS)
    ]
    if target in feature_cols:
        suspicious.append(target)
    if suspicious:
        report.failed.append(
            f"Possible label leakage (suspicious feature names): {suspicious}"
        )
        report.details["leakage_suspicious_names"] = suspicious
    else:
        report.passed.append("No suspicious feature names (leakage by name)")

    # --- 7. Label leakage: correlation with target ---
    try:
        y = df[target].dropna()
        valid_idx = y.index
        X_sub = df.loc[valid_idx, feature_cols]
        # Drop rows where any feature is NaN for correlation
        valid_mask = ~X_sub.isna().any(axis=1)
        X_sub = X_sub.loc[valid_mask]
        y_sub = y.loc[valid_mask]
        if len(y_sub) < 10:
            report.warnings.append("Too few rows to check correlation-based leakage")
        else:
            corrs = X_sub.corrwith(y_sub).abs()
            near_perfect = corrs[corrs >= leakage_corr_threshold]
            if len(near_perfect) > 0:
                report.failed.append(
                    f"Possible label leakage (|corr with target| >= {leakage_corr_threshold}): "
                    f"{near_perfect.to_dict()}"
                )
                report.details["leakage_high_corr"] = near_perfect.to_dict()
            else:
                report.passed.append("No feature with near-perfect correlation to target")
    except Exception as e:
        report.warnings.append(f"Could not run correlation leakage check: {e}")
        report.details["leakage_check_error"] = str(e)

    for msg in report.failed:
        warnings.warn(f"[validate_feature_data] {msg}", UserWarning, stacklevel=2)
    for msg in report.warnings:
        warnings.warn(f"[validate_feature_data] {msg}", UserWarning, stacklevel=2)

    if raise_on_fail and report.failed:
        report.raise_if_failed("validate_feature_data")
    return report


# ---------------------------------------------------------------------------
# Convenience: run both validations
# ---------------------------------------------------------------------------

def validate_pipeline_data(
    df_clean: Optional[pd.DataFrame] = None,
    df_features: Optional[pd.DataFrame] = None,
    target: str = "return_1d",
    **kwargs: Any,
) -> Tuple[Optional[ValidationReport], Optional[ValidationReport]]:
    """
    Run validate_clean_data and/or validate_feature_data.

    Args:
        df_clean: If provided, run validate_clean_data on it.
        df_features: If provided, run validate_feature_data on it (with target).
        target: Target column for feature validation.
        **kwargs: Passed to validate_* (e.g. raise_on_fail).

    Returns:
        (report_clean, report_features); either can be None if corresponding df not provided.

    Example:
        >>> r_clean, r_feat = validate_pipeline_data(df_clean=df_clean, df_features=df_feat)
        >>> assert r_clean.ok and r_feat.ok, "Validation failed"
    """
    r_clean = validate_clean_data(df_clean, **kwargs) if df_clean is not None else None
    r_feat = (
        validate_feature_data(df_features, target=target, **kwargs)
        if df_features is not None else None
    )
    return r_clean, r_feat

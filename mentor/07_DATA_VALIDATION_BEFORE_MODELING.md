# Data Validation Before Modeling

Step-by-step guide: what to check after **clean/** and **features/**, why it matters for ML and time series, and how to use the reusable validation module.

---

## 1. What to Check After clean/ Stage

**Output of clean/:** One DataFrame per symbol with columns: `date`, `open`, `high`, `low`, `close`, `volume` (and optionally `ticker`). Rows = one row per trading day, sorted by date.

### Why each check matters

| Check | Why it matters for ML / time series |
|-------|-------------------------------------|
| **Required columns** | Models and feature code expect `date`, OHLCV. Missing column → crash or wrong feature. |
| **date type & order** | Splits (train/val/test) and backtests assume **ascending time**. Wrong order = data leakage (future in train). |
| **No duplicate dates** | One row per date; duplicates bias averages and break “one step ahead” logic. |
| **No negative price/volume** | Invalid raw data; indicators (returns, volatility) become nonsense. |
| **OHLC logic** (high ≥ open,close,low; low ≤ …) | Violations = bad data or adjusted series; models learn noise. |
| **No zero prices** | Zero close → infinite returns; breaks scaling and models. |
| **Date gaps** | Large gaps = missing data; rolling windows and sequence models see wrong history. |
| **Minimum rows** | Too few rows → no room for train/val/test or rolling window. |

### Checklist after clean/

- [ ] All of `date`, `open`, `high`, `low`, `close`, `volume` exist.
- [ ] `date` is datetime, no nulls, sorted ascending.
- [ ] No duplicate dates.
- [ ] No negative values in price columns or volume.
- [ ] For each row: `high` ≥ open, close, low; `low` ≤ open, close, high.
- [ ] No zero in `open`, `high`, `low`, `close`.
- [ ] Max gap between consecutive dates &lt; threshold (e.g. 30 days) or acceptable.
- [ ] At least 2 rows (prefer 100+ for modeling).

### Red flags and how to fix them

| Red flag | Fix |
|----------|-----|
| Missing columns | Fix upstream (crawler/clean) or standardize column names in `clean_price`. |
| date not sorted | `df = df.sort_values('date').reset_index(drop=True)` before any split. |
| Duplicate dates | `df = df.drop_duplicates(subset=['date'], keep='first')` (or last, by policy). |
| Negative price/volume | Drop or correct bad rows; fix source. |
| OHLC violated | Drop row or fix (e.g. high = max(open, high, low, close)). |
| Zero price | Drop row or treat as missing; fix source. |
| Huge date gap | Accept or fill (forward fill / mark as missing); document. |
| Too few rows | Get more history or skip symbol for modeling. |

### Concrete Python (pandas) after clean/

```python
import pandas as pd
from src.utils.data_validation import validate_clean_data, ValidationReport

# Load cleaned OHLCV (e.g. from clean_price output)
df = pd.read_csv("data/clean/VCB.csv")
df["date"] = pd.to_datetime(df["date"])

report = validate_clean_data(df)
print("Passed:", report.passed)
print("Warnings:", report.warnings)
print("Failed:", report.failed)
print("Details:", report.details)

if not report.ok:
    # Option 1: fix and re-run
    # Option 2: raise
    report.raise_if_failed("Clean data invalid")
```

Manual checks (same ideas as in the module):

```python
# Required columns
assert set(["date", "open", "high", "low", "close", "volume"]).issubset(df.columns)

# No negatives
assert (df[["open", "high", "low", "close"]] >= 0).all().all()
assert (df["volume"] >= 0).all()

# OHLC
assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all()
assert (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all()

# Sorted, no duplicate dates
assert df["date"].is_monotonic_increasing
assert df["date"].duplicated().sum() == 0
```

---

## 2. What to Check After features/ Stage

**Output of features/:** Same rows as clean (or fewer if you drop NaNs), plus columns like `return_1d`, `ma_5`, `volatility_5`, `rsi_14`, etc. One of these (e.g. `return_1d`) is the **target** for forecasting.

### Why each check matters

| Check | Why it matters for ML / time series |
|-------|-------------------------------------|
| **Target exists and is numeric** | Training and evaluation need a well-defined target. |
| **Enough feature columns** | No features → no model inputs. |
| **NaN in features/target** | sklearn/torch often require no NaN; or explicit imputation. |
| **Enough rows** | Need train + val + test (e.g. 60/20/20); too few = overfit or no test. |
| **Time order** | Same as clean: splits and backtests assume ascending time. |
| **Label leakage (name)** | Features named "next_return" or "future_close" use future info → fake performance. |
| **Label leakage (correlation)** | A feature ≈ target (e.g. copy or 1-step shift) inflates metrics. |

### Checklist after features/

- [ ] Target column (e.g. `return_1d`) exists and is numeric.
- [ ] At least one numeric feature column (excluding date, OHLCV, ticker, target).
- [ ] NaN in feature matrix within policy (e.g. 0%); target NaN handled (drop or impute).
- [ ] Row count ≥ minimum (e.g. 100) for splits.
- [ ] If `date` present: sorted ascending.
- [ ] No feature name suggests future (next, future, tomorrow, forward, target).
- [ ] No feature has |correlation with target| ≈ 1 (leakage).

### Red flags and how to fix them

| Red flag | Fix |
|----------|-----|
| Target missing | Add target in feature build (e.g. `return_1d = close.pct_change()`) or fix column name. |
| No feature columns | Ensure build_features runs and excludes only id/raw/target (see evaluation/data.py). |
| Too many NaNs | Increase warmup (drop first N rows) or reduce rolling windows; or impute. |
| Suspicious feature name | Rename or remove (e.g. drop "return_next_day" if it’s the target in disguise). |
| Near-perfect corr with target | Remove that feature (likely target copy or future info). |

### Concrete Python (pandas) after features/

```python
import pandas as pd
from src.utils.data_validation import validate_feature_data

df = pd.read_csv("data/features/vn30/VCB.csv")
df["date"] = pd.to_datetime(df["date"])

report = validate_feature_data(df, target="return_1d")
print(report.to_dict())
if not report.ok:
    report.raise_if_failed("Feature data invalid")
```

Example for **label leakage detection**: if you accidentally add a “next day return” column as feature:

```python
# BAD: feature that is the target shifted (leakage)
df["return_1d_next"] = df["return_1d"].shift(-1)  # future info!
report = validate_feature_data(df, target="return_1d")
# report.failed or report.warnings will mention leakage (name or high correlation)
```

---

## 3. Reusable Module: `src/utils/data_validation.py`

### Functions

| Function | Purpose |
|----------|---------|
| `validate_clean_data(df, ...)` | Validates OHLCV after clean/. Returns `ValidationReport`. |
| `validate_feature_data(df, target="return_1d", ...)` | Validates feature matrix and target; checks leakage. Returns `ValidationReport`. |
| `validate_pipeline_data(df_clean=..., df_features=..., target=...)` | Runs one or both; returns `(report_clean, report_features)`. |

### ValidationReport

- **passed**: list of passed check messages.
- **warnings**: list of warning messages (e.g. zero prices, few rows).
- **failed**: list of failed check messages (report.ok is False if any).
- **details**: dict with counts, column names, correlations, etc.
- **raise_if_failed(msg)**: raises `ValueError` if there are failed checks.

All functions emit `warnings.warn()` for failed and warning checks so you see them even if you don’t inspect the report.

### Parameters (short)

- **validate_clean_data**: `required_columns`, `max_date_gap_days`, `raise_on_fail`.
- **validate_feature_data**: `target`, `date_col`, `min_rows`, `max_nan_frac`, `leakage_corr_threshold`, `raise_on_fail`.

---

## 4. Examples

### Time series (stock prices) – after clean

```python
import pandas as pd
from src.utils.data_validation import validate_clean_data

df = pd.read_csv("data/clean/FPT.csv")
df["date"] = pd.to_datetime(df["date"])
r = validate_clean_data(df, max_date_gap_days=30)
assert r.ok, r.failed
```

### Tabular ML – after features

```python
import pandas as pd
from src.utils.data_validation import validate_feature_data

df = pd.read_csv("data/features/vn30/VCB.csv")
df["date"] = pd.to_datetime(df["date"])
r = validate_feature_data(df, target="return_1d", min_rows=100)
if not r.ok:
    print("Failed:", r.failed)
print("Details:", r.details)
```

### Label leakage detection

- **By name:** Any feature whose name contains `next`, `future`, `tomorrow`, `forward`, or `target` is flagged.
- **By correlation:** Any feature with `|corr(feature, target)| >= leakage_corr_threshold` (default 0.99) is flagged.

Fix: remove or rename the feature, or fix the feature engineering so it doesn’t use future information.

---

## 5. How to Integrate Into Your Pipeline

### Option A: After clean (in script or in clean_price)

```python
# In a script that runs clean_many or after clean_price()
from src.clean.clean_price import clean_price
from src.utils.data_validation import validate_clean_data

df = clean_price("data/raw/VCB.csv", "data/clean/VCB.csv")
report = validate_clean_data(df, raise_on_fail=True)  # stop if invalid
```

### Option B: After features (in script or in build_features_single)

```python
# After build_features() or build_features_single()
from src.features.build_features import build_features_single
from src.utils.data_validation import validate_feature_data

df = build_features_single("VCB.csv", clean_dir="data/clean", features_dir="data/features/vn30")
if df is not None:
    report = validate_feature_data(df, target="return_1d", raise_on_fail=False)
    if not report.ok:
        print("Fix before training:", report.failed)
```

### Option C: Before evaluation (in run_forecasting_pipeline or before get_train_val_test_splits)

```python
from src.evaluation.splits import get_train_val_test_splits
from src.utils.data_validation import validate_feature_data

df = pd.read_csv("data/features/vn30/VCB.csv")
df["date"] = pd.to_datetime(df["date"])
validate_feature_data(df, target="return_1d", raise_on_fail=True)
train_df, val_df, test_df = get_train_val_test_splits(df, date_col="date")
```

### Option D: Validate both stages at once

```python
from src.utils.data_validation import validate_pipeline_data

df_clean = pd.read_csv("data/clean/VCB.csv")
df_clean["date"] = pd.to_datetime(df_clean["date"])
df_feat = pd.read_csv("data/features/vn30/VCB.csv")
df_feat["date"] = pd.to_datetime(df_feat["date"])

r_clean, r_feat = validate_pipeline_data(df_clean=df_clean, df_features=df_feat, target="return_1d")
assert r_clean.ok and r_feat.ok, (r_clean.failed, r_feat.failed)
```

---

## 6. Summary

1. **After clean/:** Use `validate_clean_data(df)` to check OHLCV, types, order, duplicates, OHLC logic, gaps.
2. **After features/:** Use `validate_feature_data(df, target="return_1d")` to check target, features, NaNs, time order, and leakage (name + correlation).
3. Both return a **ValidationReport** (passed/warnings/failed/details) and emit **warnings**; use **raise_on_fail=True** to stop the pipeline when validation fails.
4. Fix red flags before training so splits and metrics are valid and leakage-free.

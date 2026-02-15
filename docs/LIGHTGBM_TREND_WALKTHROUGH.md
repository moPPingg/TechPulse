# LightGBM Trend Pipeline — Line-by-Line Walkthrough for Beginners

This document explains `src/models/lightgbm_trend.py` slowly and concretely: what problem it solves, how the pipeline runs, each function’s inputs/outputs and purpose, pseudocode for the whole flow, and three small reimplementation exercises.

---

## 1. What problem is this file solving?

**In one sentence:** The file trains a **classifier** that predicts whether a stock’s **next-day return** will be “up” or “down” (or up/neutral/down in 3-class mode), using **today’s features** (and optionally news), so you can use it for trend signals or research.

**Concretely:**

- You have a table where each **row = one trading day** for one stock (e.g. FPT).
- Each row has: **date**, and many **features** (e.g. returns, moving averages, RSI, volatility, volume, and optionally news scores).
- You want to predict: **tomorrow’s** move: up (1) or down (0), or up/neutral/down (2/1/0).
- The file:
  - Loads that table (price features + optional news).
  - Defines **which column is the target** (next-day return) and **which columns are features**.
  - Splits data **in time** (no shuffling) into train / validation / test.
  - Converts **continuous next-day return** into **trend labels** (binary or 3-class).
  - Trains a **LightGBM** model, optionally **tunes** it, then **evaluates** it and gets **feature importance**.

So the problem is: **supervised classification of next-day stock trend from current-day (and optionally news) features**, with a time-aware, production-style pipeline.

---

## 2. Pipeline in execution order

When you call `LightGBMTrendPipeline(...).run()`, things happen in this order:

1. **load_data()**  
   Load the feature CSV for the symbol, optionally load news and merge by date. Decide which columns are features (store in `self.feature_cols`).

2. **split()**  
   Split the full DataFrame into `train_df`, `val_df`, `test_df` by **time** (e.g. 60% / 20% / 20%), no shuffle.

3. **preprocess(train_df, val_df, test_df)**  
   From each split, build **X** (feature matrix) and **y** (target). Target = next-day return, then converted to trend labels. Scale features using **only** the train set; fill missing values. Store `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` (and scaler).

4. **train()** (or **run_tuning()** then **train()**)  
   Train LightGBM on `(X_train, y_train)`, optionally use `(X_val, y_val)` for early stopping. Store the trained model in `self.model`.

5. **evaluate()**  
   Predict on `X_test`, compare to `y_test`, compute accuracy, precision, recall, F1, ROC-AUC, confusion matrix. Store in `self.metrics`.

6. **get_feature_importance()**  
   Ask LightGBM for each feature’s importance (e.g. gain). Store in `self.importance_df`.

So the **execution order** is: load → split → preprocess → train (maybe tune first) → evaluate → importance.

---

## 3. Each function: what goes in, what comes out, why it exists

### Top of file (lines 15–44)

- **`from __future__ import annotations`**  
  Lets you use type hints like `-> pd.DataFrame` and forward references (e.g. `"lgb.LGBMModel"`) without runtime errors in older Python.

- **Imports**  
  `logging`, `Path`, typing, `numpy`, `pandas`, optional `lightgbm`, `StandardScaler`, and sklearn metrics. So: **in** = nothing at call time; **out** = you have the names available. **Why:** the rest of the code uses these.

- **`logger = logging.getLogger(__name__)`**  
  **In:** nothing. **Out:** a logger named after this module. **Why:** so you can log (e.g. “Loaded 2720 rows…”) without printing.

- **`DEFAULT_RETURN_COL = "return_1d"`**  
  **In:** nothing. **Out:** the default column name used as the “return” that becomes the next-day target. **Why:** one place to change if your CSV uses another name (e.g. `return_1d_adj`).

---

### 1) Data loading and merge

**`load_price_features(symbol, features_dir, from_date=None, to_date=None)`**

- **In:**  
  - `symbol`: e.g. `"FPT"`.  
  - `features_dir`: folder path where CSVs live (e.g. `data/features/vn30`).  
  - `from_date`, `to_date`: optional string dates to filter rows.

- **Out:**  
  One pandas DataFrame: one row per date, columns = date + all CSV columns (e.g. open, high, low, close, volume, return_1d, ma_*, rsi, volatility_*, …). Sorted by date, index reset.

- **Why:**  
  The model needs one table per symbol; this is the single entry point to get that table from disk and optional date range.

**`load_news_daily_features(symbol, from_date, to_date, config_path=None)`**

- **In:**  
  Symbol and date range; optional config path for the news engine.

- **Out:**  
  DataFrame with at least `date` and numeric columns (e.g. composite_score, article_count, avg_sentiment, avg_relevance). Empty DataFrame if no data.

- **Why:**  
  So you can add “news” as extra features per day; the pipeline can run with or without this.

**`merge_multimodal(price_df, news_df, news_prefix="news_")`**

- **In:**  
  - `price_df`: the table from `load_price_features`.  
  - `news_df`: the table from `load_news_daily_features`.  
  - `news_prefix`: string to prepend to news column names.

- **Out:**  
  One DataFrame: all columns of `price_df` plus news columns (with prefix). Rows aligned by **date** (left-join: keep every price date; missing news → NaN).

- **Why:**  
  So price and news live in one table; prefix avoids name clashes (e.g. `news_composite_score`).

---

### 2) Target and feature columns

**`build_trend_target(df, return_col, threshold_up, threshold_down, n_classes)`**

- **In:**  
  - `df`: DataFrame with at least `return_col` (e.g. `return_1d`).  
  - `return_col`: which column is the daily return.  
  - `threshold_up`, `threshold_down`: used for 3-class (up / neutral / down).  
  - `n_classes`: 2 (up/down) or 3 (up/neutral/down).

- **Out:**  
  A 1D numpy array of integer labels. Length = `len(df) - 1` (last row has no “next” return).  
  - Binary: 1 if next-day return > threshold_up, else 0.  
  - 3-class: 2 = up, 1 = neutral, 0 = down, based on thresholds.

- **Why:**  
  The model needs **labels** (0/1 or 0/1/2), not raw returns. This converts “next-day return” into those labels. It uses `shift(-1)` so row *i*’s label = return at *i*+1.

**`get_feature_columns_for_trend(df, exclude=None)`**

- **In:**  
  - `df`: full merged DataFrame.  
  - `exclude`: optional list of extra column names to skip.

- **Out:**  
  List of column names that are **numeric** and not in the “do not use” set (date, open, high, low, close, volume, ticker, plus `exclude`). So: the columns that will become the model’s **X**.

- **Why:**  
  You must not use date as a feature, and usually not raw price/volume (leak or non-stationarity). This centralizes “what is a feature” so the rest of the pipeline stays consistent.

---

### 3) Time-aware splitting

**`time_aware_split(df, date_col, train_ratio, val_ratio, test_ratio, purge_gap)`**

- **In:**  
  - `df`: full DataFrame (e.g. after load + merge).  
  - `date_col`: column used to order (usually `"date"`).  
  - `train_ratio`, `val_ratio`, `test_ratio`: e.g. 0.6, 0.2, 0.2 (must sum to 1).  
  - `purge_gap`: optional number of rows to drop between train/val and val/test (to avoid overlap when target is overlapping, e.g. 5-day return).

- **Out:**  
  Three DataFrames: `train_df`, `val_df`, `test_df`. Consecutive in time; no shuffle.

- **Why:**  
  In time series you must not use future data in training. So split by time and never shuffle.

---

### 4) Preprocessing

**`preprocess_splits(train_df, val_df, test_df, feature_cols, return_col, fit_scaler_on_train, fill_missing)`**

- **In:**  
  - The three split DataFrames.  
  - `feature_cols`: list from `get_feature_columns_for_trend`.  
  - `return_col`: e.g. `return_1d`.  
  - `fit_scaler_on_train`: if True, fit StandardScaler only on train.  
  - `fill_missing`: value for NaN (e.g. 0.0).

- **Out:**  
  Seven things:  
  - `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` (numpy arrays).  
  - `scaler`: the fitted StandardScaler (or None).  
  `y_*` here are **continuous** (next-day return); the pipeline later converts them to trend labels with `continuous_to_trend_labels`.

  Internally it drops the last row of each split (no “next” for that date), fills NaN, replaces inf, clips huge values, then optionally scales.

- **Why:**  
  Models need numeric X and y; scaling only on train avoids leakage; handling NaN/inf keeps training stable.

**`continuous_to_trend_labels(y_continuous, threshold_up, threshold_down, n_classes)`**

- **In:**  
  - `y_continuous`: 1D array of next-day returns.  
  - Same thresholds and `n_classes` as above.

- **Out:**  
  1D array of integers (0/1 for binary, or 0/1/2 for 3-class).

- **Why:**  
  Preprocessing gives continuous `y`; the classifier needs discrete labels. This does the same mapping as `build_trend_target` but on an already-extracted `y` array.

---

### 5) Training, evaluation, importance

**`train_lightgbm(X_train, y_train, X_val, y_val, feature_names, params, num_boost_round, early_stopping_rounds, n_classes)`**

- **In:**  
  - Train (and optionally val) feature matrices and labels.  
  - Optional feature names, custom params, rounds, early stopping, and number of classes.

- **Out:**  
  A trained LightGBM model object (e.g. `lgb.Booster`).

- **Why:**  
  Single place to configure and run LightGBM so the pipeline and tuning share the same setup.

**`evaluate_trend(y_true, y_pred, y_prob, n_classes)`**

- **In:**  
  - `y_true`: true labels.  
  - `y_pred`: predicted labels (0/1 or 0/1/2).  
  - `y_prob`: predicted probabilities (optional; needed for ROC-AUC).  
  - `n_classes`: 2 or 3.

- **Out:**  
  Dict: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `confusion_matrix` (list of lists).

- **Why:**  
  One consistent way to measure trend classification so you can compare models or runs.

**`get_feature_importance(model, importance_type, feature_names)`**

- **In:**  
  Trained LightGBM model, type (e.g. `"gain"`), and list of feature names (same order as in training).

- **Out:**  
  DataFrame with columns `feature` and `importance`, sorted by importance descending.

- **Why:**  
  So you can see which inputs (e.g. volatility_5, volume_change) the model uses most.

---

### 6) Hyperparameter tuning

**`tune_lightgbm(X_train, y_train, X_val, y_val, feature_names, n_trials, timeout, n_classes)`**

- **In:**  
  Same data as training; number of Optuna trials and optional timeout.

- **Out:**  
  A tuple: `(best_params dict, retrained LightGBM model)`.

- **Why:**  
  To search learning rate, num_leaves, feature_fraction, etc., and return one best model for the pipeline to use.

---

### 7) Pipeline class

**`LightGBMTrendPipeline`**

- **In (at construction):**  
  Symbol, features_dir, date range, whether to include news, return column, thresholds, n_classes, split ratios, purge_gap, scale or not, lgb params, tune or not, tune_trials.

- **Out (over time):**  
  After `run()`: `self.df`, `self.feature_cols`, `self.X_train`, `self.y_train`, … , `self.model`, `self.metrics`, `self.importance_df`.

- **Why:**  
  One object that holds config and state so you can call `load_data()`, `split()`, `preprocess()`, `train()`, `evaluate()`, `get_feature_importance()` in order (or `run()` to do all).

---

## 4. Whole pipeline in simple pseudocode

```text
FUNCTION run():
    df = load_price_features(symbol, features_dir, from_date, to_date)
    IF include_news:
        news_df = load_news_daily_features(symbol, from_date, to_date)
        df = merge_multimodal(df, news_df)

    feature_cols = get_feature_columns_for_trend(df, exclude=[return_col])

    train_df, val_df, test_df = time_aware_split(df, train_ratio, val_ratio, test_ratio, purge_gap)

    (X_train, y_train_cont, X_val, y_val_cont, X_test, y_test_cont, scaler) =
        preprocess_splits(train_df, val_df, test_df, feature_cols, return_col)

    y_train = continuous_to_trend_labels(y_train_cont, threshold_up, threshold_down, n_classes)
    y_val   = continuous_to_trend_labels(y_val_cont,   threshold_up, threshold_down, n_classes)
    y_test  = continuous_to_trend_labels(y_test_cont,  threshold_up, threshold_down, n_classes)

    IF tune:
        best_params, model = tune_lightgbm(X_train, y_train, X_val, y_val, ...)
    ELSE:
        model = train_lightgbm(X_train, y_train, X_val, y_val, ...)

    y_pred = model.predict(X_test)
    y_prob = (same as y_pred for probabilities)
    metrics = evaluate_trend(y_test, y_pred, y_prob, n_classes)

    importance_df = get_feature_importance(model, feature_names=feature_cols)

    RETURN metrics
```

So: load → merge (optional) → pick features → split in time → preprocess (X, y_continuous, scale) → turn y_continuous into labels → train (or tune then train) → evaluate on test → importance.

---

## 5. Three small coding exercises

These ask you to reimplement **parts** of this file from scratch (no copy-paste from the real code). Use only pandas/numpy and standard Python unless stated.

---

### Exercise 1: Build trend labels from a return column

**Goal:** Recreate the core of `build_trend_target` and `continuous_to_trend_labels`.

- **Input:** A 1D numpy array `returns` of length N (e.g. daily returns), and two floats `threshold_up` and `threshold_down` (e.g. 0.0 and -0.001).
- **Output:** A 1D numpy array `labels` of length **N − 1** of integers:
  - For each index `i` in `0 .. N-2`, the label at `i` is based on **returns[i+1]** (the “next day” return):
    - If `returns[i+1] > threshold_up` → 1 (up).
    - If `returns[i+1] < threshold_down` → 0 (down).
    - Otherwise → 0 for binary (treat as down), or extend later to 1 for “neutral” in 3-class.
- **Constraint:** No pandas; only numpy. No `shift`; use indexing (e.g. `returns[1:]`).
- **Check:** For `returns = np.array([0.01, -0.005, 0.02, 0.0])` and thresholds 0.0 and -0.001, your labels for “next day” should be: down, up, neutral (or down if you do binary only).

---

### Exercise 2: Time-based train/val/test split

**Goal:** Recreate the idea of `time_aware_split` without purge.

- **Input:** A pandas DataFrame `df` with a column `date_col` (e.g. `"date"`), and three floats `train_ratio`, `val_ratio`, `test_ratio` that sum to 1.0 (e.g. 0.6, 0.2, 0.2).
- **Output:** Three DataFrames `train_df`, `val_df`, `test_df`, which are **contiguous in time** and non-overlapping. So: sort by `date_col`, then take the first 60% of rows as train, next 20% as val, last 20% as test. No shuffle.
- **Constraint:** Use only pandas (e.g. `sort_values`, `iloc`, or `head`/`tail`). No external split helpers.
- **Check:** For a DataFrame of 100 rows, train should have 60, val 20, test 20; the last date of train should be before the first date of val, and same for val vs test.

---

### Exercise 3: “Next-day” X and y from one DataFrame

**Goal:** Recreate the idea of `_to_xy` inside `preprocess_splits`: build X and y so that row i of X predicts the value at i+1.

- **Input:** A pandas DataFrame `df` with numeric columns only (e.g. 5 columns), and the name of one column `target_col` (e.g. `"return_1d"`). Assume no NaN for simplicity.
- **Output:** Two numpy arrays:
  - `X`: shape `(N-1, num_features)` where `num_features = len(columns) - 1` (all columns except the target). Row `i` of X = row `i` of `df` (excluding target).
  - `y`: shape `(N-1,)`. Element `i` of y = value of `target_col` at row **i+1** of `df` (the “next day”).
- **Constraint:** No `shift`; use integer indexing (e.g. `df.iloc[i+1][target_col]` or vectorized slices like `df[target_col].values[1:]`).
- **Check:** For a small `df` of 4 rows, X should have 3 rows and y 3 elements; `y[0]` should equal `df[target_col].iloc[1]`.

---

Doing these three exercises will fix in your mind: (1) how next-day targets and trend labels are built, (2) how time splits work, and (3) how X and y are aligned for “predict next day” without using `shift`. After that, re-reading `lightgbm_trend.py` will feel much more concrete.

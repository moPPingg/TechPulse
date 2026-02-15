# src/evaluation/

**Purpose:** Data loading (tabular/sequential), train/val/test splits (time-based), metrics (MAE, RMSE, direction_acc, etc.), and backtest. Supports both training scripts and inference (splits, feature columns).

**Fits into system:** Used by `scripts/run_forecasting_pipeline.py`, `src/inference/service` (splits, data prep, metrics). No direct API dependency.

**Data in/out:**
- **In:** Feature CSVs or DataFrames; target column (e.g. return_1d).
- **Out:** Split indices, prepared arrays, metric dicts, backtest results.

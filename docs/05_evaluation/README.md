# 05 — Evaluation

**What this folder does:** How we control data leakage and measure performance (including tail metrics) so the system is defensible in reports or theses.

**Fits into the system:** Maps to `src/evaluation/splits.py` (train/val/test by time, purged splits), `src/evaluation/metrics.py` (MAE, RMSE, direction_acc, etc.), `src/evaluation/backtest.py`, and the leakage checklist in [HOW_TO_RUN_AND_EXTEND.md](../HOW_TO_RUN_AND_EXTEND.md).

**Data in/out:** Input: feature/target data and model predictions. Output: metric dicts, backtest results, and a clear “no future info” argument.

**Files:**
- `01_LEAKAGE_CONTROL.md` — What leakage is, how it appears in time series and news, and how we avoid it (splits, features, embargo).
- `02_TAIL_METRICS.md` — Metrics for tail risk and rare events beyond MSE/MAE.

**Order:** After 01_foundations and 02_modeling. Essential for defending the methodology. See [LEARNING_PATH.md](../LEARNING_PATH.md) for full curriculum.

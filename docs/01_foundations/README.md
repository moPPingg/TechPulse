# 01 — Foundations

**What this folder does:** Introduces the ML and time-series concepts you need before touching the forecasting or evaluation code in TechPulse.

**Fits into the system:** Prevents leakage and wrong validation. Directly applies to `src/evaluation/splits.py` (time-based splits, no shuffle), `src/features/build_features.py` (past-only features), and `src/evaluation/metrics.py`.

**Data in/out:** Conceptual only (no project data). Output: mental model for “no future information” and “chronological validation.”

**Files:**
- `01_MACHINE_LEARNING_BASICS.md` — Bias–variance, cross-validation, scaling, leakage, metrics, pipeline. Study first.
- `02_DEEP_LEARNING_BASICS.md` — Neural nets and deep learning basics (optional if you only use tree/linear/ARIMA).
- `03_TIME_SERIES_FUNDAMENTALS.md` — Stationarity, autocorrelation, why we split by time. Study before 02_modeling.

**Order:** 01 → 03 (required); 02 (optional). Then proceed to [02_modeling](../02_modeling/).

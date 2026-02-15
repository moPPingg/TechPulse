# 02 — Modeling

**What this folder does:** Explains how forecasting models are used in TechPulse (baselines, training, evaluation) and how to add or change a model.

**Fits into the system:** Maps to `src/models/forecasting/` (base, linear, XGBoost, ARIMA, LSTM, PatchTST, Transformer), `src/inference/service.py` (load, predict, cache), and `src/ensemble/aggregator.py` (combine forecasts).

**Data in/out:** Input: feature matrices and target (e.g. return_1d) from `src/evaluation/data.py`. Output: predictions and ensemble forecast cached for the API.

**Files:**
- `01_BASELINE_MODELS.md` — Baseline models and how they are trained and evaluated in this project.

**Order:** After [01_foundations](../01_foundations/). Then [03_multimodal](../03_multimodal/) (news) or [../SYSTEM_ARCHITECTURE.md](../SYSTEM_ARCHITECTURE.md) (full flow).

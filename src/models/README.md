# src/models/

**Purpose:** Forecasting and trend-classification models. Subpackage `forecasting` contains return-forecasting models (Linear, XGBoost, ARIMA, LSTM, PatchTST, Transformer). At package root: **LightGBM** and **LSTM** pipelines for **stock trend classification** (binary/3-class from next-day return).

**Trend classification (same target, comparable evaluation):**
- **`lightgbm_trend.py`** – Tabular features, time-aware split, optional Optuna tuning, feature importance.
- **`lstm_trend.py`** – Sliding-window sequences (PyTorch), LSTM classifier, same metrics for comparison.
- **`itransformer_trend.py`** – iTransformer (inverted embedding: each variable = token), trend classifier; same data/splits as LSTM.
- **Scripts:** `scripts/run_lstm_vs_lightgbm_trend.py` (LSTM vs LightGBM); `scripts/run_trend_model_comparison.py` (LSTM vs iTransformer vs PatchTST [regress→threshold] vs optional LightGBM).

**Fits into system:** Used by inference service and training scripts. Inputs come from evaluation (prepared features/target). Outputs are raw predictions before ensemble (or trend metrics/importance).

**Data in/out:**
- **In:** Train: (X_train, y_train). Predict: X (feature matrix or sequences).
- **Out:** Predictions (mean, optionally std) or class labels/probs; trained model state on disk if persisted.

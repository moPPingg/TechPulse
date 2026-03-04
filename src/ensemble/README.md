# src/ensemble/

**Purpose:** Combine predictions from multiple models in two ways:
- **aggregator.py:** Regression ensemble (weighted mean, optional std) using inverse-MAE and stability weights. Used by inference for return forecasts.
- **stacking.py:** Temporal stacking for **trend classification**. Base models (LightGBM, LSTM, PatchTST, iTransformer) produce class probabilities; a meta-model (logistic or MLP) is trained on validation-set base predictions and combines them at test time. Includes model comparison (base vs ensemble metrics).

**Fits into system:** Aggregator is called from `src/inference/service`. Stacking is used by `scripts/run_temporal_ensemble.py` for trend pipelines.

**Data in/out:**
- **Aggregator:** Per-model forecasts (mean, optional std), validation metrics → ensemble mean, std, weights.
- **Stacking:** Base model proba on val/test → meta-model fit on val → ensemble predictions and evaluation (accuracy, F1, ROC-AUC).

"""
Temporal stacking ensemble for trend classification.

Combines predictions from multiple base models (LightGBM, LSTM, PatchTST, iTransformer)
using a meta-model trained on out-of-sample base predictions to avoid leakage.

Stacking strategy:
  - Base models are trained on the train set (with optional val for early stopping).
  - Base model predictions on the VAL set form the meta-features; the meta-model
    is trained on (stacked_val_probs, y_val). No base model sees val labels during
    training, so val predictions are out-of-sample.
  - At test time: base models predict on test -> stacked test probs -> meta-model
    predicts the final class.

Meta-model: Logistic regression (default) or optional MLP. Uses class probabilities
from each base model (concat) so the meta-model sees (n_val, n_models * n_classes).

Evaluation: Same metrics as single models (accuracy, F1, ROC-AUC, confusion matrix).
Model comparison: Side-by-side table of base vs ensemble metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_proba_shape(P: np.ndarray, n_classes: int, model_name: str) -> np.ndarray:
    """Ensure P is (n_samples, n_classes); convert logits or 1d to proba if needed."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        if n_classes == 2:
            P = np.column_stack([1 - P, P])
        else:
            raise ValueError(f"{model_name}: 1d preds not supported for n_classes={n_classes}")
    elif P.ndim == 2 and P.shape[1] != n_classes:
        if P.shape[1] == 1 and n_classes == 2:
            p = np.clip(P.ravel(), 1e-7, 1 - 1e-7)
            P = np.column_stack([1 - p, p])
        else:
            raise ValueError(f"{model_name}: shape {P.shape} does not match n_classes={n_classes}")
    return np.clip(P, 1e-7, 1 - 1e-7)


def stack_predictions(
    base_preds: Dict[str, np.ndarray],
    n_classes: int = 2,
) -> np.ndarray:
    """
    Stack base model probability arrays into one matrix (n_samples, n_models * n_classes).

    base_preds: {model_name: (n_samples, n_classes) or (n_samples,) for binary}
    """
    names = sorted(base_preds.keys())
    if not names:
        raise ValueError("base_preds must contain at least one model")
    n_samples = len(base_preds[names[0]])
    parts = []
    for name in names:
        P = _ensure_proba_shape(base_preds[name], n_classes, name)
        if len(P) != n_samples:
            raise ValueError(f"{name}: length {len(P)} != {n_samples}")
        parts.append(P)
    return np.hstack(parts)  # (n_samples, n_models * n_classes)


class TemporalStackingEnsemble:
    """
    Stacking ensemble: meta-model trained on base model predictions from the validation set.

    - fit(base_preds_val, y_val): base_preds_val = {model_name: (n_val, n_classes) proba}
    - predict(base_preds_test): returns (n_test,) class predictions and (n_test, n_classes) proba
    - evaluate(y_test, base_preds_test): returns metrics dict (accuracy, f1, roc_auc, ...)
    """

    def __init__(
        self,
        n_classes: int = 2,
        meta_model: str = "logistic",  # "logistic" or "mlp"
        meta_C: float = 1.0,  # LogisticRegression C (inverse regularization)
        meta_hidden: Optional[Tuple[int, ...]] = None,  # MLP hidden layers
        meta_max_iter: int = 1000,
    ):
        self.n_classes = n_classes
        self.meta_model_type = meta_model
        self.meta_C = meta_C
        self.meta_hidden = meta_hidden or (64, 32)
        self.meta_max_iter = meta_max_iter
        self._meta = None
        self._model_names: List[str] = []
        self._fitted = False

    def fit(
        self,
        base_preds_val: Dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "TemporalStackingEnsemble":
        """
        Train the meta-model on stacked validation predictions.

        base_preds_val: {model_name: array of shape (n_val,) or (n_val, n_classes)}
        y_val: (n_val,) integer labels
        """
        from sklearn.linear_model import LogisticRegression

        y_val = np.asarray(y_val).ravel()
        self._model_names = sorted(base_preds_val.keys())
        X_meta = stack_predictions(base_preds_val, self.n_classes)

        if self.meta_model_type == "logistic":
            self._meta = LogisticRegression(
                C=self.meta_C,
                max_iter=self.meta_max_iter,
                solver="lbfgs",
                random_state=42,
            )
            self._meta.fit(X_meta, y_val)
        elif self.meta_model_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            self._meta = MLPClassifier(
                hidden_layer_sizes=self.meta_hidden,
                max_iter=self.meta_max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
            self._meta.fit(X_meta, y_val)
        else:
            raise ValueError(f"meta_model must be 'logistic' or 'mlp', got {self.meta_model_type}")

        self._fitted = True
        logger.info("Stacking meta-model fitted on %d val samples, %d base models", len(y_val), len(self._model_names))
        return self

    def predict_proba(self, base_preds_test: Dict[str, np.ndarray]) -> np.ndarray:
        """Return (n_test, n_classes) ensemble probabilities."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        X_meta = stack_predictions(base_preds_test, self.n_classes)
        return self._meta.predict_proba(X_meta)

    def predict(self, base_preds_test: Dict[str, np.ndarray]) -> np.ndarray:
        """Return (n_test,) class predictions."""
        P = self.predict_proba(base_preds_test)
        return np.argmax(P, axis=1).astype(np.int32)

    def evaluate(
        self,
        y_test: np.ndarray,
        base_preds_test: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Compute metrics for the ensemble on the test set."""
        from src.models.lightgbm_trend import evaluate_trend

        y_test = np.asarray(y_test).ravel()
        y_pred = self.predict(base_preds_test)
        y_prob = self.predict_proba(base_preds_test)
        return evaluate_trend(y_test, y_pred, y_prob=y_prob, n_classes=self.n_classes)

    @property
    def model_names(self) -> List[str]:
        return list(self._model_names)


def compare_models(
    base_metrics: Dict[str, Dict[str, Any]],
    ensemble_metrics: Dict[str, Any],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a comparison table: each model (and ensemble) with selected metrics.

    base_metrics: {model_name: {accuracy: ..., f1: ..., ...}}
    ensemble_metrics: {accuracy: ..., f1: ..., ...}
    Returns: {metrics: {model_name: {accuracy, f1, ...}}, summary: best_per_metric}
    """
    metric_keys = metric_keys or ["accuracy", "precision", "recall", "f1", "roc_auc"]
    all_models = list(base_metrics.keys()) + ["Ensemble"]
    table = {}
    for m in all_models:
        table[m] = (
            ensemble_metrics if m == "Ensemble" else base_metrics.get(m, {})
        )
    best = {}
    for k in metric_keys:
        values = [(m, table[m].get(k, 0.0)) for m in all_models if isinstance(table[m].get(k), (int, float))]
        if values:
            m, v = max(values, key=lambda x: x[1])
            best[k] = {"model": m, "value": float(v)}
    return {"metrics_by_model": table, "best_per_metric": best}


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print a formatted table of model comparison."""
    table = comparison["metrics_by_model"]
    models = list(table.keys())
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    print("\n--- Model comparison ---")
    print("  " + "".join(f"{m:14s}" for m in models))
    for k in keys:
        row = "  " + "".join(f"{table[m].get(k, 0):.4f}".ljust(16) for m in models)
        print(f"  {k:12s} " + row)
    b = comparison.get("best_per_metric", {})
    print("\n  Best per metric:", {k: f"{v['model']}={v['value']:.4f}" for k, v in b.items()})

"""
LightGBM pipeline for stock trend prediction using multimodal features.

Design:
- Target: binary or 3-class trend from forward return (e.g. return_1d shifted for next day).
- Data: time-aware train/val/test split; optional news daily features merged by date.
- Preprocessing: scale numeric features on train only (no leakage); handle missing.
- Tuning: Optuna (or grid) with validation metric on val set (chronological).
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- Feature importance: native LightGBM gain/split counts and optional permutation.

Upstream: feature CSVs from src/features/build_features.py; optional news from src/news/engine.get_ml_daily_features.
Downstream: saved model artifact and metrics for inference or backtest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# LightGBM is optional at import; we check at runtime
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# Default return column used to derive trend target
DEFAULT_RETURN_COL = "return_1d"


# -----------------------------------------------------------------------------
# 1. Data loading and multimodal merge
# -----------------------------------------------------------------------------


def load_price_features(
    symbol: str,
    features_dir: Union[str, Path],
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load technical feature CSV for one symbol. Expects columns from build_features.py
    (date, open, high, low, close, volume, return_*, ma_*, rsi, volatility_*, etc.).
    """
    path = Path(features_dir) / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Feature CSV must have 'date' column")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if from_date:
        df = df[df["date"] >= from_date]
    if to_date:
        df = df[df["date"] <= to_date]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_news_daily_features(
    symbol: str,
    from_date: str,
    to_date: str,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load per-day news features from the News Intelligence Engine.
    Returns DataFrame with columns: date, symbol, composite_score, article_count,
    avg_sentiment, avg_relevance, event_breakdown (dict), horizon_breakdown (dict).
    We flatten to numeric: composite_score, article_count, avg_sentiment, avg_relevance
    plus optional event/horizon counts if needed.
    """
    from src.news.engine import get_ml_daily_features

    rows = get_ml_daily_features(symbol, from_date, to_date, config_path=config_path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Keep numeric columns; expand event_breakdown/horizon_breakdown if desired (here we keep simple)
    df["date"] = df["date"].astype(str)
    return df


def merge_multimodal(
    price_df: pd.DataFrame,
    news_df: pd.DataFrame,
    news_prefix: str = "news_",
) -> pd.DataFrame:
    """
    Left-join news daily features onto price DataFrame by date.
    News columns are prefixed to avoid clashes. Missing news days get NaN; filled later in preprocessing.
    """
    if news_df is None or news_df.empty:
        return price_df.copy()
    # Ensure we only merge numeric/news columns
    merge_cols = ["date"]
    feature_cols = [c for c in news_df.columns if c != "date" and c != "symbol"]
    if not feature_cols:
        return price_df.copy()
    news_renamed = news_df[["date"] + feature_cols].copy()
    news_renamed.columns = ["date"] + [f"{news_prefix}{c}" for c in feature_cols]
    out = price_df.merge(news_renamed, on="date", how="left")
    return out


# -----------------------------------------------------------------------------
# 2. Target and feature columns
# -----------------------------------------------------------------------------


def build_trend_target(
    df: pd.DataFrame,
    return_col: str = DEFAULT_RETURN_COL,
    threshold_up: float = 0.0,
    threshold_down: float = 0.0,
    n_classes: int = 2,
) -> np.ndarray:
    """
    Build trend label from forward return (next-day return).
    - For binary (n_classes=2): 1 if return > threshold_up, else 0.
    - For 3-class (n_classes=3): 1 up, 0 neutral, -1 down (or 0,1,2 for classifier).
    We use shift(-1) so row i has target = return at i+1 (predict next day trend).
    """
    if return_col not in df.columns:
        raise ValueError(f"Return column '{return_col}' not in DataFrame")
    # Next-period return: y[i] = return at i+1
    next_ret = df[return_col].shift(-1)
    next_ret = next_ret.iloc[:-1]  # drop last row (no next)
    if n_classes == 2:
        y = (next_ret > threshold_up).astype(np.int32)
    else:
        # 3-class: 0 down, 1 neutral, 2 up
        y = np.where(next_ret > threshold_up, 2, np.where(next_ret < threshold_down, 0, 1))
        y = y.astype(np.int32)
    return y


def get_feature_columns_for_trend(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Numeric columns only, excluding date, target-related, and raw price/volume."""
    from src.evaluation.data import EXCLUDE_FROM_FEATURES

    skip = set(EXCLUDE_FROM_FEATURES) | set(exclude or [])
    skip.add("date")
    cand = [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cand


# -----------------------------------------------------------------------------
# 3. Time-aware splitting
# -----------------------------------------------------------------------------


def time_aware_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    purge_gap: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by time order. No shuffle. Optionally use purged splits to avoid overlap leakage.
    """
    from src.evaluation.splits import (
        get_purged_train_val_test_splits,
        get_train_val_test_splits,
    )

    df = df.sort_values(date_col).reset_index(drop=True)
    if purge_gap > 0:
        return get_purged_train_val_test_splits(
            df, date_col=date_col,
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
            purge_gap=purge_gap,
        )
    return get_train_val_test_splits(
        df, date_col=date_col,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
    )


# -----------------------------------------------------------------------------
# 4. Preprocessing
# -----------------------------------------------------------------------------


def preprocess_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    return_col: str = DEFAULT_RETURN_COL,
    fit_scaler_on_train: bool = True,
    fill_missing: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Build X, y for train/val/test. Optionally scale features using StandardScaler fit on train only.
    Target is aligned: we drop the last row of each split (no next-day return for last date).
    """
    def _to_xy(df: pd.DataFrame, ret_col: str) -> Tuple[np.ndarray, np.ndarray]:
        # Align with next-day target: drop last row
        n = len(df) - 1
        if n <= 0:
            return np.empty((0, len(feature_cols))), np.empty(0)
        X = df[feature_cols].iloc[:n].copy()
        # Target for row i = return at i+1 (next day)
        y = df[ret_col].shift(-1).iloc[:n].values
        X = X.fillna(fill_missing)
        X = np.nan_to_num(X.values, nan=fill_missing, posinf=0.0, neginf=0.0)
        X = np.clip(X, -1e10, 1e10)
        y = np.nan_to_num(y, nan=0.0)
        return X.astype(np.float64), y.astype(np.float64)

    X_train, y_train = _to_xy(train_df, return_col)
    X_val, y_val = _to_xy(val_df, return_col)
    X_test, y_test = _to_xy(test_df, return_col)

    scaler = None
    if fit_scaler_on_train and len(X_train) > 0:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if len(X_val) > 0:
            X_val = scaler.transform(X_val)
        if len(X_test) > 0:
            X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# -----------------------------------------------------------------------------
# 5. Trend target from continuous return
# -----------------------------------------------------------------------------


def continuous_to_trend_labels(
    y_continuous: np.ndarray,
    threshold_up: float = 0.0,
    threshold_down: float = 0.0,
    n_classes: int = 2,
) -> np.ndarray:
    """Convert continuous returns to class labels (binary or 3-class)."""
    if n_classes == 2:
        return (y_continuous > threshold_up).astype(np.int32)
    return np.where(
        y_continuous > threshold_up, 2,
        np.where(y_continuous < threshold_down, 0, 1),
    ).astype(np.int32)


# -----------------------------------------------------------------------------
# 6. Training, tuning, evaluation, importance
# -----------------------------------------------------------------------------


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    n_classes: int = 2,
) -> "lgb.LGBMModel":
    """
    Train LightGBM classifier for trend (binary or multiclass).
    Uses validation set for early stopping if provided.
    """
    if lgb is None:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm")

    default_params = {
        "objective": "binary" if n_classes == 2 else "multiclass",
        "num_class": n_classes if n_classes > 2 else None,
        "metric": "auc" if n_classes == 2 else "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)
    if n_classes == 2 and "num_class" in default_params:
        del default_params["num_class"]

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_sets = [train_set]
    valid_names = ["train"]
    if X_val is not None and y_val is not None and len(X_val) > 0:
        valid_sets.append(lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=feature_names))
        valid_names.append("valid")

    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if early_stopping_rounds and len(valid_sets) > 1 else None

    model = lgb.train(
        default_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def evaluate_trend(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_classes: int = 2,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1, ROC-AUC, and confusion matrix (as list of lists)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # For binary, use pos_label=1; for multiclass use average
    average = "binary" if n_classes == 2 else "weighted"
    metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    if y_prob is not None and n_classes == 2 and y_prob.ndim >= 2:
        # probability of positive class
        prob_pos = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.ravel()
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
        except ValueError:
            metrics["roc_auc"] = 0.0
    elif y_prob is not None and n_classes > 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def get_feature_importance(
    model: "lgb.LGBMModel",
    importance_type: str = "gain",
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract feature importance from trained LightGBM. Returns DataFrame with feature name and importance."""
    imp = model.feature_importance(importance_type=importance_type)
    names = feature_names or [f"f{i}" for i in range(len(imp))]
    if len(names) != len(imp):
        names = [f"f{i}" for i in range(len(imp))]
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)


# -----------------------------------------------------------------------------
# 7. Hyperparameter tuning (Optuna)
# -----------------------------------------------------------------------------


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_trials: int = 30,
    timeout: Optional[float] = None,
    n_classes: int = 2,
) -> Tuple[Dict[str, Any], "lgb.LGBMModel"]:
    """
    Use Optuna to search learning_rate, num_leaves, feature_fraction, etc.
    Optimizes validation metric (AUC for binary, multi_logloss for multiclass).
    """
    if lgb is None:
        raise ImportError("lightgbm is required")
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required for tuning. Install with: pip install optuna")

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "binary" if n_classes == 2 else "multiclass",
            "num_class": n_classes if n_classes > 2 else None,
            "metric": "auc" if n_classes == 2 else "multi_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        if n_classes == 2 and "num_class" in params:
            del params["num_class"]

        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=feature_names)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(X_val)
        if n_classes == 2:
            # pred is probability of positive class
            return float(roc_auc_score(y_val, pred))
        # multiclass: pred is (n, num_class) probabilities
        return float(roc_auc_score(y_val, pred, multi_class="ovr", average="weighted"))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_trial.params.copy()
    best_params["objective"] = "binary" if n_classes == 2 else "multiclass"
    best_params["num_class"] = n_classes if n_classes > 2 else None
    best_params["metric"] = "auc" if n_classes == 2 else "multi_logloss"
    best_params["boosting_type"] = "gbdt"
    best_params["verbose"] = -1
    best_params["seed"] = 42
    best_params["n_jobs"] = -1
    if n_classes == 2 and "num_class" in best_params:
        del best_params["num_class"]

    # Retrain with best params for full round
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=feature_names)
    model = lgb.train(
        best_params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return best_params, model


# -----------------------------------------------------------------------------
# 8. Full pipeline class
# -----------------------------------------------------------------------------


class LightGBMTrendPipeline:
    """
    End-to-end LightGBM pipeline for stock trend prediction with multimodal features.

    Steps:
    1. Load price features (and optionally news daily features).
    2. Merge on date (multimodal).
    3. Build trend target from next-day return (binary or 3-class).
    4. Time-aware train/val/test split.
    5. Preprocess: scale on train, fill missing.
    6. Train (or tune then train) LightGBM.
    7. Evaluate on test; extract feature importance.
    """

    def __init__(
        self,
        symbol: str,
        features_dir: Union[str, Path],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        include_news: bool = False,
        return_col: str = DEFAULT_RETURN_COL,
        trend_threshold_up: float = 0.0,
        trend_threshold_down: float = -0.0,
        n_classes: int = 2,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        purge_gap: int = 0,
        scale_features: bool = True,
        lgb_params: Optional[Dict[str, Any]] = None,
        tune: bool = False,
        tune_trials: int = 20,
    ):
        self.symbol = symbol.upper()
        self.features_dir = Path(features_dir)
        self.from_date = from_date
        self.to_date = to_date
        self.include_news = include_news
        self.return_col = return_col
        self.trend_threshold_up = trend_threshold_up
        self.trend_threshold_down = trend_threshold_down
        self.n_classes = n_classes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_gap = purge_gap
        self.scale_features = scale_features
        self.lgb_params = lgb_params or {}
        self.tune = tune
        self.tune_trials = tune_trials

        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.X_train = self.y_train = self.X_val = self.y_val = self.X_test = self.y_test = None
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Any] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.metrics: Dict[str, Any] = {}
        self.importance_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load price features and optionally merge news daily features."""
        self.df = load_price_features(
            self.symbol,
            self.features_dir,
            from_date=self.from_date,
            to_date=self.to_date,
        )
        if self.include_news and len(self.df) > 0:
            from_d = self.df["date"].min()
            to_d = self.df["date"].max()
            news_df = load_news_daily_features(self.symbol, from_d, to_d)
            self.df = merge_multimodal(self.df, news_df, news_prefix="news_")
        self.feature_cols = get_feature_columns_for_trend(self.df, exclude=[self.return_col])
        if not self.feature_cols:
            raise ValueError("No feature columns found; check EXCLUDE_FROM_FEATURES and numeric columns")
        logger.info("Loaded %d rows, %d features for %s", len(self.df), len(self.feature_cols), self.symbol)
        return self.df

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-aware train/val/test split."""
        if self.df is None:
            self.load_data()
        train_df, val_df, test_df = time_aware_split(
            self.df,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            purge_gap=self.purge_gap,
        )
        return train_df, val_df, test_df

    def preprocess(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Build X, y and optionally scale. Aligns target with next-day return."""
        (self.X_train, y_train_c,
         self.X_val, y_val_c,
         self.X_test, y_test_c,
         self.scaler) = preprocess_splits(
            train_df, val_df, test_df,
            self.feature_cols,
            return_col=self.return_col,
            fit_scaler_on_train=self.scale_features,
        )
        # Convert continuous return to trend labels
        self.y_train = continuous_to_trend_labels(
            y_train_c, self.trend_threshold_up, self.trend_threshold_down, self.n_classes
        )
        self.y_val = continuous_to_trend_labels(
            y_val_c, self.trend_threshold_up, self.trend_threshold_down, self.n_classes
        )
        self.y_test = continuous_to_trend_labels(
            y_test_c, self.trend_threshold_up, self.trend_threshold_down, self.n_classes
        )

    def run_tuning(self) -> Dict[str, Any]:
        """Run Optuna tuning; set self.best_params and self.model."""
        if self.X_train is None or self.y_train is None or self.X_val is None or self.y_val is None:
            raise RuntimeError("Call split() and preprocess() first")
        self.best_params, self.model = tune_lightgbm(
            self.X_train, self.y_train, self.X_val, self.y_val,
            feature_names=self.feature_cols,
            n_trials=self.tune_trials,
            n_classes=self.n_classes,
        )
        logger.info("Best params: %s", self.best_params)
        return self.best_params

    def train(self) -> Any:
        """Train LightGBM (with best_params if tuning was run)."""
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call split() and preprocess() first")
        params = self.best_params if self.best_params else self.lgb_params
        self.model = train_lightgbm(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            feature_names=self.feature_cols,
            params=params,
            n_classes=self.n_classes,
        )
        return self.model

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on test set; store metrics and return dict."""
        if self.model is None or self.X_test is None or self.y_test is None:
            raise RuntimeError("Train model first")
        y_pred = self.model.predict(self.X_test)
        if self.n_classes == 2 and y_pred.ndim == 2:
            y_prob = y_pred
            y_pred = (y_pred[:, 1] > 0.5).astype(np.int32)
        else:
            y_prob = y_pred
            y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(np.int32)
        self.metrics = evaluate_trend(
            self.y_test, y_pred, y_prob=y_prob, n_classes=self.n_classes
        )
        logger.info("Test metrics: %s", {k: v for k, v in self.metrics.items() if k != "confusion_matrix"})
        return self.metrics

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return DataFrame of feature names and importance."""
        if self.model is None:
            raise RuntimeError("Train model first")
        self.importance_df = get_feature_importance(
            self.model, importance_type=importance_type, feature_names=self.feature_cols
        )
        return self.importance_df

    def run(self, tune: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run full pipeline: load_data → split → preprocess → (optional tune) → train → evaluate → importance.
        """
        tune = tune if tune is not None else self.tune
        self.load_data()
        train_df, val_df, test_df = self.split()
        self.preprocess(train_df, val_df, test_df)
        if tune:
            self.run_tuning()
        else:
            self.train()
        self.evaluate()
        self.get_feature_importance()
        return self.metrics

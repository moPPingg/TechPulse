"""
Inference service: Load cached forecast or run on-the-fly.

TEACHING:
- Cache path: data/forecasts/{symbol}.json
- Run scripts/run_inference.py to populate cache.
- If cache miss, we can run inference (slower) or fall back to feature-based proxy.

PRODUCTION: Forecast staleness is checked via max_age_days. Stale forecasts
are rejected to avoid recommendations based on outdated model outputs.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ForecastResult:
    """Forecast output for a symbol."""
    symbol: str
    as_of_date: str
    ensemble_mean: float  # % return
    ensemble_std: float
    volatility_pct: float
    model_forecasts: list
    model_mae: dict
    weights: dict
    confidence_score: float  # 0-1, higher = more confident (lower std relative to mean)


def _load_config() -> dict:
    try:
        import yaml
        path = _ROOT / "configs" / "config.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _forecasts_dir() -> Path:
    cfg = _load_config()
    base = cfg.get("data", {}).get("forecasts_dir", "data/forecasts")
    p = Path(base)
    if not p.is_absolute():
        p = _ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_forecast(
    symbol: str,
    max_age_days: Optional[int] = None,
) -> Optional[ForecastResult]:
    """
    Load cached forecast for symbol. Returns None if cache miss or stale.

    PRODUCTION: max_age_days rejects forecasts older than N days. Default from
    configs/decision.yaml (7), or 7 if not set. Set to 0 or None to disable.
    """
    symbol = symbol.strip().upper()
    path = _forecasts_dir() / f"{symbol}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_age_days is None:
            try:
                import yaml
                dec_path = _ROOT / "configs" / "decision.yaml"
                if dec_path.exists():
                    with open(dec_path, "r", encoding="utf-8") as f:
                        dec_cfg = yaml.safe_load(f) or {}
                    max_age_days = dec_cfg.get("forecast", {}).get("max_age_days", 7)
                else:
                    max_age_days = 7
            except Exception:
                max_age_days = 7

        as_of = data.get("as_of_date", "")
        if max_age_days > 0 and as_of:
            try:
                as_of_dt = datetime.strptime(str(as_of)[:10], "%Y-%m-%d")
                if (datetime.utcnow() - as_of_dt).days > max_age_days:
                    logger.warning(
                        "Forecast for %s is stale (as_of=%s, max_age=%d days); rejecting",
                        symbol, as_of, max_age_days,
                    )
                    return None
            except (ValueError, TypeError):
                pass

        return ForecastResult(
            symbol=data.get("symbol", symbol),
            as_of_date=as_of,
            ensemble_mean=float(data.get("ensemble_mean", 0)),
            ensemble_std=float(data.get("ensemble_std", 0.5)),
            volatility_pct=float(data.get("volatility_pct", 1.0)),
            model_forecasts=data.get("model_forecasts", []),
            model_mae=data.get("model_mae", {}),
            weights=data.get("weights", {}),
            confidence_score=float(data.get("confidence_score", 0.5)),
        )
    except Exception as e:
        logger.warning("Failed to load forecast for %s: %s", symbol, e)
        return None


def _save_forecast(result: ForecastResult) -> None:
    path = _forecasts_dir() / f"{result.symbol}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info("Saved forecast for %s to %s", result.symbol, path)


def run_inference_for_symbol(
    symbol: str,
    features_dir: Optional[str] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seq_len: int = 20,
) -> Optional[ForecastResult]:
    """
    Run full ML pipeline for symbol, compute ensemble, save to cache.
    Called by scripts/run_inference.py.
    """
    import pandas as pd
    import numpy as np

    from src.evaluation.splits import get_train_val_test_splits
    from src.evaluation.data import prepare_tabular, prepare_sequential, get_feature_columns, DEFAULT_TARGET
    from src.evaluation.metrics import compute_metrics
    from src.ensemble.aggregator import ModelForecast, aggregate_forecasts

    cfg = _load_config()
    base = features_dir or cfg.get("data", {}).get("features_dir", "data/features/vn30")
    if not Path(base).is_absolute():
        base = str(_ROOT / base)
    path = Path(base) / f"{symbol}.csv"
    if not path.exists():
        logger.error("Features not found: %s", path)
        return None

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Sanitize: inf/nan in feature CSV break StandardScaler and models (BCM, GVR, etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df[numeric_cols] = df[numeric_cols].clip(lower=-1e10, upper=1e10)

    train_df, val_df, test_df = get_train_val_test_splits(
        df, date_col="date", train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=0.2
    )

    data_tab = prepare_tabular(train_df, val_df, test_df, target=DEFAULT_TARGET)
    data_seq = prepare_sequential(train_df, val_df, test_df, seq_len=seq_len, target=DEFAULT_TARGET)

    feats = data_tab.feature_names
    model_results = {}

    # 1. Linear
    from src.models.forecasting.baseline_ml import LinearForecaster
    linear = LinearForecaster()
    linear.fit(data_tab.X_train, data_tab.y_train)
    pred_lin = linear.predict(data_tab.X_val)
    mae_lin = float(np.mean(np.abs(data_tab.y_val - pred_lin)))
    std_lin = float(np.std(data_tab.y_val - pred_lin)) if len(pred_lin) > 1 else 0.5
    last_lin = float(linear.predict(data_tab.X_test[-1:].reshape(1, -1))[0])
    model_results["Linear"] = {"mean": last_lin, "std": std_lin, "mae": mae_lin}

    # 2. XGBoost
    from src.models.forecasting.baseline_ml import XGBoostForecaster
    xgb = XGBoostForecaster(n_estimators=100, max_depth=4)
    xgb.fit(data_tab.X_train, data_tab.y_train)
    pred_xgb = xgb.predict(data_tab.X_val)
    mae_xgb = float(np.mean(np.abs(data_tab.y_val - pred_xgb)))
    std_xgb = float(np.std(data_tab.y_val - pred_xgb)) if len(pred_xgb) > 1 else 0.5
    last_xgb = float(xgb.predict(data_tab.X_test[-1:].reshape(1, -1))[0])
    model_results["XGBoost"] = {"mean": last_xgb, "std": std_xgb, "mae": mae_xgb}

    # 3. ARIMA
    from src.models.forecasting.arima import ARIMAForecaster
    arima = ARIMAForecaster(order=(2, 0, 2))
    arima.fit(data_tab.X_train, data_tab.y_train, use_exog=False)
    pred_arima = arima.predict(data_tab.X_val)
    mae_arima = float(np.mean(np.abs(data_tab.y_val - pred_arima)))
    std_arima = float(np.std(data_tab.y_val - pred_arima)) if len(pred_arima) > 1 else 0.5
    last_arima = float(arima.predict(data_tab.X_test[-1:])[0])
    model_results["ARIMA"] = {"mean": last_arima, "std": std_arima, "mae": mae_arima}

    # 4. LSTM
    from src.models.forecasting.lstm import LSTMForecaster
    lstm = LSTMForecaster(epochs=30, batch_size=32)
    lstm.fit(data_seq.X_train, data_seq.y_train)
    pred_lstm = lstm.predict(data_seq.X_val)
    mae_lstm = float(np.mean(np.abs(data_seq.y_val - pred_lstm)))
    std_lstm = float(np.std(data_seq.y_val - pred_lstm)) if len(pred_lstm) > 1 else 0.5
    last_lstm = float(lstm.predict(data_seq.X_test[-1:])[0])
    model_results["LSTM"] = {"mean": last_lstm, "std": std_lstm, "mae": mae_lstm}

    # 5. PatchTST
    from src.models.forecasting.patchtst import PatchTSTForecaster
    patchtst = PatchTSTForecaster(epochs=30, batch_size=32)
    patchtst.fit(data_seq.X_train, data_seq.y_train)
    pred_pt = patchtst.predict(data_seq.X_val)
    mae_pt = float(np.mean(np.abs(data_seq.y_val - pred_pt)))
    std_pt = float(np.std(data_seq.y_val - pred_pt)) if len(pred_pt) > 1 else 0.5
    last_pt = float(patchtst.predict(data_seq.X_test[-1:])[0])
    model_results["PatchTST"] = {"mean": last_pt, "std": std_pt, "mae": mae_pt}

    # 6. Transformer
    from src.models.forecasting.transformer import TransformerForecaster
    trans = TransformerForecaster(epochs=30, batch_size=32)
    trans.fit(data_seq.X_train, data_seq.y_train)
    pred_trans = trans.predict(data_seq.X_val)
    mae_trans = float(np.mean(np.abs(data_seq.y_val - pred_trans)))
    std_trans = float(np.std(data_seq.y_val - pred_trans)) if len(pred_trans) > 1 else 0.5
    last_trans = float(trans.predict(data_seq.X_test[-1:])[0])
    model_results["Transformer"] = {"mean": last_trans, "std": std_trans, "mae": mae_trans}

    # Ensemble
    forecasts = [
        ModelForecast(name=n, mean=r["mean"], std=r["std"])
        for n, r in model_results.items()
    ]
    mae_map = {n: r["mae"] for n, r in model_results.items()}
    ensemble = aggregate_forecasts(forecasts, mae_per_model=mae_map)

    # Volatility from last row of features
    vol = 1.0
    if len(df) > 0:
        row = df.iloc[-1]
        vol = float(row.get("volatility_5", 1.0) or 1.0)
    vol = max(vol, 0.1)

    # Confidence: lower std relative to abs(mean) = higher confidence
    conf = 0.5
    if ensemble.std > 0:
        conf = 1.0 / (1.0 + ensemble.std / max(abs(ensemble.mean), 0.1))
    conf = min(1.0, max(0.1, conf))

    as_of = str(df["date"].iloc[-1])[:10] if "date" in df.columns else ""

    result = ForecastResult(
        symbol=symbol,
        as_of_date=as_of,
        ensemble_mean=ensemble.mean,
        ensemble_std=ensemble.std,
        volatility_pct=vol,
        model_forecasts=[{"name": n, "mean": r["mean"], "std": r["std"]} for n, r in model_results.items()],
        model_mae=mae_map,
        weights=ensemble.weights,
        confidence_score=conf,
    )
    _save_forecast(result)
    return result

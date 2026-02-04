#!/usr/bin/env python3
"""
Unified forecasting pipeline: same splits, metrics, and backtest for 5 models.
Usage: python scripts/run_forecasting_pipeline.py [--features-dir ...] [--symbol FPT]
"""

import sys
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np

from src.evaluation.splits import get_train_val_test_splits
from src.evaluation.data import (
    prepare_tabular,
    prepare_sequential,
    get_rolling_fold_tabular,
    get_feature_columns,
    DEFAULT_TARGET,
)
from src.evaluation.metrics import compute_metrics, METRIC_NAMES
from src.evaluation.backtest import simple_backtest, rolling_backtest
from src.models.forecasting.baseline_ml import LinearForecaster, XGBoostForecaster
from src.models.forecasting.arima import ARIMAForecaster
from src.models.forecasting.lstm import LSTMForecaster
from src.models.forecasting.patchtst import PatchTSTForecaster
from src.models.forecasting.transformer import TransformerForecaster


def load_features(features_dir: str, symbol: str) -> pd.DataFrame:
    path = Path(features_dir) / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def run_pipeline(
    features_dir: str = "data/features/vn30",
    symbol: str = "FPT",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seq_len: int = 20,
    target: str = DEFAULT_TARGET,
    run_rolling: bool = False,
    rolling_test_start: int = None,
) -> dict:
    df = load_features(features_dir, symbol)
    train_df, val_df, test_df = get_train_val_test_splits(
        df, date_col="date", train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )

    # Tabular data for Linear, XGBoost, ARIMA
    data_tab = prepare_tabular(train_df, val_df, test_df, target=target)
    # Sequential for LSTM, PatchTST, Transformer
    data_seq = prepare_sequential(train_df, val_df, test_df, seq_len=seq_len, target=target)

    feats = data_tab.feature_names
    results = {}

    # 1. Linear
    results["Linear"] = simple_backtest(LinearForecaster(), data_tab)
    # 2. XGBoost
    results["XGBoost"] = simple_backtest(XGBoostForecaster(n_estimators=100, max_depth=4), data_tab)
    # 3. ARIMA (fit on y only)
    arima = ARIMAForecaster(order=(2, 0, 2))
    arima.fit(data_tab.X_train, data_tab.y_train, use_exog=False)
    pred_arima = arima.predict(data_tab.X_test)
    results["ARIMA"] = compute_metrics(data_tab.y_test, pred_arima)
    # 4. LSTM
    results["LSTM"] = simple_backtest(LSTMForecaster(epochs=30, batch_size=32), data_seq)
    # 5. PatchTST
    results["PatchTST"] = simple_backtest(PatchTSTForecaster(epochs=30, batch_size=32), data_seq)
    # 6. Transformer (iTransformer)
    results["Transformer"] = simple_backtest(TransformerForecaster(epochs=30, batch_size=32), data_seq)

    if run_rolling and len(df) > 100:
        test_start = rolling_test_start or int(len(df) * (train_ratio + val_ratio))
        def get_fold(d, t):
            return get_rolling_fold_tabular(d, t, target=target, feature_cols=feats)
        for name, factory in [
            ("Linear", lambda: LinearForecaster()),
            ("XGBoost", lambda: XGBoostForecaster(n_estimators=50, max_depth=3)),
        ]:
            try:
                results[f"{name}_rolling"] = rolling_backtest(
                    factory, df, lambda d, t: get_fold(d, t), test_start=test_start, test_end=min(test_start + 50, len(df) - 1)
                )
            except Exception as e:
                results[f"{name}_rolling"] = {m: float("nan") for m in METRIC_NAMES}
                print(f"Rolling {name} failed: {e}")

    return results


def main():
    ap = argparse.ArgumentParser(description="Unified 5-model forecasting pipeline")
    ap.add_argument("--features-dir", default="data/features/vn30", help="Features CSV directory")
    ap.add_argument("--symbol", default="FPT", help="Stock symbol (e.g. FPT)")
    ap.add_argument("--rolling", action="store_true", help="Run rolling backtest for tabular models")
    args = ap.parse_args()

    results = run_pipeline(
        features_dir=args.features_dir,
        symbol=args.symbol,
        run_rolling=args.rolling,
    )

    print("\n" + "=" * 60)
    print("FORECASTING RESULTS (same split, same metrics)")
    print("=" * 60)
    for model_name, metrics in results.items():
        if not model_name.endswith("_rolling"):
            print(f"\n{model_name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    main()

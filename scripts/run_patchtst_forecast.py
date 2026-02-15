#!/usr/bin/env python3
"""
PatchTST stock time-series forecasting: full pipeline with dataset formatting and evaluation.

Demonstrates:
  - Dataset formatting: load features -> time split -> prepare_sequential(seq_len)
  - PatchTST training with optional validation and early stopping
  - Evaluation framework: RMSE, MAE, MAPE, R2 on test set

Usage:
  python scripts/run_patchtst_forecast.py --symbol FPT
  python scripts/run_patchtst_forecast.py --symbol FPT --seq-len 40 --patch-len 8 --epochs 30
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.splits import get_train_val_test_splits
from src.evaluation.data import prepare_sequential, DEFAULT_TARGET
import pandas as pd

from src.models.forecasting.patchtst import (
    PatchTSTForecaster,
    evaluate_forecast,
)


def load_features(features_dir: Path, symbol: str):
    path = features_dir / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    parser = argparse.ArgumentParser(description="PatchTST forecasting pipeline with evaluation.")
    parser.add_argument("--symbol", type=str, default="FPT")
    parser.add_argument("--features-dir", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=20, help="Sliding window length")
    parser.add_argument("--patch-len", type=int, default=8, help="Patch length (must divide seq_len or seq_len is trimmed)")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default=None, help="Save metrics JSON here")
    args = parser.parse_args()

    features_dir = Path(args.features_dir) if args.features_dir else _ROOT / "data" / "features" / "vn30"
    df = load_features(features_dir, args.symbol)
    train_df, val_df, test_df = get_train_val_test_splits(
        df, date_col="date",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Dataset formatting: (n_samples, seq_len, n_features) for PatchTST
    data = prepare_sequential(
        train_df, val_df, test_df,
        seq_len=args.seq_len,
        target=DEFAULT_TARGET,
    )

    # Training pipeline: fit with validation for early stopping
    model = PatchTSTForecaster(
        patch_len=args.patch_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.early_stopping,
    )
    model.fit(
        data.X_train,
        data.y_train,
        X_val=data.X_val,
        y_val=data.y_val,
    )

    # Evaluation framework
    y_pred = model.predict(data.X_test)
    metrics = evaluate_forecast(data.y_test, y_pred)

    print("\n--- PatchTST evaluation (test set) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.symbol}_patchtst_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

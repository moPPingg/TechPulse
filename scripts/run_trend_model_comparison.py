#!/usr/bin/env python3
"""
Compare LSTM, iTransformer, and PatchTST for stock trend prediction on the same data.

- LSTM and iTransformer: direct trend classification (same sliding windows + trend labels).
- PatchTST: trained as regressor (predict next-day return), then predictions are thresholded
  to trend labels and evaluated with the same metrics for fair comparison.

Dataset: Same time-aware splits and sliding windows; PatchTST uses continuous return
targets for training, then we binarize its test predictions.

Usage:
  python scripts/run_trend_model_comparison.py --symbol FPT
  python scripts/run_trend_model_comparison.py --symbol FPT VCB --epochs 20 --no-lightgbm

Prerequisites: Feature CSVs in features_dir; pip install torch lightgbm
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.models.lightgbm_trend import (
    LightGBMTrendPipeline,
    DEFAULT_RETURN_COL,
    continuous_to_trend_labels,
    evaluate_trend,
)
from src.models.lstm_trend import (
    LSTMTrendPipeline,
    build_sliding_windows_from_splits,
    scale_sequences,
)
from src.models.itransformer_trend import iTransformerTrendPipeline
from src.models.forecasting.patchtst import PatchTSTForecaster


def _load_features_dir_from_config() -> Path:
    try:
        import yaml
        cfg_path = _ROOT / "configs" / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            data = cfg.get("data", {})
            features_dir = data.get("features_dir", "data/features/vn30")
            p = Path(features_dir)
            if not p.is_absolute():
                p = _ROOT / p
            return p
    except Exception:
        pass
    return _ROOT / "data" / "features" / "vn30"


def _continuous_targets_for_windows(
    full_df: pd.DataFrame,
    return_col: str,
    seq_len: int,
    n_train: int,
    n_val: int,
    n_test: int,
) -> tuple:
    """
    Return (y_train_cont, y_val_cont, y_test_cont) for the same sample indices
    as build_sliding_windows. Used to train PatchTST on regression.
    """
    n_full = len(full_df)
    next_ret = full_df[return_col].shift(-1).values.astype(np.float64)
    next_ret = np.nan_to_num(next_ret, nan=0.0)
    # Sample k has window ending at row seq_len-1+k, next return at row seq_len+k
    y_cont_all = next_ret[seq_len - 1 : n_full - 1]  # length n_full - seq_len - 1
    train_end = max(0, n_train - seq_len)
    val_end = train_end + n_val
    y_train_cont = y_cont_all[:train_end]
    y_val_cont = y_cont_all[train_end:val_end]
    y_test_cont = y_cont_all[val_end:]
    return y_train_cont, y_val_cont, y_test_cont


def _print_comparison(symbol: str, metrics_by_model: dict) -> None:
    """Print side-by-side metrics for all models."""
    print(f"\n--- Comparison: {symbol} ---")
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = list(metrics_by_model.keys())
    print("  " + "".join(f"{m:14s}" for m in models))
    for key in keys:
        row = "  " + "".join(f"{metrics_by_model[m].get(key, 0):.4f}".ljust(16) for m in models)
        print(f"  {key:12s} " + row)
    for m in models:
        print(f"  confusion_matrix {m}:", metrics_by_model[m].get("confusion_matrix"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LSTM, iTransformer, and PatchTST for stock trend prediction."
    )
    parser.add_argument("--symbol", nargs="+", required=True, help="Symbol(s), e.g. FPT VCB")
    parser.add_argument("--features-dir", type=str, default=None)
    parser.add_argument("--from-date", type=str, default=None)
    parser.add_argument("--to-date", type=str, default=None)
    parser.add_argument("--include-news", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--purge-gap", type=int, default=0)
    parser.add_argument("--threshold-up", type=float, default=0.0)
    parser.add_argument("--threshold-down", type=float, default=0.0)
    parser.add_argument("--n-classes", type=int, default=2, choices=[2, 3])
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2, help="LSTM/iTransformer lr")
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--no-lightgbm", action="store_true", help="Skip LightGBM (faster)")
    parser.add_argument("--no-patchtst", action="store_true", help="Skip PatchTST")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    features_dir = args.features_dir or str(_load_features_dir_from_config())
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "data" / "trend_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    common_kw = {
        "from_date": args.from_date,
        "to_date": args.to_date,
        "include_news": args.include_news,
        "trend_threshold_up": args.threshold_up,
        "trend_threshold_down": args.threshold_down,
        "n_classes": args.n_classes,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "purge_gap": args.purge_gap,
    }

    results = {}
    for symbol in args.symbol:
        print(f"\n========== {symbol} ==========")
        res = {"symbol": symbol, "lightgbm": {}, "lstm": {}, "itransformer": {}, "patchtst": {}}
        metrics_by_model = {}
        try:
            # --- LightGBM (optional) ---
            if not args.no_lightgbm:
                print("Training LightGBM...")
                lgb_pipe = LightGBMTrendPipeline(
                    symbol=symbol,
                    features_dir=features_dir,
                    tune=False,
                    **common_kw,
                )
                lgb_pipe.run()
                res["lightgbm"] = lgb_pipe.metrics
                metrics_by_model["LightGBM"] = lgb_pipe.metrics

            # --- LSTM ---
            print("Training LSTM...")
            lstm_pipe = LSTMTrendPipeline(
                symbol=symbol,
                features_dir=features_dir,
                seq_len=args.seq_len,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                early_stopping_patience=args.early_stopping,
                **common_kw,
            )
            lstm_pipe.run()
            res["lstm"] = lstm_pipe.metrics
            metrics_by_model["LSTM"] = lstm_pipe.metrics

            # --- iTransformer ---
            print("Training iTransformer...")
            itrans_pipe = iTransformerTrendPipeline(
                symbol=symbol,
                features_dir=features_dir,
                seq_len=args.seq_len,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1e-3,
                early_stopping_patience=args.early_stopping,
                **common_kw,
            )
            itrans_pipe.run()
            res["itransformer"] = itrans_pipe.metrics
            metrics_by_model["iTransformer"] = itrans_pipe.metrics

            # --- PatchTST: regression then threshold to trend ---
            if not args.no_patchtst and lstm_pipe.X_train is not None:
                print("Training PatchTST (regression, then threshold to trend)...")
                train_df, val_df, test_df = lstm_pipe.split()
                full = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
                n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
                y_train_cont, y_val_cont, y_test_cont = _continuous_targets_for_windows(
                    full, lstm_pipe.return_col, args.seq_len, n_train, n_val, n_test
                )
                # Build raw (unscaled) X so PatchTST fits its own scaler
                (X_train_r, _, X_val_r, _, X_test_r, _) = build_sliding_windows_from_splits(
                    train_df, val_df, test_df,
                    lstm_pipe.feature_cols,
                    return_col=lstm_pipe.return_col,
                    seq_len=args.seq_len,
                    threshold_up=args.threshold_up,
                    threshold_down=args.threshold_down,
                    n_classes=args.n_classes,
                )
                n_trim = min(len(X_train_r), len(y_train_cont))
                X_train_r, y_train_cont = X_train_r[:n_trim], np.asarray(y_train_cont[:n_trim], dtype=np.float32)
                n_trim_v = min(len(X_val_r), len(y_val_cont))
                X_val_r, y_val_cont = X_val_r[:n_trim_v], np.asarray(y_val_cont[:n_trim_v], dtype=np.float32)
                patch_len = min(8, args.seq_len // 2 or 1)
                ptst = PatchTSTForecaster(
                    patch_len=patch_len,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=1e-3,
                    early_stopping_patience=args.early_stopping,
                )
                ptst.fit(X_train_r, y_train_cont, X_val=X_val_r, y_val=y_val_cont)
                y_pred_cont = ptst.predict(X_test_r)
                trend_pred = continuous_to_trend_labels(
                    y_pred_cont,
                    args.threshold_up,
                    args.threshold_down,
                    args.n_classes,
                )
                ptst_metrics = evaluate_trend(
                    lstm_pipe.y_test, trend_pred, y_prob=None, n_classes=args.n_classes
                )
                res["patchtst"] = ptst_metrics
                metrics_by_model["PatchTST"] = ptst_metrics

            _print_comparison(symbol, metrics_by_model)
            results[symbol] = res
            out_file = out_dir / f"{symbol}_trend_models.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Wrote {out_file}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error for {symbol}: {e}")
            results[symbol] = {"symbol": symbol, "error": str(e)}

    print("\n========== Summary (F1) ==========")
    for symbol, res in results.items():
        if "error" in res:
            print(f"  {symbol}: failed - {res['error']}")
            continue
        f1s = [(m, res.get(m, {}).get("f1", 0)) for m in ["lightgbm", "lstm", "itransformer", "patchtst"] if res.get(m)]
        if f1s:
            best = max(f1s, key=lambda x: x[1])
            print(f"  {symbol}: " + " ".join(f"{m}={v:.4f}" for m, v in f1s) + f" -> best: {best[0]}")

    summary_path = out_dir / "trend_models_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

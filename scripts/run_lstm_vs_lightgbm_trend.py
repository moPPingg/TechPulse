#!/usr/bin/env python3
"""
Run LSTM and LightGBM trend pipelines on the same data and compare metrics.

Uses the same time-aware splits and trend target so comparison is fair.
Output: side-by-side metrics (accuracy, F1, ROC-AUC, etc.) and optional JSON.

Usage:
  python scripts/run_lstm_vs_lightgbm_trend.py --symbol FPT
  python scripts/run_lstm_vs_lightgbm_trend.py --symbol FPT VCB --seq-len 20 --epochs 30
  python scripts/run_lstm_vs_lightgbm_trend.py --symbol FPT --out-dir data/trend_comparison

Prerequisites:
  - Feature CSVs in features_dir (run price pipeline first).
  - pip install torch lightgbm
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.lightgbm_trend import LightGBMTrendPipeline
from src.models.lstm_trend import LSTMTrendPipeline


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


def _print_comparison(symbol: str, lgb_metrics: dict, lstm_metrics: dict) -> None:
    """Print side-by-side comparison of key metrics."""
    print(f"\n--- Comparison: {symbol} ---")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        lv = lgb_metrics.get(key, 0.0)
        rv = lstm_metrics.get(key, 0.0)
        diff = rv - lv
        print(f"  {key:12s}  LightGBM: {lv:.4f}  LSTM: {rv:.4f}  (diff {diff:+.4f})")
    print("  confusion_matrix  LightGBM:", lgb_metrics.get("confusion_matrix"))
    print("  confusion_matrix  LSTM:    ", lstm_metrics.get("confusion_matrix"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LSTM vs LightGBM for stock trend classification on the same data."
    )
    parser.add_argument("--symbol", nargs="+", required=True, help="Symbol(s), e.g. FPT VCB")
    parser.add_argument("--features-dir", type=str, default=None, help="Feature CSVs directory (default: from config)")
    parser.add_argument("--from-date", type=str, default=None)
    parser.add_argument("--to-date", type=str, default=None)
    parser.add_argument("--include-news", action="store_true", help="Merge news daily features")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--purge-gap", type=int, default=0)
    parser.add_argument("--threshold-up", type=float, default=0.0)
    parser.add_argument("--threshold-down", type=float, default=0.0)
    parser.add_argument("--n-classes", type=int, default=2, choices=[2, 3])
    # LSTM-specific
    parser.add_argument("--seq-len", type=int, default=20, help="Sliding window length for LSTM")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default=None, help="Save comparison JSON here")
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
        res = {"symbol": symbol, "lightgbm": {}, "lstm": {}, "comparison": {}}
        try:
            # --- LightGBM baseline ---
            print("Training LightGBM...")
            lgb_pipe = LightGBMTrendPipeline(
                symbol=symbol,
                features_dir=features_dir,
                tune=False,
                **common_kw,
            )
            lgb_pipe.run()
            res["lightgbm"] = lgb_pipe.metrics

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
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                **common_kw,
            )
            lstm_pipe.run()
            res["lstm"] = lstm_pipe.metrics

            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                res["comparison"][f"{key}_lightgbm"] = res["lightgbm"].get(key, 0.0)
                res["comparison"][f"{key}_lstm"] = res["lstm"].get(key, 0.0)
                res["comparison"][f"{key}_diff"] = res["lstm"].get(key, 0.0) - res["lightgbm"].get(key, 0.0)

            _print_comparison(symbol, res["lightgbm"], res["lstm"])
            results[symbol] = res

            # Save per-symbol JSON (metrics only, no raw confusion matrices if huge)
            out_file = out_dir / f"{symbol}_comparison.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Wrote {out_file}")

        except Exception as e:
            print(f"Error for {symbol}: {e}")
            results[symbol] = {"symbol": symbol, "error": str(e)}

    # Summary
    print("\n========== Summary ==========")
    for symbol, res in results.items():
        if "error" in res:
            print(f"  {symbol}: failed - {res['error']}")
            continue
        lgb_f1 = res["lightgbm"].get("f1", 0)
        lstm_f1 = res["lstm"].get("f1", 0)
        winner = "LSTM" if lstm_f1 > lgb_f1 else "LightGBM"
        print(f"  {symbol}: F1 LightGBM={lgb_f1:.4f} LSTM={lstm_f1:.4f} -> {winner} wins on F1")

    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the LightGBM trend prediction pipeline for one or more symbols.

Usage:
  python scripts/run_lightgbm_trend.py --symbol FPT
  python scripts/run_lightgbm_trend.py --symbol FPT VCB --tune --trials 25
  python scripts/run_lightgbm_trend.py --symbol FPT --include-news --features-dir data/features/vn30

Prerequisites:
  - Feature CSVs in features_dir (run price pipeline first: fetch_vn30 or run_forecasting_pipeline).
  - pip install lightgbm  (and optuna for --tune).
"""

import argparse
import json
import sys
from pathlib import Path

# Project root for imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.lightgbm_trend import LightGBMTrendPipeline


def _load_features_dir_from_config() -> Path:
    """Read features_dir from configs/config.yaml; resolve relative to project root."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM trend pipeline: train and evaluate.")
    parser.add_argument("--symbol", nargs="+", required=True, help="Symbol(s), e.g. FPT VCB")
    parser.add_argument("--features-dir", type=str, default=None, help="Directory with feature CSVs (default: from config)")
    parser.add_argument("--from-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--to-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--include-news", action="store_true", help="Merge news daily features from engine")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials (default 20)")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--purge-gap", type=int, default=0, help="Gap between splits to avoid leakage")
    parser.add_argument("--threshold-up", type=float, default=0.0, help="Return threshold for 'up' class")
    parser.add_argument("--threshold-down", type=float, default=0.0, help="Return threshold for 'down' (3-class)")
    parser.add_argument("--n-classes", type=int, default=2, choices=[2, 3], help="Binary (2) or 3-class trend")
    parser.add_argument("--out-dir", type=str, default=None, help="Save metrics and importance to this dir")
    args = parser.parse_args()

    features_dir = args.features_dir or str(_load_features_dir_from_config())
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "data" / "lightgbm_trend"
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbol:
        print(f"\n--- {symbol} ---")
        try:
            pipe = LightGBMTrendPipeline(
                symbol=symbol,
                features_dir=features_dir,
                from_date=args.from_date,
                to_date=args.to_date,
                include_news=args.include_news,
                trend_threshold_up=args.threshold_up,
                trend_threshold_down=args.threshold_down,
                n_classes=args.n_classes,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                purge_gap=args.purge_gap,
                tune=args.tune,
                tune_trials=args.trials,
            )
            metrics = pipe.run()
            importance_df = pipe.get_feature_importance()

            # Print summary
            print("Test metrics:", {k: v for k, v in metrics.items() if k != "confusion_matrix"})
            print("Confusion matrix:", metrics.get("confusion_matrix"))
            print("Top 10 features:")
            print(importance_df.head(10).to_string(index=False))

            # Save
            out_prefix = out_dir / f"{symbol}_trend"
            with open(f"{out_prefix}_metrics.json", "w", encoding="utf-8") as f:
                json.dump({k: (v if k != "confusion_matrix" else v) for k, v in metrics.items()}, f, indent=2)
            importance_df.to_csv(f"{out_prefix}_importance.csv", index=False)
            print(f"Saved to {out_prefix}_metrics.json and {out_prefix}_importance.csv")
        except FileNotFoundError as e:
            print(f"Skip {symbol}: {e}")
        except Exception as e:
            print(f"Error for {symbol}: {e}")
            raise


if __name__ == "__main__":
    main()

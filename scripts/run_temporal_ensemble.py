#!/usr/bin/env python3
"""
Temporal stacking ensemble: train LightGBM, LSTM, PatchTST, iTransformer, then stack
their predictions with a meta-model (logistic regression) and evaluate.

Flow:
  1. Load data, time-aware split (train/val/test).
  2. Train base models: LightGBM (tabular), LSTM, iTransformer, PatchTST (sequential).
  3. Collect base model probabilities on VAL and TEST. Align lengths so all models
     contribute to the same set of samples (trim to smallest val/test size).
  4. Fit stacking meta-model on (stacked_val_proba, y_val).
  5. Evaluate ensemble and each base on test; print comparison table.

Usage:
  python scripts/run_temporal_ensemble.py --symbol FPT
  python scripts/run_temporal_ensemble.py --symbol FPT VCB --epochs 25 --meta mlp
  python scripts/run_temporal_ensemble.py --symbol FPT --no-lightgbm --no-patchtst

Prerequisites: Feature CSVs; pip install torch lightgbm scikit-learn
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
    predict_lstm,
)
from src.models.itransformer_trend import iTransformerTrendPipeline, predict_itransformer
from src.models.forecasting.patchtst import PatchTSTForecaster
from src.ensemble.stacking import (
    TemporalStackingEnsemble,
    compare_models,
    print_comparison,
)


def _load_features_dir_from_config() -> Path:
    try:
        import yaml
        cfg_path = _ROOT / "configs" / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            p = Path(cfg.get("data", {}).get("features_dir", "data/features/vn30"))
            return p if p.is_absolute() else _ROOT / p
    except Exception:
        pass
    return _ROOT / "data" / "features" / "vn30"


def _continuous_targets_for_windows(full_df: pd.DataFrame, return_col: str, seq_len: int, n_train: int, n_val: int, n_test: int):
    n_full = len(full_df)
    next_ret = full_df[return_col].shift(-1).values.astype(np.float64)
    next_ret = np.nan_to_num(next_ret, nan=0.0)
    y_cont_all = next_ret[seq_len - 1 : n_full - 1]
    train_end = max(0, n_train - seq_len)
    val_end = train_end + n_val
    y_train_cont = y_cont_all[:train_end]
    y_val_cont = y_cont_all[train_end:val_end]
    y_test_cont = y_cont_all[val_end:]
    return y_train_cont, y_val_cont, y_test_cont


def _regression_to_proba(y_cont: np.ndarray, n_classes: int, threshold_up: float, threshold_down: float) -> np.ndarray:
    """Convert regression predictions to class probabilities for stacking (binary: sigmoid-style)."""
    y_cont = np.asarray(y_cont).ravel()
    if n_classes == 2:
        # P(up) increasing in y_cont; clip to (0,1)
        p_up = 1.0 / (1.0 + np.exp(-np.clip(y_cont * 10, -20, 20)))
        return np.column_stack([1 - p_up, p_up])
    # 3-class: soft assignment from distance to thresholds
    p_up = np.maximum(0, np.minimum(1, (y_cont - threshold_down) / (threshold_up - threshold_down + 1e-8)))
    p_down = np.maximum(0, np.minimum(1, (threshold_down - y_cont) / (threshold_down - threshold_up + 1e-8)))
    p_mid = 1.0 - p_up - p_down
    p_mid = np.maximum(0, p_mid)
    s = p_up + p_mid + p_down
    return np.column_stack([p_down / s, p_mid / s, p_up / s])


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal stacking ensemble (LightGBM + LSTM + PatchTST + iTransformer).")
    parser.add_argument("--symbol", nargs="+", required=True)
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
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--no-lightgbm", action="store_true")
    parser.add_argument("--no-patchtst", action="store_true")
    parser.add_argument("--meta", type=str, default="logistic", choices=["logistic", "mlp"])
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    features_dir = args.features_dir or str(_load_features_dir_from_config())
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "data" / "temporal_ensemble"
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
        try:
            # ---- 1. LightGBM ----
            lgb_pipe = None
            if not args.no_lightgbm:
                print("Training LightGBM...")
                lgb_pipe = LightGBMTrendPipeline(
                    symbol=symbol,
                    features_dir=features_dir,
                    tune=False,
                    **common_kw,
                )
                lgb_pipe.run()

            # ---- 2. LSTM ----
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

            # ---- 3. iTransformer ----
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

            train_df, val_df, test_df = lstm_pipe.split()

            # ---- 4. PatchTST (regression -> proba for stacking) ----
            ptst_pipe = None
            if not args.no_patchtst:
                print("Training PatchTST...")
                full = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
                n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
                y_train_cont, y_val_cont, y_test_cont = _continuous_targets_for_windows(
                    full, lstm_pipe.return_col, args.seq_len, n_train, n_val, n_test
                )
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
                ptst_pipe = {"ptst": ptst, "X_val_r": X_val_r, "X_test_r": X_test_r}

            # ---- Align lengths: use sequential val/test size; trim tabular (LightGBM) if needed ----
            n_val_seq = len(lstm_pipe.X_val)
            n_test_seq = len(lstm_pipe.X_test)
            n_val_tab = (len(val_df) - 1) if lgb_pipe is not None and lgb_pipe.X_val is not None else n_val_seq
            n_test_tab = (len(test_df) - 1) if lgb_pipe is not None and lgb_pipe.X_test is not None else n_test_seq
            n_val_c = min(n_val_seq, n_val_tab)
            n_test_c = min(n_test_seq, n_test_tab)

            # ---- Base predictions for stacking (proba) ----
            def get_lgb_proba(pipe, X, n_use):
                if X is None or len(X) == 0:
                    return None
                p = pipe.model.predict(X[:n_use])
                if p.ndim == 1 and args.n_classes == 2:
                    p = np.column_stack([1 - p, p])
                return p

            base_preds_val = {}
            base_preds_test = {}
            y_val_c = lstm_pipe.y_val[:n_val_c]
            y_test_c = lstm_pipe.y_test[:n_test_c]

            if lgb_pipe is not None:
                base_preds_val["LightGBM"] = get_lgb_proba(lgb_pipe, lgb_pipe.X_val, n_val_c)
                base_preds_test["LightGBM"] = get_lgb_proba(lgb_pipe, lgb_pipe.X_test, n_test_c)

            _, probs_lstm_val = predict_lstm(lstm_pipe.model, lstm_pipe.X_val[:n_val_c], device=lstm_pipe.device, batch_size=args.batch_size)
            _, probs_lstm_test = predict_lstm(lstm_pipe.model, lstm_pipe.X_test[:n_test_c], device=lstm_pipe.device, batch_size=args.batch_size)
            base_preds_val["LSTM"] = probs_lstm_val
            base_preds_test["LSTM"] = probs_lstm_test

            _, probs_itrans_val = predict_itransformer(itrans_pipe.model, itrans_pipe.X_val[:n_val_c], device=itrans_pipe.device, batch_size=args.batch_size)
            _, probs_itrans_test = predict_itransformer(itrans_pipe.model, itrans_pipe.X_test[:n_test_c], device=itrans_pipe.device, batch_size=args.batch_size)
            base_preds_val["iTransformer"] = probs_itrans_val
            base_preds_test["iTransformer"] = probs_itrans_test

            if ptst_pipe is not None:
                ptst = ptst_pipe["ptst"]
                X_val_r = ptst_pipe["X_val_r"]
                X_test_r = ptst_pipe["X_test_r"]
                y_ptst_val = ptst.predict(X_val_r[:n_val_c])
                y_ptst_test = ptst.predict(X_test_r[:n_test_c])
                base_preds_val["PatchTST"] = _regression_to_proba(
                    y_ptst_val, args.n_classes, args.threshold_up, args.threshold_down
                )
                base_preds_test["PatchTST"] = _regression_to_proba(
                    y_ptst_test, args.n_classes, args.threshold_up, args.threshold_down
                )

            # ---- Stacking ----
            print("Fitting stacking meta-model...")
            ensemble = TemporalStackingEnsemble(
                n_classes=args.n_classes,
                meta_model=args.meta,
            )
            ensemble.fit(base_preds_val, y_val_c)
            ensemble_metrics = ensemble.evaluate(y_test_c, base_preds_test)

            # ---- Base metrics on same test slice ----
            base_metrics = {}
            if "LightGBM" in base_preds_test:
                pred_lgb = np.argmax(base_preds_test["LightGBM"], axis=1)
                base_metrics["LightGBM"] = evaluate_trend(y_test_c, pred_lgb, y_prob=base_preds_test["LightGBM"], n_classes=args.n_classes)
            pred_lstm = np.argmax(base_preds_test["LSTM"], axis=1)
            base_metrics["LSTM"] = evaluate_trend(y_test_c, pred_lstm, y_prob=base_preds_test["LSTM"], n_classes=args.n_classes)
            pred_itrans = np.argmax(base_preds_test["iTransformer"], axis=1)
            base_metrics["iTransformer"] = evaluate_trend(y_test_c, pred_itrans, y_prob=base_preds_test["iTransformer"], n_classes=args.n_classes)
            if "PatchTST" in base_preds_test:
                pred_ptst = np.argmax(base_preds_test["PatchTST"], axis=1)
                base_metrics["PatchTST"] = evaluate_trend(y_test_c, pred_ptst, y_prob=base_preds_test["PatchTST"], n_classes=args.n_classes)

            comparison = compare_models(base_metrics, ensemble_metrics)
            print_comparison(comparison)

            res = {
                "symbol": symbol,
                "base_metrics": base_metrics,
                "ensemble_metrics": ensemble_metrics,
                "comparison": comparison,
                "n_val": n_val_c,
                "n_test": n_test_c,
            }
            results[symbol] = res
            out_file = out_dir / f"{symbol}_ensemble.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in res.items() if k != "comparison"}, f, indent=2)
            print(f"Wrote {out_file}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error for {symbol}: {e}")
            results[symbol] = {"symbol": symbol, "error": str(e)}

    summary_path = out_dir / "ensemble_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pre-compute forecasts for all VN30 symbols (or specified symbols).
Saves to data/forecasts/{symbol}.json for the app to consume.

Usage:
  python scripts/run_inference.py                    # all VN30
  python scripts/run_inference.py --symbol FPT VCB   # specific symbols

Requires: data/features/vn30/*.csv (run fetch_vn30 first)
"""

import sys
import argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.inference.service import run_inference_for_symbol


def main():
    ap = argparse.ArgumentParser(description="Run inference and cache forecasts")
    ap.add_argument("--symbol", nargs="*", default=None, help="Symbols (default: all VN30)")
    ap.add_argument("--features-dir", default="data/features/vn30", help="Features directory")
    args = ap.parse_args()

    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol]
    else:
        try:
            import yaml
            with open(_project_root / "configs" / "symbols.yaml", "r") as f:
                cfg = yaml.safe_load(f) or {}
            symbols = cfg.get("symbols") or cfg.get("vn30", [])
        except Exception:
            symbols = [
                "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
                "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSI", "STB", "TCB", "TPB",
                "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE", "SSB", "PDR",
            ]

    features_dir = str(_project_root / args.features_dir)
    success = 0
    for sym in symbols:
        print(f"Running inference for {sym}...")
        try:
            r = run_inference_for_symbol(sym, features_dir=features_dir)
            if r:
                print(f"  {sym}: mean={r.ensemble_mean:.3f}%, std={r.ensemble_std:.3f}, conf={r.confidence_score:.2f}")
                success += 1
            else:
                print(f"  {sym}: SKIP (no features)")
        except Exception as e:
            print(f"  {sym}: FAILED - {e}")
    print(f"\nDone. {success}/{len(symbols)} symbols cached.")


if __name__ == "__main__":
    main()

"""
Backfill script using vnstock (not vnstock3) to close the Jan-Mar 2026 data gap.
Run from project root: python scripts/backfill_missing_data.py
"""
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import vnstock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

BACKFILL_START = "2026-01-01"
BACKFILL_END   = datetime.now().strftime("%Y-%m-%d")
RAW_DIR = Path("data/raw")

success, failed = 0, []

for ticker in VN30:
    csv_path = RAW_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        logger.warning(f"[{ticker}] CSV not found at {csv_path}, skipping.")
        failed.append(ticker)
        continue

    try:
        df_new = vnstock.stock_historical_data(ticker, BACKFILL_START, BACKFILL_END)

        if df_new is None or df_new.empty:
            logger.warning(f"[{ticker}] No new data returned.")
            failed.append(ticker)
            continue

        # Normalize columns: rename 'time' -> 'date'
        df_new = df_new.rename(columns={'time': 'date'})
        df_new['date'] = pd.to_datetime(df_new['date'])

        # Keep only OHLCV columns
        keep_cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in df_new.columns]
        df_new = df_new[keep_cols]

        # Load existing CSV
        df_old = pd.read_csv(csv_path)
        df_old['date'] = pd.to_datetime(df_old['date'])

        # Merge, deduplicate, sort
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=['date'])
        df_merged = df_merged.sort_values('date').reset_index(drop=True)

        df_merged.to_csv(csv_path, index=False)

        logger.info(
            f"[{ticker}] ✅ Merged {len(df_new)} rows → total {len(df_merged)} rows. "
            f"Range: {df_merged['date'].min().date()} → {df_merged['date'].max().date()}"
        )
        success += 1

    except Exception as e:
        logger.error(f"[{ticker}] ❌ Failed: {e}")
        failed.append(ticker)

logger.info(f"\n=== BACKFILL COMPLETE: {success}/{len(VN30)} succeeded. Failed: {failed or 'None'} ===")

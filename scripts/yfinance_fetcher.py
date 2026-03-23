"""
VN30 Daily Data Fetcher — yfinance + schedule
Chạy tự động lúc 16:00 GMT+7 mỗi ngày giao dịch (sau khi HOSE đóng cửa 15:00).
"""

import os
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Yahoo Finance dùng suffix .VN cho cổ phiếu HOSE
VN30_TICKERS = [
    "ACB.VN", "BCM.VN", "BID.VN", "BVH.VN", "CTG.VN",
    "FPT.VN", "GAS.VN", "GVR.VN", "HDB.VN", "HPG.VN",
    "MBB.VN", "MSN.VN", "MWG.VN", "PLX.VN", "POW.VN",
    "SAB.VN", "SSI.VN", "STB.VN", "TCB.VN", "TPB.VN",
    "VCB.VN", "VHM.VN", "VIB.VN", "VIC.VN", "VJC.VN",
    "VNM.VN", "VPB.VN", "VRE.VN", "SSB.VN", "PDR.VN",
]

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------
def _symbol_to_filename(ticker: str) -> str:
    """ACB.VN -> ACB"""
    return ticker.replace(".VN", "")


def fetch_one(ticker: str, lookback_days: int = 5) -> bool:
    """
    Fetch the last `lookback_days` of OHLCV for `ticker` and append/update the CSV.
    Returns True on success.
    """
    name = _symbol_to_filename(ticker)
    csv_path = DATA_DIR / f"{name}.csv"

    # Determine fetch window
    end_dt = datetime.utcnow() + timedelta(hours=7)          # GMT+7
    start_dt = end_dt - timedelta(days=lookback_days)

    try:
        raw = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        log.error(f"[{name}] yfinance download failed: {exc}")
        return False

    if raw.empty:
        log.warning(f"[{name}] No data returned from yfinance.")
        return False

    # Normalise columns
    new = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })[["open", "high", "low", "close", "volume"]].copy()
    new.index.name = "date"
    new = new.reset_index()
    new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")

    # Merge with existing CSV (upsert on date)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing["date"] = existing["date"].astype(str)
        merged = (
            pd.concat([existing, new])
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        merged = new.sort_values("date").reset_index(drop=True)

    merged.to_csv(csv_path, index=False)
    log.info(f"[{name}] Updated -> {csv_path}  ({len(merged)} rows total)")
    return True


def fetch_all() -> None:
    log.info("=" * 60)
    log.info(f"Starting VN30 fetch run  ({datetime.utcnow():%Y-%m-%d %H:%M} UTC)")

    end_dt = datetime.utcnow() + timedelta(hours=7)
    start_dt = end_dt - timedelta(days=5)

    try:
        # Batch download tất cả 30 ticker trong 1 request — nhanh hơn ~10x
        raw = yf.download(
            VN30_TICKERS,
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as exc:
        log.error(f"Batch download failed: {exc}")
        return

    ok, fail = 0, 0
    for ticker in VN30_TICKERS:
        name = _symbol_to_filename(ticker)
        csv_path = DATA_DIR / f"{name}.csv"
        try:
            df = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].dropna()
            if df.empty:
                log.warning(f"[{name}] No data returned.")
                fail += 1
                continue

            new = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            }).copy()
            new.index.name = "date"
            new = new.reset_index()
            new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")

            if csv_path.exists():
                existing = pd.read_csv(csv_path)
                existing["date"] = existing["date"].astype(str)
                merged = (
                    pd.concat([existing, new])
                    .drop_duplicates(subset="date", keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
            else:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                merged = new.sort_values("date").reset_index(drop=True)

            merged.to_csv(csv_path, index=False)
            log.info(f"[{name}] Updated -> {csv_path}  ({len(merged)} rows total)")
            ok += 1
        except Exception as exc:
            log.error(f"[{name}] Failed to process: {exc}")
            fail += 1

    log.info(f"Done — success: {ok}, failed: {fail}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
def start_scheduler() -> None:
    """
    Run fetch_all once immediately, then every weekday at 09:00 UTC (= 16:00 GMT+7).
    """
    log.info("VN30 fetcher starting. Running initial fetch...")
    fetch_all()

    # 09:00 UTC = 16:00 ICT (GMT+7) — after HOSE close at 14:45 local
    schedule.every().monday.at("09:00").do(fetch_all)
    schedule.every().tuesday.at("09:00").do(fetch_all)
    schedule.every().wednesday.at("09:00").do(fetch_all)
    schedule.every().thursday.at("09:00").do(fetch_all)
    schedule.every().friday.at("09:00").do(fetch_all)

    log.info("Scheduler running. Waiting for next job...")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    start_scheduler()

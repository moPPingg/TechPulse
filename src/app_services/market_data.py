"""
MarketDataService: Single entry point for price data, indicators, and chart series.

Responsibilities:
- Load features from CSV (primary) or CafeF API (fallback)
- Compute technical indicators (MA, RSI, volatility)
- Provide chart-ready OHLCV and indicator series

Used by: api.py, recommendation/signal flow
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent.parent.parent


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


def _load_features(symbol: str) -> Optional[pd.DataFrame]:
    """Load features DataFrame for symbol. Returns None if file missing."""
    try:
        cfg = _load_config()
        base = Path(cfg.get("data", {}).get("features_dir", "data/features/vn30"))
        if not base.is_absolute():
            base = _ROOT / base
        path = base / f"{symbol.upper()}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


def _fetch_price_from_cafef(
    symbol: str,
    days: int = 90,
    end_date_str: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fallback: fetch OHLC from CafeF API when features missing.
    end_date_str: YYYY-MM-DD to fetch up to that date (for chart horizon)."""
    try:
        from datetime import datetime, timedelta
        from src.crawl.cafef_scraper import fetch_price_cafef
        if end_date_str:
            try:
                end_dt = pd.to_datetime(end_date_str)
            except Exception:
                end_dt = datetime.now()
        else:
            end_dt = datetime.now()
        start = end_dt - timedelta(days=min(days, 730))  # up to 2 years
        df = fetch_price_cafef(
            symbol=symbol.upper(),
            start_date=start.strftime("%d/%m/%Y"),
            end_date=end_dt.strftime("%d/%m/%Y"),
            page_size=500,
            timeout=15,
        )
        if df is None or len(df) == 0:
            return None
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MA, RSI, volatility for chart and signals."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df["return_1d"] = df["close"].pct_change() * 100
    for w in [5, 20, 50]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
    df["volatility_5"] = df["close"].pct_change().rolling(5).std() * 100
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def get_stock_data(
    symbol: str,
    days: int = 90,
    end_date_str: Optional[str] = None,
    fetch_days: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Get price + indicators: features CSV first, fallback CafeF API.
    end_date_str: when set, fetch/filter data up to that date (for chart horizon).
    fetch_days: when set with end_date_str, fetch this many days of history (default: days)."""
    df = _load_features(symbol)
    if df is not None and len(df) > 0:
        if end_date_str:
            try:
                end_dt = pd.to_datetime(end_date_str)
                df = df[df["date"] <= end_dt].tail(days).reset_index(drop=True)
            except Exception:
                df = df.tail(days).reset_index(drop=True)
        else:
            df = df.tail(days).reset_index(drop=True)
        return _compute_indicators(df)
    cafef_days = fetch_days if fetch_days is not None else days
    df = _fetch_price_from_cafef(symbol, days=cafef_days, end_date_str=end_date_str)
    if df is not None and len(df) > 0:
        df = df.tail(days).reset_index(drop=True)
        return _compute_indicators(df)
    return None


def get_indicators(symbol: str, days: int = 90) -> Dict[str, Any]:
    """Return last-row indicators for signals and UI."""
    df = get_stock_data(symbol, days)
    out = {"last_date": None, "close": None, "return_1d": None, "volatility_5": None, "rsi_14": None, "ma_20": None, "ma_50": None}
    if df is not None and len(df) > 0:
        row = df.iloc[-1]
        for col in ["return_1d", "volatility_5", "rsi_14", "ma_20", "ma_50"]:
            if col in row:
                v = row[col]
                out[col] = float(v) if pd.notna(v) else None
        out["last_date"] = str(row.get("date", ""))[:10]
        out["close"] = float(row.get("close", 0)) if "close" in row else None
    return out


def get_chart_data(
    symbol: str,
    days: int = 90,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Return OHLCV, MA, RSI, volatility series for chart.
    end_date: YYYY-MM-DD to show data up to that date (respects chart horizon)."""
    fetch_days = 730 if end_date else int(days)  # more history when filtering by date
    df = get_stock_data(
        symbol,
        days=int(days),
        end_date_str=end_date,
        fetch_days=fetch_days,
    )

    if df is None or len(df) == 0:
        return {"ohlcv": [], "ma": [], "rsi": [], "volatility": []}

    ohlcv = [
        {
            "date": str(row.get("date", ""))[:10],
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "volume": int(row.get("volume", 0)),
        }
        for _, row in df.iterrows()
    ]

    ma_data = []
    for _, row in df.iterrows():
        d = {"date": str(row.get("date", ""))[:10]}
        for w in [5, 20, 50]:
            col = f"ma_{w}"
            if col in row and pd.notna(row[col]):
                d[col] = float(row[col])
        ma_data.append(d)

    rsi_data = [
        {"date": str(row.get("date", ""))[:10], "rsi": float(row.get("rsi_14", 0))}
        for _, row in df.iterrows()
        if "rsi_14" in row and pd.notna(row.get("rsi_14"))
    ]

    vol_data = [
        {"date": str(row.get("date", ""))[:10], "volatility": float(row.get("volatility_5", 0))}
        for _, row in df.iterrows()
        if "volatility_5" in row and pd.notna(row.get("volatility_5"))
    ]

    return {
        "ohlcv": ohlcv,
        "ma": ma_data,
        "rsi": rsi_data,
        "volatility": vol_data,
    }

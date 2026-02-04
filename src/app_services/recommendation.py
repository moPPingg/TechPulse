"""
Recommendation engine: forecast + news + anomaly → Buy/Hold/Avoid + risk metrics + explanation.
All logic is deterministic and tunable; no training here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    """Normal CDF without scipy."""
    z = (x - mu) / sigma if sigma > 0 else 0
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# Project root for relative paths
_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class UserProfile:
    name: str
    capital: float
    years_experience: float
    risk_tolerance: str  # "low" | "medium" | "high"


@dataclass
class RiskAdvice:
    recommendation: str  # "Buy" | "Hold" | "Avoid"
    risk_of_loss_pct: float
    risk_of_ruin_pct: float
    explanation: str


def _load_config() -> dict:
    import yaml
    path = _ROOT / "configs" / "config.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_forecast_signal(symbol: str, features_dir: Optional[str] = None) -> Tuple[str, float, float]:
    """
    Returns (direction, strength, volatility_proxy).
    direction in ("up", "flat", "down"); strength in [0,1]; volatility_proxy >= 0.
    Uses last row of features CSV as proxy when no live model output is available.
    """
    cfg = _load_config()
    base = Path(features_dir or cfg.get("data", {}).get("features_dir", "data/features/vn30"))
    if not base.is_absolute():
        base = _ROOT / base
    path = base / f"{symbol.upper()}.csv"
    if not path.exists():
        logger.warning("No features file for %s at %s", symbol, path)
        return "flat", 0.0, 0.5

    import pandas as pd
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    if len(df) == 0:
        return "flat", 0.0, 0.5

    row = df.iloc[-1]
    # Use return_1d as proxy for "recent momentum" / forecast
    ret = float(row.get("return_1d", 0) or 0)
    vol = float(row.get("volatility_5", 1.0) or 1.0)
    vol = max(vol, 0.1)
    strength = min(1.0, abs(ret) / 2.0)
    if ret > 0.05:
        direction = "up"
    elif ret < -0.05:
        direction = "down"
    else:
        direction = "flat"
    return direction, strength, vol


def get_news_sentiment(symbol: str, news_db_path: Optional[str] = None, days: int = 30) -> Tuple[float, int]:
    """
    Returns (average_sentiment_score, article_count). Score in [-1, 1].
    """
    from datetime import datetime, timedelta
    cfg = _load_config()
    # Prefer news config
    try:
        import yaml
        with open(_ROOT / "configs" / "news.yaml", encoding="utf-8") as f:
            news_cfg = yaml.safe_load(f) or {}
        db_path = news_cfg.get("database", {}).get("path", "data/news/news.db")
    except Exception:
        db_path = "data/news/news.db"
    if news_db_path:
        db_path = news_db_path
    path = Path(db_path)
    if not path.is_absolute():
        path = _ROOT / path
    if not path.exists():
        return 0.0, 0

    try:
        from src.news.db import get_engine, get_articles_by_ticker_date
        conn = get_engine(str(path))
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = get_articles_by_ticker_date(conn, symbol.upper(), date_from=since, limit=100)
        if not rows:
            return 0.0, 0
        scores = [r.get("sentiment_score") for r in rows if r.get("sentiment_score") is not None]
        if not scores:
            return 0.0, len(rows)
        return float(sum(scores) / len(scores)), len(rows)
    except Exception as e:
        logger.warning("News sentiment for %s: %s", symbol, e)
        return 0.0, 0


def get_anomaly_proxy(symbol: str, vol_from_forecast: Optional[float] = None) -> Tuple[bool, float]:
    """
    Returns (detected, score). No anomaly module yet; use volatility as proxy (high vol = higher score).
    If vol_from_forecast is provided, skip extra get_forecast_signal call (faster).
    """
    if vol_from_forecast is not None:
        vol = vol_from_forecast
    else:
        _, _, vol = get_forecast_signal(symbol)
    detected = vol > 2.0
    score = min(1.0, vol / 3.0)
    return detected, score


def get_risk_advice(profile: UserProfile, symbol: str) -> RiskAdvice:
    """
    Combines forecast, news, anomaly and profile into recommendation and risk metrics.
    """
    direction, strength, vol = get_forecast_signal(symbol)
    sentiment, news_count = get_news_sentiment(symbol)
    anomaly_detected, anomaly_score = get_anomaly_proxy(symbol, vol_from_forecast=vol)

    # --- Raw view ---
    if anomaly_detected and anomaly_score > 0.6:
        raw = "Avoid"
    elif direction == "up" and sentiment >= -0.2 and not anomaly_detected:
        raw = "Buy"
    elif direction == "down" or sentiment <= -0.4:
        raw = "Avoid"
    else:
        raw = "Hold"

    # --- Adjust for risk tolerance ---
    tol = (profile.risk_tolerance or "medium").strip().lower()
    if tol == "low":
        if raw == "Buy":
            recommendation = "Hold"
        elif raw == "Avoid":
            recommendation = "Avoid"
        else:
            recommendation = "Hold"
    elif tol == "high":
        if raw == "Hold":
            recommendation = "Buy"
        elif raw == "Avoid":
            recommendation = "Hold"
        else:
            recommendation = raw
    else:
        recommendation = raw

    # --- Risk of loss: P(return < 0). Approximate with normal(mean, vol).
    # Mean from direction: up -> +0.2, flat -> 0, down -> -0.2 (simplified)
    mean = 0.2 if direction == "up" else (-0.2 if direction == "down" else 0.0)
    mean += sentiment * 0.1
    sigma = max(0.5, vol / 100.0)
    if sigma > 0:
        risk_of_loss = _norm_cdf(0, mean, sigma)
    else:
        risk_of_loss = 0.5
    risk_of_loss = max(0, min(1, risk_of_loss)) * 100

    # --- Risk of ruin: P(loss > 20% of capital). Simplified: position = f(capital, tolerance).
    frac = 0.1 if tol == "low" else (0.2 if tol == "medium" else 0.3)
    position_vol = sigma * frac
    ruin = (1 - _norm_cdf(-0.2 / frac, 0, position_vol)) if position_vol > 0 else 0.05
    risk_of_ruin = max(0, min(1, ruin)) * 100

    explanation = build_explanation(
        recommendation=recommendation,
        direction=direction,
        sentiment=sentiment,
        news_count=news_count,
        anomaly_detected=anomaly_detected,
        risk_of_loss_pct=risk_of_loss,
        risk_of_ruin_pct=risk_of_ruin,
    )
    return RiskAdvice(
        recommendation=recommendation,
        risk_of_loss_pct=round(risk_of_loss, 1),
        risk_of_ruin_pct=round(risk_of_ruin, 1),
        explanation=explanation,
    )


def build_explanation(
    recommendation: str,
    direction: str,
    sentiment: float,
    news_count: int,
    anomaly_detected: bool,
    risk_of_loss_pct: float,
    risk_of_ruin_pct: float,
) -> str:
    """Template-based explanation for research and compliance."""
    parts = []
    parts.append(f"Dựa trên xu hướng gần đây (chỉ báo kỹ thuật): {direction}.")
    if news_count > 0:
        sent_label = "tích cực" if sentiment > 0.1 else ("tiêu cực" if sentiment < -0.1 else "trung tính")
        parts.append(f"Tin tức ({news_count} bài) có sentiment {sent_label}.")
    if anomaly_detected:
        parts.append("Có cảnh báo biến động bất thường.")
    parts.append(f"Khuyến nghị: {recommendation}. Xác suất lỗ trong kỳ: khoảng {risk_of_loss_pct:.0f}%. Rủi ro sụt giảm mạnh vốn: khoảng {risk_of_ruin_pct:.0f}%.")
    return " ".join(parts)

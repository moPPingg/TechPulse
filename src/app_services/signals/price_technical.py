"""
Layer 1: Price/Technical Signal

Derives signal from price data: RSI, return, volatility.
"""

from dataclasses import dataclass
from typing import Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class PriceTechnicalSignal:
    """Signal from price and technical indicators."""
    direction: str  # "up" | "down" | "flat"
    strength: float  # 0-1
    rsi: Optional[float]
    return_1d: Optional[float]
    volatility_pct: float
    close: Optional[float]
    last_date: Optional[str]


def get_price_technical_signal(symbol: str) -> Optional[PriceTechnicalSignal]:
    """Compute price/technical signal from market data."""
    try:
        from src.app_services.market_data import get_indicators

        ind = get_indicators(symbol, days=90)
        ret = ind.get("return_1d") or 0
        vol = ind.get("volatility_5") or 1.0
        vol = max(float(vol), 0.1)
        rsi = ind.get("rsi_14")

        if ret > 0.05:
            direction = "up"
        elif ret < -0.05:
            direction = "down"
        else:
            direction = "flat"

        strength = min(1.0, abs(float(ret)) / 2.0)

        return PriceTechnicalSignal(
            direction=direction,
            strength=strength,
            rsi=float(rsi) if rsi is not None else None,
            return_1d=float(ret) if ret is not None else None,
            volatility_pct=vol,
            close=ind.get("close"),
            last_date=ind.get("last_date"),
        )
    except Exception as e:
        logger.warning("Price technical signal failed for %s: %s", symbol, e)
        return None

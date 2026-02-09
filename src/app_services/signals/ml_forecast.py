"""
Layer 2: ML/DL Forecast Signal

Ensemble forecast from inference cache or fallback from technical.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class MLForecastSignal:
    """Signal from ML/DL ensemble forecast."""
    mean: float  # % expected return
    std: float
    confidence: float  # 0-1
    model_weights: Dict[str, float]
    used_inference: bool  # True = cached inference, False = fallback
    as_of_date: str = ""  # PRODUCTION: for data_freshness


def get_ml_forecast_signal(symbol: str, technical_signal: Optional["PriceTechnicalSignal"]) -> MLForecastSignal:
    """Get ML forecast: prefer cached inference, else derive from technical."""
    symbol = symbol.strip().upper()

    try:
        from src.inference.service import get_forecast
        fc = get_forecast(symbol)
    except Exception as e:
        logger.warning("Inference get_forecast failed: %s", e)
        fc = None

    if fc:
        return MLForecastSignal(
            mean=fc.ensemble_mean,
            std=fc.ensemble_std,
            confidence=fc.confidence_score,
            model_weights=fc.weights or {},
            used_inference=True,
            as_of_date=getattr(fc, "as_of_date", "") or "",
        )

    # Fallback from technical
    if technical_signal:
        mean = 0.2 if technical_signal.direction == "up" else (-0.2 if technical_signal.direction == "down" else 0.0)
        std = max(0.5, technical_signal.volatility_pct)
    else:
        mean = 0.0
        std = 0.5

    return MLForecastSignal(
        mean=mean,
        std=std,
        confidence=0.5,
        model_weights={},
        used_inference=False,
        as_of_date="",
    )

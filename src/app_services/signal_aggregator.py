"""
SignalAggregator: Combine forecast, risk, news into structured Signals.

Single entry point for all signals used by RecommendationEngine.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Signals:
    """Aggregated signals for recommendation decision."""
    forecast_mean: float
    forecast_std: float
    confidence: float
    volatility_pct: float
    prob_loss_pct: float
    prob_ruin_pct: float
    expected_return_lower: float
    expected_return_upper: float
    news_sentiment: float
    news_count: int
    model_weights: Dict[str, float]
    used_inference: bool


def aggregate(symbol: str, user_profile: Any) -> Optional[Signals]:
    """
    Aggregate forecast, risk, news into Signals.
    user_profile: UserProfile from recommendation module.
    """
    symbol = symbol.strip().upper()

    # 1. Forecast
    forecast_result = None
    try:
        from src.inference.service import get_forecast
        forecast_result = get_forecast(symbol)
    except Exception as e:
        logger.warning("Inference get_forecast failed: %s", e)

    if forecast_result:
        ensemble_mean = forecast_result.ensemble_mean
        ensemble_std = forecast_result.ensemble_std
        vol = forecast_result.volatility_pct
        confidence = forecast_result.confidence_score
        model_weights = forecast_result.weights or {}
        used_inference = True
    else:
        from src.app_services.recommendation import get_forecast_signal
        direction, strength, vol = get_forecast_signal(symbol)
        ensemble_mean = 0.2 if direction == "up" else (-0.2 if direction == "down" else 0.0)
        ensemble_std = max(0.5, vol)
        confidence = 0.5
        model_weights = {}
        used_inference = False

    vol = max(vol, 0.1)

    # 2. News
    from src.app_services.news_service import get_sentiment
    sentiment, news_count = get_sentiment(symbol, days=30)

    # 3. Risk
    from src.app_services.recommendation import _position_frac
    from src.risk_engine.risk import compute_risk_metrics

    position_frac = _position_frac(user_profile)
    risk = compute_risk_metrics(
        forecast_mean=ensemble_mean,
        forecast_std=ensemble_std,
        volatility_pct=vol,
        position_frac=position_frac,
        drawdown_threshold_pct=20.0,
        confidence=0.95,
    )

    return Signals(
        forecast_mean=ensemble_mean,
        forecast_std=ensemble_std,
        confidence=confidence,
        volatility_pct=vol,
        prob_loss_pct=risk.prob_loss_pct,
        prob_ruin_pct=risk.prob_ruin_pct,
        expected_return_lower=risk.expected_return_lower,
        expected_return_upper=risk.expected_return_upper,
        news_sentiment=sentiment,
        news_count=news_count,
        model_weights=model_weights,
        used_inference=used_inference,
    )

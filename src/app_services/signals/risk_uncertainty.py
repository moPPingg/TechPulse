"""
Layer 4: Risk & Uncertainty Signal

P(loss), P(ruin), expected return CI, volatility.
"""

from dataclasses import dataclass
from typing import Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskUncertaintySignal:
    """Risk and uncertainty metrics."""
    prob_loss_pct: float
    prob_ruin_pct: float
    expected_return_lower: float
    expected_return_upper: float
    volatility_pct: float


def get_risk_uncertainty_signal(
    ml_forecast: "MLForecastSignal",
    volatility_pct: float,
    position_frac: float,
) -> RiskUncertaintySignal:
    """Compute risk metrics from forecast and position."""
    try:
        from src.risk_engine.risk import compute_risk_metrics

        risk = compute_risk_metrics(
            forecast_mean=ml_forecast.mean,
            forecast_std=ml_forecast.std,
            volatility_pct=max(volatility_pct, 0.1),
            position_frac=position_frac,
            drawdown_threshold_pct=20.0,
            confidence=0.95,
        )
        return RiskUncertaintySignal(
            prob_loss_pct=risk.prob_loss_pct,
            prob_ruin_pct=risk.prob_ruin_pct,
            expected_return_lower=risk.expected_return_lower,
            expected_return_upper=risk.expected_return_upper,
            volatility_pct=risk.volatility_pct,
        )
    except Exception as e:
        logger.warning("Risk uncertainty signal failed: %s", e)
        return RiskUncertaintySignal(
            prob_loss_pct=50.0,
            prob_ruin_pct=10.0,
            expected_return_lower=-2.0,
            expected_return_upper=2.0,
            volatility_pct=volatility_pct or 1.0,
        )

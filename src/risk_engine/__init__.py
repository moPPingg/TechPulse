"""
Risk Engine: Expected return distribution, VaR, drawdown, ruin proxy.

What goes in:  Forecast mean, std, volatility, position size, user profile.
What comes out: P(loss), P(ruin), expected return CI, VaR/CVaR proxy.

Why this exists: Point forecasts alone don't tell you P(bad outcome).
Risk = probability of adverse events given uncertainty.
"""

from src.risk_engine.risk import (
    RiskMetrics,
    compute_risk_metrics,
    prob_loss,
    prob_ruin_proxy,
    expected_return_ci,
)

__all__ = [
    "RiskMetrics",
    "compute_risk_metrics",
    "prob_loss",
    "prob_ruin_proxy",
    "expected_return_ci",
]

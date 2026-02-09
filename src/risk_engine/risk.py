"""
Risk Engine: P(loss), P(ruin), VaR, expected return confidence interval.

TEACHING:
- We model return as approximately normal: N(mu, sigma²).
  mu = forecast mean (%), sigma = combined uncertainty (forecast std + realized vol).
- P(loss) = P(return < 0) = Phi((0 - mu) / sigma)
- Ruin proxy: P(loss exceeding X% of capital). With leverage, position vol scales.
- VaR: percentile of loss distribution (e.g. 95% VaR = 5th percentile of return).
- CI: [mu - z*sigma, mu + z*sigma] for 95% use z=1.96.

PRODUCTION DISCLAIMER: Financial returns are often fat-tailed. The normal
assumption may underestimate tail risk (P(ruin), VaR). Use these metrics as
indicative only; consider stress-testing with fat-tailed distributions.
"""

import math
from dataclasses import dataclass
from typing import Tuple


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """Approximate inverse CDF for normal. p in (0, 1)."""
    # Rational approximation (Abramowitz & Stegun)
    if p <= 0 or p >= 1:
        return 0.0
    a = (-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0)
    b = (-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1)
    p = p - 0.5
    if abs(p) < 0.425:
        t = 0.180625 - p * p
        num = sum(a[i] * (t ** i) for i in range(6))
        den = 1 + sum(b[i] * (t ** (i + 1)) for i in range(5))
        return p * num / den
    t = math.sqrt(-math.log(0.5 - abs(p)))
    c = (7.745450142783414e-4, 2.272384498966918e-2, 2.417807251774506e-1,
         1.270458252452368e0, 3.647848324763204e0, 5.769497221460691e0, 4.630337846156545e0, 1.423437110749683e0)
    d = (1.050750071644416e-9, 5.944738425303371e-2, 1.574483117449093e0,
         1.176939508913124e1, 2.432417655059949e1, 1.971297383663816e1, 4.351787712209471e0)
    num = sum(c[i] * (t ** i) for i in range(8))
    den = 1 + sum(d[i] * (t ** (i + 1)) for i in range(6))
    x = num / den
    return -x if p < 0 else x


def prob_loss(mu: float, sigma: float) -> float:
    """
    P(return < 0) under N(mu, sigma²).
    mu, sigma in same units (e.g. %).
    """
    if sigma <= 0:
        return 0.5 if mu < 0 else 0.0
    z = (0 - mu) / sigma
    return _norm_cdf(z)


def prob_ruin_proxy(
    mu: float,
    sigma: float,
    drawdown_threshold_pct: float = 20.0,
    position_frac: float = 0.2,
) -> float:
    """
    P(loss exceeding drawdown_threshold_pct of capital).
    Approximate: position vol = sigma * sqrt(position_frac).
    Ruin = P(return < -drawdown_threshold_pct/100).

    Args:
        mu: Expected return (%)
        sigma: Volatility (%)
        drawdown_threshold_pct: Threshold (e.g. 20 = 20% drawdown)
        position_frac: Fraction of capital in position (from risk tolerance)

    Returns:
        Probability in [0, 1]
    """
    threshold = -drawdown_threshold_pct / 100.0  # e.g. -0.2
    pos_sigma = sigma * (position_frac ** 0.5)  # rough scaling
    if pos_sigma <= 0:
        return 0.0 if mu > threshold else 1.0
    z = (threshold - mu) / pos_sigma
    return _norm_cdf(z)


def expected_return_ci(mu: float, sigma: float, confidence: float = 0.95) -> Tuple[float, float]:
    """
    (lower, upper) confidence interval for return.
    """
    if confidence <= 0 or confidence >= 1:
        confidence = 0.95
    alpha = 1 - confidence
    z = _norm_ppf(1 - alpha / 2)
    return (mu - z * sigma, mu + z * sigma)


def var_percentile(mu: float, sigma: float, percentile: float = 5.0) -> float:
    """
    VaR: return at given percentile (e.g. 5% = 95% VaR).
    Returns the return level such that P(return < level) = percentile/100.
    """
    p = percentile / 100.0
    z = _norm_ppf(p)
    return mu + z * sigma


@dataclass
class RiskMetrics:
    """All risk outputs in one struct."""
    prob_loss_pct: float
    prob_ruin_pct: float
    expected_return_mean: float
    expected_return_lower: float
    expected_return_upper: float
    var_95: float  # 5th percentile return
    volatility_pct: float


def compute_risk_metrics(
    forecast_mean: float,
    forecast_std: float,
    volatility_pct: float,
    position_frac: float = 0.2,
    drawdown_threshold_pct: float = 20.0,
    confidence: float = 0.95,
) -> RiskMetrics:
    """
    Full risk computation.

    Args:
        forecast_mean: Ensemble forecast mean (%)
        forecast_std: Ensemble forecast std (%)
        volatility_pct: Realized/historical volatility (%), used to inflate uncertainty
        position_frac: Fraction of capital in position
        drawdown_threshold_pct: Ruin threshold
        confidence: CI level

    Combines forecast uncertainty with realized vol: sigma = sqrt(forecast_std² + vol²)
    """
    sigma = (forecast_std ** 2 + volatility_pct ** 2) ** 0.5
    sigma = max(sigma, 0.01)

    pl = prob_loss(forecast_mean, sigma)
    pr = prob_ruin_proxy(
        forecast_mean, sigma,
        drawdown_threshold_pct=drawdown_threshold_pct,
        position_frac=position_frac,
    )
    lo, hi = expected_return_ci(forecast_mean, sigma, confidence)
    var95 = var_percentile(forecast_mean, sigma, 5.0)

    return RiskMetrics(
        prob_loss_pct=pl * 100,
        prob_ruin_pct=pr * 100,
        expected_return_mean=forecast_mean,
        expected_return_lower=lo,
        expected_return_upper=hi,
        var_95=var95,
        volatility_pct=volatility_pct,
    )

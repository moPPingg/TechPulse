"""
Inference: Run models and get forecasts for recommendation engine.

What goes in:  Symbol, features path.
What comes out: Cached or computed ForecastResult (ensemble mean, std, per-model, volatility).

Why this exists: App needs forecasts without re-training every request.
We pre-compute via scripts/run_inference.py and cache to data/forecasts/.
"""

from src.inference.service import (
    ForecastResult,
    get_forecast,
    run_inference_for_symbol,
)

__all__ = [
    "ForecastResult",
    "get_forecast",
    "run_inference_for_symbol",
]

"""
App service layer: aggregates ML outputs and produces recommendation + risk + explanation.
Used by Streamlit (and later FastAPI) without duplicating ML logic.
"""

from src.app_services.recommendation import (
    get_forecast_signal,
    get_news_sentiment,
    get_risk_advice,
    UserProfile,
    RiskAdvice,
)

__all__ = [
    "get_forecast_signal",
    "get_news_sentiment",
    "get_risk_advice",
    "UserProfile",
    "RiskAdvice",
]

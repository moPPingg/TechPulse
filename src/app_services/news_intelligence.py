"""
News Intelligence Service — facade over the News Intelligence Engine.

This module is the app-facing API: dashboards, API routes, and recommendation
code import from here. All logic lives in src.news.engine and src.news.models.

- Ingest: use engine.run_ingest() (or scheduler runs pipeline).
- Per-ticker signal + explainability: get_stock_news_signal() → StockNewsSignal.
- Per-ticker daily signals: get_ticker_daily_signals() → List[DailyNewsSignal].
- Market shock: detect_market_shock().
- ML features: engine.get_ml_daily_features().
"""

import logging
from typing import List, Optional

# Re-export canonical types so existing imports (api, recommendation, signals) keep working.
from src.news.models import (
    DailyNewsSignal,
    EnrichedArticleView,
    ImpactItem,
    MarketShockResult,
    StockNewsSignal,
)
from src.news import engine

logger = logging.getLogger(__name__)


def get_stock_news_signal(
    symbol: str,
    days: int = 30,
    min_relevance: float = 0.2,
    limit_articles: int = 10,
    event_type_filter: Optional[str] = None,
    sentiment_method: str = "lexicon",
) -> StockNewsSignal:
    """
    Aggregate enriched articles into stock-level signal with full explainability.
    Delegates to News Intelligence Engine.
    """
    return engine.get_signal(
        symbol=symbol,
        days=days,
        min_relevance=min_relevance,
        limit_articles=limit_articles,
        event_type_filter=event_type_filter,
        sentiment_method=sentiment_method,
    )


def get_ticker_daily_signals(
    symbol: str,
    from_date: str,
    to_date: str,
    min_relevance: float = 0.2,
    sentiment_method: str = "lexicon",
) -> List[DailyNewsSignal]:
    """
    Per-ticker daily signals with per-article contribution breakdown.
    Delegates to News Intelligence Engine.
    """
    return engine.get_daily_signals(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        min_relevance=min_relevance,
        sentiment_method=sentiment_method,
    )


def detect_market_shock(conn, symbol: str, hours: int = 24, sentiment_method: str = "lexicon") -> MarketShockResult:
    """Detect abnormal sentiment spike or rare events. Delegates to engine."""
    return engine.detect_market_shock(
        conn, symbol.upper(), hours=hours, sentiment_method=sentiment_method
    )

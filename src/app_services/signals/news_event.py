"""
Layer 3: News & Event Signal

Composite from news intelligence service.
"""

from dataclasses import dataclass
from typing import Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsEventSignal:
    """Signal from news and events."""
    composite_score: float  # -1 to 1
    article_count: int
    net_impact_label: str  # bullish | bearish | neutral
    net_impact_confidence: float  # 0-100


def get_news_event_signal(symbol: str) -> NewsEventSignal:
    """Get news/event signal from intelligence service."""
    try:
        from src.app_services.news_intelligence import get_stock_news_signal
        sig = get_stock_news_signal(symbol, days=30, min_relevance=0.0, limit_articles=20)
        return NewsEventSignal(
            composite_score=sig.composite_score,
            article_count=sig.article_count,
            net_impact_label=sig.net_impact_label or "neutral",
            net_impact_confidence=sig.net_impact_confidence or 0,
        )
    except Exception as e:
        logger.warning("News event signal failed for %s: %s", symbol, e)
        return NewsEventSignal(
            composite_score=0.0,
            article_count=0,
            net_impact_label="neutral",
            net_impact_confidence=0,
        )

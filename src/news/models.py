"""
Canonical output types for the News Intelligence Engine.

All engine outputs use these models so that:
- ML pipelines get consistent feature shapes
- Trading strategies get signals + explainability
- Dashboards get per-article contribution and reasoning
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnrichedArticleView:
    """
    Single enriched article with full explainability fields.

    Every article exposes: url (original link), source, published_at,
    ticker_relevance, impact_horizon, horizon_weight, sentiment_score,
    sentiment_confidence, contribution_weight, raw_contribution.
    """

    article_id: int
    title: str
    summary: str
    url: str
    source: str
    published_at: Optional[str]
    event_type: str
    ticker_relevance: float
    sentiment_score: float
    sentiment_confidence: Optional[float]
    impact_horizon: str
    horizon_weight: float = 1.0
    contribution_weight: float = 0.0
    raw_contribution: float = 0.0


@dataclass
class ImpactItem:
    """Single high-impact article for dashboards (title, url, direction, why it matters)."""
    title: str
    why_it_matters: str
    impact_direction: str  # bullish | bearish | neutral
    time_horizon: str
    confidence: float
    url: str
    event_type: str


@dataclass
class MarketShockResult:
    """Market-shock detection result (sentiment spike or rare events)."""
    is_shock: bool
    reason: str
    summary: str
    contributing_article_titles: List[str] = field(default_factory=list)


@dataclass
class StockNewsSignal:
    """
    Aggregated news signal for one ticker.

    Usable by: dashboards (reasoning, top_articles, top_3_impact),
    trading (composite_score, net_impact_label, market_shock),
    ML (composite_score, avg_sentiment, event_breakdown as features).
    """

    symbol: str
    composite_score: float
    article_count: int
    avg_sentiment: float
    avg_relevance: float
    horizon_breakdown: Dict[str, int] = field(default_factory=dict)
    event_breakdown: Dict[str, int] = field(default_factory=dict)
    top_articles: List[EnrichedArticleView] = field(default_factory=list)
    top_3_impact: List[ImpactItem] = field(default_factory=list)
    net_impact_label: str = "neutral"
    net_impact_confidence: float = 0.0
    is_general_fallback: bool = False
    reasoning: str = ""
    top_contributors: List[Dict[str, Any]] = field(default_factory=list)
    market_shock: Optional[MarketShockResult] = None


@dataclass
class DailyNewsSignal:
    """
    Per-ticker daily aggregate with per-article contribution.

    Usable by: ML (one row per (symbol, date)), trading (daily composite),
    dashboards (articles with contribution_weight, raw_contribution).
    """

    date: str  # YYYY-MM-DD
    composite_score: float
    article_count: int
    articles: List[EnrichedArticleView]

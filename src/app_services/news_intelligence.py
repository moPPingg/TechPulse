"""
News Intelligence Service: Stock-level aggregation and API responses.

Aggregates enriched articles into a single stock-level signal.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class EnrichedArticleView:
    """Enriched article for API response."""
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


@dataclass
class ImpactItem:
    """Single impactful news for Investment Intelligence panel."""
    title: str
    why_it_matters: str
    impact_direction: str  # bullish | bearish | neutral
    time_horizon: str
    confidence: float
    url: str
    event_type: str


@dataclass
class StockNewsSignal:
    """Aggregated news signal for a symbol."""
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


def _get_db_path() -> Path:
    try:
        import yaml
        with open(_ROOT / "configs" / "news.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        p = cfg.get("database", {}).get("path", "data/news/news.db")
    except Exception:
        p = "data/news/news.db"
    path = Path(p)
    if not path.is_absolute():
        path = _ROOT / path
    return path


def _normalize_url(url: str, source: str) -> str:
    urls = {
        "cafef": "https://cafef.vn",
        "vietstock": "https://vietstock.vn",
        "vnexpress": "https://vnexpress.net",
        "hsx": "https://www.hsx.vn",
        "vneconomy": "https://vneconomy.vn",
        "tradingeconomics": "https://tradingeconomics.com",
        "ssc": "https://ssc.gov.vn",
    }
    if not url or not isinstance(url, str):
        return ""
    u = url.strip()
    if u.startswith("http"):
        return u
    base = urls.get((source or "").lower(), "")
    if not base:
        return u
    return base.rstrip("/") + (u if u.startswith("/") else "/" + u.lstrip("/"))


# Event type → why it matters (Vietnamese)
WHY_IT_MATTERS = {
    "earnings": "Kết quả kinh doanh ảnh hưởng trực tiếp đến giá cổ phiếu.",
    "legal": "Tin pháp lý có thể gây biến động giá trong ngắn hạn.",
    "macro": "Yếu tố vĩ mô tác động đến toàn thị trường.",
    "operations": "Hoạt động sản xuất kinh doanh phản ánh triển vọng công ty.",
    "guidance": "Hướng dẫn kỳ vọng của ban lãnh đạo quan trọng cho nhà đầu tư.",
    "ma": "M&A thường tạo biến động mạnh và cơ hội định giá lại.",
    "dividend": "Cổ tức ảnh hưởng dòng tiền và kỳ vọng cổ đông.",
    "other": "Tin chung liên quan đến ngành hoặc thị trường.",
}

HORIZON_LABELS = {"intraday": "Trong phiên", "short_term": "Ngắn hạn (1–2 tuần)", "long_term": "Dài hạn (1+ tháng)"}


def _why_it_matters(event_type: str, sentiment: float) -> str:
    base = WHY_IT_MATTERS.get(event_type, WHY_IT_MATTERS["other"])
    if sentiment > 0.2:
        return base + " Tín hiệu tích cực."
    if sentiment < -0.2:
        return base + " Tín hiệu cần thận trọng."
    return base


def _impact_direction(sentiment: float) -> str:
    if sentiment > 0.1:
        return "bullish"
    if sentiment < -0.1:
        return "bearish"
    return "neutral"


def _summarize(text: str, max_chars: int = 180) -> str:
    if not text or not str(text).strip():
        return ""
    text = str(text).strip()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if not sentences:
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    out = ". ".join(sentences[:2])
    if out and not out.endswith("."):
        out += "."
    return (out[: max_chars - 3].rsplit(" ", 1)[0] + "...") if len(out) > max_chars else out


def get_stock_news_signal(
    symbol: str,
    days: int = 30,
    min_relevance: float = 0.2,
    limit_articles: int = 10,
    event_type_filter: Optional[str] = None,
) -> StockNewsSignal:
    """
    Aggregate enriched articles into stock-level signal.
    Falls back to non-enriched articles when enrichments missing.
    """
    path = _get_db_path()
    if not path.exists():
        return StockNewsSignal(
            symbol=symbol.upper(),
            composite_score=0.0,
            article_count=0,
            avg_sentiment=0.0,
            avg_relevance=0.0,
        )

    from src.news.db import get_engine, init_db, get_enriched_articles_for_ticker, get_recent_general_articles
    from src.news.intelligence import get_horizon_weight
    init_db(str(path))
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_engine(str(path))

    rows = get_enriched_articles_for_ticker(
        conn, symbol.upper(),
        date_from=since,
        limit=limit_articles * 3,
        min_relevance=min_relevance,
        event_type_filter=event_type_filter,
    )

    is_fallback = False
    if not rows:
        rows = get_recent_general_articles(
            conn, date_from=since, limit=limit_articles * 3, sentiment_method=sentiment_method
        )
        is_fallback = bool(rows)

    if not rows:
        return StockNewsSignal(
            symbol=symbol.upper(),
            composite_score=0.0,
            article_count=0,
            avg_sentiment=0.0,
            avg_relevance=0.0,
        )

    # Aggregate
    horizon_breakdown: Dict[str, int] = {}
    event_breakdown: Dict[str, int] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    sentiments = []
    relevances = []

    articles_out: List[EnrichedArticleView] = []

    for r in rows[:limit_articles]:
        rel = float(r.get("ticker_relevance_score", 0.5))
        sent = float(r.get("sentiment_score", 0))
        horizon = r.get("impact_horizon") or "short_term"
        event = r.get("event_type") or "other"

        horizon_breakdown[horizon] = horizon_breakdown.get(horizon, 0) + 1
        event_breakdown[event] = event_breakdown.get(event, 0) + 1

        hw = get_horizon_weight(horizon)
        w = rel * hw
        weighted_sum += sent * w
        weight_sum += w
        sentiments.append(sent)
        relevances.append(rel)

        body = r.get("body_clean") or ""
        summary = _summarize(body)
        if not summary and r.get("title"):
            summary = str(r.get("title", ""))[:150]

        raw_url = r.get("url", "") or ""
        url = _normalize_url(raw_url, r.get("source", "")) or raw_url

        articles_out.append(EnrichedArticleView(
            article_id=r.get("article_id", 0),
            title=r.get("title", ""),
            summary=summary,
            url=url,
            source=r.get("source", ""),
            published_at=r.get("published_at"),
            event_type=event,
            ticker_relevance=round(rel, 3),
            sentiment_score=round(sent, 3),
            sentiment_confidence=r.get("sentiment_confidence"),
            impact_horizon=horizon,
        ))

    composite = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    composite = max(-1.0, min(1.0, composite))
    avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0
    avg_rel = sum(relevances) / len(relevances) if relevances else 0.0

    # Top 3 by impact: relevance * |sentiment| * horizon_weight
    impact_scores = []
    for i, r in enumerate(rows[:limit_articles]):
        rel = float(r.get("ticker_relevance_score", 0.5))
        sent = float(r.get("sentiment_score", 0))
        h = r.get("impact_horizon") or "short_term"
        hw = get_horizon_weight(h)
        score = rel * max(0.1, abs(sent)) * hw
        impact_scores.append((i, score))

    impact_scores.sort(key=lambda x: -x[1])
    top_3_indices = [idx for idx, _ in impact_scores[:3]]

    top_3_impact: List[ImpactItem] = []
    for idx in top_3_indices:
        r = rows[idx]
        rel = float(r.get("ticker_relevance_score", 0.5))
        sent = float(r.get("sentiment_score", 0))
        evt = r.get("event_type") or "other"
        h = r.get("impact_horizon") or "short_term"
        conf = float(r.get("sentiment_confidence", 0.5) or 0.5)
        body = r.get("body_clean") or ""
        summary = _summarize(body)
        if not summary and r.get("title"):
            summary = str(r.get("title", ""))[:120]
        raw_url = r.get("url", "") or ""
        url = _normalize_url(raw_url, r.get("source", "")) or raw_url
        top_3_impact.append(ImpactItem(
            title=articles_out[idx].title if idx < len(articles_out) else (r.get("title", "") or ""),
            why_it_matters=_why_it_matters(evt, sent),
            impact_direction=_impact_direction(sent),
            time_horizon=HORIZON_LABELS.get(h, h),
            confidence=round(min(1.0, rel * (0.7 + 0.3 * conf)), 2),
            url=url,
            event_type=evt,
        ))

    # Net impact label + confidence
    if composite > 0.15:
        net_label = "bullish"
    elif composite < -0.15:
        net_label = "bearish"
    else:
        net_label = "neutral"

    net_conf = min(1.0, avg_rel * (0.6 + 0.4 * (1.0 - abs(composite)))) if composite != 0 else 0.5
    net_conf = round(net_conf * 100)

    return StockNewsSignal(
        symbol=symbol.upper(),
        composite_score=round(composite, 3),
        article_count=len(rows),
        avg_sentiment=round(avg_sent, 3),
        avg_relevance=round(avg_rel, 3),
        horizon_breakdown=horizon_breakdown,
        event_breakdown=event_breakdown,
        top_articles=articles_out,
        top_3_impact=top_3_impact,
        net_impact_label=net_label,
        net_impact_confidence=net_conf,
        is_general_fallback=is_fallback,
    )

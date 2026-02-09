"""
NewsService: Filtered, scored, summarized news for investment decisions.

Responsibilities:
- Fetch articles by ticker (or symbol-in-text fallback)
- Score relevance (0–1) for investment decisions
- Summarize for display; add investment_summary for decision context
- Normalize URLs
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent.parent.parent

_SOURCE_BASE_URLS = {
    "cafef": "https://cafef.vn",
    "vietstock": "https://vietstock.vn",
    "vnexpress": "https://vnexpress.net",
    "hsx": "https://www.hsx.vn",
    "vneconomy": "https://vneconomy.vn",
    "tradingeconomics": "https://tradingeconomics.com",
    "ssc": "https://ssc.gov.vn",
}

# Stock-focused sources get higher relevance baseline
_STOCK_SOURCES = {"cafef", "vietstock", "vnexpress"}


@dataclass
class Article:
    """Article with relevance and investment summary."""
    id: int
    title: str
    summary: str
    investment_summary: str  # Why this matters for {symbol}
    url: str
    source: str
    sentiment: Optional[float]
    relevance_score: float
    published_at: Optional[str]


def _normalize_url(url: str, source: str) -> str:
    if not url or not isinstance(url, str):
        return ""
    u = url.strip()
    if u.startswith("http://") or u.startswith("https://"):
        return u
    base = _SOURCE_BASE_URLS.get((source or "").lower(), "")
    if not base:
        return u
    if u.startswith("/"):
        return base.rstrip("/") + u
    return base.rstrip("/") + "/" + u.lstrip("/")


def _summarize_extractive(text: str, max_sentences: int = 2, max_chars: int = 180) -> str:
    if not text or not str(text).strip():
        return ""
    text = str(text).strip()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if not sentences:
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    out = ". ".join(sentences[:max_sentences])
    if out and not out.endswith("."):
        out += "."
    if len(out) > max_chars:
        out = out[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return out


def _compute_relevance(row: dict, symbol: str) -> float:
    """
    Score 0–1: ticker in title, ticker in body, stock-focused source, recency, sentiment extremity.
    """
    symbol_upper = symbol.upper()
    title = (row.get("title") or "").upper()
    body = (row.get("body_clean") or "").upper()
    source = (row.get("source") or "").lower()
    pub = row.get("published_at") or ""
    sent = row.get("sentiment_score")

    score = 0.0

    if symbol_upper in title:
        score += 0.3
    if symbol_upper in body:
        score += 0.2

    if source in _STOCK_SOURCES:
        score += 0.2

    if pub:
        try:
            dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except Exception:
            dt = None
        if dt:
            days_ago = (datetime.utcnow() - dt.replace(tzinfo=None)).days if hasattr(dt, "replace") else 999
            if days_ago <= 3:
                score += 0.2
            elif days_ago <= 7:
                score += 0.1

    if sent is not None and abs(float(sent)) > 0.3:
        score += 0.1

    return min(1.0, score)


def _investment_summary(row: dict, symbol: str, sentiment: Optional[float]) -> str:
    """One-sentence 'why this matters' for the symbol."""
    sent_label = "tích cực" if sentiment and sentiment > 0.1 else ("tiêu cực" if sentiment and sentiment < -0.1 else "trung tính")
    return f"Tin {sent_label} liên quan {symbol}; đáng chú ý cho nhà đầu tư theo dõi mã này."


def get_news_db_path() -> Path:
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


def get_articles(
    symbol: str,
    days: int = 30,
    limit: int = 20,
    min_relevance: float = 0.0,
) -> List[Article]:
    """Fetch articles for symbol, score relevance, return sorted by relevance then date."""
    from src.news.db import get_engine, get_articles_by_ticker_date, get_articles_matching_symbol_in_text

    path = get_news_db_path()
    if not path.exists():
        return []

    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_engine(str(path))

    rows = get_articles_by_ticker_date(conn, symbol.upper(), date_from=since, limit=limit * 2)
    if not rows:
        rows = get_articles_matching_symbol_in_text(conn, symbol.upper(), date_from=since, limit=limit * 2)

    articles = []
    for r in rows[:limit * 2]:
        body = r.get("body_clean") or r.get("body_raw") or ""
        summary = _summarize_extractive(body, max_sentences=2, max_chars=180)
        if not summary and r.get("title"):
            summary = str(r.get("title", ""))[:150]

        rel = _compute_relevance(r, symbol)
        if rel < min_relevance:
            continue

        sent = r.get("sentiment_score")
        inv_summary = _investment_summary(r, symbol, sent)

        raw_url = r.get("url", "") or ""
        url = _normalize_url(raw_url, r.get("source", "")) or raw_url

        articles.append(Article(
            id=r.get("id", 0),
            title=r.get("title", ""),
            summary=summary,
            investment_summary=inv_summary,
            url=url,
            source=r.get("source", ""),
            sentiment=float(sent) if sent is not None else None,
            relevance_score=round(rel, 2),
            published_at=r.get("published_at"),
        ))

    articles.sort(key=lambda a: (-a.relevance_score, a.published_at or ""), reverse=False)
    return articles[:limit]


def get_sentiment(symbol: str, days: int = 30) -> Tuple[float, int]:
    """Return (avg_sentiment, article_count)."""
    arts = get_articles(symbol, days=days, limit=100)
    scores = [a.sentiment for a in arts if a.sentiment is not None]
    avg = float(sum(scores) / len(scores)) if scores else 0.0
    return avg, len(arts)

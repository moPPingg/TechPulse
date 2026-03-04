"""
News Intelligence Engine — production-grade ingestion, enrichment, and aggregation.

Architecture (layers):

  1. INGEST
     Crawl sources → normalize URL & store → content-hash dedupe → persist.
     (Implemented as pipeline: crawl → clean → sentiment → align → enrich.)

  2. NORMALIZE & DEDUPLICATE
     - URL normalization (relative → absolute by source).
     - Deduplication by (source, url) at DB; cross-source by content_hash (SHA-1 of title+body).

  3. ENRICH
     Per article: ticker relevance, sentiment score + confidence, event_type,
     impact_horizon + horizon_weight (from intelligence module).

  4. AGGREGATE
     - Rolling window: composite_score = sum(sentiment * weight) / sum(weight),
       weight = ticker_relevance * horizon_weight.
     - Per-ticker daily: group by date, same formula per day.

  5. OUTPUT (explainability + consumers)
     - Dashboards: StockNewsSignal with reasoning, top_contributors, per-article breakdown.
     - Trading: composite_score, net_impact_label, market_shock.
     - ML: get_ml_daily_features() returns flat rows (date, symbol, composite_score, counts, ...).

Design decisions:
- Single DB (SQLite) as source of truth; no message queue to keep ops simple.
- Idempotent pipeline: only process articles missing a step (clean/sentiment/align/enrich).
- Explainability is first-class: every article has contribution_weight and raw_contribution.
- Market shock: heuristic (sentiment spike vs 7d baseline, or rare event types in 24h).
- Types live in news.models so ML/trading/dashboards share the same contracts.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.news import models

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent.parent.parent

HIGH_IMPACT_EVENT_TYPES = {"legal", "ma", "macro"}

_SOURCE_BASE_URLS = {
    "cafef": "https://cafef.vn",
    "vietstock": "https://vietstock.vn",
    "vnexpress": "https://vnexpress.net",
    "hsx": "https://www.hsx.vn",
    "vneconomy": "https://vneconomy.vn",
    "tradingeconomics": "https://tradingeconomics.com",
    "ssc": "https://ssc.gov.vn",
}


def _get_db_path(config_path: Optional[str] = None) -> Path:
    try:
        import yaml
        path = Path(config_path or "configs/news.yaml")
        if not path.is_absolute():
            path = _ROOT / path
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
        p = cfg.get("database", {}).get("path", "data/news/news.db")
    except Exception:
        p = "data/news/news.db"
    path = Path(p)
    if not path.is_absolute():
        path = _ROOT / path
    return path


def _normalize_url(url: str, source: str) -> str:
    if not url or not isinstance(url, str):
        return ""
    u = url.strip()
    if u.startswith("http"):
        return u
    s = (source or "").strip().lower()
    base = _SOURCE_BASE_URLS.get(s, "")
    if not base:
        low_u = u.lower()
        if low_u.endswith(".chn"):
            base = "https://cafef.vn"
        elif low_u.endswith(".html"):
            base = "https://vnexpress.net"
        elif u.startswith("/"):
            base = "https://cafef.vn"
    if not base:
        return u
    return base.rstrip("/") + (u if u.startswith("/") else "/" + u.lstrip("/"))


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


def _impact_direction(sentiment: float) -> str:
    if sentiment > 0.1:
        return "bullish"
    if sentiment < -0.1:
        return "bearish"
    return "neutral"


# -----------------------------------------------------------------------------
# 1. INGEST — run full pipeline (crawl → clean → sentiment → align → enrich)
# -----------------------------------------------------------------------------


def run_ingest(
    config_path: Optional[str] = None,
    steps: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Run the news pipeline: crawl, normalize/dedup, enrich.
    Steps default to ["crawl", "clean", "sentiment", "align", "enrich"].
    Returns dict of step name -> count processed.
    """
    from src.news.pipeline import run_pipeline
    return run_pipeline(config_path=config_path, steps=steps or None)


# -----------------------------------------------------------------------------
# 2–4. AGGREGATE + EXPLAINABILITY — per-ticker signal and daily signals
# -----------------------------------------------------------------------------


def _get_conn(config_path: Optional[str] = None):
    path = _get_db_path(config_path)
    if not path.exists():
        return None
    from src.news.db import get_engine, init_db
    init_db(str(path))
    return get_engine(str(path))


def get_signal(
    symbol: str,
    days: int = 30,
    min_relevance: float = 0.2,
    limit_articles: int = 10,
    event_type_filter: Optional[str] = None,
    sentiment_method: str = "lexicon",
    config_path: Optional[str] = None,
) -> models.StockNewsSignal:
    """
    Aggregate enriched articles into one stock-level signal with full explainability.
    Returns StockNewsSignal: composite_score, reasoning, top_contributors, per-article breakdown.
    """
    from src.news.db import get_enriched_articles_for_ticker, get_recent_general_articles
    from src.news.intelligence import get_horizon_weight, why_it_matters_text, HORIZON_LABELS

    conn = _get_conn(config_path)
    empty = models.StockNewsSignal(
        symbol=symbol.upper(),
        composite_score=0.0,
        article_count=0,
        avg_sentiment=0.0,
        avg_relevance=0.0,
    )
    if not conn:
        return empty

    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = get_enriched_articles_for_ticker(
        conn, symbol.upper(),
        date_from=since,
        limit=limit_articles * 3,
        min_relevance=min_relevance,
        event_type_filter=event_type_filter,
        sentiment_method=sentiment_method,
    )
    is_fallback = False
    if not rows:
        rows = get_recent_general_articles(
            conn, date_from=since, limit=limit_articles * 3, sentiment_method=sentiment_method,
        )
        is_fallback = bool(rows)
    if not rows:
        return empty

    horizon_breakdown: Dict[str, int] = {}
    event_breakdown: Dict[str, int] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    sentiments: List[float] = []
    relevances: List[float] = []
    confidences: List[float] = []
    bull_count = bear_count = neutral_count = 0
    articles_out: List[models.EnrichedArticleView] = []
    limited_rows = rows[:limit_articles] if limit_articles > 0 else rows

    for r in limited_rows:
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
        raw_conf = r.get("sentiment_confidence")
        if raw_conf is None:
            raw_conf = 0.5 + 0.3 * min(1.0, abs(sent))
        conf = float(raw_conf)
        confidences.append(conf)
        if sent > 0.1:
            bull_count += 1
        elif sent < -0.1:
            bear_count += 1
        else:
            neutral_count += 1
        body = r.get("body_clean") or ""
        summary = _summarize(body)
        if not summary and r.get("title"):
            summary = str(r.get("title", ""))[:150]
        raw_url = r.get("url", "") or ""
        url = _normalize_url(raw_url, r.get("source", "")) or raw_url
        articles_out.append(
            models.EnrichedArticleView(
                article_id=r.get("article_id", 0),
                title=r.get("title", ""),
                summary=summary,
                url=url,
                source=r.get("source", ""),
                published_at=r.get("published_at"),
                event_type=event,
                ticker_relevance=round(rel, 3),
                sentiment_score=round(sent, 3),
                sentiment_confidence=round(conf, 3),
                impact_horizon=horizon,
                horizon_weight=round(hw, 3),
                contribution_weight=round(w, 3),
                raw_contribution=round(sent * w, 4),
            )
        )

    composite = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    composite = max(-1.0, min(1.0, composite))
    avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0
    avg_rel = sum(relevances) / len(relevances) if relevances else 0.0

    impact_scores = []
    for i, r in enumerate(limited_rows):
        rel = float(r.get("ticker_relevance_score", 0.5))
        sent = float(r.get("sentiment_score", 0))
        h = r.get("impact_horizon") or "short_term"
        hw = get_horizon_weight(h)
        impact_scores.append((i, rel * max(0.1, abs(sent)) * hw))
    impact_scores.sort(key=lambda x: -x[1])
    top_3_indices = [idx for idx, _ in impact_scores[:3]]

    top_3_impact: List[models.ImpactItem] = []
    for idx in top_3_indices:
        r = limited_rows[idx]
        rel = float(r.get("ticker_relevance_score", 0.5))
        sent = float(r.get("sentiment_score", 0))
        evt = r.get("event_type") or "other"
        h = r.get("impact_horizon") or "short_term"
        raw_conf = r.get("sentiment_confidence") or 0.5 + 0.3 * min(1.0, abs(sent))
        conf = float(raw_conf)
        raw_url = r.get("url", "") or ""
        url = _normalize_url(raw_url, r.get("source", "")) or raw_url
        top_3_impact.append(
            models.ImpactItem(
                title=articles_out[idx].title if idx < len(articles_out) else (r.get("title", "") or ""),
                why_it_matters=why_it_matters_text(evt, sent),
                impact_direction=_impact_direction(sent),
                time_horizon=HORIZON_LABELS.get(h, h),
                confidence=round(min(1.0, rel * (0.7 + 0.3 * conf)), 2),
                url=url,
                event_type=evt,
            )
        )

    if composite > 0.15:
        net_label = "bullish"
        agreement = bull_count / len(limited_rows) if limited_rows else 0.0
    elif composite < -0.15:
        net_label = "bearish"
        agreement = bear_count / len(limited_rows) if limited_rows else 0.0
    else:
        net_label = "neutral"
        agreement = neutral_count / len(limited_rows) if limited_rows else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    net_conf = round(min(1.0, 0.5 * agreement + 0.5 * avg_conf) * 100)

    top_contributors = []
    for idx in top_3_indices:
        if idx < len(articles_out):
            a = articles_out[idx]
            top_contributors.append({
                "title": a.title,
                "url": a.url,
                "direction": _impact_direction(a.sentiment_score),
                "raw_contribution": round(a.raw_contribution, 4),
            })
    reason_parts = [f"Prediction: {net_label}."]
    if top_contributors:
        driver_str = "; ".join(
            f'"{c["title"][:50]}..." ({c["raw_contribution"]:+.2f})' for c in top_contributors[:3]
        )
        reason_parts.append(f" Main drivers: {driver_str}.")
    if event_breakdown:
        event_str = ", ".join(f"{k}: {v}" for k, v in sorted(event_breakdown.items()) if v > 0)
        reason_parts.append(f" Event mix: {event_str}.")
    reasoning = "".join(reason_parts).strip()

    market_shock_result: Optional[models.MarketShockResult] = None
    try:
        market_shock_result = detect_market_shock(conn, symbol.upper(), hours=24, sentiment_method=sentiment_method)
    except Exception as e:
        logger.debug("Market shock check failed for %s: %s", symbol, e)

    return models.StockNewsSignal(
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
        reasoning=reasoning,
        top_contributors=top_contributors,
        market_shock=market_shock_result if (market_shock_result and market_shock_result.is_shock) else None,
    )


def detect_market_shock(
    conn,
    symbol: str,
    hours: int = 24,
    sentiment_method: str = "lexicon",
) -> models.MarketShockResult:
    """Detect abnormal sentiment spike or rare high-impact events in the last `hours`."""
    from src.news.db import get_enriched_articles_for_ticker

    symbol = symbol.upper()
    now = datetime.utcnow()
    window_start = now - timedelta(hours=hours)
    baseline_end = window_start
    baseline_start = baseline_end - timedelta(days=7)
    date_from_recent = window_start.strftime("%Y-%m-%d")
    date_to_recent = now.strftime("%Y-%m-%d")
    date_from_baseline = baseline_start.strftime("%Y-%m-%d")
    date_to_baseline = baseline_end.strftime("%Y-%m-%d")

    rows_recent = get_enriched_articles_for_ticker(
        conn, symbol, date_from=date_from_recent, date_to=date_to_recent,
        limit=200, min_relevance=0.0, sentiment_method=sentiment_method,
    )
    rows_baseline = get_enriched_articles_for_ticker(
        conn, symbol, date_from=date_from_baseline, date_to=date_to_baseline,
        limit=500, min_relevance=0.0, sentiment_method=sentiment_method,
    )

    def in_window(r: dict, start: datetime, end: datetime) -> bool:
        pt = r.get("published_at")
        if not pt:
            return True
        try:
            t = datetime.fromisoformat(str(pt).replace("Z", "+00:00")) if "T" in str(pt) else datetime.strptime(str(pt)[:10], "%Y-%m-%d")
            return start <= t <= end
        except Exception:
            return True

    recent = [r for r in rows_recent if in_window(r, window_start, now)]
    baseline = [r for r in rows_baseline if in_window(r, baseline_start, baseline_end)]

    if recent:
        avg_recent = sum(float(r.get("sentiment_score", 0)) for r in recent) / len(recent)
        avg_baseline = (sum(float(r.get("sentiment_score", 0)) for r in baseline) / len(baseline)) if baseline else 0.0
        if abs(avg_recent - avg_baseline) >= 0.5 and len(baseline) >= 3:
            return models.MarketShockResult(
                is_shock=True,
                reason="sentiment_spike",
                summary=f"Sentiment in last {hours}h (avg={avg_recent:.2f}) deviates strongly from 7d baseline (avg={avg_baseline:.2f}).",
                contributing_article_titles=[r.get("title", "") or "" for r in recent[:5]],
            )

    high_impact = [r for r in recent if (r.get("event_type") or "").lower() in HIGH_IMPACT_EVENT_TYPES]
    if len(high_impact) >= 2:
        return models.MarketShockResult(
            is_shock=True,
            reason="rare_events",
            summary=f"Multiple high-impact events (legal/M&A/macro) in last {hours}h.",
            contributing_article_titles=[r.get("title", "") or "" for r in high_impact[:5]],
        )
    if len(high_impact) == 1 and abs(float(high_impact[0].get("sentiment_score", 0))) > 0.5:
        return models.MarketShockResult(
            is_shock=True,
            reason="rare_events",
            summary="High-impact event with strong sentiment in last 24h.",
            contributing_article_titles=[high_impact[0].get("title", "") or ""],
        )

    return models.MarketShockResult(is_shock=False, reason="", summary="", contributing_article_titles=[])


def get_daily_signals(
    symbol: str,
    from_date: str,
    to_date: str,
    min_relevance: float = 0.2,
    sentiment_method: str = "lexicon",
    config_path: Optional[str] = None,
) -> List[models.DailyNewsSignal]:
    """
    Aggregate news into per-ticker daily signals with per-article contribution.
    from_date / to_date: YYYY-MM-DD (inclusive).
    """
    conn = _get_conn(config_path)
    if not conn:
        return []

    from src.news.db import get_enriched_articles_for_ticker
    from src.news.intelligence import get_horizon_weight

    symbol = symbol.upper()
    rows = get_enriched_articles_for_ticker(
        conn, symbol, date_from=from_date, date_to=to_date,
        limit=2000, min_relevance=min_relevance, sentiment_method=sentiment_method,
    )
    if not rows:
        return []

    by_date: Dict[str, List[dict]] = {}
    for r in rows:
        pt = r.get("published_at") or ""
        date_key = str(pt)[:10] if len(str(pt)) >= 10 else datetime.utcnow().strftime("%Y-%m-%d")
        if len(date_key) != 10:
            date_key = datetime.utcnow().strftime("%Y-%m-%d")
        if date_key not in by_date:
            by_date[date_key] = []
        by_date[date_key].append(r)

    out: List[models.DailyNewsSignal] = []
    for date_key in sorted(by_date.keys(), reverse=True):
        day_rows = by_date[date_key]
        weighted_sum = 0.0
        weight_sum = 0.0
        day_articles: List[models.EnrichedArticleView] = []
        for r in day_rows:
            rel = float(r.get("ticker_relevance_score", 0.5))
            sent = float(r.get("sentiment_score", 0))
            horizon = r.get("impact_horizon") or "short_term"
            hw = get_horizon_weight(horizon)
            w = rel * hw
            weighted_sum += sent * w
            weight_sum += w
            raw_conf = r.get("sentiment_confidence")
            if raw_conf is None:
                raw_conf = 0.5 + 0.3 * min(1.0, abs(sent))
            conf = float(raw_conf)
            body = r.get("body_clean") or ""
            summary = _summarize(body)
            if not summary and r.get("title"):
                summary = str(r.get("title", ""))[:150]
            raw_url = r.get("url", "") or ""
            url = _normalize_url(raw_url, r.get("source", "")) or raw_url
            day_articles.append(
                models.EnrichedArticleView(
                    article_id=r.get("article_id", 0),
                    title=r.get("title", ""),
                    summary=summary,
                    url=url,
                    source=r.get("source", ""),
                    published_at=r.get("published_at"),
                    event_type=r.get("event_type") or "other",
                    ticker_relevance=round(rel, 3),
                    sentiment_score=round(sent, 3),
                    sentiment_confidence=round(conf, 3),
                    impact_horizon=horizon,
                    horizon_weight=round(hw, 3),
                    contribution_weight=round(w, 3),
                    raw_contribution=round(sent * w, 4),
                )
            )
        composite = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        composite = max(-1.0, min(1.0, composite))
        out.append(
            models.DailyNewsSignal(
                date=date_key,
                composite_score=round(composite, 3),
                article_count=len(day_articles),
                articles=day_articles,
            )
        )
    return out


# -----------------------------------------------------------------------------
# 5. OUTPUT FOR ML — flat feature rows (one per symbol per date)
# -----------------------------------------------------------------------------


def get_ml_daily_features(
    symbol: str,
    from_date: str,
    to_date: str,
    min_relevance: float = 0.2,
    sentiment_method: str = "lexicon",
    config_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return per-day feature rows for ML models and backtests.

    Each row: date, symbol, composite_score, article_count, avg_sentiment,
    avg_relevance, event_* counts, horizon_* counts, plus optional top N raw_contribution sum.
    Usable as daily features in price forecasting or strategy signals.
    """
    daily = get_daily_signals(
        symbol, from_date, to_date,
        min_relevance=min_relevance,
        sentiment_method=sentiment_method,
        config_path=config_path,
    )
    rows = []
    for d in daily:
        event_counts = {}
        horizon_counts = {}
        for a in d.articles:
            event_counts[a.event_type] = event_counts.get(a.event_type, 0) + 1
            horizon_counts[a.impact_horizon] = horizon_counts.get(a.impact_horizon, 0) + 1
        avg_sent = sum(a.sentiment_score for a in d.articles) / len(d.articles) if d.articles else 0.0
        avg_rel = sum(a.ticker_relevance for a in d.articles) / len(d.articles) if d.articles else 0.0
        rows.append({
            "date": d.date,
            "symbol": symbol.upper(),
            "composite_score": d.composite_score,
            "article_count": d.article_count,
            "avg_sentiment": round(avg_sent, 4),
            "avg_relevance": round(avg_rel, 4),
            "event_breakdown": event_counts,
            "horizon_breakdown": horizon_counts,
        })
    return rows

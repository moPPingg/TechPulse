"""
News Intelligence: Source filtering, event classification, ticker relevance, impact horizon.

Investment-grade signal enrichment for the news pipeline.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Source tiers ---
SOURCE_STOCK = {"cafef", "vietstock"}
SOURCE_SECTOR = {"vnexpress", "vneconomy"}
SOURCE_MACRO = {"tradingeconomics"}
SOURCE_REGULATORY = {"ssc", "hsx"}

# --- Event type keywords (Vietnamese) ---
EVENT_KEYWORDS = {
    "earnings": [
        "báo cáo tài chính", "bctc", "lợi nhuận", "doanh thu", "kết quả kinh doanh",
        "lãi ròng", "lnst", "eps", "roe", "biên lợi nhuận",
    ],
    "legal": ["kiện", "phạt", "vi phạm", "ubck", "ssc", "xử phạt", "cảnh cáo"],
    "macro": ["lãi suất", "gdp", "inflation", "fed", "ngân hàng nhà nước", "chính sách tiền tệ"],
    "operations": ["mở rộng", "nhà máy", "dự án", "sản xuất", "kinh doanh", "hợp đồng"],
    "guidance": ["kỳ vọng", "mục tiêu", "dự báo doanh thu", "kế hoạch năm"],
    "ma": ["mua lại", "sáp nhập", "thâu tóm", "m&a", "deal"],
    "dividend": ["cổ tức", "chia cổ phần", "trả cổ tức"],
}
EVENT_OTHER = "other"

# --- Impact horizon keywords ---
HORIZON_INTRADAY = ["trong phiên", "mở cửa", "đóng cửa", "trong ngày"]
HORIZON_SHORT = ["tuần tới", "tháng này", "quý này", "ngắn hạn"]
HORIZON_LONG = ["năm nay", "kế hoạch", "chiến lược", "dài hạn"]

# --- Event weights for relevance (higher = more impactful) ---
EVENT_WEIGHTS = {
    "earnings": 1.0,
    "ma": 1.0,
    "guidance": 1.0,
    "legal": 0.8,
    "operations": 0.8,
    "dividend": 0.7,
    "macro": 0.6,
    "other": 0.5,
}

# --- Source weights for relevance ---
SOURCE_WEIGHTS = {
    **{s: 1.0 for s in SOURCE_STOCK},
    **{s: 0.8 for s in SOURCE_SECTOR},
    **{s: 0.6 for s in SOURCE_REGULATORY},
    **{s: 0.4 for s in SOURCE_MACRO},
}

# --- Horizon weights for aggregation ---
HORIZON_WEIGHTS = {"intraday": 1.2, "short_term": 1.0, "long_term": 0.8}


def classify_source_tier(source: str) -> str:
    """Classify source: stock, sector, macro, regulatory."""
    s = (source or "").lower()
    if s in SOURCE_STOCK:
        return "stock"
    if s in SOURCE_SECTOR:
        return "sector"
    if s in SOURCE_MACRO:
        return "macro"
    if s in SOURCE_REGULATORY:
        return "regulatory"
    return "other"


def should_drop_by_source_filter(
    source_tier: str,
    ticker_in_title: bool,
    ticker_in_body: bool,
    symbol: Optional[str] = None,
) -> bool:
    """
    Drop macro/sector articles that don't mention the ticker.
    Returns True if article should be dropped.
    """
    if source_tier == "stock":
        return False
    if source_tier == "regulatory":
        return False  # Keep regulatory (disclosures)
    if source_tier in ("sector", "macro"):
        if ticker_in_title or ticker_in_body:
            return False
        return True  # Drop macro/sector without ticker mention
    return False


def classify_event_type(title: str, body: str) -> str:
    """Classify event type from title + body."""
    text = ((title or "") + " " + (body or "")).lower()
    scores = {}
    for event_type, keywords in EVENT_KEYWORDS.items():
        scores[event_type] = sum(1 for k in keywords if k in text)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else EVENT_OTHER


def classify_impact_horizon(title: str, body: str, published_at: Optional[str]) -> str:
    """Classify impact horizon: intraday, short_term, long_term."""
    text = ((title or "") + " " + (body or "")).lower()
    for kw in HORIZON_INTRADAY:
        if kw in text:
            return "intraday"
    for kw in HORIZON_SHORT:
        if kw in text:
            return "short_term"
    for kw in HORIZON_LONG:
        if kw in text:
            return "long_term"

    # Recency heuristic: very recent = intraday
    if published_at:
        try:
            dt = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
            dt_naive = dt.replace(tzinfo=None) if hasattr(dt, "replace") else dt
            hours_ago = (datetime.utcnow() - dt_naive).total_seconds() / 3600
            if hours_ago < 6:
                return "intraday"
        except Exception:
            pass

    return "short_term"  # Default


def compute_sentiment_confidence(score: float, pos_count: int, neg_count: int) -> float:
    """Confidence 0-1 based on term count and score extremity."""
    term_count = pos_count + neg_count
    term_factor = min(1.0, term_count / 3.0)
    extremity = 0.7 + 0.3 * min(1.0, abs(score))
    return min(1.0, term_factor * extremity)


def compute_recency_score(published_at: Optional[str]) -> float:
    """Recency score 0-1. Fresher = higher."""
    if not published_at:
        return 0.3
    try:
        dt = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        dt_naive = dt.replace(tzinfo=None) if hasattr(dt, "replace") else dt
        days_ago = (datetime.utcnow() - dt_naive).days
    except Exception:
        return 0.3
    if days_ago <= 1:
        return 1.0
    if days_ago <= 3:
        return 0.8
    if days_ago <= 7:
        return 0.5
    if days_ago <= 30:
        return 0.2
    return 0.1


def compute_ticker_relevance(
    ticker_in_title: bool,
    ticker_in_body: bool,
    alias_only: bool,
    event_type: str,
    recency: float,
    source: str,
) -> float:
    """
    Ticker relevance 0-1.
    Formula: 0.4*mention + 0.25*event + 0.2*recency + 0.15*source
    """
    if ticker_in_title:
        mention = 1.0
    elif ticker_in_body:
        mention = 0.6
    elif alias_only:
        mention = 0.3
    else:
        mention = 0.1

    event_w = EVENT_WEIGHTS.get(event_type, 0.5)
    source_w = SOURCE_WEIGHTS.get((source or "").lower(), 0.5)

    relevance = 0.4 * mention + 0.25 * event_w + 0.2 * recency + 0.15 * source_w
    return min(1.0, relevance)


def enrich_article(
    source: str,
    title: str,
    body: str,
    published_at: Optional[str],
    sentiment_score: float,
    tickers: List[str],
    sentiment_pos_count: int = 0,
    sentiment_neg_count: int = 0,
) -> Tuple[dict, List[Tuple[str, float]]]:
    """
    Enrich a single article. Returns (enrichment_dict, [(ticker, relevance_score), ...]).
    tickers: list of tickers from align (exclude __NONE__).
    """
    body_clean = body or ""
    text_upper = (title + " " + body_clean).upper()

    source_tier = classify_source_tier(source)
    event_type = classify_event_type(title, body_clean)
    impact_horizon = classify_impact_horizon(title, body_clean, published_at)
    recency = compute_recency_score(published_at)
    term_count = sentiment_pos_count + sentiment_neg_count
    if term_count == 0:
        sent_conf = 0.5 + 0.3 * min(1.0, abs(sentiment_score))  # Proxy when no counts
    else:
        sent_conf = compute_sentiment_confidence(
            sentiment_score, sentiment_pos_count, sentiment_neg_count
        )

    # Build ticker list for scoring (from align or extract from text)
    tickers_to_score = [t for t in tickers if t and t != "__NONE__"]
    if not tickers_to_score:
        tickers_to_score = []  # Will try text match in pipeline

    enrichment = {
        "source_tier": source_tier,
        "event_type": event_type,
        "impact_horizon": impact_horizon,
        "sentiment_confidence": round(sent_conf, 3),
    }

    body_upper = body_clean.upper()
    ticker_scores = []
    for ticker in tickers_to_score:
        sym = ticker.upper()
        in_title = sym in (title or "").upper()
        in_body = sym in body_upper
        rel = compute_ticker_relevance(
            ticker_in_title=in_title,
            ticker_in_body=in_body,
            alias_only=False,
            event_type=event_type,
            recency=recency,
            source=source,
        )
        ticker_scores.append((ticker, round(rel, 3)))

    return enrichment, ticker_scores


def get_horizon_weight(horizon: str) -> float:
    return HORIZON_WEIGHTS.get(horizon, 1.0)

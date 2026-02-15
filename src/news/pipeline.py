"""
Orchestrate: crawl → clean → sentiment → align. All steps idempotent; DB is source of truth.
"""

import logging
from pathlib import Path
from typing import Optional
import hashlib

from src.news.db import (
    get_engine,
    init_db,
    insert_article,
    find_article_by_content_hash,
    set_article_cleaned,
    set_sentiment,
    set_article_tickers,
    set_article_enrichment,
    set_article_ticker_score,
    get_article_ids_without_cleaned,
    get_article_ids_without_sentiment,
    get_article_ids_without_tickers,
    get_article_ids_with_only_none_ticker,
    delete_article_tickers,
    get_article_ids_without_enrichment,
    get_article_by_id,
)
from src.news.clean import clean_article_body
from src.news.sentiment import sentiment_lexicon_extended
from src.news.ticker_align import align_article_to_tickers, extract_tickers_from_text, load_symbols_and_aliases
from src.news.intelligence import enrich_article
from src.news.crawlers.cafef import CafeFNewsCrawler
from src.news.crawlers.vietstock import VietstockNewsCrawler
from src.news.crawlers.vnexpress import VNExpressNewsCrawler
from src.news.crawlers.hsx import HSXNewsCrawler
from src.news.crawlers.vneconomy import VNEconomyNewsCrawler
from src.news.crawlers.tradingeconomics import TradingEconomicsCrawler
from src.news.crawlers.ssc import SSCReportCrawler

logger = logging.getLogger(__name__)


def load_news_config(config_path: Optional[str] = None) -> dict:
    import yaml
    path = Path(config_path or "configs/news.yaml")
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run_crawl(conn, config: dict) -> int:
    """Crawl enabled sources; insert/upsert articles. Returns count of processed URLs."""
    sources = config.get("sources", {})
    db_path = config.get("database", {}).get("path", "data/news/news.db")
    crawlers = {
        "cafef": CafeFNewsCrawler(
            base_url=sources.get("cafef", {}).get("base_url", "https://cafef.vn"),
            request_delay_seconds=sources.get("cafef", {}).get("request_delay_seconds", 2),
            timeout_seconds=sources.get("cafef", {}).get("timeout_seconds", 30),
        ),
        "vietstock": VietstockNewsCrawler(
            base_url=sources.get("vietstock", {}).get("base_url", "https://vietstock.vn"),
            request_delay_seconds=sources.get("vietstock", {}).get("request_delay_seconds", 2),
            timeout_seconds=sources.get("vietstock", {}).get("timeout_seconds", 30),
        ),
        "vnexpress": VNExpressNewsCrawler(
            base_url=sources.get("vnexpress", {}).get("base_url", "https://vnexpress.net"),
            request_delay_seconds=sources.get("vnexpress", {}).get("request_delay_seconds", 2),
            timeout_seconds=sources.get("vnexpress", {}).get("timeout_seconds", 30),
        ),
        "hsx": HSXNewsCrawler(
            base_url=sources.get("hsx", {}).get("base_url", "https://www.hsx.vn"),
            request_delay_seconds=sources.get("hsx", {}).get("request_delay_seconds", 2),
            timeout_seconds=sources.get("hsx", {}).get("timeout_seconds", 30),
        ),
        "vneconomy": VNEconomyNewsCrawler(
            base_url=sources.get("vneconomy", {}).get("base_url", "https://vneconomy.vn"),
            request_delay_seconds=sources.get("vneconomy", {}).get("request_delay_seconds", 2),
            timeout_seconds=sources.get("vneconomy", {}).get("timeout_seconds", 30),
        ),
        "tradingeconomics": TradingEconomicsCrawler(
            base_url=sources.get("tradingeconomics", {}).get("base_url", "https://tradingeconomics.com"),
            request_delay_seconds=sources.get("tradingeconomics", {}).get("request_delay_seconds", 2),
            timeout_seconds=sources.get("tradingeconomics", {}).get("timeout_seconds", 30),
        ),
        "ssc": SSCReportCrawler(
            base_url=sources.get("ssc", {}).get("base_url", "https://ssc.gov.vn"),
            request_delay_seconds=sources.get("ssc", {}).get("request_delay_seconds", 2),
            timeout_seconds=sources.get("ssc", {}).get("timeout_seconds", 30),
        ),
    }
    total = 0
    for name, cfg in sources.items():
        if not cfg.get("enabled") or name not in crawlers:
            continue
        crawler = crawlers[name]
        categories = cfg.get("categories", [])
        max_pages = cfg.get("max_pages_per_run", 2)
        for cat in categories:
            records = crawler.fetch_article_list(cat, max_pages)
            for rec in records:
                body = rec.body_raw
                if not body and hasattr(crawler, "fetch_article_body"):
                    body = crawler.fetch_article_body(rec.url)

                # Content-hash-based deduplication across sources.
                normalized_text = f"{(rec.title or '').strip()}\n{(body or '').strip()}"
                content_hash = hashlib.sha1(
                    normalized_text.encode("utf-8", errors="ignore")
                ).hexdigest() if normalized_text.strip() else None

                if content_hash:
                    existing_id = find_article_by_content_hash(conn, content_hash)
                    if existing_id:
                        logger.debug(
                            "Skip duplicate article %s from %s (existing id=%s)",
                            rec.url,
                            rec.source,
                            existing_id,
                        )
                        continue

                aid = insert_article(
                    conn,
                    source=rec.source,
                    url=rec.url,
                    title=rec.title,
                    body_raw=body,
                    published_at=rec.published_at,
                    content_hash=content_hash,
                )
                if aid:
                    total += 1
    logger.info("Crawl step: %d articles processed", total)
    return total


def run_clean(conn, config: dict) -> int:
    """Clean bodies for articles that don't have article_cleaned yet."""
    clean_cfg = config.get("clean", {})
    max_chars = clean_cfg.get("max_body_chars")
    ids = get_article_ids_without_cleaned(conn, limit=5000)
    count = 0
    for aid in ids:
        row = get_article_by_id(conn, aid)
        if not row:
            continue
        raw = row.get("body_raw") or ""
        cleaned = clean_article_body(
            raw,
            strip_html_flag=clean_cfg.get("strip_html", True),
            normalize_ws=clean_cfg.get("normalize_whitespace", True),
            max_chars=max_chars,
        )
        set_article_cleaned(conn, aid, cleaned)
        count += 1
    logger.info("Clean step: %d articles", count)
    return count


def run_sentiment(conn, config: dict) -> int:
    """Compute sentiment for articles that don't have it (method from config)."""
    method = config.get("sentiment", {}).get("method", "lexicon")
    ids = get_article_ids_without_sentiment(conn, method=method, limit=5000)
    count = 0
    for aid in ids:
        row = get_article_by_id(conn, aid)
        if not row:
            continue
        text = (row.get("title") or "") + " " + (row.get("body_raw") or "")
        score, _, _, _ = sentiment_lexicon_extended(text)
        set_sentiment(conn, aid, score, method)
        count += 1
    logger.info("Sentiment step: %d articles", count)
    return count


def run_align(conn, config: dict, project_root: Optional[Path] = None) -> int:
    """Align articles to tickers; store article_tickers."""
    align_cfg = config.get("ticker_align", {})
    symbols_path = align_cfg.get("symbols")
    if symbols_path and project_root:
        p = Path(symbols_path)
        if not p.is_absolute():
            symbols_path = str(project_root / p)
    ids = get_article_ids_without_tickers(conn, limit=5000)
    ids_none = get_article_ids_with_only_none_ticker(conn, limit=5000)
    for aid in ids_none:
        delete_article_tickers(conn, aid)
    ids = list(dict.fromkeys(ids + ids_none))
    count = 0
    for aid in ids:
        row = get_article_by_id(conn, aid)
        if not row:
            continue
        cur = conn.execute("SELECT body_clean FROM article_cleaned WHERE article_id = ?", (aid,))
        r = cur.fetchone()
        body_clean = r["body_clean"] if r else (row.get("body_raw") or "")
        tickers = align_article_to_tickers(
            row.get("title") or "",
            body_clean,
            symbols_path=symbols_path,
        )
        if tickers:
            set_article_tickers(conn, aid, [(t, rel) for t, rel in tickers])
            count += 1
        else:
            set_article_tickers(conn, aid, [("__NONE__", "none")])
    logger.info("Align step: %d articles with at least one ticker", count)
    return count


def run_enrich(conn, config: dict, project_root: Optional[Path] = None) -> int:
    """Enrich articles: source_tier, event_type, impact_horizon, ticker_relevance.
    Khi llm_enrich.enabled=true và có API key, dùng LLM để đọc tin và phân loại chính xác hơn."""
    align_cfg = config.get("ticker_align", {})
    symbols_path = align_cfg.get("symbols")
    if symbols_path and project_root:
        p = Path(symbols_path)
        if not p.is_absolute():
            symbols_path = str(project_root / p)
    symbols, aliases = load_symbols_and_aliases(symbols_path)
    use_llm = bool(config.get("llm_enrich", {}).get("enabled"))

    ids = get_article_ids_without_enrichment(conn, limit=5000)
    count = 0
    for aid in ids:
        row = get_article_by_id(conn, aid)
        if not row:
            continue
        cur = conn.execute("SELECT body_clean FROM article_cleaned WHERE article_id = ?", (aid,))
        r = cur.fetchone()
        body_clean = r["body_clean"] if r else (row.get("body_raw") or "")

        cur = conn.execute("SELECT score FROM sentiments WHERE article_id = ? AND method = ?", (aid, "lexicon"))
        sent_row = cur.fetchone()
        sentiment_score = float(sent_row["score"]) if sent_row else 0.0

        cur = conn.execute("SELECT ticker FROM article_tickers WHERE article_id = ?", (aid,))
        db_tickers = [r["ticker"] for r in cur.fetchall() if r["ticker"] and r["ticker"] != "__NONE__"]
        if not db_tickers:
            combined = (row.get("title") or "") + "\n" + body_clean
            extracted = extract_tickers_from_text(combined, symbols, aliases)
            db_tickers = [t for t, _ in extracted]

        text = (row.get("title") or "") + " " + (row.get("body_raw") or "")
        _, _, pos_count, neg_count = sentiment_lexicon_extended(text)

        # Thử LLM enrich nếu bật (AI đọc tin và phân loại)
        llm_result = None
        if use_llm:
            try:
                from src.news.llm_enrich import enrich_with_llm
                llm_result = enrich_with_llm(
                    title=row.get("title", ""),
                    body=body_clean,
                    config=config,
                )
            except Exception as e:
                logger.debug("LLM enrich skip for article %d: %s", aid, e)

        if llm_result:
            # Dùng kết quả từ LLM
            set_sentiment(conn, aid, llm_result["sentiment"], "llm")
            sent_conf = 0.5 + 0.3 * min(1.0, abs(llm_result["sentiment"]))
            from src.news.intelligence import classify_source_tier
            source_tier = classify_source_tier(row.get("source", ""))
            set_article_enrichment(
                conn, aid,
                source_tier=source_tier,
                event_type=llm_result["event_type"],
                impact_horizon=llm_result["impact_horizon"],
                sentiment_confidence=round(sent_conf, 3),
            )
            # Ticker scores: dùng rule-based, scale bởi llm ticker_relevance
            enrichment, ticker_scores = enrich_article(
                source=row.get("source", ""),
                title=row.get("title", ""),
                body=body_clean,
                published_at=row.get("published_at"),
                sentiment_score=llm_result["sentiment"],
                tickers=db_tickers,
                sentiment_pos_count=pos_count,
                sentiment_neg_count=neg_count,
            )
            llm_rel = llm_result.get("ticker_relevance", 0.5)
            for ticker, rel in ticker_scores:
                blended = round((rel + llm_rel) / 2, 3)
                set_article_ticker_score(conn, aid, ticker, blended)
        else:
            # Fallback: rule-based như cũ
            enrichment, ticker_scores = enrich_article(
                source=row.get("source", ""),
                title=row.get("title", ""),
                body=body_clean,
                published_at=row.get("published_at"),
                sentiment_score=sentiment_score,
                tickers=db_tickers,
                sentiment_pos_count=pos_count,
                sentiment_neg_count=neg_count,
            )
            set_article_enrichment(
                conn, aid,
                source_tier=enrichment["source_tier"],
                event_type=enrichment["event_type"],
                impact_horizon=enrichment["impact_horizon"],
                sentiment_confidence=enrichment.get("sentiment_confidence"),
            )
            for ticker, rel in ticker_scores:
                set_article_ticker_score(conn, aid, ticker, rel)
        count += 1
    logger.info("Enrich step: %d articles (LLM=%s)", count, use_llm)
    return count


def run_pipeline(config_path: Optional[str] = None, steps: Optional[list] = None) -> dict:
    """
    Run full or partial pipeline. steps = ['crawl','clean','sentiment','align'] or None for all.
    Returns dict of step -> count.
    """
    config = load_news_config(config_path)
    db_path = config.get("database", {}).get("path", "data/news/news.db")
    root = Path(__file__).resolve().parent.parent.parent
    if not Path(db_path).is_absolute():
        db_path = str(root / db_path)
    init_db(db_path)
    conn = get_engine(db_path)
    todo = steps or ["crawl", "clean", "sentiment", "align", "enrich"]
    results = {}
    if "crawl" in todo:
        results["crawl"] = run_crawl(conn, config)
    if "clean" in todo:
        results["clean"] = run_clean(conn, config)
    if "sentiment" in todo:
        results["sentiment"] = run_sentiment(conn, config)
    if "align" in todo:
        results["align"] = run_align(conn, config, project_root=root)
    if "enrich" in todo:
        results["enrich"] = run_enrich(conn, config, project_root=root)
    return results

"""
News database schema and access. SQLite by default; schema is research-usable and stable.

Why this file exists: Single source of truth for news storage (articles, cleaned, sentiments,
tickers, enrichments, ticker_scores). Used by src/news/pipeline.py (write) and src/news/engine.py (read).
Upstream: pipeline inserts; downstream: engine queries for aggregation and signals.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 1

CREATE_ARTICLES = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    body_raw TEXT,
    published_at TEXT,
    created_at TEXT NOT NULL,
    -- Optional content-based hash for cross-source deduplication
    content_hash TEXT,
    UNIQUE(source, url)
);
"""

CREATE_ARTICLE_CLEANED = """
CREATE TABLE IF NOT EXISTS article_cleaned (
    article_id INTEGER NOT NULL PRIMARY KEY,
    body_clean TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_SENTIMENTS = """
CREATE TABLE IF NOT EXISTS sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    score REAL NOT NULL,
    method TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(article_id, method),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_ARTICLE_TICKERS = """
CREATE TABLE IF NOT EXISTS article_tickers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    relevance TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(article_id, ticker),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_ARTICLE_ENRICHMENTS = """
CREATE TABLE IF NOT EXISTS article_enrichments (
    article_id INTEGER NOT NULL PRIMARY KEY,
    source_tier TEXT NOT NULL,
    event_type TEXT NOT NULL,
    impact_horizon TEXT NOT NULL,
    sentiment_confidence REAL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_ARTICLE_TICKER_SCORES = """
CREATE TABLE IF NOT EXISTS article_ticker_scores (
    article_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    relevance_score REAL NOT NULL,
    PRIMARY KEY (article_id, ticker),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)",
    "CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at)",
    "CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_article_tickers_ticker ON article_tickers(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_article_tickers_article_id ON article_tickers(article_id)",
    "CREATE INDEX IF NOT EXISTS idx_sentiments_article_id ON sentiments(article_id)",
    "CREATE INDEX IF NOT EXISTS idx_article_ticker_scores_ticker ON article_ticker_scores(ticker)",
]

_engine = None


def get_engine(db_path: str):
    global _engine
    import sqlite3
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _engine = sqlite3.connect(str(path), timeout=30)
    _engine.row_factory = sqlite3.Row
    return _engine


def init_db(db_path: str) -> None:
    """Create tables and indexes. Idempotent."""
    conn = get_engine(db_path)
    for stmt in [
        CREATE_ARTICLES,
        CREATE_ARTICLE_CLEANED,
        CREATE_SENTIMENTS,
        CREATE_ARTICLE_TICKERS,
        CREATE_ARTICLE_ENRICHMENTS,
        CREATE_ARTICLE_TICKER_SCORES,
    ]:
        conn.execute(stmt)
    # Lightweight migration for older DBs: ensure content_hash column exists.
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN content_hash TEXT")
    except Exception:
        # Column already exists or migration not needed.
        pass
    for stmt in CREATE_INDEXES:
        conn.execute(stmt)
    conn.commit()
    logger.info("News DB initialized at %s", db_path)


def _iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def find_article_by_content_hash(conn, content_hash: str) -> Optional[int]:
    """Return existing article id for a given content hash, if any."""
    if not content_hash:
        return None
    cur = conn.execute(
        "SELECT id FROM articles WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    )
    row = cur.fetchone()
    return int(row["id"]) if row else None


def insert_article(
    conn,
    source: str,
    url: str,
    title: str,
    body_raw: Optional[str] = None,
    published_at: Optional[str] = None,
    content_hash: Optional[str] = None,
) -> int:
    """Insert or replace by (source, url). Returns article id."""
    now = _iso_now()
    cursor = conn.execute(
        """
        INSERT INTO articles (source, url, title, body_raw, published_at, created_at, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, url) DO UPDATE SET
            title = excluded.title,
            body_raw = COALESCE(excluded.body_raw, body_raw),
            published_at = COALESCE(excluded.published_at, published_at),
            content_hash = COALESCE(excluded.content_hash, content_hash)
        """,
        (source, url, title, body_raw or "", published_at, now, content_hash),
    )
    conn.commit()
    rid = cursor.lastrowid
    if rid == 0:
        cur = conn.execute("SELECT id FROM articles WHERE source = ? AND url = ?", (source, url))
        row = cur.fetchone()
        rid = row["id"] if row else 0
    return rid


def set_article_cleaned(conn, article_id: int, body_clean: str) -> None:
    now = _iso_now()
    conn.execute(
        """
        INSERT INTO article_cleaned (article_id, body_clean, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(article_id) DO UPDATE SET body_clean = excluded.body_clean, updated_at = excluded.updated_at
        """,
        (article_id, body_clean, now),
    )
    conn.commit()


def set_sentiment(conn, article_id: int, score: float, method: str) -> None:
    now = _iso_now()
    conn.execute(
        """
        INSERT INTO sentiments (article_id, score, method, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(article_id, method) DO UPDATE SET score = excluded.score
        """,
        (article_id, score, method, now),
    )
    conn.commit()


def set_article_tickers(conn, article_id: int, tickers: List[Tuple[str, Optional[str]]]) -> None:
    now = _iso_now()
    conn.execute("DELETE FROM article_tickers WHERE article_id = ?", (article_id,))
    for ticker, relevance in tickers:
        conn.execute(
            "INSERT INTO article_tickers (article_id, ticker, relevance, created_at) VALUES (?, ?, ?, ?)",
            (article_id, ticker, relevance or "", now),
        )
    conn.commit()


def get_articles_by_ticker_date(
    conn,
    ticker: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 500,
    sentiment_method: str = "lexicon",
) -> List[dict]:
    """Return list of dicts: id, source, url, title, body_clean, published_at, sentiment_score."""
    sql = """
        SELECT a.id, a.source, a.url, a.title, a.published_at,
               COALESCE(c.body_clean, a.body_raw) AS body_clean,
               s.score AS sentiment_score
        FROM articles a
        JOIN article_tickers at ON at.article_id = a.id AND at.ticker = ?
        LEFT JOIN article_cleaned c ON c.article_id = a.id
        LEFT JOIN sentiments s ON s.article_id = a.id AND s.method = ?
        WHERE 1=1
    """
    params: list = [ticker, sentiment_method]
    if date_from:
        sql += " AND (a.published_at >= ? OR a.published_at IS NULL)"
        params.append(date_from)
    if date_to:
        sql += " AND (a.published_at <= ? OR a.published_at IS NULL)"
        params.append(date_to)
    sql += " ORDER BY a.published_at DESC LIMIT ?"
    params.append(limit)
    cur = conn.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def get_recent_articles(
    conn,
    limit: int = 20,
    sentiment_method: str = "lexicon",
) -> List[dict]:
    """Tin gần đây không lọc theo ticker (fallback khi không có tin theo mã)."""
    sql = """
        SELECT a.id, a.source, a.url, a.title, a.published_at,
               COALESCE(c.body_clean, a.body_raw) AS body_clean,
               s.score AS sentiment_score
        FROM articles a
        LEFT JOIN article_cleaned c ON c.article_id = a.id
        LEFT JOIN sentiments s ON s.article_id = a.id AND s.method = ?
        ORDER BY COALESCE(a.published_at, a.created_at) DESC
        LIMIT ?
    """
    cur = conn.execute(sql, (sentiment_method, limit))
    return [dict(row) for row in cur.fetchall()]


def get_articles_matching_symbol_in_text(
    conn,
    symbol: str,
    date_from: Optional[str] = None,
    limit: int = 20,
    sentiment_method: str = "lexicon",
) -> List[dict]:
    """Tin gần đây có nhắc đến symbol trong title hoặc body (fallback khi không có trong article_tickers)."""
    sql = """
        SELECT a.id, a.source, a.url, a.title, a.published_at,
               COALESCE(c.body_clean, a.body_raw) AS body_clean,
               s.score AS sentiment_score
        FROM articles a
        LEFT JOIN article_cleaned c ON c.article_id = a.id
        LEFT JOIN sentiments s ON s.article_id = a.id AND s.method = ?
        WHERE (
            a.title LIKE ? OR
            COALESCE(c.body_clean, a.body_raw) LIKE ?
        )
    """
    pattern = f"%{symbol.upper()}%"
    params: list = [sentiment_method, pattern, pattern]
    if date_from:
        sql += " AND (a.published_at >= ? OR a.published_at IS NULL)"
        params.append(date_from)
    sql += " ORDER BY COALESCE(a.published_at, a.created_at) DESC LIMIT ?"
    params.append(limit)
    cur = conn.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def get_article_ids_without_cleaned(conn, limit: int = 1000) -> List[int]:
    cur = conn.execute(
        "SELECT a.id FROM articles a LEFT JOIN article_cleaned c ON c.article_id = a.id WHERE c.article_id IS NULL LIMIT ?",
        (limit,),
    )
    return [row["id"] for row in cur.fetchall()]


def get_article_ids_without_sentiment(conn, method: str, limit: int = 1000) -> List[int]:
    cur = conn.execute(
        """
        SELECT a.id FROM articles a
        LEFT JOIN sentiments s ON s.article_id = a.id AND s.method = ?
        WHERE s.article_id IS NULL
        LIMIT ?
        """,
        (method, limit),
    )
    return [row["id"] for row in cur.fetchall()]


def get_article_ids_without_tickers(conn, limit: int = 1000) -> List[int]:
    """Articles with no ticker row at all."""
    cur = conn.execute(
        "SELECT a.id FROM articles a LEFT JOIN article_tickers at ON at.article_id = a.id WHERE at.article_id IS NULL LIMIT ?",
        (limit,),
    )
    return [row["id"] for row in cur.fetchall()]


def get_article_ids_with_only_none_ticker(conn, limit: int = 1000) -> List[int]:
    """Articles that only have __NONE__ ticker (re-align when aliases improve)."""
    cur = conn.execute(
        """
        SELECT a.id FROM articles a
        WHERE NOT EXISTS (SELECT 1 FROM article_tickers at WHERE at.article_id = a.id AND at.ticker != '__NONE__')
        AND EXISTS (SELECT 1 FROM article_tickers at WHERE at.article_id = a.id AND at.ticker = '__NONE__')
        LIMIT ?
        """,
        (limit,),
    )
    return [row["id"] for row in cur.fetchall()]


def delete_article_tickers(conn, article_id: int) -> None:
    """Remove all ticker rows for an article (for re-align)."""
    conn.execute("DELETE FROM article_tickers WHERE article_id = ?", (article_id,))
    conn.commit()


def get_article_by_id(conn, article_id: int) -> Optional[dict]:
    cur = conn.execute(
        "SELECT id, source, url, title, body_raw, published_at FROM articles WHERE id = ?",
        (article_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_recent_general_articles(
    conn,
    date_from: Optional[str] = None,
    limit: int = 20,
    sentiment_method: str = "lexicon",
    sources: Optional[List[str]] = None,
) -> List[dict]:
    """Tin thị trường gần đây (fallback khi không có tin theo mã). sources: cafef, vietstock, vnexpress, vneconomy."""
    default_sources = ("cafef", "vietstock", "vnexpress", "vneconomy")
    src = sources or default_sources
    placeholders = ",".join("?" * len(src))
    sql = f"""
        SELECT a.id AS article_id, a.source, a.url, a.title, a.published_at,
               COALESCE(c.body_clean, a.body_raw) AS body_clean,
               COALESCE(
                   (SELECT score FROM sentiments WHERE article_id = a.id AND method = 'llm' LIMIT 1),
                   (SELECT score FROM sentiments WHERE article_id = a.id AND method = ? LIMIT 1)
               ) AS sentiment_score,
               e.source_tier, e.event_type, e.impact_horizon, e.sentiment_confidence,
               0.4 AS ticker_relevance_score
        FROM articles a
        LEFT JOIN article_cleaned c ON c.article_id = a.id
        LEFT JOIN article_enrichments e ON e.article_id = a.id
        WHERE a.source IN ({placeholders})
    """
    params: list = [sentiment_method] + list(src)
    if date_from:
        sql += " AND (a.published_at >= ? OR a.published_at IS NULL)"
        params.append(date_from)
    sql += " ORDER BY COALESCE(a.published_at, a.created_at) DESC LIMIT ?"
    params.append(limit)
    cur = conn.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def set_article_enrichment(
    conn,
    article_id: int,
    source_tier: str,
    event_type: str,
    impact_horizon: str,
    sentiment_confidence: Optional[float] = None,
) -> None:
    now = _iso_now()
    conn.execute(
        """
        INSERT INTO article_enrichments (article_id, source_tier, event_type, impact_horizon, sentiment_confidence, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id) DO UPDATE SET
            source_tier = excluded.source_tier,
            event_type = excluded.event_type,
            impact_horizon = excluded.impact_horizon,
            sentiment_confidence = excluded.sentiment_confidence,
            updated_at = excluded.updated_at
        """,
        (article_id, source_tier, event_type, impact_horizon, sentiment_confidence, now),
    )
    conn.commit()


def set_article_ticker_score(conn, article_id: int, ticker: str, relevance_score: float) -> None:
    conn.execute(
        """
        INSERT INTO article_ticker_scores (article_id, ticker, relevance_score)
        VALUES (?, ?, ?)
        ON CONFLICT(article_id, ticker) DO UPDATE SET relevance_score = excluded.relevance_score
        """,
        (article_id, ticker, relevance_score),
    )
    conn.commit()


def get_article_ids_without_enrichment(conn, limit: int = 1000) -> List[int]:
    cur = conn.execute(
        """
        SELECT a.id FROM articles a
        LEFT JOIN article_enrichments e ON e.article_id = a.id
        LEFT JOIN sentiments s ON s.article_id = a.id AND s.method = 'lexicon'
        WHERE e.article_id IS NULL AND s.article_id IS NOT NULL
        LIMIT ?
        """,
        (limit,),
    )
    return [row["id"] for row in cur.fetchall()]


def get_enriched_articles_for_ticker(
    conn,
    ticker: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    min_relevance: float = 0.0,
    sentiment_method: str = "lexicon",
    event_type_filter: Optional[str] = None,
) -> List[dict]:
    """
    Get articles with enrichments and ticker scores for symbol.
    Uses article_ticker_scores when available; fallback 0.5 for unenriched.
    date_from / date_to: YYYY-MM-DD; both inclusive for filtering published_at.
    """
    ticker = ticker.upper()
    pattern = f"%{ticker}%"
    sql = """
        SELECT a.id AS article_id, a.source, a.url, a.title, a.published_at,
               COALESCE(c.body_clean, a.body_raw) AS body_clean,
               COALESCE(
                   (SELECT score FROM sentiments WHERE article_id = a.id AND method = 'llm' LIMIT 1),
                   (SELECT score FROM sentiments WHERE article_id = a.id AND method = ? LIMIT 1)
               ) AS sentiment_score,
               e.source_tier, e.event_type, e.impact_horizon, e.sentiment_confidence,
               COALESCE(ts.relevance_score, 0.5) AS ticker_relevance_score
        FROM articles a
        LEFT JOIN article_cleaned c ON c.article_id = a.id
        LEFT JOIN article_enrichments e ON e.article_id = a.id
        LEFT JOIN article_ticker_scores ts ON ts.article_id = a.id AND ts.ticker = ?
        WHERE (
            EXISTS (SELECT 1 FROM article_tickers at WHERE at.article_id = a.id AND at.ticker = ?)
            OR a.title LIKE ?
            OR COALESCE(c.body_clean, a.body_raw) LIKE ?
        )
    """
    params: list = [sentiment_method, ticker, ticker, pattern, pattern]
    if date_from:
        sql += " AND (a.published_at >= ? OR a.published_at IS NULL)"
        params.append(date_from)
    if date_to:
        sql += " AND (a.published_at <= ? OR a.published_at IS NULL)"
        end_day = (date_to + "T23:59:59") if len(date_to) == 10 else date_to
        params.append(end_day)
    if event_type_filter:
        sql += " AND (e.event_type = ? OR e.event_type IS NULL)"
        params.append(event_type_filter)
    sql += " ORDER BY COALESCE(ts.relevance_score, 0.5) DESC, a.published_at DESC LIMIT ?"
    params.append(limit)
    cur = conn.execute(sql, params)
    rows = [dict(row) for row in cur.fetchall()]
    return [r for r in rows if r.get("ticker_relevance_score", 0) >= min_relevance]

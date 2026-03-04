"""
Vietnamese stock news pipeline: crawl → clean → sentiment → align → store.
"""

from src.news.db import init_db, get_engine, insert_article, get_articles_by_ticker_date

__all__ = [
    "init_db",
    "get_engine",
    "insert_article",
    "get_articles_by_ticker_date",
]

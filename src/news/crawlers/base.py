"""
Abstract base for news crawlers. Same interface for all sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ArticleRecord:
    source: str
    url: str
    title: str
    body_raw: Optional[str] = None
    published_at: Optional[str] = None


class BaseNewsCrawler(ABC):
    """Implement fetch_article_list() and optionally fetch_article_body()."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @abstractmethod
    def fetch_article_list(self, category: str, max_pages: int) -> List[ArticleRecord]:
        """Return list of articles (at least url, title, source); body and published_at optional."""
        pass

    def fetch_article_body(self, url: str) -> Optional[str]:
        """Override to fetch full body for an article URL. Default returns None."""
        return None

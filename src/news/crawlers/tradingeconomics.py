"""
TradingEconomics Vietnam indicators crawler.
User reference: https://tradingeconomics.com/vietnam/indicators
We treat each indicator row as a pseudo-article.
"""

import time
import logging
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from src.news.crawlers.base import BaseNewsCrawler, ArticleRecord

logger = logging.getLogger(__name__)


class TradingEconomicsCrawler(BaseNewsCrawler):
    def __init__(
        self,
        base_url: str = "https://tradingeconomics.com",
        request_delay_seconds: float = 2.0,
        timeout_seconds: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.delay = request_delay_seconds
        self.timeout = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    @property
    def source_name(self) -> str:
        return "tradingeconomics"

    def fetch_article_list(self, category: str, max_pages: int) -> List[ArticleRecord]:
        # Only one page is relevant for indicators.
        out: List[ArticleRecord] = []
        path = category or "vietnam/indicators"
        url = f"{self.base_url}/{path.strip('/')}"
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            # Indicators are usually listed in tables.
            rows = soup.select("table tr")
            for row in rows:
                rec = self._parse_row(row, url)
                if rec:
                    out.append(rec)
            logger.info("TradingEconomics %s: %d indicators", url, len(out))
        except Exception as e:
            logger.warning("TradingEconomics list fetch failed for %s: %s", url, e)
        time.sleep(self.delay)
        return out

    def _parse_row(self, row, base_url: str) -> Optional[ArticleRecord]:
        try:
            cells = row.find_all("td")
            if len(cells) < 2:
                return None
            name = cells[0].get_text(strip=True)
            latest = cells[1].get_text(strip=True)
            if not name:
                return None
            body = f"Latest value: {latest}"
            return ArticleRecord(
                source=self.source_name,
                url=base_url,
                title=name,
                body_raw=body,
                published_at=None,
            )
        except Exception as e:
            logger.debug("TradingEconomics parse row error: %s", e)
            return None


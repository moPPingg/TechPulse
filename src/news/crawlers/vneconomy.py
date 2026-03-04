"""
VnEconomy crawler.
User reference: https://vneconomy.vn/
We grab recent articles from the front page or a specified section.
"""

import time
import logging
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from src.news.crawlers.base import BaseNewsCrawler, ArticleRecord

logger = logging.getLogger(__name__)


class VNEconomyNewsCrawler(BaseNewsCrawler):
    def __init__(
        self,
        base_url: str = "https://vneconomy.vn",
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
                "Accept-Language": "vi-VN,vi;q=0.9",
            }
        )

    @property
    def source_name(self) -> str:
        return "vneconomy"

    def fetch_article_list(self, category: str, max_pages: int) -> List[ArticleRecord]:
        # VnEconomy has multiple sections; by default we scan the homepage once.
        out: List[ArticleRecord] = []
        path = (category or "").strip("/")
        if path:
            url = f"{self.base_url}/{path}"
        else:
            url = self.base_url.rstrip("/") + "/"
        url = url.rstrip("/")  # VnEconomy returns 404 for .htm/ URLs
        try:
            r = self.session.get(url, timeout=self.timeout, allow_redirects=False)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            # Articles are usually wrapped in <article> tags.
            items = soup.find_all("article")
            for item in items:
                rec = self._parse_item(item)
                if rec:
                    out.append(rec)
            logger.info("VnEconomy %s: %d articles", url, len(out))
        except Exception as e:
            logger.warning("VnEconomy list fetch failed for %s: %s", url, e)
        time.sleep(self.delay)
        return out

    def _parse_item(self, item) -> Optional[ArticleRecord]:
        try:
            # Tiêu đề nằm trong h3 > a, không phải thẻ a đầu tiên (có thể bọc ảnh, rỗng)
            h3 = item.find("h3")
            if not h3:
                return None
            a = h3.find("a")
            if not a or not a.get("href"):
                return None
            title = a.get_text(strip=True)
            if len(title) < 8:
                return None
            href = a["href"]
            if href.startswith("/"):
                url = self.base_url + href
            else:
                url = href
            return ArticleRecord(
                source=self.source_name,
                url=url,
                title=title,
                body_raw=None,
                published_at=None,
            )
        except Exception as e:
            logger.debug("VnEconomy parse item error: %s", e)
            return None


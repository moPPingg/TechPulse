"""
HSX (HOSE) announcements crawler.
Target: https://www.hsx.vn/vi/ (Vietnamese site root).
We heuristically grab links that look like disclosures / PDFs.
"""

import time
import logging
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from src.news.crawlers.base import BaseNewsCrawler, ArticleRecord

logger = logging.getLogger(__name__)


class HSXNewsCrawler(BaseNewsCrawler):
    def __init__(
        self,
        base_url: str = "https://www.hsx.vn",
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
        return "hsx"

    def fetch_article_list(self, category: str, max_pages: int) -> List[ArticleRecord]:
        # HSX is heavily JS-driven; here we just scrape the main Vietnamese page once.
        out: List[ArticleRecord] = []
        url = f"{self.base_url}/{(category or 'vi').strip('/')}"
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            links = soup.find_all("a")
            seen = set()
            for a in links:
                href = a.get("href")
                title = a.get_text(strip=True)
                if not href or not title or len(title) < 8:
                    continue
                low_href = href.lower()
                low_title = title.lower()
                # Heuristic: disclosures / news / pdfs
                if not any(
                    key in low_href or key in low_title
                    for key in ["pdf", "thong-tin", "công bố", "cong-bo", "news"]
                ):
                    continue
                if href in seen:
                    continue
                seen.add(href)
                if href.startswith("/"):
                    full_url = self.base_url + href
                else:
                    full_url = href
                out.append(
                    ArticleRecord(
                        source=self.source_name,
                        url=full_url,
                        title=title,
                        body_raw=None,
                        published_at=None,
                    )
                )
            logger.info("HSX %s: %d candidate links", url, len(out))
        except Exception as e:
            logger.warning("HSX list fetch failed for %s: %s", url, e)
        time.sleep(self.delay)
        return out


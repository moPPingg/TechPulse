"""
SSC (State Securities Commission of Vietnam) disclosure crawler.
User reference: https://ssc.gov.vn/webcenter/portal/ubck
We grab PDF / disclosure links as articles.
"""

import time
import logging
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from src.news.crawlers.base import BaseNewsCrawler, ArticleRecord

logger = logging.getLogger(__name__)


class SSCReportCrawler(BaseNewsCrawler):
    def __init__(
        self,
        base_url: str = "https://ssc.gov.vn",
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
        return "ssc"

    def fetch_article_list(self, category: str, max_pages: int) -> List[ArticleRecord]:
        # Portal root the user gave; deeper navigation may be added later.
        out: List[ArticleRecord] = []
        path = category or "webcenter/portal/ubck"
        url = f"{self.base_url}/{path.strip('/')}"
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
                if not (
                    ".pdf" in low_href
                    or "báo cáo" in low_title
                    or "bao cao" in low_title
                    or "cáo bạch" in low_title
                    or "cao bach" in low_title
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
            logger.info("SSC %s: %d disclosure links", url, len(out))
        except Exception as e:
            logger.warning("SSC list fetch failed for %s: %s", url, e)
        time.sleep(self.delay)
        return out


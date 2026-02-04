"""
VNExpress chứng khoán crawler.
Target list: https://vnexpress.net/kinh-doanh/chung-khoan
"""

import time
import logging
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from src.news.crawlers.base import BaseNewsCrawler, ArticleRecord

logger = logging.getLogger(__name__)


class VNExpressNewsCrawler(BaseNewsCrawler):
    def __init__(
        self,
        base_url: str = "https://vnexpress.net",
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
        return "vnexpress"

    def _build_list_url(self, category: str, page: int) -> str:
        """
        VNExpress uses '-p{page}' suffix for pagination:
        - page 1: /kinh-doanh/chung-khoan
        - page 2: /kinh-doanh/chung-khoan-p2
        """
        if page == 1:
            return f"{self.base_url}/{category.strip('/')}"
        return f"{self.base_url}/{category.strip('/')}-p{page}"

    def fetch_article_list(self, category: str, max_pages: int) -> List[ArticleRecord]:
        out: List[ArticleRecord] = []
        cat = category or "kinh-doanh/chung-khoan"
        for page in range(1, max_pages + 1):
            url = self._build_list_url(cat, page)
            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                soup = BeautifulSoup(r.content, "html.parser")
                # VNExpress listing: articles often in <article class="item-news">
                items = soup.select("article.item-news, article.item-list")
                for item in items:
                    rec = self._parse_item(item)
                    if rec:
                        out.append(rec)
                logger.info("VNExpress %s page %d: %d items", cat, page, len(items))
            except Exception as e:
                logger.warning("VNExpress %s page %d failed: %s", cat, page, e)
            time.sleep(self.delay)
        return out

    def _parse_item(self, item) -> Optional[ArticleRecord]:
        try:
            # title typically in <h3 class="title-news"><a ...>
            h3 = item.find("h3")
            if not h3:
                return None
            a = h3.find("a")
            if not a or not a.get("href"):
                return None
            title = a.get_text(strip=True)
            href = a["href"]
            if href.startswith("/"):
                url = self.base_url + href
            else:
                url = href

            # time info (if present)
            time_el = item.find("span", class_="time-public") or item.find("span", class_="time")
            pub_time = time_el.get_text(strip=True) if time_el else None

            return ArticleRecord(
                source=self.source_name,
                url=url,
                title=title,
                body_raw=None,
                published_at=pub_time,
            )
        except Exception as e:
            logger.debug("VNExpress parse item error: %s", e)
            return None

    def fetch_article_body(self, url: str) -> Optional[str]:
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            # Main content is usually in div.fck_detail
            content = soup.find("article", class_="fck_detail") or soup.find("div", class_="fck_detail")
            if not content:
                return None
            paras = content.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
            return text or None
        except Exception as e:
            logger.warning("VNExpress fetch body %s: %s", url[:80], e)
            return None


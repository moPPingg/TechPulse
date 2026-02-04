"""
CafeF news crawler. List pages + article body. Noisy but large; clean later.
"""

import re
import time
import logging
from typing import List, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from src.news.crawlers.base import BaseNewsCrawler, ArticleRecord

logger = logging.getLogger(__name__)


class CafeFNewsCrawler(BaseNewsCrawler):
    def __init__(
        self,
        base_url: str = "https://cafef.vn",
        request_delay_seconds: float = 2.0,
        timeout_seconds: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.delay = request_delay_seconds
        self.timeout = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        })

    @property
    def source_name(self) -> str:
        return "cafef"

    def fetch_article_list(self, category: str, max_pages: int) -> List[ArticleRecord]:
        out: List[ArticleRecord] = []
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/{category}/trang-{page}.chn"
            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                soup = BeautifulSoup(r.content, "html.parser")
                items = soup.find_all("div", class_="tlitem")
                for item in items:
                    rec = self._parse_item(item)
                    if rec:
                        out.append(rec)
                logger.info("CafeF %s page %d: %d items", category, page, len(items))
            except Exception as e:
                logger.warning("CafeF %s page %d failed: %s", category, page, e)
            time.sleep(self.delay)
        return out

    def _parse_item(self, item) -> Optional[ArticleRecord]:
        try:
            title_el = item.find("h3", class_="title")
            if not title_el:
                return None
            a = title_el.find("a")
            if not a or not a.get("href"):
                return None
            title = a.get_text(strip=True)
            href = a["href"]
            if href.startswith("/"):
                url = self.base_url + href
            else:
                url = href
            sapo = item.find("div", class_="sapo")
            summary = sapo.get_text(strip=True) if sapo else ""
            time_el = item.find("div", class_="time")
            pub_time = time_el.get_text(strip=True) if time_el else None
            return ArticleRecord(
                source=self.source_name,
                url=url,
                title=title,
                body_raw=summary or None,
                published_at=pub_time,
            )
        except Exception as e:
            logger.debug("Parse item error: %s", e)
            return None

    def fetch_article_body(self, url: str) -> Optional[str]:
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            content = soup.find("div", class_="detail-content")
            if not content:
                content = soup.find("div", {"id": "main-detail-content"})
            if not content:
                return None
            paras = content.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
            return text or None
        except Exception as e:
            logger.warning("CafeF fetch body %s: %s", url[:60], e)
            return None

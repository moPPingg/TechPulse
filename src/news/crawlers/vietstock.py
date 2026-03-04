"""
Vietstock (StockVN-style) news crawler. Same interface as CafeF.
"""

import time
import logging
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from src.news.crawlers.base import BaseNewsCrawler, ArticleRecord

logger = logging.getLogger(__name__)


class VietstockNewsCrawler(BaseNewsCrawler):
    def __init__(
        self,
        base_url: str = "https://vietstock.vn",
        request_delay_seconds: float = 2.0,
        timeout_seconds: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.delay = request_delay_seconds
        self.timeout = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "vi-VN,vi;q=0.9",
        })

    @property
    def source_name(self) -> str:
        return "vietstock"

    def fetch_article_list(self, category: str, max_pages: int) -> List[ArticleRecord]:
        out: List[ArticleRecord] = []
        for page in range(1, max_pages + 1):
            # Vietstock list URL pattern.
            # User-provided example: https://vietstock.vn/chung-khoan.htm
            # For page 1 we hit exactly that URL. For page >1 we try the common
            # querystring pattern ?page=N (if the site supports pagination).
            if page == 1:
                url = f"{self.base_url}/{category}"
            else:
                url = f"{self.base_url}/{category}?page={page}"
            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                soup = BeautifulSoup(r.content, "html.parser")
                links = soup.select("a[href*='/tin-nhanh/'], a[href*='/tin-tuc/'], a[href*='/chung-khoan/']")
                seen = set()
                for a in links:
                    href = a.get("href")
                    if not href or href in seen:
                        continue
                    title = a.get_text(strip=True)
                    if len(title) < 10:
                        continue
                    if href.startswith("/"):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    seen.add(href)
                    out.append(ArticleRecord(
                        source=self.source_name,
                        url=full_url,
                        title=title,
                        body_raw=None,
                        published_at=None,
                    ))
                logger.info("Vietstock %s page %d: %d links", category, page, len(links))
            except Exception as e:
                logger.warning("Vietstock %s page %d failed: %s", category, page, e)
            time.sleep(self.delay)
        return out

    def fetch_article_body(self, url: str) -> Optional[str]:
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            content = soup.find("div", class_="content") or soup.find("article") or soup.find("div", {"id": "content"})
            if not content:
                return None
            paras = content.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
            return text or None
        except Exception as e:
            logger.warning("Vietstock fetch body %s: %s", url[:60], e)
            return None

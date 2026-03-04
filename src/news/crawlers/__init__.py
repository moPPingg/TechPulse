from src.news.crawlers.base import ArticleRecord, BaseNewsCrawler
from src.news.crawlers.cafef import CafeFNewsCrawler
from src.news.crawlers.vietstock import VietstockNewsCrawler
from src.news.crawlers.vnexpress import VNExpressNewsCrawler
from src.news.crawlers.hsx import HSXNewsCrawler
from src.news.crawlers.vneconomy import VNEconomyNewsCrawler
from src.news.crawlers.tradingeconomics import TradingEconomicsCrawler
from src.news.crawlers.ssc import SSCReportCrawler

__all__ = [
    "ArticleRecord",
    "BaseNewsCrawler",
    "CafeFNewsCrawler",
    "VietstockNewsCrawler",
    "VNExpressNewsCrawler",
    "HSXNewsCrawler",
    "VNEconomyNewsCrawler",
    "TradingEconomicsCrawler",
    "SSCReportCrawler",
]

# 📰 CRAWL TIN TỨC VIỆT NAM
## CafeF & VnExpress - Nguồn tin chứng khoán Việt

---

## 📚 MỤC LỤC

1. [Tại sao chọn CafeF & VnExpress?](#1-tại-sao-chọn-cafef--vnexpress)
2. [Kiến trúc Crawler](#2-kiến-trúc-crawler)
3. [Crawl CafeF News](#3-crawl-cafef-news)
4. [Crawl VnExpress News](#4-crawl-vnexpress-news)
5. [Data Schema](#5-data-schema)
6. [Best Practices](#6-best-practices)
7. [Bài tập thực hành](#7-bài-tập-thực-hành)

---

## 1. TẠI SAO CHỌN CAFEF & VNEXPRESS?

### 🎯 So sánh các nguồn tin Việt Nam

| Nguồn | Ưu điểm | Nhược điểm | Đánh giá |
|-------|---------|------------|----------|
| **CafeF** | Chuyên chứng khoán, có API, data sạch | Ít tin tổng hợp | ⭐⭐⭐⭐⭐ |
| **VnExpress** | Nhiều tin, uy tín, dễ crawl | Nhiều noise, cần filter | ⭐⭐⭐⭐ |
| **Vneconomy** | Chuyên kinh tế | Ít tin về cổ phiếu cụ thể | ⭐⭐⭐ |
| **Đầu tư** | Chuyên đầu tư | Website phức tạp | ⭐⭐⭐ |
| **Bloomberg VN** | Chất lượng cao | Ít tin, paywall | ⭐⭐ |

### ✅ Lý do chọn CafeF + VnExpress

**CafeF:**
- ✅ Chuyên về chứng khoán VN
- ✅ Có API/RSS feed
- ✅ Tin tức real-time
- ✅ Phân loại rõ ràng (công ty, ngành)
- ✅ Đã có sẵn price crawler

**VnExpress:**
- ✅ Nguồn tin uy tín nhất VN
- ✅ Coverage rộng (kinh tế, chính trị, xã hội)
- ✅ Dễ crawl (HTML structure ổn định)
- ✅ Nhiều tin tác động gián tiếp đến thị trường
- ✅ SEO tốt → tin được đọc nhiều

### 🎯 Chiến lược kết hợp

```
CafeF (60%):
- Tin chứng khoán trực tiếp
- Báo cáo tài chính
- Phân tích kỹ thuật
- Khuyến nghị mua/bán

VnExpress (40%):
- Tin kinh tế vĩ mô
- Chính sách mới
- Scandal, sự kiện lớn
- Sentiment thị trường
```

---

## 2. KIẾN TRÚC CRAWLER

### 📊 Tổng quan hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                    NEWS CRAWLER SYSTEM                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐         ┌──────────────┐             │
│  │ CafeF Crawler│         │VnExpress     │             │
│  │              │         │Crawler       │             │
│  └──────┬───────┘         └──────┬───────┘             │
│         │                        │                      │
│         ▼                        ▼                      │
│  ┌─────────────────────────────────────┐               │
│  │     Raw News Data Storage           │               │
│  │  (JSON files / Database)            │               │
│  └─────────────┬───────────────────────┘               │
│                │                                        │
│                ▼                                        │
│  ┌─────────────────────────────────────┐               │
│  │     News Cleaning & Processing      │               │
│  │  - Remove HTML tags                 │               │
│  │  - Extract metadata                 │               │
│  │  - Deduplicate                      │               │
│  └─────────────┬───────────────────────┘               │
│                │                                        │
│                ▼                                        │
│  ┌─────────────────────────────────────┐               │
│  │     Link with Stock Symbols         │               │
│  │  - Detect ticker mentions           │               │
│  │  - Classify relevance               │               │
│  └─────────────┬───────────────────────┘               │
│                │                                        │
│                ▼                                        │
│  ┌─────────────────────────────────────┐               │
│  │     Clean News Database             │               │
│  │  data/news/cafef/                   │               │
│  │  data/news/vnexpress/               │               │
│  └─────────────────────────────────────┘               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 🔧 Tech Stack

```python
# Core libraries
requests          # HTTP requests
BeautifulSoup4    # HTML parsing
selenium          # Dynamic content (if needed)
scrapy            # Advanced crawling (optional)

# Vietnamese NLP
underthesea       # Vietnamese tokenizer
pyvi              # Vietnamese NLP toolkit
vncorenlp         # Vietnamese CoreNLP

# Storage
pandas            # Data manipulation
sqlite3           # Local database
pymongo           # MongoDB (optional)

# Utils
schedule          # Cron jobs
logging           # Logging
tqdm              # Progress bars
```

---

## 3. CRAWL CAFEF NEWS

### 🎯 CafeF News Structure

**URL patterns:**
```
Tin tổng hợp:
https://cafef.vn/thi-truong-chung-khoan.chn

Tin theo mã:
https://cafef.vn/FPT-ctcp-tap-doan-fpt.chn

RSS Feed:
https://cafef.vn/rss/thi-truong-chung-khoan.rss
```

### 📊 HTML Structure

```html
<div class="tlitem">
    <h3 class="title">
        <a href="/link-to-article">Tiêu đề bài viết</a>
    </h3>
    <div class="sapo">Tóm tắt bài viết...</div>
    <div class="time">10:30 28/01/2026</div>
    <div class="category">Chứng khoán</div>
</div>
```

### 🔧 Implementation

**Bước 1: Basic Crawler**
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

class CafeFNewsCrawler:
    """
    Crawler cho tin tức CafeF
    """
    
    def __init__(self):
        self.base_url = "https://cafef.vn"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl_news_list(self, category='thi-truong-chung-khoan', pages=5):
        """
        Crawl danh sách tin tức
        
        Args:
            category: Danh mục tin (default: thị trường chứng khoán)
            pages: Số trang cần crawl
        
        Returns:
            List of news items
        """
        news_list = []
        
        for page in range(1, pages + 1):
            url = f"{self.base_url}/{category}/trang-{page}.chn"
            
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Tìm tất cả tin tức
                articles = soup.find_all('div', class_='tlitem')
                
                for article in articles:
                    news_item = self._parse_article_item(article)
                    if news_item:
                        news_list.append(news_item)
                
                print(f"Crawled page {page}: {len(articles)} articles")
                
                # Delay để tránh bị block
                time.sleep(2)
                
            except Exception as e:
                print(f"Error crawling page {page}: {e}")
                continue
        
        return news_list
    
    def _parse_article_item(self, article):
        """
        Parse thông tin từ 1 article item
        """
        try:
            # Tiêu đề và link
            title_tag = article.find('h3', class_='title')
            if not title_tag:
                return None
            
            link_tag = title_tag.find('a')
            title = link_tag.text.strip()
            link = self.base_url + link_tag['href']
            
            # Tóm tắt
            sapo_tag = article.find('div', class_='sapo')
            summary = sapo_tag.text.strip() if sapo_tag else ""
            
            # Thời gian
            time_tag = article.find('div', class_='time')
            pub_time = time_tag.text.strip() if time_tag else ""
            
            # Category
            cat_tag = article.find('div', class_='category')
            category = cat_tag.text.strip() if cat_tag else ""
            
            return {
                'title': title,
                'summary': summary,
                'link': link,
                'published_time': pub_time,
                'category': category,
                'source': 'CafeF',
                'crawled_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None
    
    def crawl_article_content(self, url):
        """
        Crawl nội dung chi tiết bài viết
        
        Args:
            url: URL bài viết
        
        Returns:
            Article content
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm nội dung chính
            content_div = soup.find('div', class_='detail-content')
            
            if not content_div:
                return None
            
            # Lấy tất cả paragraphs
            paragraphs = content_div.find_all('p')
            content = '\n'.join([p.text.strip() for p in paragraphs])
            
            # Lấy tags/keywords
            tags = []
            tag_div = soup.find('div', class_='tags')
            if tag_div:
                tag_links = tag_div.find_all('a')
                tags = [tag.text.strip() for tag in tag_links]
            
            return {
                'content': content,
                'tags': tags
            }
            
        except Exception as e:
            print(f"Error crawling article content: {e}")
            return None
```

**Bước 2: Crawl với Full Content**
```python
def crawl_full_news(crawler, category='thi-truong-chung-khoan', pages=5):
    """
    Crawl tin tức với full content
    """
    # Bước 1: Crawl danh sách
    print("Step 1: Crawling news list...")
    news_list = crawler.crawl_news_list(category=category, pages=pages)
    print(f"Found {len(news_list)} articles")
    
    # Bước 2: Crawl content cho từng bài
    print("\nStep 2: Crawling full content...")
    for i, news in enumerate(news_list, 1):
        print(f"[{i}/{len(news_list)}] Crawling: {news['title'][:50]}...")
        
        content_data = crawler.crawl_article_content(news['link'])
        
        if content_data:
            news['content'] = content_data['content']
            news['tags'] = content_data['tags']
        else:
            news['content'] = ""
            news['tags'] = []
        
        # Delay
        time.sleep(1)
    
    # Bước 3: Save to DataFrame
    df = pd.DataFrame(news_list)
    
    return df

# Sử dụng
crawler = CafeFNewsCrawler()
df_cafef = crawl_full_news(crawler, pages=10)

# Save
df_cafef.to_csv('data/news/cafef/news_raw.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved {len(df_cafef)} articles to data/news/cafef/news_raw.csv")
```

---

## 4. CRAWL VNEXPRESS NEWS

### 🎯 VnExpress Structure

**URL patterns:**
```
Kinh doanh:
https://vnexpress.net/kinh-doanh

Chứng khoán:
https://vnexpress.net/kinh-doanh/chung-khoan

RSS:
https://vnexpress.net/rss/kinh-doanh.rss
```

### 🔧 Implementation

```python
class VnExpressNewsCrawler:
    """
    Crawler cho tin tức VnExpress
    """
    
    def __init__(self):
        self.base_url = "https://vnexpress.net"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl_news_list(self, category='kinh-doanh/chung-khoan', pages=5):
        """
        Crawl danh sách tin VnExpress
        """
        news_list = []
        
        for page in range(1, pages + 1):
            url = f"{self.base_url}/{category}-p{page}"
            
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # VnExpress dùng class 'item-news'
                articles = soup.find_all('article', class_='item-news')
                
                for article in articles:
                    news_item = self._parse_article_item(article)
                    if news_item:
                        news_list.append(news_item)
                
                print(f"Crawled page {page}: {len(articles)} articles")
                time.sleep(2)
                
            except Exception as e:
                print(f"Error crawling page {page}: {e}")
                continue
        
        return news_list
    
    def _parse_article_item(self, article):
        """
        Parse article item VnExpress
        """
        try:
            # Title và link
            title_tag = article.find('h3', class_='title-news')
            if not title_tag:
                return None
            
            link_tag = title_tag.find('a')
            title = link_tag['title']
            link = link_tag['href']
            
            # Nếu link relative, thêm base_url
            if not link.startswith('http'):
                link = self.base_url + link
            
            # Summary
            desc_tag = article.find('p', class_='description')
            summary = desc_tag.text.strip() if desc_tag else ""
            
            # Time
            time_tag = article.find('span', class_='time')
            pub_time = time_tag.text.strip() if time_tag else ""
            
            return {
                'title': title,
                'summary': summary,
                'link': link,
                'published_time': pub_time,
                'category': 'Kinh doanh',
                'source': 'VnExpress',
                'crawled_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None
    
    def crawl_article_content(self, url):
        """
        Crawl nội dung bài viết VnExpress
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # VnExpress dùng class 'fck_detail'
            content_div = soup.find('article', class_='fck_detail')
            
            if not content_div:
                return None
            
            # Lấy paragraphs
            paragraphs = content_div.find_all('p', class_='Normal')
            content = '\n'.join([p.text.strip() for p in paragraphs])
            
            # Tags
            tags = []
            tag_div = soup.find('div', class_='tags')
            if tag_div:
                tag_links = tag_div.find_all('a')
                tags = [tag.text.strip() for tag in tag_links]
            
            return {
                'content': content,
                'tags': tags
            }
            
        except Exception as e:
            print(f"Error crawling content: {e}")
            return None
```

---

## 5. DATA SCHEMA

### 📊 News Data Schema

```python
news_schema = {
    'id': 'unique_id',                    # UUID
    'title': 'Tiêu đề bài viết',         # str
    'summary': 'Tóm tắt',                 # str
    'content': 'Nội dung đầy đủ',        # str (long text)
    'link': 'URL bài viết',               # str
    'published_time': '28/01/2026 10:30', # str (cần parse)
    'category': 'Chứng khoán',           # str
    'tags': ['FPT', 'Công nghệ'],       # list
    'source': 'CafeF',                    # str (CafeF/VnExpress)
    'crawled_at': '2026-01-28T10:30:00', # ISO format
    
    # Thêm sau khi process
    'tickers_mentioned': ['FPT', 'VCB'], # list (detected)
    'sentiment_score': 0.75,              # float [-1, 1]
    'event_type': 'earnings',             # str (classified)
    'is_relevant': True,                  # bool
}
```

### 💾 Storage Structure

```
data/news/
├── cafef/
│   ├── raw/
│   │   ├── 2026-01-28.csv
│   │   ├── 2026-01-29.csv
│   │   └── ...
│   └── processed/
│       ├── 2026-01-28_processed.csv
│       └── ...
│
├── vnexpress/
│   ├── raw/
│   │   └── ...
│   └── processed/
│       └── ...
│
└── combined/
    ├── news_all.csv
    └── news_with_tickers.csv
```

---

## 6. BEST PRACTICES

### ⚠️ Ethical Crawling

**1. Respect robots.txt:**
```python
# Check robots.txt trước khi crawl
import urllib.robotparser

rp = urllib.robotparser.RobotFileParser()
rp.set_url("https://cafef.vn/robots.txt")
rp.read()

if rp.can_fetch("*", "https://cafef.vn/thi-truong-chung-khoan.chn"):
    # OK to crawl
    pass
```

**2. Rate limiting:**
```python
import time
from datetime import datetime

class RateLimiter:
    def __init__(self, requests_per_minute=30):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = None
    
    def wait(self):
        if self.last_request:
            elapsed = (datetime.now() - self.last_request).total_seconds()
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        
        self.last_request = datetime.now()

# Sử dụng
limiter = RateLimiter(requests_per_minute=30)

for url in urls:
    limiter.wait()
    response = requests.get(url)
```

**3. User-Agent rotation:**
```python
import random

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
]

def get_random_headers():
    return {'User-Agent': random.choice(USER_AGENTS)}
```

### 🔧 Error Handling

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def crawl_with_retry(url):
    """
    Crawl với retry logic
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response
    
    except requests.Timeout:
        logger.error(f"Timeout: {url}")
        raise
    
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"404 Not Found: {url}")
            return None
        else:
            logger.error(f"HTTP Error {e.response.status_code}: {url}")
            raise
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 📅 Scheduling

```python
import schedule

def daily_crawl_job():
    """
    Job chạy hàng ngày
    """
    print(f"Starting daily crawl at {datetime.now()}")
    
    # Crawl CafeF
    cafef_crawler = CafeFNewsCrawler()
    df_cafef = crawl_full_news(cafef_crawler, pages=5)
    
    # Crawl VnExpress
    vnexpress_crawler = VnExpressNewsCrawler()
    df_vnexpress = crawl_full_news(vnexpress_crawler, pages=5)
    
    # Save
    today = datetime.now().strftime('%Y-%m-%d')
    df_cafef.to_csv(f'data/news/cafef/raw/{today}.csv', index=False)
    df_vnexpress.to_csv(f'data/news/vnexpress/raw/{today}.csv', index=False)
    
    print(f"Completed: {len(df_cafef)} CafeF + {len(df_vnexpress)} VnExpress")

# Schedule: Chạy mỗi ngày lúc 8:00 AM
schedule.every().day.at("08:00").do(daily_crawl_job)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 7. BÀI TẬP THỰC HÀNH

### 🎯 Bài tập 1: Crawl CafeF News

**Đề bài:**
Implement crawler cho CafeF, crawl 100 bài viết gần nhất

**Yêu cầu:**
- Crawl cả title, summary, content
- Save to CSV với encoding UTF-8
- Handle errors gracefully
- Implement rate limiting

**Kiểm tra:**
- [ ] Crawl được 100 bài
- [ ] Content đầy đủ, không bị lỗi encoding
- [ ] Có error handling
- [ ] Có rate limiting

---

### 🎯 Bài tập 2: Crawl VnExpress News

**Đề bài:**
Implement crawler cho VnExpress, crawl tin kinh doanh 7 ngày gần nhất

**Yêu cầu:**
- Crawl từ category "Kinh doanh"
- Filter chỉ lấy tin liên quan chứng khoán
- Detect và extract ticker mentions
- Save to database (SQLite)

**Kiểm tra:**
- [ ] Crawl được tin 7 ngày
- [ ] Filter đúng tin chứng khoán
- [ ] Detect được tickers
- [ ] Save vào SQLite

---

### 🎯 Bài tập 3: Combined Crawler

**Đề bài:**
Kết hợp 2 crawlers, tạo unified news database

**Yêu cầu:**
- Crawl đồng thời CafeF + VnExpress
- Deduplicate (loại tin trùng)
- Link với VN30 tickers
- Create daily reports

**Kiểm tra:**
- [ ] Crawl được cả 2 nguồn
- [ ] Deduplicate thành công
- [ ] Link với tickers
- [ ] Generate reports

---

## ✅ KIỂM TRA HIỂU BÀI

Trước khi sang bài tiếp theo, hãy đảm bảo bạn:

- [ ] Hiểu tại sao chọn CafeF & VnExpress
- [ ] Implement được CafeF crawler
- [ ] Implement được VnExpress crawler
- [ ] Hiểu HTML structure của 2 sites
- [ ] Handle được errors và rate limiting
- [ ] Save được data với encoding đúng
- [ ] Làm được 3 bài tập thực hành

**Nếu chưa pass hết checklist, đọc lại phần tương ứng!**

---

## 📚 TÀI LIỆU THAM KHẢO

**Libraries:**
- BeautifulSoup4: HTML parsing
- Scrapy: Advanced crawling framework
- Selenium: Dynamic content

**Vietnamese NLP:**
- underthesea: Vietnamese NLP toolkit
- pyvi: Vietnamese word segmentation
- vncorenlp: Vietnamese CoreNLP

**Best Practices:**
- "Web Scraping with Python" - Ryan Mitchell
- Scrapy documentation
- robots.txt guidelines

---

## 🚀 BƯỚC TIẾP THEO

Sau khi hoàn thành bài này, sang:
- `02_VIETNAMESE_TEXT_PROCESSING.md` - Xử lý tiếng Việt & sentiment

**Chúc bạn học tốt! 🎓**

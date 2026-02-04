# Crawl và Xử lý Tin tức Việt Nam
## CafeF & VnExpress - Nguồn tin chứng khoán Việt

---

## Mục lục

1. [Tại sao chọn CafeF & VnExpress?](#1-tại-sao-chọn-cafef--vnexpress)
2. [Kiến trúc Crawler](#2-kiến-trúc-crawler)
3. [Crawl CafeF News](#3-crawl-cafef-news)
4. [Crawl VnExpress News](#4-crawl-vnexpress-news)
5. [Data Schema](#5-data-schema)
6. [NLP Preprocessing Pipeline cho Financial News](#6-nlp-preprocessing-pipeline-cho-financial-news)
7. [Vietnamese Tokenization](#7-vietnamese-tokenization)
8. [Text Representations: TF-IDF vs Embeddings vs Transformers](#8-text-representations-tf-idf-vs-embeddings-vs-transformers)
9. [Financial Sentiment Modeling](#9-financial-sentiment-modeling)
10. [Aligning News với Price Candles](#10-aligning-news-với-price-candles)
11. [Noise Filtering và Irrelevant News Removal](#11-noise-filtering-và-irrelevant-news-removal)
12. [Best Practices](#12-best-practices)
13. [Bài tập thực hành](#13-bài-tập-thực-hành)

---

## 1. TẠI SAO CHỌN CAFEF & VNEXPRESS?

### So sánh các nguồn tin Việt Nam

| Nguồn | Ưu điểm | Nhược điểm | Đánh giá |
|-------|---------|------------|----------|
| **CafeF** | Chuyên chứng khoán, có API, data sạch | Ít tin tổng hợp | Rất tốt |
| **VnExpress** | Nhiều tin, uy tín, dễ crawl | Nhiều noise, cần filter | Tốt |
| **Vneconomy** | Chuyên kinh tế | Ít tin về cổ phiếu cụ thể | Trung bình |
| **Đầu tư** | Chuyên đầu tư | Website phức tạp | Trung bình |
| **Bloomberg VN** | Chất lượng cao | Ít tin, paywall | Hạn chế |

### Lý do chọn CafeF + VnExpress

**CafeF:**
- Chuyên về chứng khoán VN
- Có API/RSS feed
- Tin tức real-time
- Phân loại rõ ràng (công ty, ngành)
- Đã có sẵn price crawler

**VnExpress:**
- Nguồn tin uy tín nhất VN
- Coverage rộng (kinh tế, chính trị, xã hội)
- Dễ crawl (HTML structure ổn định)
- Nhiều tin tác động gián tiếp đến thị trường
- SEO tốt → tin được đọc nhiều

### Chiến lược kết hợp

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

### Tổng quan hệ thống

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

### Tech Stack

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

### CafeF News Structure

**URL patterns:**
```
Tin tổng hợp:
https://cafef.vn/thi-truong-chung-khoan.chn

Tin theo mã:
https://cafef.vn/FPT-ctcp-tap-doan-fpt.chn

RSS Feed:
https://cafef.vn/rss/thi-truong-chung-khoan.rss
```

### Implementation

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

class CafeFNewsCrawler:
    """Crawler cho tin tức CafeF"""
    
    def __init__(self):
        self.base_url = "https://cafef.vn"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl_news_list(self, category='thi-truong-chung-khoan', pages=5):
        """Crawl danh sách tin tức"""
        news_list = []
        
        for page in range(1, pages + 1):
            url = f"{self.base_url}/{category}/trang-{page}.chn"
            
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all('div', class_='tlitem')
                
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
        """Parse thông tin từ 1 article item"""
        try:
            title_tag = article.find('h3', class_='title')
            if not title_tag:
                return None
            
            link_tag = title_tag.find('a')
            title = link_tag.text.strip()
            link = self.base_url + link_tag['href']
            
            sapo_tag = article.find('div', class_='sapo')
            summary = sapo_tag.text.strip() if sapo_tag else ""
            
            time_tag = article.find('div', class_='time')
            pub_time = time_tag.text.strip() if time_tag else ""
            
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
        """Crawl nội dung chi tiết bài viết"""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content_div = soup.find('div', class_='detail-content')
            
            if not content_div:
                return None
            
            paragraphs = content_div.find_all('p')
            content = '\n'.join([p.text.strip() for p in paragraphs])
            
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

---

## 4. CRAWL VNEXPRESS NEWS

### VnExpress Structure

**URL patterns:**
```
Kinh doanh:
https://vnexpress.net/kinh-doanh

Chứng khoán:
https://vnexpress.net/kinh-doanh/chung-khoan

RSS:
https://vnexpress.net/rss/kinh-doanh.rss
```

### Implementation

```python
class VnExpressNewsCrawler:
    """Crawler cho tin tức VnExpress"""
    
    def __init__(self):
        self.base_url = "https://vnexpress.net"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl_news_list(self, category='kinh-doanh/chung-khoan', pages=5):
        """Crawl danh sách tin VnExpress"""
        news_list = []
        
        for page in range(1, pages + 1):
            url = f"{self.base_url}/{category}-p{page}"
            
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
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
        """Parse article item VnExpress"""
        try:
            title_tag = article.find('h3', class_='title-news')
            if not title_tag:
                return None
            
            link_tag = title_tag.find('a')
            title = link_tag['title']
            link = link_tag['href']
            
            if not link.startswith('http'):
                link = self.base_url + link
            
            desc_tag = article.find('p', class_='description')
            summary = desc_tag.text.strip() if desc_tag else ""
            
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
```

---

## 5. DATA SCHEMA

### News Data Schema

```python
news_schema = {
    'id': 'unique_id',                    # UUID
    'title': 'Tiêu đề bài viết',         # str
    'summary': 'Tóm tắt',                 # str
    'content': 'Nội dung đầy đủ',        # str (long text)
    'link': 'URL bài viết',               # str
    'published_time': '28/01/2026 10:30', # str (cần parse)
    'category': 'Chứng khoán',           # str
    'tags': ['FPT', 'Công nghệ'],        # list
    'source': 'CafeF',                    # str (CafeF/VnExpress)
    'crawled_at': '2026-01-28T10:30:00', # ISO format
    
    # Thêm sau khi process
    'tickers_mentioned': ['FPT', 'VCB'], # list (detected)
    'sentiment_score': 0.75,              # float [-1, 1]
    'event_type': 'earnings',             # str (classified)
    'is_relevant': True,                  # bool
}
```

### Storage Structure

```
data/news/
├── cafef/
│   ├── raw/
│   │   ├── 2026-01-28.csv
│   │   └── ...
│   └── processed/
│       └── ...
│
├── vnexpress/
│   ├── raw/
│   └── processed/
│
└── combined/
    ├── news_all.csv
    └── news_with_tickers.csv
```

---

## 6. NLP PREPROCESSING PIPELINE CHO FINANCIAL NEWS

### 6.1. Tại sao cần Pipeline riêng cho Financial News?

**Đặc thù của tin tài chính:**
```
1. Nhiều số liệu: "FPT tăng 15%, đạt 100,500 đồng"
2. Ticker symbols: "VCB, FPT, VNM..."
3. Thuật ngữ chuyên ngành: "PE, EPS, margin call, T+3"
4. Tên công ty có dấu: "Tập đoàn Vingroup", "Ngân hàng Vietcombank"
5. Thời gian quan trọng: "Phiên chiều", "Tuần tới"
```

### 6.2. Full Preprocessing Pipeline

```python
import re
import unicodedata
from typing import List, Dict, Optional
from datetime import datetime

class FinancialNewsPreprocessor:
    """
    Complete NLP preprocessing pipeline cho financial news tiếng Việt
    """
    
    def __init__(self):
        # Vietnamese stopwords (mở rộng cho financial context)
        self.stopwords = self._load_stopwords()
        
        # Financial abbreviations
        self.financial_abbrevs = {
            'BCTC': 'báo cáo tài chính',
            'ĐHCĐ': 'đại hội cổ đông',
            'HĐQT': 'hội đồng quản trị',
            'KQKD': 'kết quả kinh doanh',
            'LNST': 'lợi nhuận sau thuế',
            'DTT': 'doanh thu thuần',
            'VĐL': 'vốn điều lệ',
            'CP': 'cổ phiếu',
            'CK': 'chứng khoán',
            'NĐT': 'nhà đầu tư',
            'TTCK': 'thị trường chứng khoán',
        }
        
        # VN30 tickers
        self.vn30_tickers = [
            'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR',
            'HDB', 'HPG', 'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX',
            'POW', 'SAB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM',
            'VIC', 'VJC', 'VNM', 'VPB', 'VRE', 'SSB'
        ]
        
        # Company name mapping
        self.company_names = {
            'fpt': 'FPT',
            'vietcombank': 'VCB',
            'vingroup': 'VIC',
            'vinhomes': 'VHM',
            'hòa phát': 'HPG',
            'masan': 'MSN',
            'vinamilk': 'VNM',
            'vietinbank': 'CTG',
            'bidv': 'BID',
            'techcombank': 'TCB',
            'vpbank': 'VPB',
            'mb': 'MBB',
            'sacombank': 'STB',
        }
    
    def _load_stopwords(self) -> set:
        """Load Vietnamese stopwords"""
        # Common Vietnamese stopwords
        stopwords = {
            'và', 'của', 'cho', 'là', 'với', 'được', 'trong', 'có',
            'đã', 'sẽ', 'đang', 'để', 'này', 'đó', 'như', 'từ',
            'một', 'các', 'những', 'nhiều', 'tại', 'về', 'theo',
            'qua', 'khi', 'còn', 'hay', 'hoặc', 'nhưng', 'mà',
            'thì', 'rằng', 'nên', 'vì', 'cũng', 'bởi', 'đến',
            'trên', 'dưới', 'ra', 'vào', 'lại', 'đi', 'sau',
            'trước', 'giữa', 'ngoài', 'hơn', 'rất', 'quá',
        }
        return stopwords
    
    def preprocess(self, text: str, config: Dict = None) -> Dict:
        """
        Main preprocessing function
        
        Args:
            text: Raw news text
            config: Configuration dict
        
        Returns:
            Dict with processed text and metadata
        """
        config = config or {}
        
        result = {
            'original': text,
            'cleaned': None,
            'tokens': None,
            'tickers': [],
            'numbers': [],
            'dates': [],
            'financial_terms': []
        }
        
        # Step 1: Basic cleaning
        cleaned = self._basic_clean(text)
        
        # Step 2: Extract tickers BEFORE lowercasing
        result['tickers'] = self._extract_tickers(cleaned)
        
        # Step 3: Extract numbers and percentages
        result['numbers'] = self._extract_numbers(cleaned)
        
        # Step 4: Extract dates
        result['dates'] = self._extract_dates(cleaned)
        
        # Step 5: Normalize text
        normalized = self._normalize_text(cleaned)
        
        # Step 6: Expand abbreviations
        expanded = self._expand_abbreviations(normalized)
        
        # Step 7: Extract financial terms
        result['financial_terms'] = self._extract_financial_terms(expanded)
        
        # Step 8: Remove stopwords (optional)
        if config.get('remove_stopwords', False):
            expanded = self._remove_stopwords(expanded)
        
        result['cleaned'] = expanded
        
        return result
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep Vietnamese, numbers, basic punctuation)
        text = re.sub(r'[^\w\s\.,!?%\-\(\)àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', '', text)
        
        return text.strip()
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        tickers = []
        
        # Direct ticker mentions (3-letter uppercase)
        pattern = r'\b([A-Z]{3})\b'
        matches = re.findall(pattern, text)
        for match in matches:
            if match in self.vn30_tickers:
                tickers.append(match)
        
        # Company name to ticker
        text_lower = text.lower()
        for name, ticker in self.company_names.items():
            if name in text_lower:
                tickers.append(ticker)
        
        return list(set(tickers))
    
    def _extract_numbers(self, text: str) -> List[Dict]:
        """Extract numbers and their context"""
        numbers = []
        
        # Percentages
        pattern_pct = r'([\-+]?\d+[,.]?\d*)\s*%'
        for match in re.finditer(pattern_pct, text):
            numbers.append({
                'value': match.group(1),
                'type': 'percentage',
                'position': match.start()
            })
        
        # Currency (VND)
        pattern_vnd = r'(\d+[,.]?\d*)\s*(đồng|VND|vnđ|tỷ|triệu)'
        for match in re.finditer(pattern_vnd, text, re.IGNORECASE):
            numbers.append({
                'value': match.group(1),
                'unit': match.group(2),
                'type': 'currency',
                'position': match.start()
            })
        
        # Stock prices (format: 100,500)
        pattern_price = r'\b(\d{2,3},\d{3})\b'
        for match in re.finditer(pattern_price, text):
            numbers.append({
                'value': match.group(1),
                'type': 'stock_price',
                'position': match.start()
            })
        
        return numbers
    
    def _extract_dates(self, text: str) -> List[Dict]:
        """Extract dates from text"""
        dates = []
        
        # Format: DD/MM/YYYY
        pattern1 = r'(\d{1,2})/(\d{1,2})/(\d{4})'
        for match in re.finditer(pattern1, text):
            dates.append({
                'raw': match.group(0),
                'day': match.group(1),
                'month': match.group(2),
                'year': match.group(3)
            })
        
        # Format: "ngày DD tháng MM"
        pattern2 = r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})'
        for match in re.finditer(pattern2, text, re.IGNORECASE):
            dates.append({
                'raw': match.group(0),
                'day': match.group(1),
                'month': match.group(2)
            })
        
        return dates
    
    def _normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text"""
        # Lowercase
        text = text.lower()
        
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand financial abbreviations"""
        for abbrev, full in self.financial_abbrevs.items():
            text = re.sub(
                r'\b' + abbrev.lower() + r'\b',
                full,
                text,
                flags=re.IGNORECASE
            )
        return text
    
    def _extract_financial_terms(self, text: str) -> List[str]:
        """Extract financial terminology"""
        financial_terms = [
            'lợi nhuận', 'doanh thu', 'vốn hóa', 'cổ tức',
            'biên lợi nhuận', 'p/e', 'eps', 'roe', 'roa',
            'tăng trưởng', 'sụt giảm', 'phá sản', 'nợ xấu',
            'margin', 'volume', 'thanh khoản', 'biến động',
            'mua ròng', 'bán ròng', 'room ngoại', 'định giá',
            'breakout', 'support', 'resistance', 'trend',
        ]
        
        found_terms = []
        for term in financial_terms:
            if term in text:
                found_terms.append(term)
        
        return found_terms
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove Vietnamese stopwords"""
        words = text.split()
        filtered = [w for w in words if w.lower() not in self.stopwords]
        return ' '.join(filtered)
```

### 6.3. Batch Processing Pipeline

```python
def process_news_batch(df: pd.DataFrame, 
                       text_col: str = 'content',
                       config: Dict = None) -> pd.DataFrame:
    """
    Process batch of news articles
    
    Args:
        df: DataFrame with news
        text_col: Column containing text
        config: Preprocessing config
    
    Returns:
        DataFrame with processed columns
    """
    preprocessor = FinancialNewsPreprocessor()
    
    results = []
    for idx, row in df.iterrows():
        text = row[text_col] if pd.notna(row[text_col]) else ''
        
        processed = preprocessor.preprocess(text, config)
        
        results.append({
            'idx': idx,
            'text_cleaned': processed['cleaned'],
            'tickers': processed['tickers'],
            'numbers': processed['numbers'],
            'dates': processed['dates'],
            'financial_terms': processed['financial_terms']
        })
    
    results_df = pd.DataFrame(results).set_index('idx')
    
    # Merge with original
    df_processed = df.join(results_df)
    
    return df_processed

# Usage
df_news = pd.read_csv('data/news/cafef/raw/2026-01-28.csv')
df_processed = process_news_batch(df_news, text_col='content')
```

---

## 7. VIETNAMESE TOKENIZATION

### 7.1. Đặc thù của tiếng Việt

**Vấn đề:**
```
English: "I love Vietnam" → ["I", "love", "Vietnam"]
Vietnamese: "Tôi yêu Việt Nam" → ???

Sai: ["Tôi", "yêu", "Việt", "Nam"]
Đúng: ["Tôi", "yêu", "Việt_Nam"]

"Việt Nam" là MỘT từ, không phải HAI từ!
```

### 7.2. Các công cụ Tokenization

**Tool 1: Underthesea (Recommended)**

```python
# pip install underthesea
from underthesea import word_tokenize, pos_tag, ner

class VietnameseTokenizer:
    """Vietnamese tokenization using underthesea"""
    
    @staticmethod
    def tokenize(text: str, format: str = 'text') -> str:
        """
        Tokenize Vietnamese text
        
        Args:
            text: Input text
            format: 'text' or 'list'
        
        Returns:
            Tokenized text with '_' joining compound words
        """
        result = word_tokenize(text, format=format)
        return result
    
    @staticmethod
    def pos_tagging(text: str) -> List[tuple]:
        """Part-of-speech tagging"""
        return pos_tag(text)
    
    @staticmethod
    def named_entity_recognition(text: str) -> List[tuple]:
        """Named entity recognition"""
        return ner(text)

# Usage examples
tokenizer = VietnameseTokenizer()

text = "Tập đoàn FPT công bố kết quả kinh doanh quý 4"
print(tokenizer.tokenize(text))
# Output: "Tập_đoàn FPT công_bố kết_quả kinh_doanh quý 4"

print(tokenizer.pos_tagging(text))
# Output: [('Tập_đoàn', 'N'), ('FPT', 'Np'), ('công_bố', 'V'), ...]

print(tokenizer.named_entity_recognition(text))
# Output: [('Tập_đoàn', 'B-ORG'), ('FPT', 'I-ORG'), ...]
```

**Tool 2: VnCoreNLP**

```python
# pip install py_vncorenlp
# Download model: py_vncorenlp.download_model(save_dir='/path/to/vncorenlp')

from py_vncorenlp import VnCoreNLP

class VnCoreTokenizer:
    """Vietnamese tokenization using VnCoreNLP"""
    
    def __init__(self, model_path: str = 'vncorenlp'):
        self.nlp = VnCoreNLP(save_dir=model_path)
    
    def tokenize(self, text: str) -> List[str]:
        """Word segmentation"""
        result = self.nlp.word_segment(text)
        return result
    
    def pos_tag(self, text: str) -> List[List[Dict]]:
        """POS tagging"""
        result = self.nlp.pos_tag(text)
        return result
    
    def ner(self, text: str) -> List[List[Dict]]:
        """Named entity recognition"""
        result = self.nlp.ner(text)
        return result
    
    def annotate(self, text: str) -> List[Dict]:
        """Full annotation"""
        result = self.nlp.annotate(text)
        return result
```

**Tool 3: PyVi (Lightweight)**

```python
# pip install pyvi
from pyvi import ViTokenizer, ViPosTagger

class PyViTokenizer:
    """Vietnamese tokenization using PyVi"""
    
    @staticmethod
    def tokenize(text: str) -> str:
        """Tokenize text"""
        return ViTokenizer.tokenize(text)
    
    @staticmethod
    def pos_tag(text: str) -> List[tuple]:
        """POS tagging"""
        return ViPosTagger.postagging(ViTokenizer.tokenize(text))

# Usage
text = "Chứng khoán Việt Nam tăng mạnh"
print(PyViTokenizer.tokenize(text))
# Output: "Chứng_khoán Việt_Nam tăng mạnh"
```

### 7.3. So sánh các công cụ

| Tool | Speed | Accuracy | Features | Memory |
|------|-------|----------|----------|--------|
| **Underthesea** | Fast | Good | Tokenize, POS, NER | Low |
| **VnCoreNLP** | Medium | Best | Full NLP pipeline | High |
| **PyVi** | Very Fast | Medium | Tokenize, POS | Very Low |

**Recommendations:**
```
Research/Production: VnCoreNLP (best accuracy)
Quick experiments: Underthesea (good balance)
Large-scale processing: PyVi (fastest)
```

### 7.4. Financial-Specific Tokenization

```python
class FinancialVietnameseTokenizer:
    """
    Tokenizer optimized for financial text
    """
    
    def __init__(self, base_tokenizer='underthesea'):
        self.base_tokenizer = base_tokenizer
        
        # Financial compound words to preserve
        self.compound_words = {
            'thị trường chứng khoán': 'thị_trường_chứng_khoán',
            'vốn hóa thị trường': 'vốn_hóa_thị_trường',
            'tỷ suất lợi nhuận': 'tỷ_suất_lợi_nhuận',
            'biên lợi nhuận gộp': 'biên_lợi_nhuận_gộp',
            'dòng tiền tự do': 'dòng_tiền_tự_do',
            'nhà đầu tư nước ngoài': 'nhà_đầu_tư_nước_ngoài',
            'phiên giao dịch': 'phiên_giao_dịch',
            'giá trị giao dịch': 'giá_trị_giao_dịch',
            'khối lượng giao dịch': 'khối_lượng_giao_dịch',
        }
    
    def tokenize(self, text: str) -> str:
        """Tokenize with financial domain awareness"""
        # Pre-process: Replace compound words
        text_processed = text.lower()
        for phrase, replacement in self.compound_words.items():
            text_processed = text_processed.replace(phrase, replacement)
        
        # Base tokenization
        if self.base_tokenizer == 'underthesea':
            tokens = word_tokenize(text_processed)
        else:
            tokens = text_processed  # fallback
        
        return tokens
    
    def extract_financial_entities(self, text: str) -> Dict:
        """Extract financial entities"""
        entities = {
            'tickers': [],
            'prices': [],
            'percentages': [],
            'organizations': [],
            'dates': []
        }
        
        # Use NER
        ner_results = ner(text)
        
        for token, label in ner_results:
            if 'ORG' in label:
                entities['organizations'].append(token)
        
        # Extract tickers (3-letter caps)
        tickers = re.findall(r'\b[A-Z]{3}\b', text)
        entities['tickers'] = tickers
        
        # Extract percentages
        percentages = re.findall(r'[\-+]?\d+[,.]?\d*\s*%', text)
        entities['percentages'] = percentages
        
        return entities
```

---

## 8. TEXT REPRESENTATIONS: TF-IDF VS EMBEDDINGS VS TRANSFORMERS

### 8.1. TF-IDF (Traditional)

**Ý tưởng:**
```
TF-IDF = Term Frequency × Inverse Document Frequency

TF: Từ xuất hiện nhiều trong doc → quan trọng cho doc đó
IDF: Từ xuất hiện ít trong corpus → quan trọng hơn từ phổ biến
```

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfidfNewsVectorizer:
    """TF-IDF vectorization for Vietnamese news"""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms appearing in < 2 docs
            max_df=0.95,  # Ignore terms appearing in > 95% docs
            sublinear_tf=True  # Use 1 + log(tf) instead of tf
        )
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to TF-IDF vectors"""
        # Tokenize first
        tokenized = [word_tokenize(text) for text in texts]
        
        # Fit TF-IDF
        vectors = self.vectorizer.fit_transform(tokenized)
        self.is_fitted = True
        
        return vectors.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted")
        
        tokenized = [word_tokenize(text) for text in texts]
        return self.vectorizer.transform(tokenized).toarray()
    
    def get_top_features(self, n=20) -> List[str]:
        """Get top features by IDF score"""
        feature_names = self.vectorizer.get_feature_names_out()
        idf_scores = self.vectorizer.idf_
        
        top_indices = np.argsort(idf_scores)[-n:]
        return [feature_names[i] for i in top_indices]

# Usage
vectorizer = TfidfNewsVectorizer(max_features=3000)
X = vectorizer.fit_transform(df_news['content'].tolist())
print(f"TF-IDF shape: {X.shape}")
print(f"Top features: {vectorizer.get_top_features(10)}")
```

### 8.2. Word Embeddings (Word2Vec, FastText)

**Ý tưởng:**
```
Word → Dense vector (100-300 dimensions)
"chứng khoán" → [0.2, -0.5, 0.1, ...]

Similar words → Similar vectors
cosine_similarity("FPT", "VCB") > cosine_similarity("FPT", "bánh mì")
```

**Implementation với Pre-trained Vietnamese Embeddings:**

```python
from gensim.models import KeyedVectors
import numpy as np

class VietnameseEmbeddings:
    """
    Vietnamese word embeddings using pre-trained models
    
    Download: https://github.com/vietnlp/vi-word-embeddings
    """
    
    def __init__(self, model_path: str = 'vi.vec'):
        """Load pre-trained embeddings"""
        self.model = KeyedVectors.load_word2vec_format(model_path)
        self.dim = self.model.vector_size
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """Get embedding for a single word"""
        try:
            return self.model[word]
        except KeyError:
            return np.zeros(self.dim)
    
    def get_document_vector(self, text: str, method='mean') -> np.ndarray:
        """
        Get document vector by aggregating word vectors
        
        Methods: 'mean', 'max', 'concat'
        """
        # Tokenize
        tokens = word_tokenize(text).split()
        
        # Get vectors for each token
        vectors = []
        for token in tokens:
            vec = self.get_word_vector(token)
            if np.any(vec):  # Skip zero vectors
                vectors.append(vec)
        
        if not vectors:
            return np.zeros(self.dim)
        
        vectors = np.array(vectors)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'max':
            return np.max(vectors, axis=0)
        elif method == 'concat':
            return np.concatenate([
                np.mean(vectors, axis=0),
                np.max(vectors, axis=0)
            ])
        
        return np.mean(vectors, axis=0)
    
    def most_similar(self, word: str, topn: int = 10) -> List[tuple]:
        """Find most similar words"""
        try:
            return self.model.most_similar(word, topn=topn)
        except KeyError:
            return []

# Usage
embeddings = VietnameseEmbeddings('vi.vec')

# Single word
vec = embeddings.get_word_vector('chứng_khoán')
print(f"Vector shape: {vec.shape}")

# Document
doc_vec = embeddings.get_document_vector("FPT công bố lợi nhuận tăng 20%")
print(f"Document vector shape: {doc_vec.shape}")

# Similar words
similar = embeddings.most_similar('cổ_phiếu')
print(f"Similar to 'cổ_phiếu': {similar[:5]}")
```

### 8.3. Transformer Embeddings (PhoBERT)

**Ý tưởng:**
```
Word2Vec: Fixed embedding per word
BERT: Context-dependent embedding

"bank" in "river bank" ≠ "bank" in "investment bank"
BERT captures this, Word2Vec doesn't
```

**Implementation với PhoBERT:**

```python
# pip install transformers torch

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

class PhoBERTEmbeddings:
    """
    Vietnamese BERT embeddings using PhoBERT
    
    Model: vinai/phobert-base or vinai/phobert-large
    """
    
    def __init__(self, model_name: str = 'vinai/phobert-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def get_embedding(self, text: str, pooling: str = 'cls') -> np.ndarray:
        """
        Get embedding for text
        
        Pooling methods:
        - 'cls': Use [CLS] token embedding
        - 'mean': Mean of all token embeddings
        - 'max': Max pooling
        """
        # Tokenize (PhoBERT expects word-segmented input)
        text_tokenized = word_tokenize(text)
        
        # Encode
        inputs = self.tokenizer(
            text_tokenized,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
        
        # Pooling
        if pooling == 'cls':
            embedding = hidden_states[:, 0, :]  # [CLS] token
        elif pooling == 'mean':
            # Mean pooling (exclude padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            embedding = masked_hidden.sum(1) / attention_mask.sum(1)
        elif pooling == 'max':
            embedding = hidden_states.max(dim=1)[0]
        
        return embedding.cpu().numpy().squeeze()
    
    def get_batch_embeddings(self, texts: List[str], 
                            batch_size: int = 32,
                            pooling: str = 'cls') -> np.ndarray:
        """Get embeddings for batch of texts"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_tokenized = [word_tokenize(t) for t in batch]
            
            inputs = self.tokenizer(
                batch_tokenized,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
            
            if pooling == 'cls':
                batch_embeddings = hidden_states[:, 0, :]
            elif pooling == 'mean':
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = hidden_states * attention_mask
                batch_embeddings = masked_hidden.sum(1) / attention_mask.sum(1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

# Usage
phobert = PhoBERTEmbeddings('vinai/phobert-base')

# Single text
embedding = phobert.get_embedding("FPT báo lãi kỷ lục quý 4")
print(f"PhoBERT embedding shape: {embedding.shape}")  # (768,)

# Batch
texts = df_news['content'].tolist()[:100]
embeddings = phobert.get_batch_embeddings(texts, batch_size=16)
print(f"Batch embeddings shape: {embeddings.shape}")  # (100, 768)
```

### 8.4. So sánh các phương pháp

| Method | Dimensionality | Context-aware | Speed | Use case |
|--------|---------------|---------------|-------|----------|
| **TF-IDF** | Sparse, high | No | Very fast | Baseline, interpretable |
| **Word2Vec** | Dense, 100-300 | No | Fast | General NLP tasks |
| **FastText** | Dense, 100-300 | No (but handles OOV) | Fast | OOV words important |
| **PhoBERT** | Dense, 768 | Yes | Slow | Best quality needed |

**Recommendations cho Financial News:**
```
Baseline/Quick experiments: TF-IDF
Production with limited compute: Word2Vec + aggregation
Best performance: PhoBERT (fine-tune if possible)
```

---

## 9. FINANCIAL SENTIMENT MODELING

### 9.1. Đặc thù của Financial Sentiment

**Khác biệt với general sentiment:**
```
General: "The stock is falling" → Negative (về ngữ nghĩa)
Finance: "The stock is falling" → 
    - Negative nếu bạn đang hold
    - Positive nếu bạn muốn mua
    - Neutral nếu đang chờ đợi

"Lãi suất tăng" → 
    - Negative cho cổ phiếu growth
    - Positive cho cổ phiếu ngân hàng
```

### 9.2. Lexicon-based Sentiment

```python
class FinancialSentimentLexicon:
    """
    Lexicon-based sentiment analysis cho tiếng Việt financial
    """
    
    def __init__(self):
        # Positive financial terms
        self.positive_terms = {
            'tăng': 1.0, 'tăng trưởng': 1.2, 'tăng mạnh': 1.5,
            'lợi nhuận': 0.8, 'lãi': 0.8, 'lãi lớn': 1.2,
            'kỷ lục': 1.0, 'đột phá': 1.2, 'vượt kỳ vọng': 1.3,
            'phục hồi': 0.8, 'hồi phục': 0.8, 'bứt phá': 1.2,
            'thắng lớn': 1.3, 'khởi sắc': 0.9, 'triển vọng': 0.7,
            'tích cực': 0.8, 'lạc quan': 0.8, 'thuận lợi': 0.7,
            'cổ tức': 0.6, 'chia cổ tức': 0.8, 'mua ròng': 0.9,
        }
        
        # Negative financial terms
        self.negative_terms = {
            'giảm': -1.0, 'sụt giảm': -1.2, 'giảm mạnh': -1.5,
            'lỗ': -1.2, 'thua lỗ': -1.3, 'lỗ nặng': -1.5,
            'suy thoái': -1.3, 'khủng hoảng': -1.5, 'phá sản': -1.8,
            'vỡ nợ': -1.5, 'nợ xấu': -1.2, 'margin call': -1.3,
            'bán tháo': -1.4, 'hoảng loạn': -1.5, 'bi quan': -1.0,
            'rủi ro': -0.8, 'cảnh báo': -0.9, 'tiêu cực': -1.0,
            'bán ròng': -0.9, 'rút vốn': -1.0, 'thoái lui': -0.8,
        }
        
        # Intensifiers
        self.intensifiers = {
            'rất': 1.3, 'cực kỳ': 1.5, 'vô cùng': 1.5,
            'hơi': 0.7, 'một chút': 0.6, 'nhẹ': 0.7,
            'kỷ lục': 1.4, 'lịch sử': 1.3, 'chưa từng có': 1.4,
        }
        
        # Negators
        self.negators = {'không', 'chưa', 'chẳng', 'không hề'}
    
    def calculate_sentiment(self, text: str) -> Dict:
        """Calculate sentiment score from text"""
        # Tokenize
        tokens = word_tokenize(text.lower()).split()
        
        sentiment_score = 0.0
        positive_count = 0
        negative_count = 0
        matched_terms = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check for negator
            is_negated = False
            if i > 0 and tokens[i-1] in self.negators:
                is_negated = True
            
            # Check bigrams first
            if i < len(tokens) - 1:
                bigram = f"{token} {tokens[i+1]}"
                if bigram in self.positive_terms:
                    score = self.positive_terms[bigram]
                    if is_negated:
                        score = -score
                    sentiment_score += score
                    positive_count += 1
                    matched_terms.append((bigram, score))
                    i += 2
                    continue
                elif bigram in self.negative_terms:
                    score = self.negative_terms[bigram]
                    if is_negated:
                        score = -score
                    sentiment_score += score
                    negative_count += 1
                    matched_terms.append((bigram, score))
                    i += 2
                    continue
            
            # Check unigrams
            if token in self.positive_terms:
                score = self.positive_terms[token]
                if is_negated:
                    score = -score
                sentiment_score += score
                positive_count += 1
                matched_terms.append((token, score))
            elif token in self.negative_terms:
                score = self.negative_terms[token]
                if is_negated:
                    score = -score
                sentiment_score += score
                negative_count += 1
                matched_terms.append((token, score))
            
            i += 1
        
        # Normalize
        total_count = positive_count + negative_count
        if total_count > 0:
            sentiment_score = sentiment_score / total_count
        
        # Clip to [-1, 1]
        sentiment_score = max(-1, min(1, sentiment_score))
        
        return {
            'score': sentiment_score,
            'label': self._get_label(sentiment_score),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'matched_terms': matched_terms
        }
    
    def _get_label(self, score: float) -> str:
        """Convert score to label"""
        if score > 0.2:
            return 'positive'
        elif score < -0.2:
            return 'negative'
        else:
            return 'neutral'
```

### 9.3. ML-based Sentiment

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class MLSentimentClassifier:
    """
    ML-based sentiment classifier cho financial news
    """
    
    def __init__(self, vectorizer=None, classifier='logistic'):
        self.vectorizer = vectorizer or TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2)
        )
        
        if classifier == 'logistic':
            self.model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000
            )
        elif classifier == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            )
    
    def fit(self, texts: List[str], labels: List[int]):
        """
        Train classifier
        
        Labels: 0=negative, 1=neutral, 2=positive
        """
        # Tokenize
        texts_tokenized = [word_tokenize(t) for t in texts]
        
        # Vectorize
        X = self.vectorizer.fit_transform(texts_tokenized)
        
        # Train
        self.model.fit(X, labels)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict sentiment"""
        texts_tokenized = [word_tokenize(t) for t in texts]
        X = self.vectorizer.transform(texts_tokenized)
        return self.model.predict(X)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities"""
        texts_tokenized = [word_tokenize(t) for t in texts]
        X = self.vectorizer.transform(texts_tokenized)
        return self.model.predict_proba(X)
    
    def evaluate(self, texts: List[str], labels: List[int], cv=5):
        """Cross-validation evaluation"""
        texts_tokenized = [word_tokenize(t) for t in texts]
        X = self.vectorizer.fit_transform(texts_tokenized)
        
        scores = cross_val_score(self.model, X, labels, cv=cv, scoring='f1_macro')
        return {
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores': scores
        }
```

### 9.4. PhoBERT-based Sentiment

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    """Dataset for sentiment classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = word_tokenize(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

class PhoBERTSentimentClassifier:
    """
    PhoBERT-based sentiment classifier
    """
    
    def __init__(self, num_labels=3, model_name='vinai/phobert-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    def train(self, train_texts, train_labels, val_texts, val_labels, 
              output_dir='./sentiment_model', epochs=3):
        """Fine-tune PhoBERT for sentiment"""
        
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer
        )
        val_dataset = SentimentDataset(
            val_texts, val_labels, self.tokenizer
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
    
    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment"""
        self.model.eval()
        predictions = []
        
        for text in texts:
            text_tokenized = word_tokenize(text)
            inputs = self.tokenizer(
                text_tokenized,
                return_tensors='pt',
                truncation=True,
                max_length=256
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
        
        return predictions
```

---

## 10. ALIGNING NEWS VỚI PRICE CANDLES

### 10.1. Vấn đề Timestamp Alignment

**Challenges:**
```
News: "28/01/2026 10:30" - Tin về FPT
Price: 
    - 2026-01-28 09:00-09:15 candle
    - 2026-01-28 09:15-09:30 candle
    - ...

Q: Tin lúc 10:30 ảnh hưởng đến candle nào?
- 10:30 candle? (concurrent)
- 10:45 candle? (next candle)
- Cả ngày? (daily aggregation)
```

### 10.2. Timestamp Parsing

```python
from datetime import datetime, timedelta
import pytz

class NewsTimestampParser:
    """Parse và normalize timestamps từ Vietnamese news"""
    
    def __init__(self, timezone='Asia/Ho_Chi_Minh'):
        self.tz = pytz.timezone(timezone)
        
        # Common formats
        self.formats = [
            '%d/%m/%Y %H:%M',      # 28/01/2026 10:30
            '%d/%m/%Y - %H:%M',    # 28/01/2026 - 10:30
            '%H:%M %d/%m/%Y',      # 10:30 28/01/2026
            '%d-%m-%Y %H:%M',      # 28-01-2026 10:30
            '%Y-%m-%d %H:%M:%S',   # 2026-01-28 10:30:00
        ]
        
        # Relative time patterns
        self.relative_patterns = {
            'phút trước': 'minutes',
            'giờ trước': 'hours',
            'ngày trước': 'days',
        }
    
    def parse(self, time_str: str, reference_time: datetime = None) -> Optional[datetime]:
        """Parse timestamp string to datetime"""
        if not time_str:
            return None
        
        time_str = time_str.strip()
        reference_time = reference_time or datetime.now(self.tz)
        
        # Try relative time first
        for pattern, unit in self.relative_patterns.items():
            if pattern in time_str:
                match = re.search(r'(\d+)\s*' + pattern, time_str)
                if match:
                    value = int(match.group(1))
                    if unit == 'minutes':
                        return reference_time - timedelta(minutes=value)
                    elif unit == 'hours':
                        return reference_time - timedelta(hours=value)
                    elif unit == 'days':
                        return reference_time - timedelta(days=value)
        
        # Try absolute formats
        for fmt in self.formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                return self.tz.localize(dt)
            except ValueError:
                continue
        
        return None
    
    def to_trading_session(self, dt: datetime) -> Dict:
        """Convert datetime to trading session info"""
        if dt is None:
            return None
        
        hour = dt.hour
        minute = dt.minute
        time_decimal = hour + minute / 60
        
        # Vietnam market hours
        # Morning: 9:00 - 11:30
        # Afternoon: 13:00 - 15:00
        
        if time_decimal < 9:
            session = 'pre_market'
            next_candle = dt.replace(hour=9, minute=0)
        elif 9 <= time_decimal < 11.5:
            session = 'morning'
            next_candle = dt + timedelta(minutes=15 - (minute % 15))
        elif 11.5 <= time_decimal < 13:
            session = 'lunch_break'
            next_candle = dt.replace(hour=13, minute=0)
        elif 13 <= time_decimal < 15:
            session = 'afternoon'
            next_candle = dt + timedelta(minutes=15 - (minute % 15))
        else:
            session = 'after_market'
            next_candle = (dt + timedelta(days=1)).replace(hour=9, minute=0)
        
        return {
            'datetime': dt,
            'date': dt.date(),
            'session': session,
            'next_candle_start': next_candle,
            'is_trading_hours': session in ['morning', 'afternoon']
        }
```

### 10.3. News-Price Alignment Strategies

```python
class NewsPriceAligner:
    """
    Align news với price data
    """
    
    def __init__(self, price_df: pd.DataFrame, timestamp_col='datetime'):
        """
        Args:
            price_df: DataFrame with OHLCV data
            timestamp_col: Column name for timestamps
        """
        self.price_df = price_df.copy()
        self.price_df[timestamp_col] = pd.to_datetime(self.price_df[timestamp_col])
        self.price_df = self.price_df.set_index(timestamp_col).sort_index()
        self.parser = NewsTimestampParser()
    
    def align_to_candle(self, news_time: datetime, 
                        method: str = 'next') -> Optional[pd.Series]:
        """
        Align news timestamp to price candle
        
        Methods:
        - 'next': Next candle after news (most common)
        - 'current': Current candle (if during trading)
        - 'same_day': End of day candle
        - 'next_day': Next trading day open
        """
        session_info = self.parser.to_trading_session(news_time)
        
        if method == 'next':
            # Find next candle after news time
            mask = self.price_df.index > news_time
            if mask.any():
                return self.price_df[mask].iloc[0]
        
        elif method == 'current':
            # Find candle containing news time
            # Assumes candles are 15-min intervals
            candle_start = news_time.replace(
                minute=(news_time.minute // 15) * 15,
                second=0, microsecond=0
            )
            if candle_start in self.price_df.index:
                return self.price_df.loc[candle_start]
        
        elif method == 'same_day':
            # Last candle of the day
            date = news_time.date()
            day_data = self.price_df[self.price_df.index.date == date]
            if len(day_data) > 0:
                return day_data.iloc[-1]
        
        elif method == 'next_day':
            # First candle of next trading day
            date = news_time.date()
            next_day_data = self.price_df[self.price_df.index.date > date]
            if len(next_day_data) > 0:
                return next_day_data.iloc[0]
        
        return None
    
    def create_aligned_dataset(self, news_df: pd.DataFrame,
                              news_time_col: str = 'published_time',
                              method: str = 'next',
                              forward_returns: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Create aligned dataset for modeling
        
        Args:
            news_df: DataFrame with news
            news_time_col: Column with news timestamp
            method: Alignment method
            forward_returns: Periods for forward returns
        
        Returns:
            DataFrame with news + aligned price data
        """
        aligned_data = []
        
        for idx, row in news_df.iterrows():
            # Parse news time
            news_time = self.parser.parse(row[news_time_col])
            
            if news_time is None:
                continue
            
            # Get aligned candle
            candle = self.align_to_candle(news_time, method=method)
            
            if candle is None:
                continue
            
            record = {
                'news_idx': idx,
                'news_time': news_time,
                'aligned_candle_time': candle.name,
                'open': candle.get('open'),
                'high': candle.get('high'),
                'low': candle.get('low'),
                'close': candle.get('close'),
                'volume': candle.get('volume'),
            }
            
            # Calculate forward returns
            candle_idx = self.price_df.index.get_loc(candle.name)
            
            for period in forward_returns:
                future_idx = candle_idx + period
                if future_idx < len(self.price_df):
                    future_price = self.price_df.iloc[future_idx]['close']
                    current_price = candle['close']
                    record[f'return_{period}'] = (future_price - current_price) / current_price
                else:
                    record[f'return_{period}'] = None
            
            aligned_data.append(record)
        
        aligned_df = pd.DataFrame(aligned_data)
        
        # Merge with news data
        result = news_df.merge(
            aligned_df,
            left_index=True,
            right_on='news_idx',
            how='inner'
        )
        
        return result

# Usage
price_df = pd.read_csv('data/price/FPT_15min.csv')
aligner = NewsPriceAligner(price_df)

aligned_dataset = aligner.create_aligned_dataset(
    df_news,
    method='next',
    forward_returns=[1, 5, 10, 20]
)
```

---

## 11. NOISE FILTERING VÀ IRRELEVANT NEWS REMOVAL

### 11.1. Tại sao cần Filtering?

**Vấn đề:**
```
Crawled 1000 articles from CafeF
But:
- 300 articles: Không liên quan (quảng cáo, lifestyle)
- 200 articles: Duplicate hoặc rehash
- 100 articles: Old news (đã phản ánh vào giá)
- 400 articles: Actually relevant

Noise ratio: 60%!
```

### 11.2. Rule-based Filtering

```python
class NewsFilter:
    """
    Filter noise và irrelevant news
    """
    
    def __init__(self):
        # Irrelevant keywords
        self.irrelevant_keywords = [
            'quảng cáo', 'sponsored', 'pr ', 'advertorial',
            'lifestyle', 'du lịch', 'ẩm thực', 'giải trí',
            'thể thao', 'bóng đá', 'sao việt', 'hot girl',
        ]
        
        # Financial keywords (keep these)
        self.financial_keywords = [
            'cổ phiếu', 'chứng khoán', 'lợi nhuận', 'doanh thu',
            'vn-index', 'vnindex', 'hnx', 'upcom',
            'mua bán', 'giao dịch', 'đầu tư', 'cổ tức',
            'báo cáo tài chính', 'kết quả kinh doanh',
            'ngân hàng', 'bất động sản', 'năng lượng',
        ]
        
        # Minimum content length
        self.min_content_length = 100
    
    def filter_single(self, news: Dict) -> Dict:
        """
        Filter single news article
        
        Returns dict with 'keep' boolean and 'reason'
        """
        title = news.get('title', '').lower()
        content = news.get('content', '').lower()
        full_text = title + ' ' + content
        
        # Check minimum length
        if len(content) < self.min_content_length:
            return {'keep': False, 'reason': 'too_short'}
        
        # Check irrelevant keywords
        for keyword in self.irrelevant_keywords:
            if keyword in full_text:
                return {'keep': False, 'reason': f'irrelevant_{keyword}'}
        
        # Check financial relevance
        has_financial = False
        for keyword in self.financial_keywords:
            if keyword in full_text:
                has_financial = True
                break
        
        if not has_financial:
            return {'keep': False, 'reason': 'not_financial'}
        
        return {'keep': True, 'reason': 'relevant'}
    
    def filter_batch(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter batch of news"""
        results = []
        
        for idx, row in news_df.iterrows():
            result = self.filter_single(row.to_dict())
            results.append({
                'idx': idx,
                'keep': result['keep'],
                'reason': result['reason']
            })
        
        results_df = pd.DataFrame(results).set_index('idx')
        news_df = news_df.join(results_df)
        
        # Statistics
        total = len(news_df)
        kept = news_df['keep'].sum()
        print(f"Filtered: {total} → {kept} ({kept/total*100:.1f}% kept)")
        print(f"\nRemoval reasons:")
        print(news_df[~news_df['keep']]['reason'].value_counts())
        
        return news_df[news_df['keep']].drop(columns=['keep', 'reason'])
```

### 11.3. Deduplication

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

class NewsDeduplicator:
    """
    Remove duplicate and near-duplicate news
    """
    
    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(max_features=1000)
    
    def exact_dedup(self, news_df: pd.DataFrame, 
                   key_cols: List[str] = ['title']) -> pd.DataFrame:
        """Remove exact duplicates based on key columns"""
        original_count = len(news_df)
        
        # Create hash of key columns
        news_df['_hash'] = news_df[key_cols].apply(
            lambda x: hashlib.md5('|'.join(x.astype(str)).encode()).hexdigest(),
            axis=1
        )
        
        # Keep first occurrence
        news_df = news_df.drop_duplicates(subset='_hash', keep='first')
        news_df = news_df.drop(columns=['_hash'])
        
        print(f"Exact dedup: {original_count} → {len(news_df)}")
        return news_df
    
    def semantic_dedup(self, news_df: pd.DataFrame,
                      text_col: str = 'content') -> pd.DataFrame:
        """Remove semantically similar articles"""
        original_count = len(news_df)
        
        # Vectorize
        texts = news_df[text_col].fillna('').tolist()
        texts_tokenized = [word_tokenize(t) for t in texts]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts_tokenized)
        except ValueError:
            # Empty vocabulary
            return news_df
        
        # Calculate pairwise similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicates
        keep_indices = []
        seen = set()
        
        for i in range(len(news_df)):
            if i in seen:
                continue
            
            keep_indices.append(i)
            
            # Mark similar articles as seen
            similar = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
            for j in similar:
                if j != i:
                    seen.add(j)
        
        news_df_deduped = news_df.iloc[keep_indices]
        
        print(f"Semantic dedup: {original_count} → {len(news_df_deduped)}")
        return news_df_deduped
    
    def deduplicate(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Full deduplication pipeline"""
        # Exact first (faster)
        news_df = self.exact_dedup(news_df)
        
        # Then semantic
        news_df = self.semantic_dedup(news_df)
        
        return news_df
```

### 11.4. Relevance Scoring

```python
class RelevanceScorer:
    """
    Score news relevance to specific ticker
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.company_info = self._load_company_info(ticker)
    
    def _load_company_info(self, ticker: str) -> Dict:
        """Load company info for ticker"""
        # Hardcoded for demo, should load from DB
        company_db = {
            'FPT': {
                'name': 'FPT Corporation',
                'aliases': ['fpt', 'tập đoàn fpt', 'fpt corp'],
                'keywords': ['công nghệ', 'phần mềm', 'it', 'outsourcing', 
                            'trương gia bình', 'fintech', 'ai'],
                'competitors': ['CMC', 'FTS', 'ONE'],
                'sector': 'technology'
            },
            'VCB': {
                'name': 'Vietcombank',
                'aliases': ['vietcombank', 'ngân hàng ngoại thương'],
                'keywords': ['ngân hàng', 'tín dụng', 'lãi suất', 
                            'nợ xấu', 'cho vay'],
                'competitors': ['BID', 'CTG', 'TCB'],
                'sector': 'banking'
            }
        }
        return company_db.get(ticker, {})
    
    def score(self, text: str) -> Dict:
        """
        Score relevance of text to ticker
        
        Returns: {'score': float, 'signals': list}
        """
        text_lower = text.lower()
        score = 0.0
        signals = []
        
        # Direct ticker mention
        if self.ticker.lower() in text_lower:
            score += 1.0
            signals.append(('ticker_mention', 1.0))
        
        # Company name mention
        for alias in self.company_info.get('aliases', []):
            if alias in text_lower:
                score += 0.8
                signals.append(('alias_mention', 0.8))
                break
        
        # Keyword matches
        keywords = self.company_info.get('keywords', [])
        matched_keywords = [kw for kw in keywords if kw in text_lower]
        if matched_keywords:
            keyword_score = min(0.5, len(matched_keywords) * 0.1)
            score += keyword_score
            signals.append(('keywords', keyword_score, matched_keywords))
        
        # Competitor mention (indirect relevance)
        for competitor in self.company_info.get('competitors', []):
            if competitor.lower() in text_lower:
                score += 0.3
                signals.append(('competitor_mention', 0.3, competitor))
        
        # Normalize score to [0, 1]
        score = min(1.0, score)
        
        return {
            'score': score,
            'signals': signals,
            'is_relevant': score > 0.3
        }
    
    def filter_by_relevance(self, news_df: pd.DataFrame,
                           text_col: str = 'content',
                           min_score: float = 0.3) -> pd.DataFrame:
        """Filter news by relevance score"""
        scores = []
        
        for idx, row in news_df.iterrows():
            text = row[text_col] if pd.notna(row[text_col]) else ''
            result = self.score(text)
            scores.append({
                'idx': idx,
                'relevance_score': result['score'],
                'is_relevant': result['is_relevant']
            })
        
        scores_df = pd.DataFrame(scores).set_index('idx')
        news_df = news_df.join(scores_df)
        
        # Filter
        filtered = news_df[news_df['relevance_score'] >= min_score]
        
        print(f"Relevance filter ({self.ticker}): {len(news_df)} → {len(filtered)}")
        return filtered

# Usage
scorer = RelevanceScorer('FPT')
df_fpt_news = scorer.filter_by_relevance(df_news, min_score=0.3)
```

### 11.5. Complete Filtering Pipeline

```python
def filter_news_pipeline(news_df: pd.DataFrame,
                        tickers: List[str] = None) -> pd.DataFrame:
    """
    Complete news filtering pipeline
    
    Steps:
    1. Basic filtering (remove irrelevant)
    2. Deduplication
    3. Relevance scoring (if tickers specified)
    """
    print(f"Starting with {len(news_df)} articles")
    
    # Step 1: Basic filtering
    filter = NewsFilter()
    news_df = filter.filter_batch(news_df)
    
    # Step 2: Deduplication
    dedup = NewsDeduplicator(similarity_threshold=0.85)
    news_df = dedup.deduplicate(news_df)
    
    # Step 3: Relevance scoring
    if tickers:
        relevant_dfs = []
        for ticker in tickers:
            scorer = RelevanceScorer(ticker)
            ticker_news = scorer.filter_by_relevance(news_df.copy())
            ticker_news['primary_ticker'] = ticker
            relevant_dfs.append(ticker_news)
        
        news_df = pd.concat(relevant_dfs, ignore_index=True)
        news_df = news_df.drop_duplicates(subset=['link'], keep='first')
    
    print(f"\nFinal: {len(news_df)} articles")
    return news_df

# Usage
df_clean = filter_news_pipeline(
    df_raw,
    tickers=['FPT', 'VCB', 'VNM']
)
```

---

## 12. BEST PRACTICES

### Ethical Crawling

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

### Rate Limiting

```python
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
```

### Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def crawl_with_retry(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response
```

---

## 13. BÀI TẬP THỰC HÀNH

### Bài tập 1: Full NLP Pipeline

**Yêu cầu:**
1. Crawl 100 bài từ CafeF
2. Implement preprocessing pipeline
3. Extract tickers và financial terms
4. Output: Clean dataset ready for modeling

### Bài tập 2: Sentiment Analysis

**Yêu cầu:**
1. Build lexicon-based sentiment
2. Train ML classifier (if labeled data available)
3. Compare với PhoBERT
4. Evaluate on test set

### Bài tập 3: News-Price Alignment

**Yêu cầu:**
1. Align news với 15-min candles
2. Calculate forward returns
3. Analyze: Sentiment có predict returns không?
4. Build simple trading signal

---

## Kiểm tra hiểu bài

- [ ] Implement được full NLP pipeline
- [ ] Tokenize được tiếng Việt đúng cách
- [ ] So sánh được TF-IDF vs Embeddings vs Transformers
- [ ] Build được sentiment model
- [ ] Align được news với price data
- [ ] Filter được noise và irrelevant news

---

## Tài liệu tham khảo

**Vietnamese NLP:**
- Underthesea: Vietnamese NLP toolkit
- VnCoreNLP: Vietnamese CoreNLP
- PhoBERT: Pre-trained Vietnamese BERT

**Financial NLP:**
- "Financial Sentiment Analysis" - Loughran & McDonald
- FinBERT: Financial domain BERT

---

## Bước tiếp theo

Sau khi hoàn thành:
- `02_VIETNAMESE_TEXT_PROCESSING.md` - Xử lý tiếng Việt nâng cao
- `03_MULTIMODAL_FUSION.md` - Kết hợp text và price data

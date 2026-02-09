# News Intelligence Pipeline — Investment-Grade Design

> **Purpose:** Redesign the news pipeline into a professional investment-grade signal system for stock prediction.

---

## 1. Pipeline Stages

```
Crawl → Clean → [NEW: Source Filter] → [NEW: Enrich] → Sentiment → Align → [NEW: Ticker Scoring] → [NEW: Aggregation]
                                    │
                                    ├── Event type classification
                                    ├── Impact horizon
                                    └── Sentiment confidence
```

### Stage 1: Source Filtering

**Goal:** Remove generic macro news unless directly relevant to the stock.

| Source Type | Sources | Keep When |
|-------------|---------|-----------|
| **Stock** | cafef, vietstock | Always (stock-focused) |
| **Sector** | vnexpress, vneconomy | When ticker in title/body |
| **Macro** | tradingeconomics | When ticker mentioned or sector keyword |
| **Regulatory** | ssc, hsx | When ticker in disclosure or sector |

**Filter rule:** Drop macro/sector articles with no ticker mention and no VN index keywords.

---

### Stage 2: Ticker Relevance Scoring

**Goal:** Quantify how strongly the article affects a given stock (0–1).

**Formula:**
```
relevance = 0.4 * mention_strength + 0.25 * event_weight + 0.2 * recency + 0.15 * source_weight
```

| Factor | Formula |
|--------|---------|
| **mention_strength** | 1.0 if ticker in title; 0.6 if ticker in body; 0.3 if alias only; 0.1 if sector only |
| **event_weight** | 1.0 (earnings, M&A, guidance); 0.8 (legal, operations); 0.6 (macro); 0.5 (other) |
| **recency** | 1.0 if ≤1 day; 0.8 if ≤3 days; 0.5 if ≤7 days; 0.2 if ≤30 days |
| **source_weight** | 1.0 (cafef, vietstock); 0.8 (vnexpress); 0.6 (vneconomy, ssc, hsx); 0.4 (tradingeconomics) |

---

### Stage 3: Event Type Classification

**Types:** `earnings`, `legal`, `macro`, `operations`, `guidance`, `ma`, `dividend`, `other`

**Keyword-based (Vietnamese):**
- earnings: "báo cáo tài chính", "lợi nhuận", "doanh thu", "kết quả kinh doanh"
- legal: "kiện", "phạt", "vi phạm", "ubck", "ssc"
- macro: "lãi suất", "gdp", "inflation", "fed"
- operations: "mở rộng", "nhà máy", "dự án", "sản xuất"
- guidance: "kỳ vọng", "mục tiêu", "dự báo doanh thu"
- ma: "mua lại", "sáp nhập", "thâu tóm"
- dividend: "cổ tức", "chia cổ phần"

---

### Stage 4: Sentiment Scoring (with Confidence)

**Output:** `(score, confidence)` where score ∈ [-1, 1], confidence ∈ [0, 1].

- **score:** Lexicon-based (existing) or future PhoBERT.
- **confidence:** Based on term count and extremity.
  - `confidence = min(1, (pos_count + neg_count) / 3) * (0.7 + 0.3 * abs(score))`

---

### Stage 5: Impact Horizon Estimation

| Horizon | Keywords / Heuristics |
|---------|------------------------|
| **intraday** | "trong phiên", "mở cửa", "đóng cửa", breaking news (<6h old) |
| **short_term** | "tuần tới", "tháng này", earnings, guidance |
| **long_term** | "năm nay", "kế hoạch", "chiến lược", M&A |

---

### Stage 6: Stock-Level Aggregation

**Output:** Single signal per symbol.

```
stock_signal = Σ (relevance_i * sentiment_i * horizon_weight_i) / Σ relevance_i
```

- `horizon_weight`: intraday=1.2, short_term=1.0, long_term=0.8
- Only articles with `relevance >= 0.2` and `source_tier != "dropped"`

---

## 2. Data Structures

### EnrichedArticle (per article, per ticker)

```python
@dataclass
class EnrichedArticle:
    article_id: int
    ticker: str
    title: str
    summary: str
    url: str
    source: str
    published_at: Optional[str]
    # Enrichments
    source_tier: str           # "stock" | "sector" | "macro" | "regulatory" | "dropped"
    event_type: str            # "earnings" | "legal" | "macro" | ...
    ticker_relevance: float    # 0–1
    sentiment_score: float     # -1 to 1
    sentiment_confidence: float  # 0–1
    impact_horizon: str        # "intraday" | "short_term" | "long_term"
```

### StockNewsSignal (aggregated)

```python
@dataclass
class StockNewsSignal:
    symbol: str
    composite_score: float      # -1 to 1, aggregated
    article_count: int
    avg_sentiment: float
    avg_relevance: float
    horizon_breakdown: Dict[str, int]   # intraday: 2, short_term: 5, ...
    event_breakdown: Dict[str, int]     # earnings: 1, macro: 3, ...
    top_articles: List[EnrichedArticle]  # top 5–10 by relevance
```

---

## 3. Scoring Formulas (Summary)

| Metric | Formula |
|--------|---------|
| **ticker_relevance** | 0.4×mention + 0.25×event + 0.2×recency + 0.15×source |
| **sentiment_confidence** | min(1, term_count/3) × (0.7 + 0.3×\|score\|) |
| **composite_score** | Σ(relevance × sentiment × horizon_weight) / Σ relevance |

---

## 4. API Interface

### GET /api/stock/{symbol}/news/intelligence

**Query params:** `days=30`, `min_relevance=0.2`, `event_type=earnings` (optional filter)

**Response:**
```json
{
  "symbol": "FPT",
  "signal": {
    "composite_score": 0.35,
    "article_count": 12,
    "avg_sentiment": 0.28,
    "avg_relevance": 0.62,
    "horizon_breakdown": {"intraday": 1, "short_term": 6, "long_term": 5},
    "event_breakdown": {"earnings": 2, "operations": 4, "macro": 3, "other": 3}
  },
  "articles": [
    {
      "article_id": 123,
      "title": "...",
      "summary": "...",
      "url": "...",
      "source": "cafef",
      "published_at": "2025-02-04T...",
      "event_type": "earnings",
      "ticker_relevance": 0.85,
      "sentiment_score": 0.45,
      "sentiment_confidence": 0.82,
      "impact_horizon": "short_term"
    }
  ]
}
```

### GET /api/stock/{symbol}/news/signal

**Lightweight:** Just the aggregated signal (no article list).

```json
{
  "symbol": "FPT",
  "composite_score": 0.35,
  "article_count": 12,
  "avg_sentiment": 0.28,
  "avg_relevance": 0.62,
  "horizon_breakdown": {...},
  "event_breakdown": {...}
}
```

---

## 5. Database Schema Additions

```sql
CREATE TABLE IF NOT EXISTS article_enrichments (
    article_id INTEGER PRIMARY KEY,
    source_tier TEXT NOT NULL,      -- stock, sector, macro, regulatory
    event_type TEXT NOT NULL,       -- earnings, legal, macro, ...
    impact_horizon TEXT NOT NULL,   -- intraday, short_term, long_term
    sentiment_confidence REAL,      -- 0-1
    updated_at TEXT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

-- Ticker-specific relevance: extend article_tickers or new table
-- Option: add ticker_relevance_score to a new table
CREATE TABLE IF NOT EXISTS article_ticker_scores (
    article_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    relevance_score REAL NOT NULL,   -- 0-1
    PRIMARY KEY (article_id, ticker),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
```

---

## 6. Implementation Order

1. DB migrations (article_enrichments, article_ticker_scores)
2. Source filter + event type + impact horizon classifiers
3. Ticker relevance + sentiment confidence scoring
4. Enrichment pipeline step (run after sentiment + align)
5. Stock-level aggregation service
6. API endpoints

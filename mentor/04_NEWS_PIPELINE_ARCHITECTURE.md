# Vietnamese Stock News Pipeline – Architecture

## 1. Full architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VIETNAMESE STOCK NEWS PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  SOURCES                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ CafeF       │  │ Vietstock   │  │ Other (ext.) │   Config: configs/news.yaml
│  │ (noisy, big)│  │ (StockVN)   │  │              │                         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                         │
│         │                │                │                                  │
│         ▼                ▼                ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ CRAWL (daily / scheduled)                                              │  │
│  │  - Fetch list pages → article URLs                                    │  │
│  │  - Fetch article body per URL                                         │  │
│  │  - Dedupe by url (idempotent upsert)                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STORAGE (SQLite / Postgres)                                           │  │
│  │  articles(id, source, url, title, body_raw, published_at, created_at)  │  │
│  │  article_cleaned(article_id, body_clean)                               │  │
│  │  sentiments(article_id, score, method)                                 │  │
│  │  article_tickers(article_id, ticker, relevance)                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ CLEAN                                                                 │  │
│  │  - Strip HTML, scripts, ads                                           │  │
│  │  - Normalize whitespace & Unicode (Vietnamese)                        │  │
│  │  - Optional: remove boilerplate (byline, “đọc thêm”)                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ SENTIMENT                                                             │  │
│  │  - Lexicon / rule-based (research baseline)                            │  │
│  │  - Optional: PhoBERT sentiment / fine-tuned model                     │  │
│  │  - Output: score + method stored in DB                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ TICKER ALIGNMENT                                                      │  │
│  │  - VN30 ticker list + company name / alias mapping                    │  │
│  │  - Match in title + body_clean (keyword / regex)                     │  │
│  │  - Store (article_id, ticker, relevance) by date                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ RESEARCH OUTPUT                                                       │  │
│  │  - Query: news by (ticker, date range)                                │  │
│  │  - Join with price data for event-aware / sentiment studies           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Design choices (research-grade)

- **Single DB**: One database for raw + cleaned + sentiment + tickers; no demo-only in-memory state.
- **Idempotency**: Crawl upserts by `(source, url)`; re-runs do not duplicate.
- **Config-driven**: Sources, DB path, crawl limits in `configs/news.yaml` (and env for secrets if needed).
- **Logging**: Structured logs (source, step, counts, errors) for reproducibility.
- **Extensibility**: New source = new crawler module + config entry; pipeline unchanged.

## 3. Making it real-time later

1. **Same pipeline, higher frequency**  
   Run the same crawl → clean → sentiment → align → store pipeline on a schedule (e.g. cron every 15–30 minutes). Only new URLs are inserted (upsert by url).

2. **Incremental crawl**  
   Each source crawler accepts a “since” time or “last N pages”. Store `last_crawled_at` per source and pass it next run so only new articles are fetched.

3. **No change to DB or downstream**  
   Research code continues to query by `(ticker, date)`. Real-time just fills the DB more frequently.

4. **Optional: queue-based**  
   For very low latency: a watcher pushes new article URLs to a queue (e.g. Redis); workers run clean → sentiment → align → store. Schema and research queries stay the same.

5. **Rate limiting**  
   Respect robots.txt and add per-source delays to avoid blocks when increasing frequency.

## 4. File layout (implemented)

```
configs/news.yaml
src/news/
  __init__.py
  db.py           # Schema, init, insert, query
  clean.py        # Text cleaning
  sentiment.py    # Sentiment extraction
  ticker_align.py # Align to VN30 tickers
  crawlers/
    __init__.py
    base.py      # Abstract crawler interface
    cafef.py     # CafeF news
    vietstock.py # Vietstock (StockVN-style)
  pipeline.py    # Orchestrate: crawl → clean → sentiment → align
scripts/run_news_pipeline.py
```

## 5. How to make it real-time later

- **Same code, higher frequency:** Run `python scripts/run_news_pipeline.py` on a schedule (cron/Task Scheduler every 15–30 minutes). Upsert by `(source, url)` keeps the DB deduplicated.
- **Incremental crawl:** Add a “since” or “last N pages” parameter to each crawler and store `last_crawled_at` per source in config or a small table; pass it into `fetch_article_list` so only new pages are requested.
- **Rate limiting:** Keep `request_delay_seconds` and respect robots.txt when increasing frequency.
- **Optional queue:** For sub-minute latency, a separate watcher can push new URLs to Redis/RabbitMQ; workers run the same clean → sentiment → align → store. Schema and research queries stay unchanged.

# src/news/

**Purpose:** Full news intelligence path: ingest (crawl, clean, sentiment, align, enrich), store in SQLite, and aggregate into per-ticker signals with explainability. Single responsibility: turn raw news into stored, enriched articles and signal outputs.

**Fits into system:** News crawlers here are separate from `src/crawl` (price). Pipeline runs on a schedule from `api.py`. Engine output is consumed by `src/app_services/news_intelligence` and API.

**Data in/out:**
- **In:** Config (sources, DB path); HTTP from news sites.
- **Out:** SQLite `news.db`; `StockNewsSignal`, `DailyNewsSignal`, ML feature rows from engine.

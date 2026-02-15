# src/crawl/

**Purpose:** Fetch **price** (OHLCV) data from CafeF only. Does not crawl news (news crawlers live in `src/news/crawlers/`).

**Fits into system:** Used by `src/pipeline/runcrawler/run_crawler.py` and `src/app_services/market_data.py` when live price or history is needed. Output is raw CSVs or DataFrames for the price pipeline.

**Data in/out:**
- **In:** Symbol(s), date range, save path.
- **Out:** CSV files in `data/raw/` or DataFrame(s).

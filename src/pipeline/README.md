# src/pipeline/

**Purpose:** Orchestrate the **price** data pipeline (crawl → clean → features) for VN30. Does not run the news pipeline (news is in `src/news/pipeline.py`).

**Fits into system:** `vnindex30/fetch_vn30.py` is called by the API scheduler. It uses `runcrawler/run_crawler.py` (crawl), `src/clean`, and `src/features` to produce raw → processed → feature data.

**Data in/out:**
- **In:** Config (symbols, dates, paths).
- **Out:** Updated `data/raw`, `data/processed`, and feature outputs.

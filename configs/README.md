# configs/

**Purpose:** YAML configuration for the whole system. Single source of truth for symbols, paths, pipeline options, and news sources.

**Fits into system:** Loaded by `api.py` (scheduler), `src/news/pipeline.py`, `src/pipeline/vnindex30/fetch_vn30.py`, and inference/decision code. No business logic here.

**Data in/out:**
- **In:** None (files are read at runtime).
- **Out:** `config.yaml` → pipeline dates, paths, crawl/clean/feature options; `symbols.yaml` → VN30 list; `news.yaml` → DB path, sources, crawl_interval_minutes; `decision.yaml` → decision thresholds if used.

# 03 — Multimodal (News)

**What this folder does:** Covers how news data is used in TechPulse: sources, ingestion, and conversion from raw articles to a per-ticker news signal.

**Fits into the system:** Maps to `src/news/` (crawl, clean, sentiment, align, enrich, engine) and `src/app_services/news_intelligence.py` (facade). See also root [NEWS_INTELLIGENCE_ENGINE.md](../NEWS_INTELLIGENCE_ENGINE.md) for the engine API and design.

**Data in/out:** Input: HTTP from news sites; config (sources, DB path). Output: SQLite `news.db`; `StockNewsSignal`, `DailyNewsSignal`, and ML-ready daily features from the engine.

**Files:**
- `01_NEWS_DATA_VIETNAM.md` — News sources and data in Vietnam.
- `02_AI_NEWS_ANALYSIS.md` — How AI/news analysis is applied in this system (sentiment, event type, aggregation).

**Order:** After 02_modeling (optional to do in parallel). Read with [NEWS_INTELLIGENCE_ENGINE.md](../NEWS_INTELLIGENCE_ENGINE.md) for full design.

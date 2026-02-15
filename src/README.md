# src/

**Purpose:** All application and pipeline code. No UI assets (those live in `web/`).

**Fits into system:** Root of the Python package. `api.py` and scripts import from `src.*`. Subpackages: `app_services` (API-facing logic), `news` (news ingest + engine), `pipeline` (price pipeline), `crawl`, `clean`, `features` (price data path), `evaluation`, `models`, `ensemble`, `inference`, `risk_engine`, `utils`.

**Data in/out:**
- **In:** Configs, `data/` files, news DB.
- **Out:** Signals, recommendations, forecast cache, and in-memory responses to API.

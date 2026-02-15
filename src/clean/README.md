# src/clean/

**Purpose:** Clean and validate **price** data (OHLCV CSVs). Deduplicate, handle missing values, optional normalization. Used only by the price pipeline, not by news.

**Fits into system:** Called from `src/pipeline/vnindex30/fetch_vn30.py` after crawl. Output is consumed by `src/features/build_features`.

**Data in/out:**
- **In:** Raw CSV paths (e.g. `data/raw/*.csv`).
- **Out:** Cleaned CSVs (e.g. `data/processed/*.csv`) or in-memory DataFrames.

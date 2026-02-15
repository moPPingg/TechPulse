# src/features/

**Purpose:** Build **price** technical features (returns, MAs, RSI, volatility, etc.) from cleaned OHLCV. Used only in the price pipeline.

**Fits into system:** Called from `src/pipeline/vnindex30/fetch_vn30.py` after clean. Output feeds evaluation and forecasting models.

**Data in/out:**
- **In:** Cleaned price DataFrames or CSV paths.
- **Out:** Feature DataFrame (or saved CSVs) with target column for ML.

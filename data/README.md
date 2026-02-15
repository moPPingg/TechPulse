# data/

**Purpose:** All persisted data (raw price CSVs, processed/cleaned CSVs, feature CSVs, forecast cache, and news DB path). Config points here; code reads/writes via paths in config.

**Fits into system:** Downstream of crawl/clean/features (price) and news pipeline (news DB). Upstream of evaluation (features), inference (forecasts), and API (chart data, news).

**Data in/out:**
- **In:** Raw CSVs from price crawler; cleaned CSVs from clean step; feature CSVs from build_features; news DB written by `src/news`.
- **Out:** Features and forecasts consumed by inference and backtest; news DB queried by news engine; chart data from processed/features.

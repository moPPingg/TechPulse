# src/inference/

**Purpose:** Load trained models (or train on demand), run inference per symbol, cache forecast to disk. Exposes `get_forecast(symbol)` for API and signal_aggregator.

**Fits into system:** Consumes evaluation (splits, data, metrics) and models/ensemble. Called by `src/app_services/signal_aggregator` and API stock endpoints.

**Data in/out:**
- **In:** Symbol; feature data from disk; model artifacts.
- **Out:** Cached JSON per symbol (mean, std, confidence); ForecastResult in memory.

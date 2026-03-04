# src/app_services/

**Purpose:** Application-facing services used by the API and recommendation flow. Aggregates lower-level modules (news, inference, risk) into signals and final recommendation.

**Fits into system:** Sits between API and domain layers. Calls `src.news` (engine/news_intelligence), `src.inference`, `src.risk_engine`, `src.app_services.market_data`. Exposes `get_risk_advice`, `aggregate`, `get_articles`, `get_stock_news_signal`, etc.

**Data in/out:**
- **In:** Symbol, user profile; reads forecasts, news signals, risk metrics, market data.
- **Out:** Recommendation (BUY/HOLD/AVOID), decision explanation, aggregated signals, article lists, chart data.

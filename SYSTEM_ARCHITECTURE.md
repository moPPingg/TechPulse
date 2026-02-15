# TechPulse — System Architecture

This document describes the **full data flow** from raw inputs (news + price data) through processing, signals, ML, API, and frontend. Every transformation step is explained.

---

## 1. High-Level Data Flow

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                     EXTERNAL SOURCES                               │
                    │  News: CafeF, Vietstock, VNExpress, HSX, ...  │  Price: CafeF      │
                    └───────────────────────────┬─────────────────┴──────────┬──────────┘
                                                │                            │
         ┌─────────────────────────────────────▼────────────────────────────▼─────────────────────────────────────┐
         │                                        INGEST LAYER                                                         │
         │  News: src/news/pipeline (crawl → clean → sentiment → align → enrich)  →  SQLite (data/news/news.db)     │
         │  Price: src/pipeline/vnindex30 (crawl_many → clean_many → build_features)  →  data/raw, data/processed   │
         └───────────────────────────┬────────────────────────────────────────────┬─────────────────────────────────┘
                                     │                                              │
         ┌───────────────────────────▼──────────────────────────────────────────────▼─────────────────────────────────┐
         │                                        STORED ARTIFACTS                                                     │
         │  • news.db: articles, article_cleaned, sentiments, article_tickers, article_enrichments, article_ticker_scores │
         │  • data/raw/*.csv: raw OHLCV  •  data/processed/*.csv: cleaned  •  data/features/*.csv: technical features   │
         │  • data/forecasts/*.json: cached model forecasts per symbol                                                │
         └───────────────────────────┬────────────────────────────────────────────┬─────────────────────────────────┘
                                     │                                              │
         ┌───────────────────────────▼──────────────────┐    ┌─────────────────────▼───────────────────────────────┐
         │         NEWS INTELLIGENCE ENGINE               │    │              PRICE / ML PIPELINE                       │
         │  src/news/engine.py                           │    │  src/evaluation/data → features + target                │
         │  • get_signal(symbol) → composite, reasoning  │    │  src/models/forecasting/* → train & predict            │
         │  • get_daily_signals(symbol, from, to)        │    │  src/ensemble/aggregator → ensemble forecast            │
         │  • get_ml_daily_features(symbol, from, to)    │    │  src/inference/service → cached forecast per symbol     │
         │  Input: DB rows  Output: StockNewsSignal,     │    │  src/risk_engine → P(loss), P(ruin), VaR                 │
         │  DailyNewsSignal, flat feature rows           │    │  Input: features  Output: mean, std, risk metrics        │
         └───────────────────────────┬──────────────────┘    └─────────────────────┬───────────────────────────────────┘
                                     │                                              │
                                     └──────────────────────┬───────────────────────┘
                                                            │
         ┌──────────────────────────────────────────────────▼──────────────────────────────────────────────────────────┐
         │                                    APPLICATION SERVICES (src/app_services)                                   │
         │  • news_intelligence: facade over news engine (get_stock_news_signal, get_ticker_daily_signals)              │
         │  • news_service: article list + relevance for UI                                                             │
         │  • market_data: OHLCV + indicators for chart                                                                   │
         │  • signal_aggregator: combines forecast + risk + news_sentiment → one struct per symbol                       │
         │  • signals/*: price_technical, ml_forecast, news_event, risk_uncertainty (each produces a signal object)      │
         │  • decision_engine: maps signals + user profile → decision (e.g. BUY/HOLD/AVOID)                               │
         │  • recommendation: get_risk_advice(profile, symbol) → full recommendation + explanation                       │
         └──────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                            │
         ┌──────────────────────────────────────────────────▼──────────────────────────────────────────────────────────┐
         │                                         API LAYER (api.py — FastAPI)                                          │
         │  GET /  → static web  │  GET /api/symbols  │  POST /api/recommend  │  GET /api/stock/{symbol}  │  ...        │
         │  All responses are JSON (or HTML for /). Downstream of API: no direct DB or model access.                   │
         └──────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                            │
         ┌──────────────────────────────────────────────────▼──────────────────────────────────────────────────────────┐
         │                                         FRONTEND (web/)                                                       │
         │  index.html + static/css, static/js  →  Consumes API only. Displays recommendation, chart, news, signals.   │
         └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Transformation Steps (Detail)

### 2.1 Raw news → stored articles

| Step | Where | Input | Output |
|------|--------|--------|--------|
| Crawl | `src/news/pipeline.run_crawl` | Config (sources, categories) | HTTP → article list (url, title, body_raw, published_at, source) |
| Normalize URL | In pipeline before insert | Relative URL + source | Absolute URL |
| Dedupe | `src/news/db.find_article_by_content_hash` | content_hash (SHA-1 of title+body) | Skip if hash exists |
| Insert | `src/news/db.insert_article` | One article record | articles.id |
| Clean | `src/news/pipeline.run_clean` | body_raw | body_clean (strip HTML, normalize whitespace) → article_cleaned |
| Sentiment | `src/news/pipeline.run_sentiment` | title+body | score, method → sentiments |
| Align | `src/news/pipeline.run_align` | title+body_clean + symbols | (ticker, relevance) → article_tickers |
| Enrich | `src/news/pipeline.run_enrich` | article + sentiment + tickers | event_type, impact_horizon, sentiment_confidence → article_enrichments; relevance_score → article_ticker_scores |

**Data exit:** SQLite `news.db` with all tables populated for recent articles.

---

### 2.2 Stored articles → news signals (per ticker)

| Step | Where | Input | Output |
|------|--------|--------|--------|
| Query | `src/news/db.get_enriched_articles_for_ticker` | symbol, date_from, date_to, min_relevance | Rows: article_id, url, title, sentiment_score, event_type, impact_horizon, ticker_relevance_score |
| Aggregate | `src/news/engine.get_signal` | Rows | weight = ticker_relevance × horizon_weight; composite = Σ(sentiment×weight)/Σ(weight) |
| Explainability | Same | Same rows | Per-article: contribution_weight, raw_contribution; top_contributors; reasoning string |
| Market shock | `src/news/engine.detect_market_shock` | Same DB, last 24h vs 7d baseline | MarketShockResult (sentiment_spike or rare_events) |

**Data exit:** `StockNewsSignal` (composite_score, reasoning, top_articles with contribution math, market_shock). Used by API and recommendation.

---

### 2.3 Raw price → features → ML forecast

| Step | Where | Input | Output |
|------|--------|--------|--------|
| Crawl | `src/crawl/cafef_scraper.fetch_price_cafef` | symbol, start_date, end_date | CSV in data/raw |
| Clean | `src/clean/clean_price.clean_many` | raw CSV paths | Dedupe, validate → data/processed |
| Features | `src/features/build_features.build_features` | processed CSV | Returns, MAs, RSI, volatility, etc. → data/features (or in-memory) |
| Splits | `src/evaluation/splits` | Feature DataFrame | train/val/test by time (no shuffle) |
| Train | `src/models/forecasting/*` | X_train, y_train | Fitted model (e.g. XGBoost, LSTM) |
| Predict | Same models | X_test / live features | Mean forecast (e.g. return_1d) |
| Ensemble | `src/ensemble/aggregator` | Per-model forecasts | Weighted mean + std |
| Cache | `src/inference/service` | symbol | Read/write data/forecasts/{symbol}.json |
| Risk | `src/risk_engine/risk` | forecast mean + std + horizon | P(loss), P(ruin), VaR |

**Data exit:** Forecast (ensemble_mean, ensemble_std), risk metrics. Consumed by signal_aggregator and recommendation.

---

### 2.4 Signals + profile → recommendation

| Step | Where | Input | Output |
|------|--------|--------|--------|
| Gather signals | `src/app_services/signal_aggregator.aggregate` | symbol, UserProfile | Forecast, volatility, risk, news_sentiment, model_weights |
| Decide | `src/app_services/decision_engine.decide` | Signals + profile | Primary signal, blocking/supporting factors, action |
| Explain | `src/app_services/recommendation.get_risk_advice` | profile, symbol | Recommendation (BUY/HOLD/AVOID), explanation, decision_explanation, data_freshness |

**Data exit:** Recommendation response (JSON). Consumed by API and then frontend.

---

### 2.5 API → frontend

| Step | Where | Input | Output |
|------|--------|--------|--------|
| HTTP | api.py routes | GET/POST + params | JSON (symbols, recommendation, stock detail, chart, news) |
| Static | FastAPI StaticFiles | web/static/*, web/index.html | CSS, JS, HTML |
| Client | web/static/js/app.js | User actions | Fetches API, renders DOM |

**Data exit:** User sees recommendation, chart, news list, and explainability text.

---

## 3. Module Responsibility Summary

| Folder / module | Single responsibility |
|-----------------|------------------------|
| **api.py** | HTTP entrypoint; schedule jobs; serve static and JSON. |
| **configs/** | YAML configuration (symbols, pipeline, news, decision). |
| **src/news/** | Ingest news (crawl, clean, sentiment, align, enrich); store in SQLite; engine aggregates to signals. |
| **src/crawl/** | Fetch OHLCV from CafeF (price data only). |
| **src/clean/** | Clean price CSVs (dedupe, validate). |
| **src/features/** | Build technical features from cleaned price. |
| **src/pipeline/** | Orchestrate price pipeline (vnindex30) and price crawler (runcrawler). |
| **src/evaluation/** | Splits, metrics, backtest, data loaders for ML. |
| **src/models/forecasting/** | Train and predict (linear, XGBoost, ARIMA, LSTM, PatchTST, Transformer). |
| **src/ensemble/** | Aggregate model forecasts with weights. |
| **src/inference/** | Load models, run inference, cache forecast per symbol. |
| **src/risk_engine/** | Compute P(loss), P(ruin), VaR from forecast distribution. |
| **src/app_services/** | Combine signals, decision engine, recommendation, market_data, news facade. |
| **web/** | Static frontend; calls API only. |

---

## 4. Scheduler (Runtime)

- **News pipeline:** Every N minutes (config: `crawl_interval_minutes`). Runs full pipeline: crawl → clean → sentiment → align → enrich. Only unprocessed articles are touched (idempotent).
- **Data pipeline (price):** Every 4 hours. Runs VN30 price crawl → clean → features. Config: `configs/config.yaml` (start_date, paths).

Both are started in `api.py` lifespan; no separate process required for continuous ingest.

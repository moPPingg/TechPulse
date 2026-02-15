# News Intelligence Engine — Architecture, Tradeoffs, Design Decisions

This document describes the **production-grade News Intelligence Engine**: layers, public API, how outputs serve ML models, trading strategies, and dashboards, and the main design tradeoffs.

---

## 1. Requirements (Summary)

1. **Ingest** news articles continuously (streaming-style).
2. **Normalize, deduplicate, enrich** each article with:
   - Ticker relevance score  
   - Sentiment score + confidence  
   - Event type  
   - Impact horizon + horizon_weight  
3. **Aggregate** articles into **per-ticker daily signals**.
4. **Explainability**: top contributing articles and per-article contribution math.
5. **Output** usable by ML models, trading strategies, and dashboards.

---

## 2. Architecture (Layers)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INGEST                                                                  │
│  Crawl sources → normalize URL → content-hash dedupe → persist           │
│  (Pipeline: crawl → clean → sentiment → align → enrich)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  NORMALIZE & DEDUPLICATE                                                 │
│  • URL: relative → absolute by source base URL                           │
│  • Dedupe: (source, url) at DB; cross-source by content_hash (SHA-1)     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ENRICH                                                                  │
│  Per article: ticker relevance, sentiment + confidence, event_type,     │
│  impact_horizon, horizon_weight (src.news.intelligence)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AGGREGATE                                                               │
│  • Rolling: composite = Σ(sentiment × weight) / Σ(weight)               │
│    weight = ticker_relevance × horizon_weight                           │
│  • Daily: group by date, same formula per day                            │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT (explainability + consumers)                                     │
│  • Dashboards: reasoning, top_contributors, per-article breakdown         │
│  • Trading: composite_score, net_impact_label, market_shock             │
│  • ML: get_ml_daily_features() → flat rows (date, symbol, features)     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Map

| Component | Role | Location |
|-----------|------|----------|
| **Models** | Canonical output types (EnrichedArticleView, StockNewsSignal, DailyNewsSignal, etc.) | `src/news/models.py` |
| **Engine** | Ingest, aggregate, explainability, ML features API | `src/news/engine.py` |
| **Pipeline** | Crawl, clean, sentiment, align, enrich (idempotent steps) | `src/news/pipeline.py` |
| **DB** | Schema, insert/query, content_hash dedupe | `src/news/db.py` |
| **Intelligence** | Event type, horizon, relevance formula, why_it_matters text | `src/news/intelligence.py` |
| **News Intelligence (facade)** | App-facing API; delegates to engine; re-exports types | `src/app_services/news_intelligence.py` |

---

## 4. Public API (Engine)

- **`run_ingest(config_path=None, steps=None)`**  
  Runs pipeline (crawl → clean → sentiment → align → enrich). Returns `{ step_name: count }`.  
  Used by: scheduler (continuous ingest), scripts.

- **`get_signal(symbol, days=30, min_relevance=0.2, limit_articles=10, ...)`**  
  Returns **StockNewsSignal**: composite_score, reasoning, top_contributors, top_articles (with per-article contribution_weight, raw_contribution), market_shock.  
  Used by: dashboards, API `/api/stock/{symbol}/news/intelligence`, recommendation layer.

- **`get_daily_signals(symbol, from_date, to_date, ...)`**  
  Returns **List[DailyNewsSignal]**: one per date with composite_score and list of articles (each with contribution math).  
  Used by: API `/api/stock/{symbol}/news/daily`, backtests, time-series views.

- **`get_ml_daily_features(symbol, from_date, to_date, ...)`**  
  Returns **List[dict]**: one row per (symbol, date) with composite_score, article_count, avg_sentiment, avg_relevance, event_breakdown, horizon_breakdown.  
  Used by: ML feature tables, forecasting models, strategy research.

- **`detect_market_shock(conn, symbol, hours=24, ...)`**  
  Returns **MarketShockResult**: is_shock, reason (sentiment_spike | rare_events), contributing_article_titles.  
  Used by: get_signal (embedded in StockNewsSignal), alerts.

---

## 5. Explainability (Per-Article Contribution Math)

Every article in the engine output exposes:

- **url**, **source**, **published_at** — traceability and link to original.
- **ticker_relevance**, **impact_horizon**, **horizon_weight** — how the article is weighted.
- **sentiment_score**, **sentiment_confidence** — sentiment and its confidence.
- **contribution_weight** = `ticker_relevance × horizon_weight`.
- **raw_contribution** = `sentiment_score × contribution_weight`.

Composite score formula:

```
composite_score = Σ(raw_contribution) / Σ(contribution_weight)
```

So the **contribution math** is explicit: each article’s share of the total weighted sum is visible. Top contributors are chosen by impact score `relevance × |sentiment| × horizon_weight` and exposed as **top_contributors** and **top_3_impact** (with “why it matters” text for dashboards).

---

## 6. Design Decisions and Tradeoffs

| Decision | Rationale | Tradeoff |
|----------|------------|----------|
| **Single DB (SQLite)** | One source of truth; no extra infra; good for research and moderate scale. | Concurrency and scale ceiling; for very high throughput, would add a queue + worker or move to PostgreSQL. |
| **Idempotent pipeline** | Only process articles missing a step (clean/sentiment/align/enrich). Safe to run on a schedule (e.g. every 15 min). | Slightly more DB reads to find “missing” rows; acceptable for current volume. |
| **Content-hash dedupe** | SHA-1 of normalized title+body avoids storing the same story from multiple sources. | Small risk of hash collision; acceptable for text. Normalization (e.g. whitespace) is shared with cleaning. |
| **Explainability first-class** | Every article has contribution_weight and raw_contribution; reasoning string and top_contributors are part of the API. | Slightly more computation and payload size; necessary for trust and debugging. |
| **Market shock = heuristic** | No trained model: sentiment spike vs 7d baseline, or rare event types (legal/ma/macro) in 24h. | Fast, interpretable; may miss complex “shocks” that an ML classifier would catch. |
| **Types in `src/news/models.py`** | Single contract for engine output so ML, trading, and dashboards share the same shapes. | App layer (news_intelligence) re-exports these types so existing imports remain valid. |
| **Facade in app_services** | `news_intelligence` delegates to engine and re-exports types. API and recommendation code keep importing from `news_intelligence`. | Clear separation: engine = core, app_services = HTTP/app adapter. |

---

## 7. Output Usability by Consumer

- **ML models**: Use **get_ml_daily_features(symbol, from_date, to_date)** for flat feature rows; optionally use **get_daily_signals** if you need per-article vectors or contribution breakdown in the model.
- **Trading strategies**: Use **get_signal(symbol)** for composite_score, net_impact_label, market_shock; use **get_daily_signals** for daily series and backtests.
- **Dashboards**: Use **get_signal** for reasoning, top_contributors, top_3_impact, and top_articles (each with url, contribution_weight, raw_contribution); use **get_daily_signals** for time-series and per-day breakdown.

---

## 8. File Layout (Refactor Summary)

- **`src/news/models.py`** — New. Canonical dataclasses only.
- **`src/news/engine.py`** — New. Ingest, aggregate, explainability, ML features; uses pipeline, db, intelligence.
- **`src/news/intelligence.py`** — Extended with `why_it_matters_text`, `HORIZON_LABELS` for engine/dashboards.
- **`src/app_services/news_intelligence.py`** — Refactored to a thin facade: re-exports models and delegates get_stock_news_signal, get_ticker_daily_signals, detect_market_shock to engine.
- **`src/news/pipeline.py`**, **`src/news/db.py`** — Unchanged in contract; engine and pipeline use them as-is.

Existing callers (api.py, recommendation.py, signals/news_event.py) continue to import from **src.app_services.news_intelligence**; no change required.

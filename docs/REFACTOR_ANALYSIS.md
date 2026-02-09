# TechPulse Architecture Analysis — Current Weaknesses

> **Purpose:** Identify architectural weaknesses before refactoring into a professional-grade financial analytics product.

---

## 1. Executive Summary

The current TechPulse codebase is functionally complete for a school demo but exhibits several architectural and UX weaknesses that prevent it from being a **real trading workflow tool**:

1. **News layer** is noisy, unfiltered, and low-value for investment decisions.
2. **UI** feels like a demo form, not a fintech dashboard.
3. **Model outputs** are shown but not explained in decision terms (why Buy/Hold/Avoid).
4. **No clear separation** between Market Data, Signals, News, and Final Recommendation.

---

## 2. Current Architecture Overview

```
api.py (FastAPI) ──┬── POST /api/recommend ──► get_risk_advice() ──► recommendation.py
                   │                                    │
                   │                                    ├── get_forecast()      (inference.service)
                   │                                    ├── get_news_sentiment() (news.db, recommendation)
                   │                                    ├── compute_risk_metrics (risk_engine)
                   │                                    └── build_explanation / build_conclusion
                   │
                   └── GET /api/stock/{symbol} ──► _load_features, get_news_sentiment, get_forecast
                         └── Returns: forecast, news, indicators (flat JSON)
```

- **Single monolithic API file** (`api.py`): ~440 lines; mixes routing, data loading, indicator computation, and chart formatting.
- **Recommendation engine** (`recommendation.py`): ~460 lines; combines forecast, news, risk, decision, explanation. Tightly coupled.
- **Frontend** (`app.js`): Single 430-line file; form → result → detail panel. No component separation.

---

## 3. Detailed Weaknesses

### 3.1 News Layer — Noisy, Unfiltered, Low-Value

| Issue | Location | Impact |
|-------|----------|--------|
| **No investment relevance scoring** | `get_news_sentiment()` | All articles treated equally; no prioritization of earnings, M&A, regulatory vs. generic fluff. |
| **Weak ticker alignment** | `ticker_align.py` | Many articles tagged `__NONE__`; fallback shows symbol-in-text match (LIKE '%VNM%') → false positives. |
| **No summarization for decisions** | `_summarize_text()` | Extractive 2-sentence summary; no "investment impact" angle. |
| **Mixed source types** | News DB | TradingEconomics (macro indicators), SSC (regulatory), CafeF (stock news) mixed together without type filtering. |
| **No recency weighting** | DB queries | Week-old news same weight as yesterday's. |

**Design gap:** News should be **scored for investment relevance** (e.g., earnings, management change, sector news) and **ranked**, not just filtered by ticker.

---

### 3.2 UI — Demo Feel, Not Fintech

| Issue | Location | Impact |
|-------|----------|--------|
| **Form-first flow** | `index.html` | User fills form before seeing anything; no market overview or watchlist. |
| **Recommendation buried** | Result card | Buy/Hold/Avoid appears among metrics; no prominent "Action" section. |
| **No clear sections** | `app.js` | Chart, indicators, news in one long panel; no tabs or collapsible regions. |
| **Inter font** | `style.css` | Generic; fintech typically uses JetBrains Mono for numbers, distinctive sans for headings. |
| **Max-width 820px** | `.app` | Constrained; dashboards need sidebar + main content, wider charts. |
| **No dark mode** | — | Traders often prefer dark UIs. |

**Design gap:** A real dashboard should have: **sidebar (symbols/watchlist)**, **main (chart + signals + recommendation)**, **news feed** in a dedicated panel with relevance badges.

---

### 3.3 Model Outputs — Shown, Not Explained

| Issue | Location | Impact |
|-------|----------|--------|
| **Explanation is paragraph** | `build_explanation()` | Long prose; hard to scan. Traders want bullet points: "Forecast: +0.5% → Bullish. Risk: 45% P(loss) → Caution." |
| **Signal breakdown minimal** | `signal_breakdown` | trend, volatility, news count; no per-model contribution ("LSTM says +0.3%, XGBoost says -0.1%"). |
| **Conclusion buried** | `conclusion` | advice, situation, why_not_buy, market_analysis — all in one block; not decision-structured. |
| **No "why not Buy" when Hold** | `build_conclusion()` | Generic "chờ tín hiệu rõ ràng"; doesn't say *which* signal is blocking. |

**Design gap:** Outputs should be **structured for decisions**: `{ primary_signal, blocking_factors, supporting_factors, action_summary }`.

---

### 3.4 No Strong Separation of Concerns

| Concern | Current State | Desired State |
|---------|---------------|---------------|
| **Market data** | Mixed in `api.py`: `_load_features`, `_fetch_price_from_cafef`, `_compute_indicators` | Dedicated `MarketDataService` with single entry point |
| **Signals** | Spread across `get_risk_advice`, `get_forecast`, `get_news_sentiment` | `SignalAggregator`: forecast, risk, news → structured `Signals` |
| **News** | `get_news_sentiment` in recommendation; DB access in app_services | `NewsService`: relevance scoring, filtering, summarization |
| **Recommendation** | Monolithic `get_risk_advice` does everything | `RecommendationEngine`: inputs `Signals` + `UserProfile`, outputs `RiskAdvice` + `DecisionExplanation` |

**Design gap:** Clear **layer contracts** and **dependency injection** so each module can be tested and extended independently.

---

### 3.5 Data Layer Coupling

| Issue | Location | Impact |
|-------|----------|--------|
| **api.py imports from 10+ modules** | Top of `api.py` | Hard to mock; integration tests only. |
| **recommendation.py loads config and DB** | `get_news_sentiment`, `_load_config` | Inline config/DB; no injectable `NewsRepository`. |
| **Forecast cache path hardcoded** | `inference/service.py` | Tied to `config.yaml`; no interface for alternate sources. |

---

### 3.6 API Design

| Issue | Location | Impact |
|-------|----------|--------|
| **GET /api/stock/{symbol} returns everything** | `get_stock_detail()` | Forecast + news + indicators in one blob; frontend can't lazy-load. |
| **No pagination for news** | `get_news_sentiment` | Returns top 10; no offset/limit for "load more". |
| **Chart endpoint separate** | `/api/stock/{symbol}/chart` | Good; but indicators duplicated between `/stock` and chart response. |

---

## 4. Summary of Gaps

| Category | Gap |
|----------|-----|
| **News** | Relevance scoring, type filtering, investment-impact summarization |
| **UI** | Dashboard layout, sidebar, clear sections, decision-focused presentation |
| **Explanations** | Structured decision format, blocking factors, per-model contribution |
| **Architecture** | Modular services (MarketData, News, Signals, Recommendation), clear contracts |
| **API** | Separation of market/signals/news; optional pagination |

---

## 5. Next Steps

See `docs/REFACTOR_TARGET_ARCHITECTURE.md` for the proposed target architecture and implementation plan.

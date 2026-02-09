# TechPulse Target Architecture — Professional Fintech Product

> **Purpose:** Define the clean target architecture for refactoring TechPulse into a professional-grade financial analytics product.

---

## 1. Design Principles

1. **Modular and extensible:** Each layer has a clear contract; new data sources or models plug in without touching core logic.
2. **Decision-centric:** Outputs explain *why* a recommendation is made, not just metrics.
3. **Fintech-grade UI:** Dashboard layout with clear separation: Market Data | Signals | News | Recommendation.
4. **News for investment:** Filtered, scored, summarized for relevance to trading decisions.

---

## 2. Target Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND — Fintech Dashboard                                        │
│  ┌──────────────┐  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Sidebar      │  │ Main: Chart │ Signals │ News (filtered, scored) │ Recommendation     │  │
│  │ - Watchlist  │  │ Decision: Buy/Hold/Avoid + WHY (blocking factors, supporting)        │  │
│  │ - Symbols    │  └─────────────────────────────────────────────────────────────────────┘  │
│  └──────────────┘                                                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                                              │
│  /api/symbols  /api/stock/{symbol}  /api/stock/{symbol}/signals  /api/stock/{symbol}/news    │
│  /api/stock/{symbol}/chart  POST /api/recommend                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│  MARKET DATA SERVICE    │  │  NEWS SERVICE           │  │  RECOMMENDATION ENGINE  │
│  - load_features()      │  │  - get_articles()       │  │  - get_advice()         │
│  - get_indicators()     │  │  - relevance_score()    │  │  - build_decision()     │
│  - get_chart_data()     │  │  - investment_summary() │  │  Input: Signals + Profile│
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
            │                               │                               │
            └───────────────────────────────┼───────────────────────────────┘
                                            ▼
                            ┌─────────────────────────────┐
                            │  SIGNAL AGGREGATOR          │
                            │  - forecast (from inference)│
                            │  - risk (from risk_engine)  │
                            │  - news_sentiment           │
                            │  Output: Structured Signals │
                            └─────────────────────────────┘
```

---

## 3. Layer Contracts

### 3.1 MarketDataService

**Responsibilities:** Load price data, compute indicators, provide chart-ready series.

| Method | Input | Output |
|--------|-------|--------|
| `get_indicators(symbol, days)` | symbol, days | `{ close, return_1d, volatility_5, rsi_14, ma_20, ma_50, last_date }` |
| `get_chart_data(symbol, days, end_date?)` | symbol, days, optional end | `{ ohlcv, ma, rsi, volatility }` |

**Data sources:** features CSV (primary), CafeF API (fallback). Logic moves out of `api.py` into `src/app_services/market_data.py`.

---

### 3.2 NewsService

**Responsibilities:** Fetch articles, score relevance, summarize for investment decisions.

| Method | Input | Output |
|--------|-------|--------|
| `get_articles(symbol, days, limit, min_relevance?)` | symbol, days, limit | `List[Article]` with `relevance_score`, `investment_summary` |
| `get_sentiment(symbol, days)` | symbol, days | `(avg_sentiment, count)` |

**Article shape:**
```python
@dataclass
class Article:
    id: int
    title: str
    summary: str          # Extractive (existing)
    investment_summary: str  # NEW: 1 sentence "Why this matters for {symbol}"
    url: str
    source: str
    sentiment: float
    relevance_score: float   # NEW: 0–1, based on type + recency + ticker match
    published_at: str
```

**Relevance scoring (simple v1):**
- Ticker in title: +0.3
- Ticker in body: +0.2
- Source = cafef/vietstock (stock-focused): +0.2
- Published within 3 days: +0.2
- Sentiment extreme (|s| > 0.3): +0.1
- Normalize to [0, 1]

---

### 3.3 SignalAggregator

**Responsibilities:** Combine forecast, risk, news into a single `Signals` struct.

```python
@dataclass
class Signals:
    forecast_mean: float
    forecast_std: float
    confidence: float
    volatility_pct: float
    prob_loss_pct: float
    prob_ruin_pct: float
    expected_return_ci: Tuple[float, float]
    news_sentiment: float
    news_count: int
    model_weights: dict  # For explainability
```

**Single entry:** `aggregate(symbol, user_profile) -> Signals`

---

### 3.4 RecommendationEngine

**Responsibilities:** Take Signals + UserProfile → RiskAdvice + DecisionExplanation.

```python
@dataclass
class DecisionExplanation:
    primary_signal: str       # "Forecast: bullish (+0.5%)"
    blocking_factors: List[str]   # "P(loss)=52%"
    supporting_factors: List[str] # "News sentiment positive"
    action_summary: str       # "Có thể cân nhắc mua với tỷ trọng vừa phải."
```

**Refactor:** Extract from current `get_risk_advice`; use `Signals` as input instead of fetching internally.

---

## 4. API Endpoints (Target)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/symbols` | VN30 list |
| `GET /api/stock/{symbol}` | Indicators + forecast summary (light) |
| `GET /api/stock/{symbol}/signals` | Full Signals (forecast, risk, news sentiment) |
| `GET /api/stock/{symbol}/news?limit=10&min_relevance=0.3` | Articles with relevance |
| `GET /api/stock/{symbol}/chart?days=90` | OHLC + MA + RSI |
| `POST /api/recommend` | RiskAdvice + DecisionExplanation |

**Design decision:** Separate `/signals` and `/news` so frontend can lazy-load and display in distinct sections.

---

## 5. UI Structure (Target)

### Layout
```
┌─────────────────────────────────────────────────────────────────┐
│ Header: TechPulse | Date | Server time                          │
├────────────┬────────────────────────────────────────────────────┤
│ Sidebar    │ Main                                                │
│ - Symbols  │ ┌────────────────────────────────────────────────┐ │
│   (VN30)   │ │ Chart (candlestick + MA)                        │ │
│ - Selected │ └────────────────────────────────────────────────┘ │
│   symbol   │ ┌────────────────────────────────────────────────┐ │
│            │ │ SIGNALS: Forecast | Risk | News sentiment       │ │
│            │ └────────────────────────────────────────────────┘ │
│            │ ┌────────────────────────────────────────────────┐ │
│            │ │ RECOMMENDATION: Buy/Hold/Avoid                  │ │
│            │ │ WHY: blocking + supporting factors              │ │
│            │ └────────────────────────────────────────────────┘ │
│            │ ┌────────────────────────────────────────────────┐ │
│            │ │ NEWS (filtered, relevance badge)                │ │
│            │ └────────────────────────────────────────────────┘ │
└────────────┴────────────────────────────────────────────────────┘
```

### Visual Hierarchy
- **Primary:** Recommendation card (large, color-coded)
- **Secondary:** Decision explanation (bullet list)
- **Tertiary:** Signals (compact grid), News (cards with relevance)
- **Chart:** Full width, collapsible

---

## 6. Implementation Order

1. **Data layer:** Create `MarketDataService`, refactor `NewsService` with relevance scoring.
2. **Logic layer:** Create `SignalAggregator`, refactor `RecommendationEngine` to use `Signals` and `DecisionExplanation`.
3. **API:** Add `/api/stock/{symbol}/signals`, `/api/stock/{symbol}/news`; move data logic into services.
4. **UI:** Dashboard layout (sidebar + main), sectioned content, decision-focused recommendation block.

---

## 7. File Structure (Target)

```
src/
├── app_services/
│   ├── __init__.py
│   ├── market_data.py      # NEW: MarketDataService
│   ├── news_service.py     # NEW: NewsService with relevance
│   ├── signal_aggregator.py # NEW: SignalAggregator
│   └── recommendation.py   # REFACTOR: use Signals, output DecisionExplanation
api.py                       # REFACTOR: thinner, use services
web/
├── index.html              # REFACTOR: dashboard layout
└── static/
    ├── css/style.css       # REFACTOR: fintech theme, sections
    └── js/app.js           # REFACTOR: modular, sections
```

---

## 8. Backward Compatibility

- `POST /api/recommend` keeps same request/response shape (add `decision_explanation` as optional).
- `GET /api/stock/{symbol}` keeps returning forecast, news, indicators (for existing clients); new clients use `/signals` and `/news`.
- Gradual migration; no breaking changes in first phase.

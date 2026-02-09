# TechPulse: Production Architecture

> **Purpose:** This document describes the full system architecture for the AI-powered stock investment advisor. It serves as both a technical reference and a teaching guide.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React + Tailwind)                                  │
│  Profile Form │ Stock Selector │ Recommendation Card │ Candlestick Chart │ Explainability │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                                          │
│  /api/symbols  /api/recommend  /api/stock/{symbol}  /api/stock/{symbol}/chart  /api/...   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           RECOMMENDATION ENGINE                                           │
│  Combines: forecast ensemble + risk metrics + news sentiment + user profile               │
│  Outputs: Buy/Hold/Avoid, P(loss), P(ruin), expected return, explanation                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                            │                   │                   │
                            ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  ENSEMBLE LAYER  │  │   RISK ENGINE    │  │   NEWS LAYER     │  │  FEATURE LAYER   │
│  - Model weights │  │  - Return dist   │  │  - Sentiment     │  │  - Indicators    │
│  - Uncertainty   │  │  - VaR / CVaR    │  │  - Ticker align  │  │  - Regime        │
│  - Aggregation   │  │  - Drawdown prob │  │                  │  │                  │
│                  │  │  - Ruin proxy    │  │                  │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL LAYER (5 Forecasters)                                  │
│  Linear │ XGBoost │ ARIMA │ LSTM │ PatchTST │ Transformer                                 │
│  Each: fit(X,y), predict(X) → mean ± uncertainty                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              EVALUATION LAYER                                             │
│  Splits (60/20/20) │ Rolling Backtest │ Metrics (MAE, RMSE, direction_acc, stability)     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYERS                                                  │
│  Raw (crawl) → Clean → Features → News DB                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Data Flow (Raw → Decision → UI)

### Stage 1: Data Ingestion
- **Crawl:** CafeF/Vietstock → raw OHLCV CSV
- **News:** Crawlers → SQLite (articles, sentiment, ticker_align)
- **Validation:** Integrity checks, no look-ahead

### Stage 2: Cleaning
- Deduplicate by date
- Handle missing (interpolate or drop)
- Corporate action adjustments (if applicable)
- Time alignment (business days)

### Stage 3: Feature Engineering
- Returns (1d, 5d, 10d, 20d)
- Moving averages (5, 10, 20, 50, 200)
- Volatility (5, 10, 20)
- RSI, MACD, Bollinger, ATR, momentum
- **No future data:** Feature at t uses only data up to t

### Stage 4: Model Training & Inference
- Train on 60% (chronological)
- Validate on 20%
- Test on 20%
- Rolling backtest for stability
- Each model outputs: **mean forecast + uncertainty** (or point estimate with empirical uncertainty)

### Stage 5: Ensemble
- Weight by historical MAE stability (penalize high variance)
- Uncertainty-aware aggregation
- Output: **ensemble mean, ensemble std**

### Stage 6: Risk Engine
- Input: ensemble mean, ensemble std, volatility, user position size
- Compute: P(return < 0), P(drawdown > threshold), ruin proxy
- Output: risk_of_loss_pct, risk_of_ruin_pct, expected_return_ci

### Stage 7: Recommendation Engine
- Combine: forecast + risk + news sentiment + user (capital, experience, leverage, risk appetite)
- Probabilistic decision (not hard rules)
- Build explanation: "Why Buy/Hold/Avoid", "What risks dominate", "Which signals matter"

### Stage 8: API → UI
- Stateless endpoints
- Typed Pydantic schemas
- Frontend renders: decision card, chart, indicators, news, explainability panel

---

## 3. Folder Structure

```
d:\techpulse\
├── api.py                    # FastAPI app entry
├── configs/
│   ├── config.yaml
│   ├── news.yaml
│   └── symbols.yaml
├── data/
│   ├── raw/vn30/
│   ├── clean/vn30/
│   ├── features/vn30/
│   ├── news/news.db
│   └── forecasts/            # Cached model outputs
├── docs/
│   └── ARCHITECTURE.md
├── src/
│   ├── app_services/         # Recommendation engine (orchestrator)
│   ├── clean/                # Price cleaning
│   ├── crawl/                # Price crawlers
│   ├── evaluation/           # Splits, metrics, backtest
│   ├── ensemble/             # Model weighting, uncertainty aggregation (NEW)
│   ├── features/             # Technical indicators
│   ├── models/forecasting/   # 5 models
│   ├── news/                 # News pipeline
│   ├── pipeline/             # Data orchestration
│   ├── risk_engine/          # VaR, drawdown, ruin (NEW)
│   └── utils/
├── scripts/
│   ├── run_forecasting_pipeline.py
│   ├── run_news_pipeline.py
│   └── run_inference.py      # Pre-compute forecasts for app (NEW)
├── web/                      # Vanilla HTML/CSS/JS (fallback)
├── web-react/                # React + Tailwind (NEW, optional)
├── tests/
└── requirements.txt
```

---

## 4. Layer Contracts (Input/Output)

| Layer | Input | Output |
|-------|-------|--------|
| **Data** | Config, symbols | Raw CSV, News DB |
| **Clean** | Raw CSV | Clean CSV (dedup, validated) |
| **Features** | Clean CSV | Feature CSV (40+ columns) |
| **Model** | (X_train, y_train), X_test | y_pred (1d), optional std |
| **Ensemble** | List of (mean, std) per model | (ensemble_mean, ensemble_std) |
| **Risk Engine** | mean, std, vol, position_pct | P(loss), P(ruin), VaR, CI |
| **Recommendation** | UserProfile, symbol | RiskAdvice (rec, metrics, explanation) |

---

## 5. No Leakage Guarantees

1. **Features:** All computed at time t using only rows 0..t.
2. **Splits:** Strict chronological; no shuffle.
3. **Backtest:** Expanding window; fit on [0,t], predict t+1.
4. **News:** Align by published_at; never use future news.
5. **Inference:** App uses last known date only; no "today" until market close.

---

## 6. Teaching Notes

- **Why Ensemble?** Single models overfit; weighting by stability reduces variance.
- **Why Risk Engine?** Forecasts are point estimates; risk = P(bad outcome).
- **Why Probabilistic Rec?** User capital and tolerance change the optimal action.
- **Why Explainability?** Compliance + trust; users need "why" not just "what".

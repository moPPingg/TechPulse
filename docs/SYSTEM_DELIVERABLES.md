# TechPulse: Full System Deliverables

This document summarizes all deliverables for the production-quality AI stock advisor.

---

## 1. Architecture Overview

See `docs/ARCHITECTURE.md` for the full diagram. Summary:

```
Frontend (HTML/JS + Lightweight Charts)
    ↓
API (FastAPI): /api/symbols, /api/recommend, /api/stock/{symbol}, /api/stock/{symbol}/chart
    ↓
Recommendation Engine (probabilistic decision)
    ↓
Ensemble Layer ← Model Layer (5 forecasters)
Risk Engine (P(loss), P(ruin), VaR, CI)
News Layer (sentiment)
Feature Layer (technical indicators)
```

---

## 2. End-to-End Data Flow

1. **Raw** → Crawl CafeF → `data/raw/vn30/*.csv`
2. **Clean** → Dedup, validate → `data/clean/vn30/*.csv`
3. **Features** → Returns, MA, RSI, volatility, etc. → `data/features/vn30/*.csv`
4. **Models** → Train on 60%, validate 20%, test 20% → predict next-step return
5. **Inference** → `run_inference.py` → cache to `data/forecasts/{symbol}.json`
6. **Ensemble** → Weight by inverse MAE → aggregate mean + std
7. **Risk** → P(loss), P(ruin), expected return CI
8. **Recommendation** → Combine forecast + risk + news + user profile → Buy/Hold/Avoid
9. **API** → Stateless endpoints → JSON
10. **UI** → Form → Result card → Chart + indicators + news

---

## 3. Folder Structure

```
src/
  app_services/    # Recommendation engine
  clean/           # Price cleaning
  crawl/           # Crawlers
  ensemble/        # Model weighting (NEW)
  evaluation/      # Splits, metrics, backtest
  features/        # Technical indicators
  inference/       # Forecast cache & service (NEW)
  models/forecasting/  # Linear, XGBoost, ARIMA, LSTM, PatchTST, Transformer
  news/            # News pipeline
  risk_engine/     # P(loss), ruin, VaR (NEW)
  pipeline/        # Data orchestration
  utils/
```

---

## 4. Backend API Design

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/symbols | VN30 symbols |
| POST | /api/recommend | Recommend(Buy/Hold/Avoid) + risk + explanation |
| GET | /api/stock/{symbol} | Stock detail: forecast, news, indicators |
| GET | /api/stock/{symbol}/chart?days=90 | OHLC + MA + RSI + volatility |

**RecommendRequest:** name, capital, years_experience, risk_tolerance, leverage, symbol  
**RecommendResponse:** recommendation, risk_of_loss_pct, risk_of_ruin_pct, expected_return_lower, expected_return_upper, confidence_score, explanation, signal_breakdown

---

## 5. ML Pipeline Implementation

- **Ensemble** (`src/ensemble/aggregator.py`): Inverse-MAE weighting, stability penalty, uncertainty propagation
- **Risk** (`src/risk_engine/risk.py`): Normal model for returns, P(loss), ruin proxy, VaR, CI
- **Inference** (`src/inference/service.py`): Load cache or run 5 models, aggregate, save
- **Recommendation** (`src/app_services/recommendation.py`): Probabilistic decision, explanation builder

---

## 6. Risk Engine

- `prob_loss(mu, sigma)` → P(return < 0)
- `prob_ruin_proxy(mu, sigma, threshold, position_frac)` → P(loss > X%)
- `expected_return_ci(mu, sigma, confidence)` → [lower, upper]
- `compute_risk_metrics(...)` → RiskMetrics dataclass

---

## 7. Recommendation Engine

- **Input:** UserProfile (capital, experience, leverage, risk_tolerance), symbol
- **Output:** RiskAdvice (recommendation, risk metrics, CI, confidence, explanation, signal_breakdown)
- **Logic:** Probabilistic score from forecast mean, P(loss), P(ruin), sentiment, confidence; adjusted by risk tolerance
- **Fallback:** If no cached forecast, use last-row feature proxy (return_1d, volatility_5)

---

## 8. Frontend UI

- Profile form: name, capital, experience, risk tolerance, leverage, symbol
- Result: recommendation card (Buy/Hold/Avoid), metrics grid, explanation, signal breakdown
- "Xem biểu đồ & chi tiết": candlestick chart (Lightweight Charts), indicators, news
- Dark theme, responsive

---

## 9. End-to-End Walkthrough

1. User fills form → POST /api/recommend
2. API loads UserProfile, calls get_risk_advice(profile, symbol)
3. Recommendation loads cached forecast (or feature proxy), news sentiment
4. Risk engine computes P(loss), P(ruin), CI
5. Probabilistic decision combines signals + tolerance
6. Explanation built from trend, volatility, news, model weights
7. Response returned → UI renders card, metrics, explanation
8. User clicks "Xem chi tiết" → GET /api/stock/{symbol} + /chart
9. Chart renders candlestick + MA; indicators and news shown

---

## 10. How to Run

```bash
pip install -r requirements.txt
python src/pipeline/vnindex30/fetch_vn30.py
python scripts/run_inference.py --symbol FPT
uvicorn api:app --reload --host 0.0.0.0
# Open http://localhost:8000
```

---

## 11. Debugging & Validation Checklist

See `docs/HOW_TO_RUN_AND_EXTEND.md` for:
- Data leakage checks
- Model output validation
- Inference cache verification
- API endpoint tests

---

## 12. How to Extend

See `docs/HOW_TO_RUN_AND_EXTEND.md` for extension ideas:
- Portfolio optimization
- Multi-asset allocation
- Regime switching
- Reinforcement learning
- Bayesian uncertainty
- Explainable AI dashboards (SHAP, attention)

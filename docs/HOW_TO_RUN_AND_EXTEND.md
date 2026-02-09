# TechPulse: How to Run, Debug, and Extend

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- pip

### 1. Install Dependencies

```bash
cd d:\techpulse
pip install -r requirements.txt
```

### 2. Get Data (Price + Features)

```bash
python src/pipeline/vnindex30/fetch_vn30.py
```

This crawls CafeF, cleans, and builds features for VN30. Output:
- `data/raw/vn30/*.csv`
- `data/clean/vn30/*.csv`
- `data/features/vn30/*.csv`

### 3. (Optional) News Pipeline

```bash
python scripts/run_news_pipeline.py
```

Output: `data/news/news.db`

### 4. Pre-compute Forecasts (Recommended)

```bash
python scripts/run_inference.py --symbol FPT VCB
# or all VN30:
python scripts/run_inference.py
```

Output: `data/forecasts/{symbol}.json`

Without this, the app falls back to feature-based proxy (weaker logic).

### 5. Run the Web App

```bash
uvicorn api:app --reload --host 0.0.0.0
```

Open: http://localhost:8000

---

## Debugging & Validation Checklist

### Data Leakage
- [ ] Features: No future data. Check `build_features.py` — all indicators use `.shift()` or rolling with past-only.
- [ ] Splits: Chronological only. No shuffle in `get_train_val_test_splits`.
- [ ] Backtest: Expanding window. `get_rolling_fold_tabular` uses rows 0..t for train, t+1 for test.
- [ ] News: Aligned by `published_at`; never use future articles.

### Model Outputs
- [ ] Each model returns 1d array, same length as X_test.
- [ ] ARIMA: fit on y only; predict steps = len(X_test).
- [ ] Sequential models: X shape (n, seq_len, n_features).

### Inference Cache
- [ ] `data/forecasts/FPT.json` exists after `run_inference.py --symbol FPT`.
- [ ] JSON has `ensemble_mean`, `ensemble_std`, `model_forecasts`, `weights`.

### API
- [ ] `GET /api/symbols` returns VN30 list.
- [ ] `POST /api/recommend` returns `recommendation`, `risk_of_loss_pct`, `explanation`.
- [ ] `GET /api/stock/FPT` returns forecast, news, indicators.
- [ ] `GET /api/stock/FPT/chart` returns ohlcv, ma, rsi, volatility.

---

## How to Extend

### 1. Portfolio Optimization
- Add `src/portfolio/` with mean-variance or risk-parity allocation.
- Input: list of (symbol, forecast_mean, forecast_std).
- Output: weights per symbol.
- API: `POST /api/portfolio/optimize`.

### 2. Multi-Asset Allocation
- Extend features to support multiple tickers.
- Add correlation matrix from returns.
- Use `cvxpy` or `scipy.optimize` for constraint optimization.

### 3. Regime Switching
- Add `src/regime/` — detect bull/bear/sideways from volatility + trend.
- Use HMM or threshold-based rules.
- In recommendation: adjust risk when regime = high volatility.

### 4. Reinforcement Learning
- Add `src/rl/` — DQN or PPO for position sizing.
- State: features, position, PnL.
- Action: buy/hold/sell, size.
- Reward: risk-adjusted return (Sharpe, etc.).

### 5. Bayesian Uncertainty
- Replace point forecasts with posterior samples.
- Use Bayesian NN (dropout at test) or MC Dropout.
- Ensemble: combine posterior means and variances.
- Risk engine: use full posterior for P(loss), VaR.

### 6. Explainable AI Dashboards
- SHAP/LIME for feature importance per prediction.
- Add `src/explain/` — compute SHAP for XGBoost, attention for Transformer.
- API: `GET /api/stock/{symbol}/explain` → feature_importance, attention_weights.
- Frontend: render bar chart of top features.

---

## File Reference

| Path | Purpose |
|------|---------|
| `api.py` | FastAPI app, routes |
| `src/app_services/recommendation.py` | Recommendation engine |
| `src/ensemble/aggregator.py` | Model weighting, aggregation |
| `src/risk_engine/risk.py` | P(loss), P(ruin), VaR, CI |
| `src/inference/service.py` | Forecast cache, inference |
| `src/evaluation/` | Splits, metrics, backtest |
| `src/models/forecasting/` | 5 models |
| `scripts/run_inference.py` | Pre-compute forecasts |

---

## Troubleshooting

- **"No features file for FPT"** → Run `fetch_vn30.py` first.
- **"No cached forecast"** → Run `scripts/run_inference.py --symbol FPT`.
- **Chart not rendering** → Check browser console; Lightweight Charts CDN may be blocked.
- **News sentiment = 0** → Run `scripts/run_news_pipeline.py`.

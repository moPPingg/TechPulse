# Stock Forecasting Web App – Architecture & Decisions

## 1. Design goals

- **User-facing:** One recommendation (Buy / Hold / Avoid), risk metrics (loss/ruin), and a short explanation.
- **Inputs:** Name, capital, years of experience, risk tolerance (and implicitly: symbol, or default).
- **Backend:** Reuse existing forecasting, news, and (future) anomaly outputs; no duplicate ML in the app.
- **Deployable:** Start with Streamlit for a single process and fast iteration; keep a clear service layer so FastAPI can replace the UI later.

---

## 2. Backend ML architecture

### 2.1. Data flow (high level)

```
User profile (name, capital, experience, risk_tolerance)
         +
Symbol (e.g. FPT)
         |
         v
┌─────────────────────────────────────────────────────────────────┐
│  AGGREGATION LAYER (no ML here; only reads from ML outputs)      │
│  - Forecast signal & uncertainty (from forecasting pipeline)     │
│  - News sentiment (from news DB)                                  │
│  - Anomaly flag / score (from anomaly module or proxy)            │
│  - Latest indicators (from features or cache)                     │
└─────────────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────────────┐
│  RECOMMENDATION ENGINE (deterministic rules + risk math)          │
│  - Combines signal + sentiment + anomaly → raw view (bullish/     │
│    neutral/bearish)                                               │
│  - Adjusts for risk_tolerance → Buy / Hold / Avoid                │
│  - Computes risk of loss (e.g. P(return < 0)), risk of ruin       │
│    (simplified from volatility + position size vs capital)       │
└─────────────────────────────────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────────────────────────────────┐
│  EXPLANATION GENERATOR                                            │
│  - Turns recommendation + indicators + news + anomaly into        │
│    a short, consistent text (templates + key numbers)              │
└─────────────────────────────────────────────────────────────────┘
         |
         v
Output: { recommendation, risk_of_loss, risk_of_ruin, explanation }
```

**Why this split:**  
ML stays in existing pipelines (forecasting, news sentiment, anomaly). The web app only **reads** their outputs and applies **fixed rules** and **risk formulas**. That keeps training and serving concerns separate and makes the app testable without running heavy models on every request.

### 2.2. Where each signal comes from

| Signal | Source | Fallback if missing |
|--------|--------|----------------------|
| **Forecast direction / expected return** | Forecasting pipeline (e.g. ensemble or single model on latest features) | Neutral (Hold) if no data |
| **Forecast uncertainty** | Prediction interval or model std | Use historical volatility proxy |
| **News sentiment** | News DB by ticker (lexicon or PhoBERT score) | “No recent news” → neutral |
| **Anomaly** | Anomaly module (e.g. reconstruction error, threshold) | “No anomaly” or volatility-based proxy |
| **Indicators** | Latest row from features CSV or cache | Skip in explanation |

**Decision:** We do **not** retrain or run the full 5-model stack on each request. We either (a) run the pipeline on a schedule and cache the latest per-symbol outputs, or (b) run one lightweight model (e.g. Linear/XGBoost) on the latest features in request time. (a) is preferred for production; (b) is acceptable for a minimal demo. The architecture doc and code assume a **service layer** that returns “forecast signal”, “sentiment”, “anomaly”; the implementation of that layer can be cache or live inference.

### 2.3. Recommendation engine (model output → advice)

**Step 1 – Raw view (before risk tolerance):**  
- Inputs: `forecast_direction` (up / flat / down), `forecast_strength` (e.g. |expected_return|), `sentiment_score`, `anomaly_detected`.  
- Rule (tunable):  
  - If anomaly_detected strong → lean **Avoid** regardless.  
  - Else if forecast up and sentiment ≥ threshold → **Buy** (raw).  
  - Else if forecast down or sentiment very negative → **Avoid** (raw).  
  - Else → **Hold** (raw).

**Step 2 – Adjust for risk tolerance:**  
- User risk: e.g. Low / Medium / High (or 1–5).  
- Mapping:  
  - **Low:** Raw Buy → Hold; Raw Avoid → Avoid.  
  - **Medium:** Keep raw.  
  - **High:** Raw Hold → Buy allowed; Raw Avoid → Hold.  
- So: **Recommendation = f(raw_view, risk_tolerance)**. This gives a single Buy / Hold / Avoid.

**Step 3 – Risk of loss:**  
- Definition: probability that return over the horizon is negative.  
- Use: forecast distribution if available (e.g. from ensemble or quantile model); else approximate from historical volatility and point forecast (e.g. normal with mean = point forecast, sigma = recent volatility).  
- Output: **P(return < 0)** in [0, 1].

**Step 4 – Risk of ruin (simplified):**  
- Definition: probability of losing a large fraction of **capital** (e.g. >20% or >30%).  
- Use: position size (from capital and risk tolerance), volatility of the symbol, and horizon. Simplified formula (e.g. normal approximation or one-period ruin) so we don’t require full Monte Carlo in v1.  
- Output: **P(loss > X% of capital)** or a single “ruin risk” score in [0, 1].

**Decision:** We use **deterministic formulas** and **tunable thresholds** so that (1) behavior is explainable, (2) we can change thresholds without retraining, and (3) the app stays fast.

### 2.4. Explanation generator

- **Template-based:** e.g. “Dựa trên [chỉ báo kỹ thuật / dự báo mô hình], [tin tức gần đây], và [cảnh báo bất thường nếu có], chúng tôi đưa ra khuyến nghị [Buy/Hold/Avoid]. Rủi ro lỗ: X%. Rủi ro sụt giảm mạnh vốn: Y%.”  
- Fill with: last forecast summary, last sentiment, anomaly yes/no, and the two risk numbers.  
- **Decision:** No free-form LLM in v1 to keep latency and cost low and outputs stable.

---

## 3. Frontend UX proposal

### 3.1. User flow

1. **Landing / intro** (optional): One sentence: “Nhập thông tin và mã cổ phiếu để nhận khuyến nghị và đánh giá rủi ro.”
2. **Form (one page):**  
   - Name (text).  
   - Capital (number, VND or USD).  
   - Years of experience (number or band: &lt;1, 1–3, 3–5, 5+).  
   - Risk tolerance (dropdown or slider: Low / Medium / High).  
   - Symbol (dropdown VN30 or text, default e.g. FPT).
3. **Submit:** “Xem khuyến nghị”.
4. **Result:**  
   - **Recommendation:** one card: Buy / Hold / Avoid (with color).  
   - **Risk of loss:** one number (e.g. 35%).  
   - **Risk of ruin:** one number or band (e.g. 8%).  
   - **Explanation:** one short paragraph (indicators + news + anomaly).

**Decision:** Single form + single result view to avoid multi-step wizard and keep the demo simple. No login in v1.

### 3.2. Layout (web thuần)

- Sidebar or top: inputs (name, capital, experience, risk tolerance, symbol).  
- Main: after submit, show recommendation card, then two columns (risk of loss | risk of ruin), then explanation in a box.  
- Footer: disclaimer (not investment advice).

### 3.3. Why Streamlit first

- **Speed:** One script, no separate frontend build; easy to run locally and deploy (e.g. Streamlit Cloud).  
- **Consistency:** Python end-to-end; same service layer can be called from Streamlit or, later, from FastAPI.  
- **Limitation:** Streamlit reruns the script on interaction; for production at scale we’d move to FastAPI + static frontend and call the same recommendation service from the API.

---

## 4. Converting model outputs into risk advice (summary)

| Model / data output | Use in recommendation | Use in risk |
|----------------------|------------------------|-------------|
| **Forecast (direction, mean, std)** | Direction → raw Buy/Hold/Avoid; strength → confidence | Mean + std → P(return &lt; 0) |
| **News sentiment** | Positive/negative → nudge Buy/Avoid; “no news” → neutral | Not directly in risk math in v1 |
| **Anomaly** | If high → force Avoid or Hold | Can cap position size later |
| **Volatility (from features)** | — | Sigma for loss/ruin formulas |
| **Capital + risk tolerance** | — | Position size → ruin risk |

So: **recommendation** = rule(raw_view, risk_tolerance); **risk of loss** = P(return &lt; 0) from forecast (or volatility); **risk of ruin** = simplified P(large loss of capital) from position size and volatility.

---

## 5. File layout (implemented)

```
mentor/05_WEB_APP_ARCHITECTURE.md   # This file
src/app_services/
  __init__.py
  recommendation.py   # get_forecast_signal(), get_news_sentiment(), get_risk_advice(), build_explanation()
api.py               # FastAPI: GET /, GET /api/symbols, POST /api/recommend; serve web/
web/
  index.html         # Form + khu vực hiển thị kết quả
  static/
    css/style.css    # Giao diện (bạn có thể chỉnh theo ý)
    js/app.js        # Gọi /api/symbols, /api/recommend; render kết quả
configs/config.yaml  # data paths (features_dir, news db) cho recommendation
```

---

## 6. How to run (pure web)

```bash
# Từ thư mục gốc project
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0
```

Mở http://localhost:8000. Điền form và bấm "Xem khuyến nghị". App đọc từ `data/features/vn30/<SYMBOL>.csv` và (nếu có) `data/news/news.db`; không chạy train model trong request.

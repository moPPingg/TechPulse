# Project structure, what each folder does, and how to run

This document explains: (1) the overall layout of the repo, (2) **every folder under `src/`** — what it means and how the technique runs, and (3) **how to run the project** step by step. No code without explanation first.

---

## Part 1: Overall project structure (top level)

Think of the project in three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA                                                    │
│  configs/   → settings (paths, symbols, crawl options)            │
│  data/      → raw prices, cleaned prices, features, news DB       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: LOGIC (src/)                                            │
│  crawl → clean → features  (price pipeline)                        │
│  news: crawl → clean → sentiment → align  (news pipeline)         │
│  evaluation + models/forecasting  (train & compare 5 models)      │
│  app_services  (turn model + news into Buy/Hold/Avoid + risk)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: RUNNERS & UI                                             │
│  scripts/     → run_forecasting_pipeline.py, run_news_pipeline.py  │
│  api.py + web/  → backend FastAPI + frontend HTML/CSS/JS → recommendation │
│  docs/        → learning material (you read, no execution)        │
│  mentor/      → teaching notes (this file + 01–05)                 │
└─────────────────────────────────────────────────────────────────┘
```

- **configs/**  
  - **What it solves:** One place for paths, symbols, and options so code doesn’t hardcode them.  
  - **Why:** Changing data location or VN30 list doesn’t require editing source files.  
  - **Main files:** `config.yaml` (prices, features, crawl), `news.yaml` (news DB, sources), `symbols.yaml` (VN30 list).

- **data/**  
  - **What it solves:** All generated data lives here: raw CSV, cleaned CSV, feature CSV, news SQLite.  
  - **Why:** Clear separation; easy to backup or point to another disk.  
  - **Typical layout:** `data/raw/vn30/`, `data/clean/vn30/`, `data/features/vn30/`, `data/news/news.db`.

- **docs/**  
  - **What it solves:** Deep-dive reading (ML basics, time series, baselines, news, evaluation).  
  - **Why:** Understanding concepts before or while you run code.  
  - **No execution:** These are markdown (and some HTML) for you to read.

- **mentor/**  
  - **What it solves:** Teaching notes that explain *what* each part does and *why* (including this file).  
  - **Why:** You asked for deep understanding; mentor files map “this folder / this function” to “this problem / this technique.”

- **scripts/**  
  - **What it solves:** Single entry points to run a full pipeline (forecasting, news).  
  - **Why:** You don’t have to remember which Python module to call with which args.  
  - **Input:** Config (and sometimes CLI flags). **Output:** Files in `data/` or logs.

- **api.py + web/**  
  - **What it solves:** Web app thuần: backend FastAPI (api.py) phục vụ API và static frontend (web/). User điền form → gọi POST /api/recommend → hiển thị Buy/Hold/Avoid + risk + explanation.  
  - **Why:** Giao diện web chuẩn, dễ tùy chỉnh UI (HTML/CSS/JS); backend tách riêng, có thể gọi từ app khác.  
  - **Runs:** `uvicorn api:app --reload`; backend đọc `data/features/` và `data/news/`, gọi `src.app_services.recommendation`.

- **tests/**  
  - **What it solves:** Automated checks so changes don’t break core behavior.  
  - **Why:** Safe refactoring and onboarding.

---

## Part 2: Folder-by-folder under `src/`

Each subsection: (1) what the folder is for, (2) what problem it solves, (3) how the technique runs, (4) main inputs/outputs, (5) one small example idea.

---

### `src/crawl/`

**Meaning:** Get **raw price data** from the internet (e.g. CafeF API).

**What problem it solves:** You need daily OHLCV (open, high, low, close, volume) for VN30 symbols. Without this, there is no input for cleaning or features.

**How the technique runs:**  
- Code sends **HTTP requests** to CafeF’s price API with symbol and date range.  
- API returns JSON; we parse it and turn it into a table (e.g. pandas DataFrame) with columns like `date`, `open`, `high`, `low`, `close`, `volume`, `ticker`.  
- That table is saved as CSV under `data/raw/` (one file per symbol or one combined file, depending on the script).

**Input:** Symbol (e.g. `"FPT"`), start date, end date, maybe page size.  
**Output:** CSV file(s) in `data/raw/vn30/` (or path from config).

**Common mistakes:**  
- Using the wrong date format (API often expects DD/MM/YYYY).  
- Not handling empty response or rate limits.  
- Saving to a path that doesn’t exist (create directories first).

**Small example idea:**  
“For symbol FPT from 01/01/2024 to 31/01/2024, request one page of data → get 20 rows → write `data/raw/vn30/FPT.csv`.”

---

### `src/clean/`

**Meaning:** Turn **raw price CSV** into **clean, consistent** tables (no duplicates, valid numbers, sorted by date).

**What problem it solves:** Raw data can have duplicates, missing values, or wrong types. Downstream (features, models) expect one row per date and numeric columns.

**How the technique runs:**  
- Read raw CSV.  
- **Drop duplicates** (e.g. by date).  
- **Validate** columns (date parseable, numeric columns are numbers).  
- Optionally drop rows with null in key columns.  
- Sort by date and write to `data/clean/vn30/`.

**Input:** Path to raw CSV (or directory of CSVs).  
**Output:** Clean CSV(s) in `data/clean/vn30/`.

**Common mistakes:**  
- Deleting too many rows (e.g. dropping any row with a single NaN).  
- Not sorting by date (breaks time-series logic later).  
- Changing column names inconsistently (features expect e.g. `close`, `date`).

**Small example idea:**  
“Read `FPT.csv` from raw → 3 duplicate dates removed, 2 rows with null volume dropped → sorted by date → save to clean/FPT.csv.”

---

### `src/features/`

**Meaning:** From **clean OHLCV**, compute **technical indicators** (returns, moving averages, RSI, MACD, volatility, etc.) and save one table per symbol.

**What problem it solves:** Models don’t use raw price only; they need derived signals (e.g. return_1d, ma_20, rsi_14). This module creates those columns once so every model uses the same features.

**How the technique runs:**  
- For each clean CSV: load → ensure `date` and `close` exist.  
- **Returns:** `close.pct_change(periods=k)` → columns like `return_1d`, `return_5d`.  
- **Moving averages:** rolling mean of `close` (e.g. windows 5, 10, 20, 50).  
- **RSI:** formula from gains/losses over a window (e.g. 14).  
- **MACD:** EMA(fast) − EMA(slow), then signal line.  
- **Volatility:** rolling std of returns.  
- Same for other indicators (Bollinger, volume features, momentum).  
- Drop rows where indicators are NaN (e.g. first 200 rows for ma_200), then save to `data/features/vn30/<SYMBOL>.csv`.

**Input:** Clean CSV path (or directory).  
**Output:** Feature CSV(s) with many extra columns (40+).  
**Shape idea:** One row per trading day; columns = date, open, high, low, close, volume + return_1d, return_5d, ma_5, ma_20, rsi_14, macd, volatility_5, ….

**Common mistakes:**  
- Using future data (e.g. tomorrow’s close) in a feature — everything must be computable at time *t* from data up to *t*.  
- Different symbols having different column sets (models expect same features).  
- Not dropping NaN rows and then breaking models that don’t handle NaN.

**Small example idea:**  
“Clean FPT has 2500 rows. After computing ma_200, first 200 rows have NaN → drop them → 2300 rows with 45 columns written to features/FPT.csv.”

---

### `src/utils/`

**Meaning:** **Shared helpers** (read/write files, load YAML, dates, logging) so the rest of `src/` doesn’t repeat the same I/O logic.

**What problem it solves:** Many modules need to load config, save CSV, or parse dates. Centralizing this avoids bugs and keeps paths/encodings consistent.

**How the technique runs:**  
- **file_utils:** e.g. `load_yaml(path)`, `save_csv(df, path)`, `load_csv(path)`.  
- **date_utils:** parse or format dates in one place.  
- **logger:** one logging setup for the app.

**Input/Output:** Depends on the function (path in, dict or DataFrame out; or df + path in, file written).  
**No ML here** — only I/O and small helpers.

**Common mistakes:**  
- Assuming project root is current working directory (scripts may run from another folder; use `Path(__file__).resolve().parent` and go up to root).  
- Wrong encoding (use UTF-8 for Vietnamese text).

---

### `src/pipeline/`

**Meaning:** **Orchestration** that chains crawl → clean → features (and optionally news) so one script can “run the whole data pipeline” for VN30.

**What problem it solves:** You don’t want to run crawl, then clean, then features by hand in the right order. This folder wires them together and reads config (symbols, paths).

**How the technique runs:**  
- **runcrawler:** calls crawl for each symbol from config, saves to raw.  
- **vnindex30/fetch_vn30.py:**  
  - Load symbols (e.g. from `symbols.yaml` or fallback list).  
  - Call crawler for each symbol → raw.  
  - Call clean on raw → clean.  
  - Call `build_features` on clean → features.  
- So one command runs: Crawl → Clean → Features for all VN30 (or a subset).

**Input:** Config files (and maybe CLI).  
**Output:** Populated `data/raw/`, `data/clean/`, `data/features/`.

**Common mistakes:**  
- Running features before clean (wrong order).  
- Not adding project root to `sys.path` when running as a script (so `import src.xxx` fails).

**Small example idea:**  
“Run `fetch_vn30.py` → it crawls 30 symbols, cleans 30 files, builds features for 30 files → 30 CSVs in each of raw/clean/features.”

---

### `src/evaluation/`

**Meaning:** **Same rules for every model:** same train/val/test split (by time), same metrics (MAE, RMSE, MAPE, R², direction accuracy), same way to build tabular vs sequence data, and optional rolling backtest.

**What problem it solves:** If each model used different splits or metrics, you couldn’t compare them fairly. This folder defines one split, one metric set, and one data prep so all 5 models (Linear, XGBoost, ARIMA, LSTM, PatchTST, Transformer) are evaluated the same way.

**How the technique runs:**  
- **splits:** Given a DataFrame sorted by date, cut at 60% / 20% / 20% (or configurable) → train_df, val_df, test_df. No shuffle.  
- **metrics:** `compute_metrics(y_true, y_pred)` returns a dict `{mae, rmse, mape, r2, direction_acc}`.  
- **data:**  
  - **Tabular:** Each row = one day; X = features that day, y = target (e.g. next-day return). Used by Linear, XGBoost, ARIMA.  
  - **Sequential:** Each sample = a window of consecutive days; X shape = (samples, seq_len, features), y = next value. Used by LSTM, PatchTST, Transformer.  
- **backtest:** Optional: refit model on expanding window and predict one step ahead for each test date; aggregate same metrics.

**Input:** DataFrame with date, features, and target column (e.g. return_1d).  
**Output:** For tabular: (X_train, y_train, X_val, y_val, X_test, y_test) plus maybe dates. For sequential: same but X are 3D. Metrics: dict of floats.

**Common mistakes:**  
- Shuffling the data (destroys time order).  
- Using future information in X (e.g. tomorrow’s return).  
- Different models using different train/val/test (they must share the same split).

**Small example idea:**  
“1000 rows of features → train 0–599, val 600–799, test 800–999. Tabular: X_train (600, n_features), y_train (600,). Sequential with seq_len=20: X_train (580, 20, n_features), y_train (580,).”

---

### `src/models/` and `src/models/forecasting/`

**Meaning:**  
- **models/ml.py:** Old/mini examples (KFold, TimeSeriesSplit, scaling, confusion matrix) for learning — not the main forecasting pipeline.  
- **models/forecasting/:** The **five forecasters** with one interface: `fit(X, y)` and `predict(X)` so the evaluation layer can call them the same way.

**What problem it solves:** You need one API (fit/predict) for evaluation and for the app. Each model (Linear, XGBoost, ARIMA, LSTM, PatchTST, Transformer) implements that API; internally they differ (tabular vs sequence, sklearn vs statsmodels vs PyTorch).

**How the technique runs:**  
- **baseline_ml (Linear, XGBoost):** X is 2D (samples, features). Fit a regressor; predict one value per row.  
- **ARIMA:** Uses only the target series (y) to fit an ARIMA(p,d,q). Predict returns N steps (N = len(X_test)).  
- **LSTM:** X is 3D (samples, seq_len, features). Network outputs one value per sample. Trained with MSE.  
- **PatchTST:** X is 3D; series is split into patches, then a Transformer encoder; head outputs one value per sample.  
- **Transformer (iTransformer-style):** X is 3D; variables are treated as the sequence dimension; attention over variables; head outputs one value per sample.

**Input:** For tabular models: X (n_samples, n_features), y (n_samples,). For sequence models: X (n_samples, seq_len, n_features), y (n_samples,).  
**Output:** predict(X) → 1D array of length n_samples.

**Common mistakes:**  
- Passing 2D X to LSTM/PatchTST/Transformer (they expect 3D).  
- ARIMA predicting with exog when it was fitted without (or the opposite).  
- Forgetting to scale inputs for neural models (or scaling with test set statistics → leakage).

**Small example idea:**  
“Tabular: 1000 train samples, 200 test. Linear fits on (1000, 45), predicts (200,) from (200, 45). LSTM: train (980, 20, 45), test (200, 20, 45) → predict (200,).”

---

### `src/news/`

**Meaning:** **News pipeline:** crawl articles (CafeF, Vietstock) → store in DB → clean text → sentiment → align to VN30 tickers. Used later by the app for “news sentiment” in the explanation.

**What problem it solves:** You want to link “recent news about FPT” to the FPT recommendation. That requires: articles in a DB, cleaned body, a sentiment score, and a link (article_id, ticker).

**How the technique runs:**  
- **crawlers:** Fetch list pages → get article URLs → fetch each article body → save (source, url, title, body_raw, published_at) into DB (upsert by url).  
- **db:** SQLite tables: articles, article_cleaned, sentiments, article_tickers.  
- **clean:** Strip HTML, normalize spaces, optional boilerplate removal → body_clean.  
- **sentiment:** Lexicon-based (Vietnamese positive/negative words) → score in [-1, 1].  
- **ticker_align:** Match VN30 symbols and optional company names in title+body → rows (article_id, ticker).

**Input:** Config (sources, DB path); for crawl: list of categories and max pages.  
**Output:** DB with articles, cleaned text, sentiment per article, and which tickers each article mentions.

**Common mistakes:**  
- Crawling too aggressively (blocked by site).  
- Sentiment on raw HTML (clean first).  
- Matching ticker “FPT” inside unrelated words (use word boundaries).

**Small example idea:**  
“Crawl 3 pages of CafeF → 60 URLs → fetch 60 bodies → insert 60 rows. Clean 60 → sentiment 60 → align: 20 articles mention FPT, 15 mention VCB → article_tickers has 35 rows.”

---

### `src/app_services/`

**Meaning:** **Recommendation engine:** reads forecast signal (from features or cache), news sentiment (from news DB), and optional anomaly proxy, then applies **rules** + **risk formulas** to produce Buy/Hold/Avoid, risk of loss, risk of ruin, and an explanation string.

**What problem it solves:** The web app needs one recommendation and two risk numbers, not raw model outputs. This layer converts “forecast up, sentiment positive, volatility 1.5%” and user “risk tolerance = low” into “Hold” and “P(loss) ≈ 35%, ruin ≈ 8%” with a short text.

**How the technique runs:**  
- **get_forecast_signal(symbol):** Read last row of feature CSV for that symbol; use return_1d and volatility as proxy for direction and uncertainty.  
- **get_news_sentiment(symbol):** Query news DB for articles with that ticker in last 30 days; average sentiment score.  
- **get_anomaly_proxy(symbol):** For now, high volatility → “elevated” anomaly.  
- **get_risk_advice(profile, symbol):**  
  - Combine direction + sentiment + anomaly → raw view (Buy/Hold/Avoid).  
  - Adjust for risk_tolerance (low → more Hold; high → more Buy).  
  - P(return < 0) from normal(mean, vol).  
  - Ruin from position size (from tolerance) and volatility.  
- **build_explanation:** Template: “Dựa trên … sentiment … khuyến nghị … rủi ro lỗ … rủi ro sụt giảm …”.

**Input:** UserProfile (name, capital, experience, risk_tolerance), symbol.  
**Output:** RiskAdvice (recommendation, risk_of_loss_pct, risk_of_ruin_pct, explanation).

**Common mistakes:**  
- Using test-set or future data in the “forecast” (we use last row of features as proxy; in production this would be last known day).  
- Hardcoding thresholds (should come from config later).

**Small example idea:**  
“User: risk low, symbol FPT. Last row: return_1d = 0.5%, vol_5 = 1.2 → direction up. News: 5 articles, avg sentiment 0.1. Raw = Buy; low tolerance → Hold. Vol → P(loss) 40%, ruin 6%. Explanation string built from these.”

---

## Part 3: How to run this project

Assumptions: you have Python 3.10+ and are in the project root (`d:\techpulse` or wherever the repo is).

### Step 0: Environment and dependencies

```bash
cd d:\techpulse
pip install -r requirements.txt
```

This installs pandas, numpy, requests, scikit-learn, xgboost, statsmodels, torch, beautifulsoup4, PyYAML, fastapi, uvicorn, etc. If something fails (e.g. torch on your machine), you can still run the price pipeline and news pipeline; forecasting may need torch for LSTM/PatchTST/Transformer.

### Step 1: Get price data and features (required for forecasting and app)

**Option A – Full VN30 pipeline (crawl → clean → features):**

```bash
python src/pipeline/vnindex30/fetch_vn30.py
```

- **What it does:** Reads `configs/config.yaml` and `configs/symbols.yaml`, crawls CafeF for each symbol, cleans, then builds features.  
- **Output:** `data/raw/vn30/*.csv`, `data/clean/vn30/*.csv`, `data/features/vn30/*.csv`.  
- **Time:** Depends on network and number of symbols (can be several minutes).

**Option B – You already have feature CSVs:**  
If `data/features/vn30/` already has e.g. `FPT.csv`, you can skip Step 1 for testing the app or the forecasting script (they read from this folder).

### Step 2: (Optional) News pipeline

If you want the app to use “news sentiment” in the explanation:

```bash
python scripts/run_news_pipeline.py
```

Or only crawl first:

```bash
python scripts/run_news_pipeline.py --steps crawl
```

Then clean, sentiment, align:

```bash
python scripts/run_news_pipeline.py --steps clean sentiment align
```

- **Output:** `data/news/news.db` with articles, cleaned text, sentiment, and article–ticker links.  
- If the DB is missing, the app still runs but uses “no recent news” (neutral sentiment).

### Step 3: Run the forecasting comparison (5 models)

Requires feature CSVs from Step 1 (at least one symbol, e.g. FPT):

```bash
python scripts/run_forecasting_pipeline.py --symbol FPT
```

- **What it does:** Loads feature CSV for FPT, splits into train/val/test, prepares tabular data for Linear/XGBoost/ARIMA and sequential data for LSTM/PatchTST/Transformer, runs each model, prints same metrics for all.  
- **Output:** Printed table of metrics (MAE, RMSE, MAPE, R², direction_acc) per model.  
- **Optional:** `--rolling` runs a small rolling backtest for tabular models.

### Step 4: Run the web app (thuần web: FastAPI + HTML/CSS/JS)

```bash
uvicorn api:app --reload --host 0.0.0.0
```

- **What it does:** Starts a local web server; browser opens (e.g. http://localhost:8501). You fill Name, Capital, Experience, Risk tolerance, Symbol, then click “Xem khuyến nghị”.  
- **Backend:** Reads `data/features/vn30/<SYMBOL>.csv` and, if present, `data/news/news.db`; calls `src.app_services.recommendation.get_risk_advice`; shows Buy/Hold/Avoid, risk of loss %, risk of ruin %, and explanation.  
- **No training** in the app; it uses the last row of features and recent news.

### Order summary

| Goal | Commands |
|------|----------|
| Only run the app (with existing features) | `uvicorn api:app --reload` |
| Full data + app | Step 1 → (Step 2 optional) → Step 4 |
| Compare 5 models | Step 1 → Step 3 |
| News in app | Step 2 → Step 4 |

### If something fails

- **“No features file for FPT”:** Run Step 1 (or ensure `data/features/vn30/FPT.csv` exists).  
- **“No module named src”:** Run from project root and/or add the project root to `PYTHONPATH`.  
- **“No module named torch”:** Install PyTorch or run only Linear/XGBoost/ARIMA (they don’t need torch).  
- **News DB missing:** App still runs; sentiment will be neutral and explanation will say no recent news.

---

## Part 4: How the techniques connect (flow)

1. **Crawl** → raw CSV (one row per day per symbol).  
2. **Clean** → same, but deduplicated and validated.  
3. **Features** → same rows, many new columns (returns, MAs, RSI, …).  
4. **Evaluation** → reads feature CSV, splits by time, builds tabular or sequential (X, y).  
5. **Models** → each gets (X_train, y_train), then predict(X_test); evaluation computes same metrics.  
6. **News** (parallel to 1–3): crawl → DB → clean → sentiment → align to tickers.  
7. **App** → for a symbol, last row of features = “forecast” proxy, DB = news sentiment; rules + risk formulas → recommendation + explanation.

So: **data flows from crawl/clean/features into evaluation and models; news flows from news pipeline into app_services; app_services uses both feature data and news to produce the user-facing output.**

---

Do you want a **simpler** version of this (e.g. one page: “only what I need to run”) or a **deeper** one (e.g. one document per folder with line-by-line code walkthrough and tiny examples)?

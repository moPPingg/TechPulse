# Green Dragon: Quantitative Trading System for the VN30 Market

Green Dragon is a research framework that formalizes Smart Money Concepts (SMC) into a quantitative trading pipeline for Vietnamese equities (VN30 index). The core hypothesis is that LSTM networks provide meaningful tail-risk protection during market crashes—a property not observed in gradient boosting baselines.

The system integrates four layers: (1) a daily OHLCV data pipeline from Yahoo Finance, (2) a mathematical feature engineering layer encoding liquidity sweeps from raw price data, (3) an LSTM classifier producing a continuous action score per trading day, and (4) an event-driven alert system that monitors live signals and dispatches structured email notifications.

---

## Results

Evaluation conditions: long-only execution, transaction cost 0.25% (0.15% commission + 0.10% slippage), chronological train/val/test split (64/16/20%).

| Model | Threshold | Sharpe — Stable | Sharpe — Normal | Sharpe — Extreme | Sharpe — Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| LSTM | 0.635 | -0.08 | 0.05 | **1.06** | **0.47** |
| PatchTST | 0.589 | 0.00 | 0.00 | 0.58 | 0.28 |
| iTransformer | 0.569 | 0.17 | -0.29 | 0.93 | 0.39 |
| LightGBM | 0.729 | 0.00 | 0.00 | 0.00 | 0.00 |

Bootstrap 95% CI for LSTM overall Sharpe: [0.31, 0.64] (B=2000, block size=20).

---

## Repository Structure

```
data/
  raw/              OHLCV CSV files per symbol (VN30 constituents, auto-updated)
  features/         Engineered feature matrices (smc_features.csv)
  oos_returns/      Out-of-sample daily return series per model

src/
  smc.py                Liquidity sweep and order block feature engineering
  lstm.py               PyTorch LSTM model definition
  process_data.py       Data cleaning and RobustScaler preprocessing
  optimize_models.py    Optuna hyperparameter and threshold search
  backtest.py           Vectorized backtester with regime classification
  run_benchmark.py      Out-of-sample evaluation across all models

scripts/
  yfinance_fetcher.py   Daily OHLCV data fetcher (runs on schedule)
  alert_system.py       Live signal monitor + news scraper + email alerts
  sentiment_regime.py   FinBERT sentiment scoring + regime mapping
  visualize_cat1.py     Figure 1 — Liquidity sweep anatomy, Figure 2 — Volume histogram
  visualize_cat2.py     Figure 6 — Optuna importance, Figure 7 — Threshold, Figure 8 — Benchmark
  visualize_cat3.py     Figure 9 — Cumulative returns, Figure 10 — Drawdown
  visualize_cat4.py     Figure 4 — XAI signal overlay
  visualize_regimes.py  Figure 5 — Market regimes
  q1_style.py           Shared matplotlib academic style settings

paper/
  techpulse_paper.tex   LaTeX manuscript (Springer LNCS)
  gen_diagram.py        Figure 3 — System architecture diagram
  images/               All generated figures

results/
  optuna_benchmark_table.csv   Regime-stratified Sharpe results
  bootstrap_sharpe.csv         Bootstrap CI results

Dockerfile              Single image used by both Docker services
docker-compose.yml      Two services: vn30-fetcher and alert-monitor
requirements.txt        Core ML/backtest dependencies
requirements-fetcher.txt  Lightweight deps for data + alert pipeline
```

---

## Reproducing the Experiments

**Requirements:** Python 3.10+, PyTorch, pandas, scikit-learn, optuna, matplotlib.

```bash
pip install -r requirements.txt
```

**Step 1 — Feature engineering:**
```bash
python src/process_data.py
```

**Step 2 — Hyperparameter search (Optuna, ~2–4 hours on CPU):**
```bash
python src/optimize_models.py
```

**Step 3 — Out-of-sample benchmark:**
```bash
python src/run_benchmark.py
```

**Step 4 — Regenerate all figures:**
```bash
python scripts/visualize_cat1.py
python scripts/visualize_cat2.py
python scripts/visualize_cat3.py
python scripts/visualize_cat4.py
python scripts/visualize_regimes.py
python paper/gen_diagram.py
```

All figures are saved to `paper/images/`.

---

## Live Data Pipeline and Alert System

### Running without Docker

Open two terminals:

```bash
# Terminal 1 — fetch VN30 data daily at 16:00 ICT
set DATA_DIR=data/raw
python scripts/yfinance_fetcher.py
```

```bash
# Terminal 2 — monitor signals and send email alerts
set GMAIL_USER=your@gmail.com
set GMAIL_APP_PASSWORD=your-app-password
set ALERT_TO_EMAIL=your@gmail.com
python scripts/alert_system.py
```

### Running with Docker

Copy `.env.example` to `.env` and fill in your Gmail credentials, then:

```bash
docker compose --env-file .env up --build -d   # start both services in background
docker compose logs -f                          # follow logs
docker compose down                             # stop
```

### Gmail App Password setup

1. Enable 2-Step Verification at `myaccount.google.com`
2. Navigate to Security → App Passwords
3. Generate a password for "Mail" and paste it into `GMAIL_APP_PASSWORD`

### Alert types

| Trigger | Subject | Content |
|:---|:---|:---|
| Every 30 min | `[Green Dragon] Tin tuc thi truong` | 15 latest headlines from VnExpress, CafeF, TinNhanhCK with FinBERT sentiment scores |
| Liquidity Sweep detected | `[Green Dragon] LIQUIDITY SWEEP — {symbol}` | Symbol, price, three trigger conditions |
| Regime 3 (Extreme) entered | `[Green Dragon] CANH BAO: THI TRUONG EXTREME REGIME` | Rolling volatility, recommended risk posture |
| Strongly negative sentiment | `[Green Dragon] CANH BAO: TIN TUC TIEU CUC MANH` | Top negative headlines, aggregate sentiment score |

---

## Key Design Decisions

**No lookahead bias:** All rolling statistics use `.shift(1)` before window computation. Train/val/test splits are strictly chronological with no shuffling.

**Scaling:** RobustScaler is fit on the training fold only and applied to OHLCV features. SMC binary indicators are passed through unscaled.

**Execution threshold:** The entry threshold (0.635 for LSTM) is optimized on the validation set to maximize Sharpe after transaction costs. It is not tuned on test data.

**Regime classification:** Market regimes are defined by a 252-day rolling annualized volatility surface computed on the full VN30 index. Regime boundaries are fixed before backtesting begins.

**Sentiment integration:** FinBERT sentiment scoring operates on a separate analytical cycle from the core trading pipeline. Sentiment signals supplement regime monitoring and alert dispatch but are not included in the LSTM input tensor.

---

## Authors

Thien Khoi Nguyen, Thu Le, Minh Triet Nguyen, Hung Anh Tran, Viet Hoang Van — FPT University, Ho Chi Minh Campus, Vietnam.

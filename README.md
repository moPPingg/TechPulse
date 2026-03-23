# Green Dragon

A quantitative trading framework for Vietnam's VN30 index that formalizes Smart Money Concepts (SMC) as deterministic, learnable features. The core finding is that LSTM with regime conditioning provides statistically supported tail-risk protection during market crashes—a property not observed in gradient boosting or Transformer baselines.

## Results

Evaluation: long-only execution, TC = 0.25% (0.15% commission + 0.10% slippage), chronological 64/16/20% train/val/test split, no data leakage.

| Model | Threshold | Sharpe (Stable) | Sharpe (Normal) | Sharpe (Extreme) | Overall [95% CI] | MDD (Overall) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LSTM | 0.635 | -0.08 | 0.05 | **1.06** | **0.23 [0.00, 0.37]** | -17.54% |
| iTransformer | 0.569 | 0.17 | -0.29 | 0.93 | 0.14 [-0.10, 0.61] | -7.66% |
| PatchTST | 0.589 | 0.00 | 0.00 | 0.58 | -0.05 [-0.32, 0.16] | -0.25% |
| LightGBM | 0.729 | -- | -- | -- | -- (no trades) | 0.00% |

Block bootstrap 95% CI: B=2000, block size=20, seed=42. LSTM is the only model whose CI lower bound is non-negative.

---

## Repository Structure

```
src/
  smc.py                Liquidity sweep feature engineering
  lstm.py               PyTorch LSTM model definition
  process_data.py       Data preprocessing and RobustScaler pipeline
  optimize_models.py    Optuna hyperparameter and threshold search
  backtest.py           Vectorized backtester with regime classification
  run_benchmark.py      Out-of-sample evaluation across all models

scripts/
  yfinance_fetcher.py   Scheduled OHLCV downloader for all 30 VN30 tickers
  alert_system.py       Signal monitor, news scraper, and Gmail alert dispatcher
  sentiment_regime.py   FinBERT sentiment scoring and dual-trigger regime mapping
  visualize_cat1.py     Liquidity sweep anatomy + volume histogram
  visualize_cat2.py     Optuna importance, threshold curve, benchmark bar chart
  visualize_cat3.py     Cumulative returns + drawdown underwater plot
  visualize_cat4.py     XAI signal overlay with BOS/CHoCH annotations
  visualize_regimes.py  Market regime switching timeline
  q1_style.py           Shared matplotlib style for all figures

paper/
  techpulse_paper.tex   LaTeX manuscript (Springer LNCS)
  gen_diagram.py        System architecture diagram
  images/               All generated figures

data/
  raw/                  Per-ticker OHLCV CSVs (auto-updated by fetcher)
  features/             Engineered feature matrix (smc_features.csv)
  oos_returns/          Out-of-sample daily return series

results/
  optuna_benchmark_table.csv
  bootstrap_sharpe.csv

Dockerfile
docker-compose.yml
requirements.txt
requirements-fetcher.txt
```

---

## Reproducing the Experiments

**Requirements:** Python 3.10+, PyTorch, pandas, scikit-learn, optuna, matplotlib.

```bash
pip install -r requirements.txt
```

```bash
# 1. Preprocess data and engineer SMC features
python src/process_data.py

# 2. Hyperparameter and threshold search via Optuna (~2–4 hours on CPU)
python src/optimize_models.py

# 3. Out-of-sample benchmark across all four models
python src/run_benchmark.py

# 4. Regenerate all paper figures
python scripts/visualize_cat1.py
python scripts/visualize_cat2.py
python scripts/visualize_cat3.py
python scripts/visualize_cat4.py
python scripts/visualize_regimes.py
python paper/gen_diagram.py
```

Figures are saved to `paper/images/`.

---

## Live Pipeline and Alert System

### Without Docker

```bash
# Terminal 1 — fetch VN30 OHLCV data daily at 16:00 ICT
DATA_DIR=data/raw python scripts/yfinance_fetcher.py

# Terminal 2 — monitor signals and dispatch email alerts
GMAIL_USER=your@gmail.com \
GMAIL_APP_PASSWORD=your-app-password \
ALERT_TO_EMAIL=your@gmail.com \
python scripts/alert_system.py
```

### With Docker

Copy `.env.example` to `.env`, fill in Gmail credentials, then:

```bash
docker compose --env-file .env up --build -d
docker compose logs -f
docker compose down
```

### Gmail App Password

1. Enable 2-Step Verification at myaccount.google.com
2. Security → App Passwords → generate for "Mail"
3. Paste the 16-character password into `GMAIL_APP_PASSWORD`

### Alert types

| Trigger | Email subject |
|:---|:---|
| Every 30 minutes | `[Green Dragon] Tin tuc thi truong` — 15 headlines with FinBERT scores |
| Liquidity sweep detected | `[Green Dragon] LIQUIDITY SWEEP — {symbol}` |
| Regime 3 entered (vol ≥ 30%) | `[Green Dragon] CANH BAO: THI TRUONG EXTREME REGIME` |
| Sentiment score < -0.40 | `[Green Dragon] CANH BAO: TIN TUC TIEU CUC MANH` |

---

## Design Notes

**No lookahead bias.** All rolling statistics use `.shift(1)` before window computation. Train/val/test splits are strictly chronological.

**Scaling.** RobustScaler is fit on the training partition only and applied to OHLCV channels. SMC indicators pass through unscaled.

**Threshold optimization.** Entry threshold is tuned on the validation set to maximize post-cost Sharpe. It is never fit on test data.

**Regime classification.** A 252-day expanding rolling volatility on the full VN30 index assigns each day to one of three regimes. Boundaries are fixed before any model sees the data.

**Sentiment.** FinBERT scoring operates offline during regime labeling and for live alert dispatch. It does not enter the LSTM input tensor.

---

## Authors

Thien Khoi Nguyen, Thu Le, Minh Triet Nguyen, Hung Anh Tran, Viet Hoang Van
FPT University, Ho Chi Minh Campus, Vietnam

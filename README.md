# Green Dragon: Quantitative Trading System for the VN30 Market

Green Dragon is a research framework that formalizes Smart Money Concepts (SMC) into a quantitative trading pipeline for Vietnamese equities (VN30 index). The core hypothesis is that LSTM networks—due to their sequential memory—provide meaningful tail-risk protection during market crashes, a property not observed in gradient boosting models.

The system integrates three components: (1) a mathematical feature engineering layer that encodes liquidity sweeps and order blocks from raw OHLCV data, (2) an LSTM classifier that outputs a continuous action score per trading day, and (3) a vectorized backtester with regime-aware evaluation across stable, normal, and extreme market conditions.

---

## Results

Evaluation conditions: long-only execution, transaction cost = 0.25% (0.15% commission + 0.10% slippage), chronological train/val/test split (64/16/20%).

| Model | Threshold | Sharpe — Stable | Sharpe — Normal | Sharpe — Extreme | Sharpe — Overall |
|:---|:---:|:---:|:---:|:---:|:---:|
| LSTM | 0.635 | -0.08 | 0.05 | **1.06** | **0.47** |
| PatchTST | 0.589 | 0.00 | 0.00 | 0.58 | 0.28 |
| iTransformer | 0.569 | 0.17 | -0.29 | 0.93 | 0.39 |
| LightGBM | 0.729 | 0.00 | 0.00 | 0.00 | 0.00 |

Bootstrap 95% confidence interval for LSTM overall Sharpe: [0.31, 0.64] (B=2000, block size=20).

---

## Repository Structure

```
data/
  raw/          OHLCV CSV files per symbol (VN30 constituents)
  features/     Engineered feature matrices (smc_features.csv)
  oos_returns/  Out-of-sample daily return series per model

src/
  smc.py              Liquidity sweep and order block feature engineering
  lstm.py             PyTorch LSTM model definition
  process_data.py     Data cleaning and RobustScaler preprocessing
  optimize_models.py  Optuna hyperparameter and threshold search
  backtest.py         Vectorized backtester with regime classification
  run_benchmark.py    Final out-of-sample evaluation across all models

scripts/
  visualize_cat1.py   Figure 1 — Liquidity sweep anatomy, Figure 2 — Volume histogram
  visualize_cat2.py   Figure 3 — Optuna importance, Figure 4 — Dynamic threshold, Figure 5 — Benchmark
  visualize_cat3.py   Figure 6 — Cumulative returns, Figure 7 — Drawdown
  visualize_cat4.py   Figure 8 — Signal overlay
  visualize_regimes.py  Figure 9 — Market regimes
  q1_style.py         Shared matplotlib academic style settings

results/
  optuna_benchmark_table.csv   Regime-stratified Sharpe results
  bootstrap_sharpe.csv         Bootstrap CI results

paper/
  techpulse_paper.tex          LaTeX manuscript
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

## Key Design Decisions

**No lookahead bias:** All rolling statistics use `.shift(1)` before window computation. Train/val/test splits are strictly chronological with no shuffling.

**Scaling:** RobustScaler is fit on the training fold only and applied to OHLCV features. SMC binary indicators are passed through unscaled.

**Execution threshold:** The entry threshold (0.635 for LSTM) is optimized on the validation set to maximize Sharpe after transaction costs. It is not tuned on test data.

**Regime classification:** Market regimes are defined by a 252-day rolling annualized volatility surface computed on the full VN30 index. Regime boundaries are fixed before backtesting begins.

---

## Authors

Thien Khoi Nguyen, Thu Le, Minh Triet Nguyen, Hung Anh Tran, Viet Hoang Van — FPT University, Ho Chi Minh Campus, Vietnam.

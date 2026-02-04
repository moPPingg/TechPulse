# Why Each Model, Inductive Bias, and When It Fails

Same train/val/test splits, same metrics (MAE, RMSE, MAPE, R², direction accuracy), and same rolling-window backtest apply to all five models. Below: why each is chosen, what inductive bias it has, and when it tends to fail.

---

## 1. Linear Regression / XGBoost (ML baseline)

**Why chosen:**  
Fast, interpretable (linear) or flexible (XGBoost) baselines. Establish how much signal is linearly (or tree-) explainable from the same features. No temporal structure; purely cross-sectional given the same feature vector.

**Inductive bias:**  
- **Linear:** Relationship between features and target is linear; additive effects.  
- **XGBoost:** Piecewise constant (tree splits); can capture interactions and non-linearity; no explicit time ordering (each row is i.i.d. in the loss).

**When it fails:**  
- When the target depends strongly on **order** (sequence) or **long-range** history; these models see only the current feature vector.  
- When the data has **regime shifts** and the same feature vector maps to different outcomes in different periods.  
- When useful structure is in the **dynamics** (e.g. momentum of momentum) rather than in a single snapshot of indicators.

---

## 2. ARIMA

**Why chosen:**  
Classic univariate (or with exog) time-series model. Captures **autocorrelation** and **trend** (via differencing). Good baseline for “how much is explained by the past of the target alone?”

**Inductive bias:**  
- Linear in past values and past errors.  
- Assumes **stationarity** (after differencing).  
- Short memory (order p, q); exponential decay of the influence of old lags.

**When it fails:**  
- **Non-stationarity** that can’t be removed by simple differencing (e.g. structural breaks, changing variance).  
- **Multivariate** structure: ignores other features unless added as exog; often worse than ML/DL when many signals matter.  
- **Non-linear** effects and **long-range** dependencies; ARIMA is linear and short-memory.  
- **Regime changes** and **tail events**; parameters are fixed.

---

## 3. LSTM

**Why chosen:**  
Designed for sequences; can learn **long-range** dependencies and **temporal** structure. Standard choice for financial time series before Transformers.

**Inductive bias:**  
- **Recursive** structure: one step at a time; strong bias toward **local** and **smooth** temporal evolution.  
- **Gating** (forget/input/output) allows selective memory; can learn which lags matter.  
- **Causal**: at each step, only past and present are visible (no lookahead).

**When it fails:**  
- **Very long** dependencies (e.g. yearly effects with daily data); gradients can vanish.  
- **Parallelization** is limited (sequential recurrence).  
- **Interpretability** is low.  
- Can **overfit** on small or noisy series; needs enough data and regularization.

---

## 4. PatchTST

**Why chosen:**  
Patch-based Transformer: segments the series into **patches**, then applies Transformer on patch tokens. Reduces sequence length and can capture **local + global** patterns; good balance for forecasting.

**Inductive bias:**  
- **Local smoothness** within a patch; **global** relations across patches via self-attention.  
- **Channel independence** in many variants (each series/feature as separate patch sequence).  
- Assumes that **patch-level** representation is useful (fixed patch length).

**When it fails:**  
- When important patterns are **at scales** not aligned with patch size (too short or too long).  
- **Very small** datasets; Transformers need data.  
- If **channel mixing** is important and the model is channel-independent, it may underperform.

---

## 5. Transformer (iTransformer / TimesNet style)

**Why chosen:**  
**iTransformer**: inverts dimensions so each **variable (channel)** is a token; attention over variables can capture **cross-series** relations (e.g. between returns and volatility). **TimesNet**-style: 2D representation and multi-scale periods. Here we use an iTransformer-style variant.

**Inductive bias:**  
- **Cross-variable** structure: which features interact for the target.  
- **Full attention** over the chosen dimension (here, channels); no explicit locality unless built in.  
- **Position** can be encoded along time within each channel.

**When it fails:**  
- **Small sample**: many parameters, risk of overfitting.  
- When the **best** signal is univariate or strictly **local in time**; the model may over-emphasize cross-channel attention.  
- **Computational cost** and need for tuning (depth, heads, lr).

---

## Summary Table

| Model        | Why chosen              | Inductive bias              | When it fails                    |
|-------------|--------------------------|-----------------------------|----------------------------------|
| Linear      | Fast, interpretable      | Linear, no time             | Non-linear, order-dependent      |
| XGBoost     | Flexible, strong baseline| Trees, no time              | Strong sequence / regime         |
| ARIMA       | Univariate baseline      | Linear, short memory, stationary | Non-stationary, multivariate, non-linear |
| LSTM        | Sequence, long-range     | Recursive, causal, local+smooth | Very long deps, small data       |
| PatchTST    | Patch + Transformer      | Local patches + global      | Wrong scale, small data          |
| Transformer | Cross-variable          | Attention over channels      | Small data, univariate-best      |

All models are evaluated on the **same** train/val/test split, **same** metrics (MAE, RMSE, MAPE, R², direction accuracy), and **same** rolling-window backtest where applicable.

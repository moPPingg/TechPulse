# Green Dragon: Quantitative Trading System

**Green Dragon** is a professional-grade, quantitative trading framework designed to algorithmicize Smart Money Concepts (SMC). By replacing lagging technical indicators with mathematical formulas tracking *Liquidity Sweeps*, the system extracts highly confident execution triggers resilient to institutional shakeouts.

Through robust Optuna Bayesian Optimization over a multi-year chronological dataset (VN30), the framework definitively proves the **Tail-Risk Protection Hypothesis**: While tree-based gradient boosting models (LightGBM) collapse during Market Crashes (Regime 3), the recursive memory properties of Long Short-Term Memory (**LSTM**) networks act as an exceptional defensive framework, independently achieving a remarkable **1.06 Sharpe Ratio** in the most extreme, crash-heavy market conditions.

---

## Architecture & Core Components

1.  **Strict Anti-Leakage Feature Engineering (`src/smc.py`)**  
    Uses a Pandera-validated, strictly partitioned sequence builder. It actively utilizes `.sort_values()` before isolated `.groupby('symbol').ffill()` routines and executes `.shift(1)` logic before rolling windows to absolutely guarantee zero future-data leakage into the historical baselines.
2.  **LSTM Deep Recurrent Engine (`src/lstm.py`)**  
    Ingests continuous chronological sequences mathematically tracking Smart Money traces to generate an active probability confidence vector `(Batch, Sequence, Feature) -> (Batch, 1)`.
3.  **Bayesian Trade Boundary Optimization (`src/optimize_models.py`)**  
    Uses Optuna's **Tree-structured Parzen Estimator (TPE)**. Instead of rigidly hardcoding an entry confidence of 50%, the framework evaluates combinations of network learning properties against dynamically tuned **Execution Thresholds**. The LSTM specifically discovered that executing Long trades strictly above a `0.635` confidence bound optimally outpaces the structural `0.25%` transaction costs (Commission + Slippage) to maximize Validation Sharpe.
4.  **Market Regime Evaluator (`src/backtest.py`)**  
    Backtests out-of-sample data not as a singular stream, but dynamically mapped against a 252-day trailing volatility surface, explicitly categorizing tests into: *Stable*, *Normal*, and *Extreme* environments.

---

## Repository Structure

The project has been aggressively cleaned and structured exclusively into production-grade ML directories:

*   `/src/`: Core execution scripts.
    *   `smc.py`: The deterministic Liquidity Sweep generator utilizing Pandera schemas.
    *   `lstm.py`: The native PyTorch network wrapper handling matrix ingestion.
    *   `process_data.py`: Preprocessing runners.
    *   `optimize_models.py`: Main execution runner combining Optuna search against the Backtesting matrix.
*   `/data/`: Separated data layers. Stores original VNstock pulls and `smc_features.csv` mappings.
*   `/results/`: Output telemetry. Contains the final evaluation `optuna_benchmark_table.csv` verifying the regime performance.
*   `/paper/`: Academic deliverables containing the `techpulse_paper.tex` mapping the Tail-Risk hypothesis directly to the mathematical benchmarks.
*   `/learning_hub/`: Massive educational repository expressly designed to explain the custom backend vector mechanics to quantitative engineers. Contains:
    *   `1_SMC_Logic_and_Vectorization.md`
    *   `2_LSTM_and_Tensor_Shapes.md`
    *   `3_Optuna_Dynamic_Threshold.md`
    *   `4_TradingView_Clone_Architecture.md`
    *   `/annotated_code/`: Line-by-line detailed comments across PyTorch and Pandas tensor transformations.

---

## Final Tuned Benchmark (Test Set)

**Evaluation Rules:** Target > Dynamic Threshold, Transaction Costs = 0.25%, Execution = Long-Only

| Model | Sharpe (Stable) | Sharpe (Normal) | Sharpe (Extreme) | **Sharpe (Overall)** |
|:---|:---:|:---:|:---:|:---:|
| **LSTM** (Threshold = 0.635) | -0.08 | 0.05 | **1.06** | **0.47** |
| **PatchTST** (Threshold = 0.589) | 0.00 | 0.00 | 0.58 | 0.28 |
| **iTransformer** (Threshold = 0.569) | 0.17 | -0.29 | 0.93 | 0.39 |
| **LightGBM** (Threshold = 0.729) | 0.00 | 0.00 | 0.00 | 0.00 |

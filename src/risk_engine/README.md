# src/risk_engine/

**Purpose:** Compute risk metrics from forecast distribution: P(loss), P(ruin), VaR, expected return interval. Assumes parametric (e.g. normal) return distribution.

**Fits into system:** Called by `src/app_services/signal_aggregator` and recommendation. Consumes forecast mean and std from inference/ensemble.

**Data in/out:**
- **In:** Forecast mean, std, horizon; optional position/capital for ruin.
- **Out:** prob_loss_pct, prob_ruin_pct, VaR, confidence intervals.

# Production-Grade System Review & Redesign

## Executive Summary

This document provides a strict technical review of TechPulse as an AI finance system, identifies flaws across six dimensions, and proposes a production-grade redesign. All findings are documented for audit and compliance.

---

## 1. ARCHITECTURAL FLAWS

### 1.1 No Purged Split / Embargo

**Finding:** `get_train_val_test_splits` uses a simple 60/20/20 time-based split with no purge gap or embargo.

**Risk:** Overlapping returns at split boundaries can leak information. For 1-day horizon this is minor, but for multi-day or event-aware targets it becomes severe.

**Fix:** Implement `PurgedTimeSeriesSplit` with `purge_gap >= horizon` and optional `embargo_pct`.

### 1.2 Forecast Cache Staleness

**Finding:** `get_forecast()` loads from `data/forecasts/{symbol}.json` with no TTL or as_of_date validation. Stale forecasts can be served indefinitely.

**Risk:** User receives a recommendation based on a forecast from weeks ago. Market regime may have shifted.

**Fix:** Add `max_age_days` check; return `None` if `as_of_date` is older than threshold. Log staleness.

### 1.3 Ad-Hoc Decision Weights

**Finding:** Decision engine uses hardcoded magic numbers (0.2, 0.3, 0.15, etc.) for combining signals. No calibration, no audit trail.

**Risk:** Arbitrary weights; cannot justify to regulator or user why these values. Not reproducible.

**Fix:** Move weights to config; add `DecisionConfig` dataclass; document rationale in comments.

### 1.4 No Walk-Forward / Retraining Schedule

**Finding:** Inference trains once on static split. No rolling retrain, no walk-forward validation.

**Risk:** Model drift. Market conditions change; model becomes stale.

**Fix:** Document retraining cadence (e.g., monthly); add `last_retrain_date` to forecast metadata.

### 1.5 Tight Coupling

**Finding:** Signal layers, decision engine, and recommendation are coupled. Hard to test in isolation or swap implementations.

**Fix:** Define clear interfaces (protocols); dependency injection for services.

---

## 2. MODELING FLAWS

### 2.1 Sequential Target Alignment

**Finding:** In `prepare_sequential`, window is `M[i-seq_len:i]` and `y = y_all[i]`. Window ends at row `i-1`; target is row `i`. This is correct for 1-step-ahead prediction. ✓ No leakage here.

### 2.2 Tabular Target Alignment

**Finding:** `y = df[target].shift(-1)` — we predict next-day return. Features at row `i` exclude `return_1d` (target). ✓ Correct.

### 2.3 Feature Leakage in build_features

**Finding:** Rolling windows use `center=False` (default). ✓ No look-ahead.

### 2.4 Normal-Distribution Assumption for Risk

**Finding:** `prob_loss` and `prob_ruin_proxy` assume returns are N(μ, σ²). Financial returns are fat-tailed.

**Risk:** Underestimation of tail risk (P(ruin), VaR). Users may take larger positions than warranted.

**Fix:** Document assumption; consider Student-t or empirical tail; add tail-risk disclaimer.

### 2.5 Ensemble Weight Source

**Finding:** Weights from validation MAE. Val set is used for model selection — acceptable. Test is held out for final evaluation. ✓ No test leakage in training.

---

## 3. UX FLAWS

### 3.1 Insufficient Risk Disclosure

**Finding:** Disclaimer "Không phải lời khuyên đầu tư" is in footer only. Users may not see it.

**Fix:** Add prominent disclaimer before recommendation; require acknowledgment for sensitive actions.

### 3.2 Confidence Not Calibrated

**Finding:** `confidence_score` is derived from std/mean ratio. Not calibrated to actual forecast accuracy.

**Risk:** 80% confidence may not mean 80% of predictions within CI. Misleading.

**Fix:** Show as "model confidence" not "accuracy"; add tooltip explaining meaning.

### 3.3 Position Size Ambiguity

**Finding:** "Tỷ trọng gợi ý 10% vốn" — 10% of total portfolio? Of allocated capital? Per position?

**Fix:** Clarify in UI: "Gợi ý tỷ trọng tối đa cho vị thế này: 10% tổng vốn."

### 3.4 No Historical Validation

**Finding:** Users cannot see if past recommendations would have performed well.

**Fix:** Add "Backtest" or "Historical accuracy" section (when data available).

### 3.5 Profile Form Mapping

**Finding:** Form sends `risk_tolerance: "Thấp"` but API maps to `"low"`. Ensure consistent mapping. ✓ Already handled in API.

---

## 4. DATA LEAKAGE RISKS

### 4.1 Price/Technical Same-Day Return

**Finding:** `get_indicators` returns `return_1d` from the last row. For a pre-market recommendation, we do not yet know today's close. Using `return_1d` implicitly assumes we're making the recommendation at market close.

**Risk:** If API is called during market hours, we may use partial-day data or stale prior-day data. Clarification needed.

**Fix:** Document that recommendations assume point-in-time = last known close. Add `as_of_date` to response. For live trading, use prior close only.

### 4.2 News Sentiment Timing

**Finding:** `get_stock_news_signal` aggregates articles by `date_from` (days ago). No explicit check that article `published_at` is before prediction timestamp.

**Risk:** News published during the trading day could be used to predict that day's return — leakage.

**Fix:** Filter news by `published_at < prediction_time` (e.g., prior day close). Implement point-in-time sentiment.

### 4.3 No Purge Between Splits

**Finding:** Adjacent train/val/test with no purge. For 1d target, overlap is minimal. For multi-day, add purge.

**Fix:** Implement purge in splits module.

---

## 5. MISLEADING METRICS

### 5.1 Confidence Score

**Finding:** `confidence_score = 1/(1+std/max(|mean|,0.1))`. Not a probability. Does not reflect actual hit rate of predictions.

**Fix:** Rename to `model_confidence`; add disclaimer: "Độ tin cậy mô hình, không phải xác suất chính xác."

### 5.2 P(loss) and P(ruin)

**Finding:** Assumes normality. Real returns have excess kurtosis. P(ruin) may be understated.

**Fix:** Add disclaimer; consider reporting "P(ruin) dựa trên giả định phân phối chuẩn."

### 5.3 No Out-of-Sample Backtest

**Finding:** Recommendation quality is never validated. We don't know if Buy/Hold/Avoid would have been profitable historically.

**Fix:** Add backtest module; report directional accuracy, Sharpe, max drawdown on held-out period.

---

## 6. PRODUCTION-GRADE REDESIGN

### 6.1 Configuration-Driven Weights

- `configs/decision.yaml`: weights for each signal layer, thresholds, position caps.
- All magic numbers moved to config with comments.

### 6.2 Staleness Controls

- Forecast: `max_age_days: 7` — reject if older.
- Add `data_freshness` to API response.

### 6.3 Purged Splits

- `PurgedTimeSeriesSplit` in evaluation module.
- `purge_gap=1` for 1d horizon; configurable.

### 6.4 Point-in-Time Semantics

- All data queries accept `as_of_date`.
- News: filter by `published_at < as_of_date`.
- Price: use close as of `as_of_date`.

### 6.5 Disclaimers and UX

- Prominent disclaimer before recommendation.
- Tooltips for P(loss), P(ruin), confidence.
- Position size clarified.

### 6.6 Audit and Logging

- Log all recommendation requests with inputs and outputs.
- Add `recommendation_id` for traceability.

---

## Implementation Checklist

- [ ] Add `PurgedTimeSeriesSplit` (optional use)
- [ ] Add forecast staleness check
- [ ] Add `configs/decision.yaml` with documented weights
- [ ] Add `as_of_date` semantics where applicable
- [ ] Add prominent disclaimer and tooltips
- [ ] Document normal-distribution assumption for risk metrics
- [ ] Add `data_freshness` and `recommendation_id` to API

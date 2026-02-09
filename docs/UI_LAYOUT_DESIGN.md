# TechPulse UI Layout — Bloomberg/TradingView-Grade Design

## User Journey (Sequential)

1. **Input stock ticker** — Primary CTA; sidebar + header symbol selector
2. **See market context** — Chart + price/volume (DATA)
3. **See technical + model signals** — Indicators + forecast (SIGNAL)
4. **See news intelligence** — Composite score + articles (SIGNAL)
5. **See final decision with reasoning** — Buy/Hold/Avoid + why (DECISION + RISK)

---

## Layout Structure

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│ HEADER                                                                              │
│ [Logo] TechPulse    │  [Symbol: FPT ▼]  │  HOSE · VN30  │  📅 05/02/2025  │  Server │
└────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┬───────────────────────────────────────────────────────────────────────┐
│ SIDEBAR     │ MAIN (scrollable, single column)                                      │
│             │                                                                       │
│ VN30        │ ┌─ SECTION: DATA (Market Context) ─────────────────────────────────┐ │
│             │ │  Candlestick chart 90d  [1d][7d][1M] horizon                     │ │
│ ACB  BCM    │ │  Price | Volume | Last update                                    │ │
│ BID  BVH    │ └──────────────────────────────────────────────────────────────────┘ │
│ ...         │                                                                       │
│             │ ┌─ SECTION: SIGNAL (Technical + Model + News) ─────────────────────┐ │
│ ─────────   │ │  Technical     │  Model Forecast  │  News Intelligence           │ │
│ Profile     │ │  RSI  Vol Ret  │  +0.5% ±0.8%    │  Composite +0.3  (12 bài)    │ │
│ [collapsed] │ └──────────────────────────────────────────────────────────────────┘ │
└─────────────┴───────────────────────────────────────────────────────────────────────┘

│ ┌─ SECTION: RISK ─────────────────────────────────────────────────────────────────┐ │
│ │  P(loss) 45%  │  P(ruin) 8%  │  Expected return -2% to 3%  │  Confidence 72%   │ │
│ └──────────────────────────────────────────────────────────────────────────────────┘ │

│ ┌─ SECTION: DECISION ─────────────────────────────────────────────────────────────┐ │
│ │                                                                                 │ │
│ │  HOLD                                                                           │ │
│ │  Primary: Dự báo đi ngang, biến động thấp.                                      │ │
│ │  Blocking: Xác suất lỗ cao (45%).                                               │ │
│ │  Supporting: Tin tức tích cực (12 bài).                                         │ │
│ │  Action: Nên quan sát thêm; chờ tín hiệu rõ ràng hơn.                           │ │
│ │                                                                                 │ │
│ └──────────────────────────────────────────────────────────────────────────────────┘ │

│ ┌─ SECTION: NEWS (Intelligence) ──────────────────────────────────────────────────┐ │
│ │  Top articles with event_type, relevance%, sentiment badge                      │ │
│ │  [earnings] FPT công bố Q4... | 85% | +0.4  🔗                                  │ │
│ └──────────────────────────────────────────────────────────────────────────────────┘ │
```

---

## Cognitive Separation

| Block | Purpose | Content |
|-------|---------|---------|
| **DATA** | Raw market context | Chart, OHLCV, volume, last date |
| **SIGNAL** | Derived inputs | Technical (RSI, vol, return), Model forecast, News composite |
| **RISK** | Quantified risk | P(loss), P(ruin), expected return CI, confidence |
| **DECISION** | Final output | Buy/Hold/Avoid + primary/blocking/supporting/action |
| **NEWS** | Supporting feed | Enriched articles (event, relevance, sentiment) |

---

## Design Principles

1. **Ticker-first:** Select symbol → entire view loads. No multi-step form.
2. **Profile collapsed:** Risk tolerance, capital, leverage in sidebar collapsible; default values OK.
3. **Sections with clear headers:** DATA | SIGNAL | RISK | DECISION | NEWS
4. **Reduce clutter:** Remove "Giải thích thuật ngữ" from main flow; tooltips suffice
5. **Monospace for numbers:** JetBrains Mono for metrics
6. **Color coding:** Buy=green, Hold=amber, Avoid=red; sentiment badges
7. **Single scroll:** No modal for "Xem biểu đồ"; chart in main flow

---

## Responsive

- Desktop: Sidebar + main (1200px max)
- Tablet: Sidebar collapsible / horizontal symbol strip
- Mobile: Symbol selector full-width; sections stack

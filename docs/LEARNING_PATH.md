# TechPulse Learning Path

Structured self-study curriculum to **understand, implement, and defend** this financial intelligence system. Order is deliberate: each section builds on the previous and ties directly to the codebase.

---

## Order of Study

| Stage | Section | What you'll learn | Skills built | Maps to project |
|-------|---------|-------------------|--------------|------------------|
| **Beginner** | [01_foundations](01_foundations/) | ML basics (bias–variance, validation, leakage), time-series concepts (stationarity, autocorrelation) | How to avoid overfitting and leakage in any ML project | `src/evaluation/splits.py`, `src/features/build_features.py` (no future data) |
| **Intermediate** | [02_modeling](02_modeling/) | Baseline forecasting models (linear, tree, ARIMA) and how they are trained and evaluated here | How our forecast pipeline works and how to add a model | `src/models/forecasting/`, `src/inference/service.py` |
| **Intermediate** | [03_multimodal](03_multimodal/) | News data in Vietnam, how news is crawled and turned into a signal in this system | How news flows from raw articles to `StockNewsSignal` | `src/news/`, `src/app_services/news_intelligence.py` |
| **System-level** | [../SYSTEM_ARCHITECTURE.md](../SYSTEM_ARCHITECTURE.md) + [ARCHITECTURE.md](ARCHITECTURE.md) | Full data flow: raw → ingest → storage → signals → ML → API → frontend | End-to-end mental model of the system | All of `src/`, `api.py`, `web/` |
| **Advanced** | [04_advanced](04_advanced/) | Event-aware and tail-aware training (how to use news/events and handle fat tails) | Research-level extensions and caveats | `src/evaluation/`, `src/risk_engine/` |
| **Research-level** | [05_evaluation](05_evaluation/) | Leakage control and tail metrics: how we validate that the system is scientifically sound | Defending methodology in reports or theses | `src/evaluation/splits.py`, `src/evaluation/metrics.py`, backtest |

---

## What Each Section Builds

- **01_foundations:** Prevents the most common ML mistakes (leakage, wrong validation). You need this before touching forecasting or evaluation code.
- **02_modeling:** Lets you read and modify the forecasting stack (models, ensemble, inference) with confidence.
- **03_multimodal:** Connects “raw news” to “composite score” and “per-article contribution” in the UI and API.
- **System architecture:** Gives you the full pipeline in one place so you can trace any feature or signal from source to API.
- **04_advanced:** Optional but important if you extend the system with event features or better risk (e.g. tail-aware loss).
- **05_evaluation:** Gives you the language and checks to defend the system (no future leakage, proper metrics, backtest design).

---

## How This Maps to the Project

```
LEARNING_PATH.md (you are here)
       │
       ├── 01_foundations  →  evaluation/splits, features (no future info)
       ├── 02_modeling     →  models/forecasting, inference, ensemble
       ├── 03_multimodal   →  news/pipeline, news/engine, news_intelligence
       ├── ARCHITECTURE    →  SYSTEM_ARCHITECTURE.md (root), NEWS_INTELLIGENCE_ENGINE.md
       ├── 04_advanced    →  evaluation, risk_engine, possible extensions
       └── 05_evaluation   →  leakage controls, tail metrics, backtest
```

---

## How This Maps to AI / Quant Career Paths

- **ML engineer:** Focus on 01 → 02 → 05 (data and model correctness, evaluation).
- **NLP / multimodal:** Add 03 and NEWS_INTELLIGENCE_ENGINE (news as a signal).
- **Quant / research:** Add 04 and 05 (event-aware, tail-aware, leakage and metrics).
- **Full-stack / system:** Add SYSTEM_ARCHITECTURE and HOW_TO_RUN_AND_EXTEND (run, extend, debug).

---

## Constraints Used in This Curriculum

- **No unrelated theory:** Every topic ties to a file or flow in this repo.
- **No fluff:** Only what is needed to understand, implement, or explain the system.
- **Deep understanding over coverage:** One clear path; optional deep dives are linked from each section README where applicable.

Start with **01_foundations**, then **02_modeling** and **03_multimodal**, then read **SYSTEM_ARCHITECTURE.md**. Use **04_advanced** and **05_evaluation** when you work on extensions or need to defend the methodology.

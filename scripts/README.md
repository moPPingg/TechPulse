# scripts/

**Purpose:** Runnable entrypoints for pipelines and one-off tasks (news pipeline, forecasting pipeline, inference, URL tests). Not part of the API server.

**Fits into system:** Call into `src/news/pipeline`, `src/pipeline/vnindex30`, `src/evaluation`, `src/models`, `src/inference`. Used for batch training, scheduled runs, or local verification.

**Data in/out:**
- **In:** Config files; data in `data/` (raw, processed, features).
- **Out:** Updated DB (news), new CSVs or forecast cache; console logs.

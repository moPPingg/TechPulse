# web/

**Purpose:** Static frontend: HTML, CSS, JavaScript. Served by FastAPI from `api.py` (mount at `/`, static at `/static`). Calls API only; no direct access to DB or Python code.

**Fits into system:** Consumes all data via REST (e.g. /api/recommend, /api/stock/{symbol}, /api/stock/{symbol}/chart). Renders recommendation, chart, news, and explainability.

**Data in/out:**
- **In:** User input (symbol, profile form); API responses (JSON).
- **Out:** Rendered DOM; no server-side state.

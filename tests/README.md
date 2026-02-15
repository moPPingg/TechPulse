# tests/

**Purpose:** Test suite for the project. Entrypoint: `test.py` or pytest. Should cover critical paths (e.g. splits, metrics, engine output) without requiring live crawlers.

**Fits into system:** Imports from `src.*`; does not run API or schedulers. Run before releases or after refactors.

**Data in/out:**
- **In:** Fixtures or small in-memory data.
- **Out:** Assertions; no side effects on `data/` or DB in CI if isolated.

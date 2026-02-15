# 04 — Advanced Training

**What this folder does:** Research-level topics: event-aware training (using news/events in the model) and tail-aware training (handling fat tails and extreme returns).

**Fits into the system:** Informs possible extensions to `src/evaluation/` (e.g. event windows, purged splits) and `src/risk_engine/` (e.g. tail metrics, non-normal assumptions). Not all of this is implemented yet; the docs explain the concepts and how they would plug in.

**Data in/out:** Conceptual and design. Input: event timestamps, news signals; return distributions. Output: design choices for training and risk.

**Files:**
- `01_EVENT_AWARE_TRAINING.md` — Using events/news in training and evaluation.
- `02_TAIL_AWARE_TRAINING.md` — Tail risk and robust loss or metrics.

**Order:** After system-level understanding ([SYSTEM_ARCHITECTURE.md](../../SYSTEM_ARCHITECTURE.md)). Use when extending the system or writing research-oriented docs.

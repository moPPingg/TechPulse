"""
Sentiment-Augmented Regime Classifier
--------------------------------------
Uses FinBERT (ProsusAI/finbert) to score financial news headlines,
then maps the aggregated sentiment score to one of three market regimes:

    Regime 1 — Stable   : low volatility, neutral-to-positive sentiment
    Regime 2 — Normal   : moderate volatility, mixed sentiment
    Regime 3 — Extreme  : high volatility or strongly negative sentiment

Install:
    pip install transformers torch

Usage:
    python scripts/sentiment_regime.py
"""

import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# FinBERT loader — lazy init so import is fast
# ---------------------------------------------------------------------------
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        print("Loading FinBERT (first run downloads ~440MB)...")
        _pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            # Run on CPU; change to device=0 if CUDA available
            device=-1,
        )
        print("FinBERT ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------
LABEL_SCORE = {
    "positive": 1.0, "Positive": 1.0,
    "neutral":  0.0, "Neutral":  0.0,
    "negative": -1.0, "Negative": -1.0,
}


def score_headline(text: str) -> float:
    """
    Returns a sentiment score in [-1, 1]:
        +1.0  strongly positive
         0.0  neutral
        -1.0  strongly negative
    """
    pipe = _get_pipeline()
    result = pipe(text, truncation=True, max_length=512)[0]
    label = result["label"]       # "Positive" | "Neutral" | "Negative"
    confidence = result["score"]  # [0, 1]
    direction = LABEL_SCORE.get(label, 0.0)
    return direction * confidence


def score_headlines(headlines: List[str]) -> Tuple[List[float], float]:
    """
    Score a list of headlines and return (per_headline_scores, aggregate_mean).
    """
    scores = [score_headline(h) for h in headlines]
    mean_score = float(np.mean(scores)) if scores else 0.0
    return scores, mean_score


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------
def classify_regime(
    sentiment_score: float,
    realized_vol: float,
    vol_stable_thresh: float = 0.15,
    vol_extreme_thresh: float = 0.30,
    sentiment_extreme_thresh: float = -0.40,
) -> Tuple[int, str]:
    """
    Map (sentiment_score, realized_vol) -> (regime_id, regime_label).

    Parameters
    ----------
    sentiment_score        : Aggregated FinBERT score in [-1, 1].
    realized_vol           : Annualised realised volatility (e.g. 0.20 = 20%).
    vol_stable_thresh      : Upper bound of Regime 1 volatility (default 15%).
    vol_extreme_thresh     : Lower bound of Regime 3 volatility (default 30%).
    sentiment_extreme_thresh: Sentiment below this triggers Regime 3 regardless of vol.

    Returns
    -------
    (int, str) — regime id (1/2/3) and human label.
    """
    # Strongly negative sentiment alone is sufficient for Extreme
    if sentiment_score < sentiment_extreme_thresh:
        return 3, "Extreme"

    if realized_vol >= vol_extreme_thresh:
        return 3, "Extreme"
    elif realized_vol <= vol_stable_thresh:
        return 1, "Stable"
    else:
        return 2, "Normal"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
DEMO_HEADLINES = [
    # --- Extreme scenario ---
    "VN-Index plunges 6% as global recession fears trigger panic selling across all sectors.",
    "Foreign investors dump Vietnamese equities amid currency crisis and banking liquidity crunch.",
    "State Bank of Vietnam intervenes as dong hits record low; market circuit breakers activated.",
    # --- Stable scenario ---
    "Vietnam GDP growth beats expectations at 7.1% for Q1 2025, boosting investor confidence.",
    "FPT Corporation reports record quarterly profit driven by software export revenue.",
    "VN30 blue-chips inch higher as domestic retail flows remain steady.",
]


def run_demo():
    print("\n" + "=" * 65)
    print("SENTIMENT-AUGMENTED REGIME CLASSIFIER — DEMO")
    print("=" * 65)

    # Split into two batches for illustration
    extreme_headlines = DEMO_HEADLINES[:3]
    stable_headlines = DEMO_HEADLINES[3:]

    for label, batch, vol in [
        ("Scenario A (Extreme news)", extreme_headlines, 0.38),
        ("Scenario B (Stable news)",  stable_headlines,  0.12),
    ]:
        scores, mean_score = score_headlines(batch)
        regime_id, regime_label = classify_regime(mean_score, realized_vol=vol)

        print(f"\n{label}")
        print(f"  Realised vol : {vol:.0%}")
        for h, s in zip(batch, scores):
            print(f"  [{s:+.3f}]  {h[:70]}")
        print(f"  Aggregate sentiment : {mean_score:+.4f}")
        print(f"  --> Regime {regime_id} ({regime_label})")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run_demo()

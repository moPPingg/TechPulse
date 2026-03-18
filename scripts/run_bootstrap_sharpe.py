"""
Block Bootstrap Confidence Intervals for Out-of-Sample Annualized Sharpe Ratio.

Usage:
    python scripts/run_bootstrap_sharpe.py --returns_dir data/oos_returns/

Expected input: CSV files named {model}_returns.csv with a single column
'daily_return' containing out-of-sample daily strategy returns (decimal form).
e.g., data/oos_returns/lstm_returns.csv

Output: prints 95% CI table and saves results to results/bootstrap_sharpe.csv
"""

import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
N_BOOTSTRAP = 2_000
BLOCK_SIZE = 20          # ~1 trading month — matches window w in the paper
CONFIDENCE = 0.95
RANDOM_SEED = 42

MODELS = ["lstm", "patchtst", "itransformer", "lightgbm"]


# ── Core Functions ─────────────────────────────────────────────────────────────

def annualized_sharpe(returns: np.ndarray, trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compute annualized Sharpe ratio from daily returns (assumes 0 risk-free rate)."""
    if returns.std() == 0 or len(returns) == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(trading_days)


def block_bootstrap_sharpe(
    returns: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    block_size: int = BLOCK_SIZE,
    confidence: float = CONFIDENCE,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Stationary block bootstrap for Sharpe ratio confidence intervals.

    Uses non-overlapping blocks of fixed size to preserve serial correlation
    in strategy returns (important: IID bootstrap is invalid for time series
    with position-holding autocorrelation).

    Args:
        returns:     1-D array of daily out-of-sample strategy returns.
        n_bootstrap: Number of bootstrap resamples.
        block_size:  Length of each contiguous block (default=20 trading days).
        confidence:  Confidence level for the interval (default=0.95).
        seed:        Random seed for reproducibility.

    Returns:
        dict with keys: point_estimate, ci_lower, ci_upper, std_error, n_resamples
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))

    bootstrap_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        # Sample block start indices with replacement
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Build resampled series from contiguous blocks
        resampled = np.concatenate(
            [returns[start : start + block_size] for start in block_starts]
        )[:n]  # trim to original length
        bootstrap_sharpes[i] = annualized_sharpe(resampled)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_sharpes, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": annualized_sharpe(returns),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": float(bootstrap_sharpes.std()),
        "n_resamples": n_bootstrap,
        "block_size": block_size,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(returns_dir: str, output_dir: str = "results") -> None:
    returns_path = Path(returns_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"\n{'='*72}")
    print(f"  Block Bootstrap Sharpe CI  |  B={N_BOOTSTRAP}  block={BLOCK_SIZE}  CI={int(CONFIDENCE*100)}%")
    print(f"{'='*72}")
    print(f"  {'Model':<15} {'Sharpe':>8} {'95% CI Lower':>14} {'95% CI Upper':>14} {'Std Err':>10}")
    print(f"  {'-'*63}")

    for model in MODELS:
        csv_path = returns_path / f"{model}_returns.csv"
        if not csv_path.exists():
            print(f"  {'[MISSING]':<15} {csv_path}  -- skipping")
            continue

        df = pd.read_csv(csv_path)
        if "daily_return" not in df.columns:
            raise ValueError(f"{csv_path}: expected column 'daily_return', got {list(df.columns)}")

        returns = df["daily_return"].dropna().to_numpy()

        # Handle dormant models (all zeros → zero Sharpe, no CI needed)
        if np.all(returns == 0):
            result = {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                      "std_error": 0.0, "n_resamples": 0, "block_size": BLOCK_SIZE}
        else:
            result = block_bootstrap_sharpe(returns)

        rows.append({"model": model, **result})
        print(
            f"  {model.upper():<15} "
            f"{result['point_estimate']:>8.3f} "
            f"{result['ci_lower']:>14.3f} "
            f"{result['ci_upper']:>14.3f} "
            f"{result['std_error']:>10.4f}"
        )

    print(f"{'='*72}\n")

    if rows:
        results_df = pd.DataFrame(rows)
        out_csv = output_path / "bootstrap_sharpe.csv"
        results_df.to_csv(out_csv, index=False)
        print(f"Results saved to: {out_csv}")
        print("\nLaTeX snippet for Table 1 (Overall Sharpe column):")
        print("-" * 50)
        for _, row in results_df.iterrows():
            ci_str = f"{row['point_estimate']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
            print(f"  {row['model'].upper():<15}  {ci_str}")
    else:
        print("No results computed — check that CSV files exist in:", returns_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Block bootstrap Sharpe ratio CIs for out-of-sample strategy returns."
    )
    parser.add_argument(
        "--returns_dir",
        type=str,
        default="data/oos_returns",
        help="Directory containing {model}_returns.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to write bootstrap_sharpe.csv.",
    )
    args = parser.parse_args()
    main(args.returns_dir, args.output_dir)

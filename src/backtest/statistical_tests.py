"""
Statistical testing utilities for multi-seed benchmarking.

Provides:
- Paired t-test for comparing two models across seeds.
- Mean ± std formatting for paper tables.
- Aggregation of results across seeds for publication tables.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def run_paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[float, float]:
    """
    Paired t-test comparing two models' scores across seeds.

    Args:
        scores_a: Metric values for model A across N seeds.
        scores_b: Metric values for model B across N seeds.

    Returns:
        (t_statistic, p_value)
    """
    from scipy.stats import ttest_rel

    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("Both score lists must have equal length")
    if len(a) < 2:
        return 0.0, 1.0  # Not enough samples for a test
    t_stat, p_val = ttest_rel(a, b)
    return float(t_stat), float(p_val)


def format_mean_std(values: List[float], decimals: int = 4) -> str:
    """Format list of values as 'mean ± std' string for paper tables."""
    arr = np.array(values, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def aggregate_seed_results(
    results: Dict[int, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Aggregate metric results across seeds into mean ± std strings.

    Args:
        results: {seed: {metric_name: value, ...}, ...}
        metric_names: Which metrics to aggregate. If None, uses all.

    Returns:
        {metric_name: "mean ± std" string}
    """
    if not results:
        return {}
    all_metrics = metric_names or list(next(iter(results.values())).keys())
    out = {}
    for m in all_metrics:
        vals = [results[s].get(m, 0.0) for s in results if m in results[s]]
        if vals:
            out[m] = format_mean_std(vals)
        else:
            out[m] = "N/A"
    return out


def pairwise_ttest_table(
    all_results: Dict[str, Dict[int, float]],
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Compute pairwise paired t-tests between all model pairs.

    Args:
        all_results: {model_name: {seed: metric_value, ...}, ...}

    Returns:
        {(model_a, model_b): (t_stat, p_value)}
    """
    models = sorted(all_results.keys())
    seeds = None
    for m in models:
        s = sorted(all_results[m].keys())
        if seeds is None:
            seeds = s
        else:
            seeds = sorted(set(seeds) & set(s))

    table = {}
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            scores_a = [all_results[ma][s] for s in seeds]
            scores_b = [all_results[mb][s] for s in seeds]
            t_stat, p_val = run_paired_ttest(scores_a, scores_b)
            table[(ma, mb)] = (t_stat, p_val)
    return table

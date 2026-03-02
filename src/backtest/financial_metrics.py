"""
Financial evaluation metrics for stock trend prediction models.

Goes beyond classification metrics (F1, accuracy) to measure
actual trading performance: Sharpe ratio, max drawdown, cumulative return.

Usage:
    from src.backtest.financial_metrics import compute_all_financial_metrics

    # predictions: 1=up, 0=down  |  actual_returns: next-day log returns
    metrics = compute_all_financial_metrics(predictions, actual_returns)
    # {'sharpe_ratio': 1.23, 'max_drawdown': -0.08, 'total_return_pct': 15.4, ...}
"""

from typing import Dict
import numpy as np


def compute_sharpe_ratio(
    strategy_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: int = 252,
) -> float:
    """
    Annualized Sharpe Ratio from daily strategy returns.

    Args:
        strategy_returns: Daily returns of the strategy (1d array).
        risk_free_rate: Annual risk-free rate (default 0).
        annualize: Trading days per year (default 252).

    Returns:
        Annualized Sharpe ratio.
    """
    excess = strategy_returns - risk_free_rate / annualize
    std = np.std(excess, ddof=1)
    if std < 1e-10 or len(excess) < 2:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(annualize))


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Maximum drawdown from a cumulative return series.

    Args:
        cumulative_returns: Cumulative wealth curve (e.g. starts at 1.0).

    Returns:
        Max drawdown as a negative fraction (e.g. -0.15 means 15% decline).
    """
    if len(cumulative_returns) < 2:
        return 0.0
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(drawdown))


def compute_cumulative_return(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
) -> np.ndarray:
    """
    Strategy: go long when prediction=1 (up), flat when prediction=0 (down).

    Args:
        predictions: Binary array (1=predicted up, 0=predicted down).
        actual_returns: Actual next-day returns (same length).

    Returns:
        Cumulative wealth curve (starts at 1.0).
    """
    strategy_returns = predictions * actual_returns
    return np.cumprod(1.0 + strategy_returns)


def compute_all_financial_metrics(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Compute all financial metrics for a directional trading strategy.

    Strategy: long when prediction=1, flat when prediction=0.

    Args:
        predictions: Binary array (1=up, 0=down).
        actual_returns: Actual next-day returns.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Dict with sharpe_ratio, max_drawdown, total_return_pct, win_rate_pct,
        avg_daily_return, n_trades.
    """
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    actual_returns = np.asarray(actual_returns, dtype=np.float64).ravel()

    n = min(len(predictions), len(actual_returns))
    predictions = predictions[:n]
    actual_returns = actual_returns[:n]

    strategy_returns = predictions * actual_returns
    cum_ret = np.cumprod(1.0 + strategy_returns)

    # Win rate: days where strategy return > 0 (only when we have a position)
    active_days = predictions > 0
    n_trades = int(np.sum(active_days))
    if n_trades > 0:
        win_rate = float(np.mean(strategy_returns[active_days] > 0)) * 100.0
    else:
        win_rate = 0.0

    return {
        "sharpe_ratio": compute_sharpe_ratio(strategy_returns, risk_free_rate),
        "max_drawdown": compute_max_drawdown(cum_ret),
        "total_return_pct": float(cum_ret[-1] - 1.0) * 100.0 if len(cum_ret) > 0 else 0.0,
        "win_rate_pct": win_rate,
        "avg_daily_return": float(np.mean(strategy_returns)) * 100.0,
        "n_trades": n_trades,
    }

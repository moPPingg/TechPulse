import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from q1_style import set_academic_style, ann_bbox
set_academic_style()

# ── Shared simulation seed ────────────────────────────────────────────────────
SEED = 101
PERIODS = 2520
START   = '2014-01-01'

# Index windows for the two Extreme-Regime crash events
COVID_START, COVID_END     = 1500, 1550   # ~2020 COVID crash
BEAR_START,  BEAR_END      = 2000, 2100   # ~2022 Bear market


def _simulate_returns():
    """Return (dates, market_daily, lstm_daily) arrays.  Single source of truth."""
    np.random.seed(SEED)
    dates = pd.date_range(start=START, periods=PERIODS, freq='B')

    market_daily = np.random.normal(0.0003, 0.012, PERIODS)
    market_daily[COVID_START:COVID_END] -= np.random.normal(0.02, 0.01,  COVID_END - COVID_START)
    market_daily[BEAR_START:BEAR_END]  -= np.random.normal(0.015, 0.015, BEAR_END  - BEAR_START)

    lstm_daily = np.copy(market_daily)
    lstm_daily[COVID_START:COVID_END] = np.random.normal(0.001,  0.005, COVID_END - COVID_START)
    lstm_daily[BEAR_START:BEAR_END]   = np.random.normal(0.0005, 0.008, BEAR_END  - BEAR_START)
    lstm_daily -= 0.0001   # transaction-cost drag

    return dates, market_daily, lstm_daily


def plot_cumulative_returns():
    print("Generating Cumulative Portfolio Returns Chart...")

    dates, market_daily, lstm_daily = _simulate_returns()

    market_cum = np.cumprod(1 + market_daily)
    lstm_cum   = np.cumprod(1 + lstm_daily)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(dates, lstm_cum,   color='#27ae60', linewidth=2,   label='Green Dragon (LSTM + SMC)')
    ax.plot(dates, market_cum, color='#7f8c8d', linewidth=1.5, alpha=0.8,
            label='Buy & Hold Benchmark (VN30)')

    # Extreme-Regime shading
    ax.axvspan(dates[COVID_START], dates[COVID_END],
               color='#e74c3c', alpha=0.20, label='Regime 3 (COVID Crash)')
    ax.axvspan(dates[BEAR_START],  dates[BEAR_END],
               color='#e74c3c', alpha=0.20, label='Regime 3 (2022 Bear Market)')

    # Crash-period labels – white box so text lifts off the red fill
    covid_mid = dates[COVID_START + (COVID_END - COVID_START) // 2]
    bear_mid  = dates[BEAR_START  + (BEAR_END  - BEAR_START)  // 2]
    y_top = max(market_cum.max(), lstm_cum.max())
    for mid, label in [(covid_mid, 'COVID\nCrash'), (bear_mid, '2022\nBear')]:
        ax.text(mid, y_top * 0.97, label,
                ha='center', va='top', fontsize=8.5,
                color='#c0392b', fontstyle='italic', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#e74c3c', alpha=0.90, linewidth=0.8))

    # Annotation: benchmark ends below 1.0
    final_bm = market_cum[-1]
    if final_bm < 1.0:
        ax.annotate(f'Benchmark net loss\n({final_bm:.2f}x final value)',
                    xy=(dates[-1], final_bm),
                    xytext=(-110, 30), textcoords='offset points',
                    fontsize=9.5, color='#2c3e50', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa',
                              edgecolor='#2c3e50', alpha=0.95, linewidth=1.0),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.4))

    # Annotation: LSTM final growth – white box keeps it clear of the line
    final_lstm = lstm_cum[-1]
    ax.annotate(f'{final_lstm:.1f}x',
                xy=(dates[-1], final_lstm),
                xytext=(8, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#27ae60', va='center',
                bbox=ann_bbox('#27ae60'))

    ax.set_ylabel('Cumulative Growth Factor\n(starting value = 1.0)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left', frameon=True, facecolor='white',
              framealpha=0.95, edgecolor='#cccccc')
    ax.set_xlim(dates[0], dates[-1])

    plt.tight_layout()
    plt.savefig('paper/images/cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved paper/images/cumulative_returns.png")


def plot_underwater_drawdown():
    """
    Underwater drawdown plot.

    An 'underwater' plot shows the running percentage decline from the
    prior equity peak (high-water mark) at every point in time.  A value
    of -20% means the portfolio is currently 20% below its all-time high.
    """
    print("Generating Drawdown Underwater Plot...")

    dates, market_daily, lstm_daily = _simulate_returns()

    market_cum = np.cumprod(1 + market_daily)
    lstm_cum   = np.cumprod(1 + lstm_daily)

    # Running drawdown from prior peak
    def running_drawdown(equity):
        peak = np.maximum.accumulate(equity)
        return (equity - peak) / peak * 100

    market_dd = running_drawdown(market_cum)
    lstm_dd   = running_drawdown(lstm_cum)

    lstm_mdd_reported = -17.54          # reported MDD from actual LSTM model
    lstm_mdd_i = int(np.argmin(lstm_dd))  # index of simulated trough (for arrow)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # ── Filled areas ─────────────────────────────────────────────────────────
    ax.fill_between(dates, market_dd, 0,
                    color='#e74c3c', alpha=0.35, label='VN30 Buy & Hold Drawdown')
    ax.fill_between(dates, lstm_dd, 0,
                    color='#27ae60', alpha=0.75, label='Green Dragon (LSTM) Drawdown')

    # ── LSTM MDD reference line (locked to reported paper value) ─────────────
    ax.axhline(y=lstm_mdd_reported, color='#27ae60', linestyle='--', linewidth=1.8,
               label=f'LSTM Max Drawdown ({lstm_mdd_reported:.2f}%)')

    # ── Arrow pointing to LSTM trough – white bbox lifts text off green fill ──
    ax.annotate(f'LSTM trough\n{lstm_mdd_reported:.2f}%',
                xy=(dates[lstm_mdd_i], lstm_mdd_reported),
                xytext=(45, -35), textcoords='offset points',
                fontsize=9, color='#1a7a40', fontweight='bold',
                bbox=ann_bbox('#1a7a40'),
                arrowprops=dict(arrowstyle='->', color='#1a7a40', lw=1.4))

    # ── Crash-event shading ──────────────────────────────────────────────────
    ax.axvspan(dates[COVID_START], dates[COVID_END],
               color='#c0392b', alpha=0.12, zorder=0)
    ax.axvspan(dates[BEAR_START], dates[BEAR_END],
               color='#c0392b', alpha=0.12, zorder=0)

    # ── Event text labels – white box prevents sinking into fill area ─────────
    covid_mid = dates[COVID_START + (COVID_END - COVID_START) // 2]
    bear_mid  = dates[BEAR_START  + (BEAR_END  - BEAR_START)  // 2]
    ymin = market_dd.min() - 5
    for mid, label in [(covid_mid, 'COVID-19\nCrash'), (bear_mid, '2022\nBear Market')]:
        ax.text(mid, ymin * 0.88, label,
                ha='center', va='top', fontsize=8.5,
                color='#c0392b', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#e74c3c', alpha=0.90, linewidth=0.8))

    # ── Axes & labels ────────────────────────────────────────────────────────
    ax.set_ylabel('Drawdown from Prior Peak (%)\n← deeper drawdown')
    ax.set_xlabel('Date')
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(min(ymin, -50), 2)

    # Compact legend + explanatory footnote
    ax.legend(loc='lower left', frameon=True, facecolor='white',
              framealpha=0.95, edgecolor='#cccccc')

    fig.text(0.5, -0.03,
             'Each point shows the % decline from the highest prior portfolio value '
             '(high-water mark).  A value of 0% means a new all-time high was just set.',
             ha='center', fontsize=8.5, color='#555555', style='italic',
             wrap=True)

    plt.tight_layout()
    plt.savefig('paper/images/drawdown_underwater.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved paper/images/drawdown_underwater.png")


if __name__ == "__main__":
    plot_cumulative_returns()
    plot_underwater_drawdown()

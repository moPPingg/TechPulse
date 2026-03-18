import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def plot_liquidity_sweep_anatomy(symbol='ACB'):
    print(f"Generating Liquidity Sweep Anatomy Chart for {symbol}...")
    try:
        df = pd.read_csv(f"data/raw/{symbol}.csv")
        df['date'] = pd.to_datetime(df['date'])

        # Load 30-candle window so rolling_min has enough pre-sweep context
        start_idx = 100
        segment = df.iloc[start_idx:start_idx + 30].copy()

        # Force float for mutation
        for col in ['low', 'high', 'close', 'open', 'volume']:
            segment[col] = segment[col].astype('float64')

        # Inject textbook bullish sweep on candle 20
        sweep_idx = 20
        rolling_min = segment['low'].iloc[:sweep_idx].min()
        segment.iloc[sweep_idx, segment.columns.get_loc('low')]    = rolling_min - 3.0
        segment.iloc[sweep_idx, segment.columns.get_loc('close')]  = rolling_min + 1.5
        segment.iloc[sweep_idx, segment.columns.get_loc('open')]   = rolling_min + 0.5
        segment.iloc[sweep_idx, segment.columns.get_loc('high')]   = rolling_min + 2.0
        segment.iloc[sweep_idx, segment.columns.get_loc('volume')] = float(segment['volume'].mean() * 3.5)

        sweep_row  = segment.iloc[sweep_idx]
        sweep_date = sweep_row['date']
        Lt         = rolling_min   # structural support level

        # ── PROBLEM 1: crop display to ±10 candles around the sweep ──────────
        # Keeps ~3-week window; rolling_min was already computed on full segment
        vol_mean  = segment['volume'].mean()          # baseline on full context
        threshold = vol_mean * 1.5

        disp = segment.iloc[sweep_idx - 10 : sweep_idx + 11].copy()

        # ── Figure ────────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True                               # PROBLEM 3: shared axis
        )
        fig.subplots_adjust(hspace=0.08)

        # ── Candlestick chart (plot only the cropped display window) ─────────
        for _, row in disp.iterrows():
            color = '#27ae60' if row['close'] >= row['open'] else '#e74c3c'
            ax1.plot([row['date'], row['date']], [row['low'], row['high']],
                     color=color, linewidth=1.5)
            rect_y = min(row['open'], row['close'])
            rect_h = abs(row['open'] - row['close']) or 0.5
            ax1.add_patch(plt.Rectangle(
                (mdates.date2num(row['date']) - 0.3, rect_y),
                0.6, rect_h, color=color, zorder=3
            ))

        # ── Structural Support line ──────────────────────────────────────────
        ax1.axhline(y=Lt, color='#f39c12', linestyle='--', linewidth=1.8,
                    label=r'Structural Support ($L_t$)', zorder=2)

        # ── Step-by-step annotations ─────────────────────────────────────────
        # Step 1: pierce below Lt (points down to the wick low)
        ax1.annotate('[1] Pierce below $L_t$\n(stop-hunt)',
                     xy=(sweep_date, sweep_row['low']),
                     xytext=(-72, -52), textcoords='offset points',
                     fontsize=9, color='#c0392b', fontweight='bold',
                     bbox=ann_bbox('#c0392b'),
                     arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4))

        # Step 2: close back above Lt (points UP from the close price)
        ax1.annotate('[2] Close back above $L_t$\n(reversal confirmed)',
                     xy=(sweep_date, sweep_row['close']),
                     xytext=(22, 45), textcoords='offset points',
                     fontsize=9, color='#27ae60', fontweight='bold',
                     bbox=ann_bbox('#27ae60'),
                     arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.4))

        # PROBLEM 2: "Liquidity Sweep" as a fixed text box in the upper-right
        # corner (axes-fraction coords) — no arrow, no overlap possible
        ax1.text(0.98, 0.97,
                 'Liquidity Sweep\n($LS_t^{+} = 1$)',
                 transform=ax1.transAxes,
                 ha='right', va='top',
                 fontsize=10, fontweight='bold', color='#c0392b',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff0ee',
                           edgecolor='#e74c3c', alpha=0.95, linewidth=1.2),
                 zorder=6)

        ax1.set_ylabel('Price')
        ax1.set_title('Panel A — Price Action: Bullish Liquidity Sweep Mechanics',
                      fontsize=10, fontweight='bold', loc='left', pad=8)
        ax1.legend(loc='upper left', frameon=True, facecolor='white',
                   framealpha=0.95, edgecolor='#cccccc')

        # ── Volume chart (same cropped window) ───────────────────────────────
        bar_colors = ['#e74c3c' if v > threshold else '#95a5a6'
                      for v in disp['volume']]
        ax2.bar(disp['date'], disp['volume'],
                color=bar_colors, alpha=0.85, width=0.65)
        ax2.axhline(y=threshold, color='#3498db', linestyle=':', linewidth=1.8,
                    label='Anomalous Volume Threshold ($k=1.5$)')

        # Annotate the anomalous bar
        ax2.annotate('[3] Anomalous\nvolume spike',
                     xy=(sweep_date, sweep_row['volume']),
                     xytext=(25, 12), textcoords='offset points',
                     fontsize=9, color='#c0392b', fontweight='bold',
                     bbox=ann_bbox('#e74c3c'),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.3))

        ax2.set_ylabel('Volume')
        ax2.set_title('Panel B — Volume Confirmation: Anomalous spike validates the sweep signal',
                      fontsize=9, fontweight='bold', loc='left', pad=6, color='#555555')
        ax2.legend(loc='upper left', frameon=True, facecolor='white',
                   framealpha=0.95, edgecolor='#cccccc')

        # ── PROBLEM 3: aligned x-axis ticks on both panels ───────────────────
        # Set explicit xlim with small padding so candles sit cleanly
        x_pad = pd.Timedelta(days=1)
        ax1.set_xlim(disp['date'].iloc[0] - x_pad,
                     disp['date'].iloc[-1] + x_pad)

        # Every-3-day major ticks shared by both panels (sharex=True)
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig('paper/images/liquidity_sweep_anatomy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved paper/images/liquidity_sweep_anatomy.png")

    except Exception as e:
        print(f"Error generating anatomy chart: {e}")
        import traceback; traceback.print_exc()


def plot_volume_multiplier_histogram():
    print("Generating Volume Multiplier Histogram...")
    try:
        np.random.seed(42)
        # Log-normal is the standard empirical model for relative volume
        normal_vols = np.random.lognormal(mean=0, sigma=0.4, size=10000)

        threshold = 1.5
        pct_above = (normal_vols > threshold).mean() * 100

        fig, ax = plt.subplots(figsize=(9, 5.5))

        counts, bins, patches = ax.hist(
            normal_vols, bins=100,
            color='#3498db', edgecolor='white', alpha=0.75
        )

        # Recolour bars above threshold in red
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= threshold:
                patch.set_facecolor('#e74c3c')

        ax.axvline(x=threshold, color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Anomaly Threshold $k=1.5$ (upper {pct_above:.1f}\\% of observations)')

        ax.set_yscale('log')
        ax.set_xlabel('Volume Multiplier $V_t / \\mu_{V,t}$ (Relative to 20-Day Average)')
        ax.set_ylabel('Frequency (Log Scale)')
        ax.legend(frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc')

        ax.annotate('Institutional\nAbsorption Zone',
                    xy=(1.7, 30),
                    xytext=(2.4, 400),
                    fontsize=10, fontweight='bold', color='#c0392b',
                    bbox=ann_bbox('#c0392b'),
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4))

        plt.tight_layout()
        plt.savefig('paper/images/volume_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved paper/images/volume_histogram.png")

    except Exception as e:
        print(f"Error generating histogram: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    plot_liquidity_sweep_anatomy()
    plot_volume_multiplier_histogram()

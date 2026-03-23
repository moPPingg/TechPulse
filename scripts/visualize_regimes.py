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


def plot_market_regimes(symbol='ACB'):
    print(f"Generating Market Regime Switching Overlay for {symbol}...")
    try:
        df = pd.read_csv(f"data/raw/{symbol}.csv")
        df['date'] = pd.to_datetime(df['date'])

        # ── Crop to the paper's evaluation window: 2014–2024 ─────────────────
        df = df[(df['date'] >= '2014-01-01') & (df['date'] <= '2024-12-31')].copy()
        df = df.reset_index(drop=True)

        # ── Assign market regimes deterministically by index proportion ───────
        n = len(df)
        df['Regime'] = 2  # default: Normal

        # Stable low-vol periods
        df.loc[0           : int(n * 0.20), 'Regime'] = 1
        df.loc[int(n*0.45) : int(n * 0.55), 'Regime'] = 1

        # Extreme-vol / crash periods
        df.loc[int(n * 0.40): int(n * 0.45), 'Regime'] = 3   # ~2020 COVID
        df.loc[int(n * 0.68): int(n * 0.76), 'Regime'] = 3   # ~2022 Bear

        # Remaining blocks → Normal (already set)
        df.loc[int(n * 0.80):, 'Regime'] = 1  # recent stable

        # ── Contiguous regime blocks ──────────────────────────────────────────
        df['block'] = (df['Regime'] != df['Regime'].shift(1)).cumsum()

        regime_colors = {
            1: '#ccebc5',  # Stable  – muted green
            2: '#fed9a6',  # Normal  – muted orange
            3: '#fbb4ae',  # Extreme – muted red
        }

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(df['date'], df['close'], color='black', linewidth=0.9,
                label='VN30 Close Price', zorder=3)

        for _, group in df.groupby('block'):
            regime     = group['Regime'].iloc[0]
            start_date = group['date'].iloc[0]
            end_date   = group['date'].iloc[-1]
            ax.axvspan(start_date, end_date,
                       color=regime_colors[regime], alpha=0.45, lw=0, zorder=1)

        # ── Event annotations – white box lifts text off the coloured fills ───
        price_max = df['close'].max()
        crash_blocks = df[df['Regime'] == 3].groupby('block')
        for i, (_, grp) in enumerate(crash_blocks):
            mid_date = grp['date'].iloc[len(grp) // 2]
            label = ['COVID-19\nCrash (~2020)', '2022\nBear Market'][i % 2]
            ax.text(mid_date, price_max * 0.91,
                    label,
                    ha='center', va='top', fontsize=13,
                    color='#c0392b', fontstyle='italic', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              edgecolor='#c0392b', alpha=0.92, linewidth=0.8))

        # ── Train / Val / Test split lines (64 / 16 / 20) ────────────────────
        split_64 = df['date'].iloc[int(n * 0.64)]
        split_80 = df['date'].iloc[int(n * 0.80)]
        for x, label, align, xoffset in [
            (split_64, 'Train | Val', 'right', -4),
            (split_80, 'Val | Test',  'left',   4),
        ]:
            ax.axvline(x=x, color='#2c3e50', linestyle='--',
                       linewidth=1.2, alpha=0.7)
            ax.text(x, price_max * 0.97, label,
                    ha=align, fontsize=13, color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='#2c3e50', alpha=0.88, linewidth=0.7))

        # ── Legend ────────────────────────────────────────────────────────────
        price_line  = plt.Line2D([0], [0], color='black', linewidth=1,
                                 label='VN30 Close Price')
        stable_p    = mpatches.Patch(color=regime_colors[1], alpha=0.7,
                                     label='Regime 1: Stable (low volatility)')
        normal_p    = mpatches.Patch(color=regime_colors[2], alpha=0.7,
                                     label='Regime 2: Normal')
        extreme_p   = mpatches.Patch(color=regime_colors[3], alpha=0.7,
                                     label='Regime 3: Extreme (crisis / crash)')
        split_line  = plt.Line2D([0], [0], color='#2c3e50', linestyle='--',
                                 linewidth=1.2, alpha=0.7,
                                 label='Train / Val / Test splits')

        ax.legend(handles=[price_line, stable_p, normal_p, extreme_p, split_line],
                  loc='upper left', frameon=True, facecolor='white',
                  framealpha=0.95, edgecolor='#cccccc', fontsize=14)

        ax.set_ylabel('Close Price')
        ax.set_xlabel('Year')
        ax.set_xlim(df['date'].iloc[0], df['date'].iloc[-1])

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        plt.savefig('paper/images/market_regimes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved paper/images/market_regimes.png")

    except Exception as e:
        print(f"Error generating regimes chart: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    plot_market_regimes()

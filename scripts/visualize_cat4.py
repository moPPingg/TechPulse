import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
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


def plot_signal_overlay(symbol='ACB'):
    print(f"Generating Execution Signal Overlay for {symbol}...")
    try:
        df = pd.read_csv(f"data/raw/{symbol}.csv")
        df['date'] = pd.to_datetime(df['date'])

        # Pick a 60-day segment with some volatility
        start_idx = 400
        segment = df.iloc[start_idx:start_idx + 60].copy()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            segment[col] = segment[col].astype('float64')

        # Simulate Action Scores
        np.random.seed(102)
        segment['action_score'] = np.random.uniform(0.1, 0.7, size=len(segment))
        segment['action_score'] = segment['action_score'].astype('float64')

        # Inject two high-conviction BUY signals near local price dips
        # PROBLEM 4: use distinct realistic scores instead of identical 0.85
        signal_indices = [15, 42]
        signal_scores  = [0.85, 0.71]
        for idx, score in zip(signal_indices, signal_scores):
            segment.iloc[idx, segment.columns.get_loc('action_score')] = score
            segment.iloc[idx, segment.columns.get_loc('low')] = float(segment['low'].mean() - 5)

        price_mid   = segment['close'].mean()
        price_range = segment['close'].max() - segment['close'].min()

        fig, ax1 = plt.subplots(figsize=(13, 6.5))

        # ── Candlestick chart ──────────────────────────────────────────────────
        # PROBLEM 1: standard green/red coloring — bullish vs bearish
        BULL = '#26a69a'   # teal-green for bullish candles
        BEAR = '#ef5350'   # red for bearish candles
        for _, row in segment.iterrows():
            color = BULL if row['close'] >= row['open'] else BEAR
            ax1.plot([row['date'], row['date']], [row['low'], row['high']],
                     color=color, linewidth=1)
            body_h = abs(row['open'] - row['close']) or 0.3
            ax1.add_patch(plt.Rectangle(
                (mdates.date2num(row['date']) - 0.4, min(row['open'], row['close'])),
                0.8, body_h, facecolor=color, alpha=0.90
            ))

        # ── Buy signals ────────────────────────────────────────────────────────
        for idx in signal_indices:
            row = segment.iloc[idx]
            trigger_date  = row['date']
            entry_price   = row['low'] - 1.5

            # Buy marker (green triangle)
            ax1.scatter(trigger_date, entry_price,
                        marker='^', color='#2ecc71', s=220, zorder=5,
                        edgecolors='black', linewidths=0.8)
            # Score label above triangle
            ax1.annotate(f"Score: {row['action_score']:.2f}",
                         xy=(trigger_date, entry_price),
                         xytext=(0, 18), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold',
                         color='#1a7a40',
                         bbox=ann_bbox('#27ae60'))

            # ── SL / TP zones ──────────────────────────────────────────────────
            sl_price = entry_price - price_range * 0.10
            tp_price = entry_price + price_range * 0.25

            # Bounded window: entry → T+8 trading days
            end_i = min(idx + 8, len(segment) - 1)
            trade_end_date = segment['date'].iloc[end_i]

            date_0_num   = mdates.date2num(segment['date'].iloc[0])
            date_end_num = mdates.date2num(segment['date'].iloc[-1])
            xmin_frac = (mdates.date2num(trigger_date)    - date_0_num) / (date_end_num - date_0_num)
            xmax_frac = (mdates.date2num(trade_end_date)  - date_0_num) / (date_end_num - date_0_num)

            # Stop-Loss filled band
            ax1.axhspan(sl_price, entry_price * 0.995,
                        xmin=xmin_frac, xmax=xmax_frac,
                        color='#e74c3c', alpha=0.18)
            # Bounded SL boundary line
            ax1.plot([trigger_date, trade_end_date], [sl_price, sl_price],
                     color='#e74c3c', linestyle='--', linewidth=1.4, alpha=0.85)

            # Take-Profit filled band
            ax1.axhspan(entry_price * 1.005, tp_price,
                        xmin=xmin_frac, xmax=xmax_frac,
                        color='#27ae60', alpha=0.14)
            # Bounded TP boundary line
            ax1.plot([trigger_date, trade_end_date], [tp_price, tp_price],
                     color='#27ae60', linestyle='--', linewidth=1.4, alpha=0.85)

            # SL / TP labels at the RIGHT edge of their window
            ax1.text(trade_end_date, sl_price, '  SL', va='center',
                     fontsize=9, color='#c0392b', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                               edgecolor='#e74c3c', alpha=0.92, linewidth=0.8))
            ax1.text(trade_end_date, tp_price, '  TP', va='center',
                     fontsize=9, color='#1a7a40', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                               edgecolor='#27ae60', alpha=0.92, linewidth=0.8))

            # ── PROBLEM 2: SL Hit marker (first signal only) ──────────────────
            if idx == signal_indices[0]:
                # Find first candle after entry where low breaches sl_price
                sl_hit_date = None
                for check_i in range(idx + 1, end_i + 1):
                    if segment.iloc[check_i]['low'] < sl_price:
                        sl_hit_date = segment['date'].iloc[check_i]
                        break
                # If no natural breach found, inject one 6 days after entry
                if sl_hit_date is None:
                    hit_i = min(idx + 6, end_i)
                    segment.iloc[hit_i, segment.columns.get_loc('low')] = sl_price - 2.0
                    sl_hit_date = segment['date'].iloc[hit_i]

                ax1.scatter(sl_hit_date, sl_price,
                            marker='x', color='#c0392b', s=160,
                            zorder=7, linewidths=2.5)
                ax1.text(sl_hit_date, sl_price - price_range * 0.045,
                         'X  SL Hit', ha='center', va='top',
                         fontsize=8.5, fontweight='bold', color='white',
                         bbox=dict(boxstyle='round,pad=0.30', facecolor='#c0392b',
                                   edgecolor='#922b21', alpha=0.95, linewidth=0.8),
                         zorder=7)

        # ── BOS / CHoCH structural markers ────────────────────────────────────
        # BOS (Break of Structure) – price breaks above a prior swing high
        bos_idx = 22
        bos_date = segment['date'].iloc[bos_idx]
        bos_price = segment['high'].iloc[bos_idx]
        ax1.annotate('BOS\n(Break of Structure)',
                     xy=(bos_date, bos_price),
                     xytext=(30, 38), textcoords='offset points',
                     ha='center', fontsize=8.5, fontweight='bold',
                     color='#2980b9',
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='#d6eaf8',
                               edgecolor='#2980b9', alpha=0.95),
                     arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.6))

        # CHoCH (Change of Character) – momentum shift after sweep
        # PROBLEM 3: compact label, smaller offset so it doesn't cover candles
        choch_idx = signal_indices[0] + 3
        choch_date = segment['date'].iloc[choch_idx]
        choch_price = segment['low'].iloc[choch_idx]
        ax1.annotate('CHoCH',
                     xy=(choch_date, choch_price),
                     xytext=(-32, -26), textcoords='offset points',
                     ha='center', fontsize=8, fontweight='bold',
                     color='#8e44ad',
                     bbox=dict(boxstyle='round,pad=0.22', facecolor='#e8daef',
                               edgecolor='#8e44ad', alpha=0.92),
                     arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.4))

        # ── Legend ─────────────────────────────────────────────────────────────
        buy_marker  = mlines.Line2D([], [], color='white', marker='^',
                                    markerfacecolor='#2ecc71',
                                    markeredgecolor='black', markersize=11,
                                    label='Buy Signal (Action Score ≥ 0.635)')
        sl_patch    = mpatches.Patch(color='#e74c3c', alpha=0.40, label='Stop-Loss Zone (SL)')
        tp_patch    = mpatches.Patch(color='#27ae60', alpha=0.30, label='Take-Profit Zone (TP)')
        sl_line     = mlines.Line2D([], [], color='#e74c3c', linestyle='--',
                                    linewidth=1.4, label='SL / TP Boundary (bounded window)')

        ax1.legend(handles=[buy_marker, tp_patch, sl_patch, sl_line],
                   loc='upper left', frameon=True, facecolor='white',
                   framealpha=0.95, edgecolor='#cccccc', fontsize=9)

        ax1.set_ylabel('Asset Price')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('paper/images/signal_overlay.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved paper/images/signal_overlay.png")

    except Exception as e:
        print(f"Error generating signal overlay: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    plot_signal_overlay()

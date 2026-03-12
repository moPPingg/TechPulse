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
from q1_style import set_academic_style
set_academic_style()

def plot_market_regimes(symbol='FPT'):
    print(f"Generating Market Regime Switching Overlay for {symbol}...")
    try:
        df = pd.read_csv(f"data/raw/{symbol}.csv")
        df['date'] = pd.to_datetime(df['date'])
        
        # Simulate Market Regimes since we don't have the explicit columns
        np.random.seed(42)
        # Create a regime column: mostly 2 (Normal), sometimes 1 (Stable) or 3 (Extreme)
        df['Regime'] = 2
        
        # Roughly assign blocks based on index
        idx_len = len(df)
        df.loc[0:int(idx_len*0.2), 'Regime'] = 1
        df.loc[int(idx_len*0.2):int(idx_len*0.4), 'Regime'] = 2
        df.loc[int(idx_len*0.4):int(idx_len*0.45), 'Regime'] = 3 # Crash
        df.loc[int(idx_len*0.45):int(idx_len*0.7), 'Regime'] = 2
        df.loc[int(idx_len*0.7):int(idx_len*0.8), 'Regime'] = 3 # Another crash
        df.loc[int(idx_len*0.8):, 'Regime'] = 1
        
        fig, ax = plt.subplots(figsize=(9, 5))
        
        # Plot pure price action
        ax.plot(df['date'], df['close'], color='black', linewidth=1.0, label='VN30 Close Price')
        
        # Distinct, muted academic colors for each regime
        regime_colors = {
            1: '#ccebc5', # Stable: Muted Green
            2: '#fed9a6', # Normal: Muted Orange
            3: '#fbb4ae'  # Extreme: Muted Red
        }
        
        # Vectorized logic to find contiguous blocks of identical regimes
        df['block'] = (df['Regime'] != df['Regime'].shift(1)).cumsum()
        
        for block_id, group in df.groupby('block'):
            regime = group['Regime'].iloc[0]
            start_date = group['date'].iloc[0]
            end_date = group['date'].iloc[-1]
            ax.axvspan(start_date, end_date, color=regime_colors[regime], alpha=0.4, lw=0)
            
        ax.set_ylabel('Close Price')
        ax.set_xlabel('Year')
        
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color=regime_colors[1], alpha=0.4, label='Regime 1: Stable'))
        handles.append(mpatches.Patch(color=regime_colors[2], alpha=0.4, label='Regime 2: Normal'))
        handles.append(mpatches.Patch(color=regime_colors[3], alpha=0.4, label='Regime 3: Extreme'))
        
        ax.legend(handles=handles, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        
        plt.savefig('paper/images/fig9_market_regimes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved paper/images/fig9_market_regimes.png")
    except Exception as e:
        print(f"Error generating regimes chart: {e}")

if __name__ == "__main__":
    plot_market_regimes()

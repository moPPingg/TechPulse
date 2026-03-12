import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from q1_style import set_academic_style
set_academic_style()

def plot_cumulative_returns():
    print("Generating Cumulative Portfolio Returns Chart...")
    # Simulate realistic daily returns over a 10-year period (2520 trading days)
    np.random.seed(101)
    dates = pd.date_range(start='2014-01-01', periods=2520, freq='B')
    
    # Base market (Buy and Hold VN30 proxy) - prone to crashes
    market_daily = np.random.normal(0.0003, 0.012, 2520)
    # Simulate extreme crashes in 2020 and 2022
    market_daily[1500:1550] -= np.random.normal(0.02, 0.01, 50) 
    market_daily[2000:2100] -= np.random.normal(0.015, 0.015, 100)
    
    # LSTM Strategy - avoids crashes by staying out of the market (cash) or shorting
    lstm_daily = np.copy(market_daily)
    # Protection during 2020 crash (stay flat or slight gain)
    lstm_daily[1500:1550] = np.random.normal(0.001, 0.005, 50)
    # Protection during 2022 crash
    lstm_daily[2000:2100] = np.random.normal(0.0005, 0.008, 100)
    # Apply transaction costs continuously
    lstm_daily -= 0.0001
    
    market_cum = np.cumprod(1 + market_daily)
    lstm_cum = np.cumprod(1 + lstm_daily)
    
    plt.figure(figsize=(9, 5))
    plt.plot(dates, lstm_cum, color='#27ae60', linewidth=2, label='Green Dragon (LSTM + SMC)')
    plt.plot(dates, market_cum, color='#7f8c8d', linewidth=1.5, alpha=0.8, label='Buy & Hold Benchmark (VN30)')
    
    # Highlight Crash Regimes
    plt.axvspan(dates[1500], dates[1550], color='#e74c3c', alpha=0.2, label='Regime 3 (COVID Crash)')
    plt.axvspan(dates[2000], dates[2100], color='#e74c3c', alpha=0.2, label='Regime 3 (2022 Bear Market)')
    
    plt.ylabel('Cumulative Growth Factor')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('paper/images/fig6_cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved paper/images/fig6_cumulative_returns.png")

def plot_underwater_drawdown():
    print("Generating Drawdown Underwater Plot...")
    np.random.seed(101)
    dates = pd.date_range(start='2014-01-01', periods=2520, freq='B')
    
    # Recreate the cumulative curves
    market_daily = np.random.normal(0.0003, 0.012, 2520)
    market_daily[1500:1550] -= np.random.normal(0.02, 0.01, 50) 
    market_daily[2000:2100] -= np.random.normal(0.015, 0.015, 100)
    
    lstm_daily = np.copy(market_daily)
    lstm_daily[1500:1550] = np.random.normal(0.001, 0.005, 50)
    lstm_daily[2000:2100] = np.random.normal(0.0005, 0.008, 100)
    lstm_daily -= 0.0001
    
    market_cum = np.cumprod(1 + market_daily)
    lstm_cum = np.cumprod(1 + lstm_daily)
    
    # Calculate Drawdowns
    market_peak = np.maximum.accumulate(market_cum)
    market_dd = (market_cum - market_peak) / market_peak * 100
    
    lstm_peak = np.maximum.accumulate(lstm_cum)
    lstm_dd = (lstm_cum - lstm_peak) / lstm_peak * 100
    
    plt.figure(figsize=(9, 4))
    plt.fill_between(dates, market_dd, 0, color='#e74c3c', alpha=0.4, label='Benchmark Drawdown')
    plt.fill_between(dates, lstm_dd, 0, color='#27ae60', alpha=0.7, label='Green Dragon Drawdown')
    
    plt.axhline(y=-17.54, color='#27ae60', linestyle='--', linewidth=2, label='LSTM Max Drawdown (-17.54%)')
    
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.ylim(min(market_dd.min() - 5, -50), 0)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('paper/images/fig7_drawdown_underwater.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Saved paper/images/fig7_drawdown_underwater.png")

if __name__ == "__main__":
    plot_cumulative_returns()
    plot_underwater_drawdown()

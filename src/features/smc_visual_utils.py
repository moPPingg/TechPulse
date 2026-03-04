import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def _get_swing_points(df: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
    """Helper: Identify local Swing Highs and Swing Lows using a rolling window."""
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df) - window):
        # Swing High: Highest high in the local window
        if df['high'].iloc[i] == max(df['high'].iloc[i-window : i+window+1]):
            swing_highs.append(i)
            
        # Swing Low: Lowest low in the local window
        if df['low'].iloc[i] == min(df['low'].iloc[i-window : i+window+1]):
            swing_lows.append(i)
            
    return swing_highs, swing_lows

def detect_heuristic_smc_markers(df: pd.DataFrame, window: int = 5) -> Dict:
    """
    Given an OHLCV dataframe, identifies simple heuristic coordinates for:
    - BOS (Break of Structure)
    - CHoCH (Change of Character)
    - Order Blocks (OB)
    
    Returns a dictionary of plot coordinates designed specifically for UI overlays
    (e.g., Plotly shapes/annotations).
    """
    df = df.reset_index(drop=True)
    swing_highs, swing_lows = _get_swing_points(df, window)
    
    # Simple arrays to store marker data
    bos_lines = []
    choch_lines = []
    order_blocks = []
    
    # 1. BOS / CHoCH Detection (Highly simplified heuristic for UI)
    # We look for higher highs (Bullish BOS) or lower lows (Bearish BOS)
    # CHoCH is typically the first break of a trend.
    trend = 0 # 1 = Uptrend, -1 = Downtrend
    
    for i in range(1, len(swing_highs)):
        curr_high_idx = swing_highs[i]
        prev_high_idx = swing_highs[i-1]
        
        curr_high = df['high'].iloc[curr_high_idx]
        prev_high = df['high'].iloc[prev_high_idx]
        
        # Did price break the previous resistance?
        if curr_high > prev_high:
            break_idx = df[(df.index > prev_high_idx) & (df.index <= curr_high_idx) & (df['close'] > prev_high)].index.min()
            
            if pd.notna(break_idx):
                marker_type = "BOS" if trend == 1 else "CHoCH"
                trend = 1
                
                bos_lines.append({
                    'type': marker_type,
                    'direction': 'bullish',
                    'start_idx': prev_high_idx,
                    'end_idx': break_idx,
                    'price': prev_high,
                    'date_start': df['date'].iloc[prev_high_idx],
                    'date_end': df['date'].iloc[break_idx]
                })

    for i in range(1, len(swing_lows)):
        curr_low_idx = swing_lows[i]
        prev_low_idx = swing_lows[i-1]
        
        curr_low = df['low'].iloc[curr_low_idx]
        prev_low = df['low'].iloc[prev_low_idx]
        
        if curr_low < prev_low:
             break_idx = df[(df.index > prev_low_idx) & (df.index <= curr_low_idx) & (df['close'] < prev_low)].index.min()
             if pd.notna(break_idx):
                 marker_type = "BOS" if trend == -1 else "CHoCH"
                 trend = -1
                 
                 choch_lines.append({
                    'type': marker_type,
                    'direction': 'bearish',
                    'start_idx': prev_low_idx,
                    'end_idx': break_idx,
                    'price': prev_low,
                    'date_start': df['date'].iloc[prev_low_idx],
                    'date_end': df['date'].iloc[break_idx]
                 })

    # 2. Order Block (OB) Detection
    # Simplest definition: The last down candle before a strong impulsive up move (and vice versa)
    # We will look for large body candles
    df['body_size'] = abs(df['close'] - df['open'])
    avg_body = df['body_size'].mean()
    
    for i in range(1, len(df)-2):
        # Bullish OB: Bearish candle followed by a massive Bullish candle engulfing it
        if df['close'].iloc[i] < df['open'].iloc[i]: # Bearish
            # Next candle is huge bullish
            if (df['close'].iloc[i+1] > df['open'].iloc[i+1]) and (df['body_size'].iloc[i+1] > avg_body * 1.5):
                if df['close'].iloc[i+1] > df['high'].iloc[i]: # Engulfed
                    order_blocks.append({
                        'type': 'OB',
                        'direction': 'bullish',
                        'top': df['high'].iloc[i],
                        'bottom': df['low'].iloc[i],
                        'start_date': df['date'].iloc[i],
                        # Arbitrary end date 20 periods out for drawing the zone
                        'end_date': df['date'].iloc[min(i+20, len(df)-1)] 
                    })
                    
        # Bearish OB
        if df['close'].iloc[i] > df['open'].iloc[i]: # Bullish
            if (df['close'].iloc[i+1] < df['open'].iloc[i+1]) and (df['body_size'].iloc[i+1] > avg_body * 1.5):
                if df['close'].iloc[i+1] < df['low'].iloc[i]:
                    order_blocks.append({
                        'type': 'OB',
                        'direction': 'bearish',
                        'top': df['high'].iloc[i],
                        'bottom': df['low'].iloc[i],
                        'start_date': df['date'].iloc[i],
                        'end_date': df['date'].iloc[min(i+20, len(df)-1)]
                    })

    return {
        'bos': bos_lines,
        'choch': choch_lines,
        'order_blocks': order_blocks
    }

# Example usage/test if run directly
if __name__ == "__main__":
    try:
        sample_df = pd.read_csv("data/raw/FPT.csv")
        sample_df['date'] = pd.to_datetime(sample_df['date'])
        markers = detect_heuristic_smc_markers(sample_df.tail(200))
        print(f"Detected {len(markers['bos'])} Bullish BOS/CHoCH markers.")
        print(f"Detected {len(markers['choch'])} Bearish BOS/CHoCH markers.")
        print(f"Detected {len(markers['order_blocks'])} Order Blocks.")
    except Exception as e:
        print(f"Error test parsing: {e}")

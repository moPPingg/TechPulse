import pandas as pd
import numpy as np
import pandera as pa
from pandera.typing import DataFrame, Series

# 1. Schema Validation (Pandera)
SMCSchema = pa.DataFrameSchema({
    "symbol": pa.Column("object", coerce=True, description="Asset ticker symbol"),
    "date": pa.Column("datetime64[ns]", coerce=True, description="Observation date"),
    "open": pa.Column("float64", coerce=True, nullable=False),
    "high": pa.Column("float64", coerce=True, nullable=False),
    "low": pa.Column("float64", coerce=True, nullable=False),
    "close": pa.Column("float64", coerce=True, nullable=False),
    "volume": pa.Column("float64", coerce=True, nullable=False, checks=pa.Check.ge(0)),
    "news_sentiment": pa.Column("float64", coerce=True, nullable=True)
}, strict=False)

# Output Schema
SMCOutputSchema = pa.DataFrameSchema({
    "symbol": pa.Column("object"),
    "date": pa.Column("datetime64[ns]"),
    "ls_binary": pa.Column("int32", checks=pa.Check.isin([0, 1])),
    "ls_strength": pa.Column("float64")
})

@pa.check_input(SMCSchema)
@pa.check_output(SMCOutputSchema)
def compute_smc_features(
    df: pd.DataFrame, 
    lookback_window: int = 20, 
    volume_multiplier: float = 1.5,
    nan_strategy: str = 'ffill'
) -> pd.DataFrame:
    """
    Computes Smart Money Concepts (SMC) Liquidity Sweep features on multi-asset panel data.
    
    A Liquidity Sweep is defined mathematically when:
      1. Price pierces a structural support: low < rolling_min of previous N periods.
      2. Price aggressively reverts back up: close > rolling_min.
      3. An anomalous spike in volume occurs: volume > rolling_mean(N) * k.
      
    Args:
        df: Input panel DataFrame containing ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'news_sentiment']
        lookback_window: Window size 'N' for structural minimum and volume mean.
        volume_multiplier: The 'k' factor to identify anomalous volume.
        nan_strategy: Strategy to handle missing data ('ffill' or 'drop').
        
    Returns:
        DataFrame containing ONLY: ['symbol', 'date', 'ls_binary', 'ls_strength']
        
    Architecture Decisions & Assumptions:
    - Groupby is strictly applied to 'symbol' to absolutely prevent cross-asset data leakage.
    - Dates must be sorted per symbol before rolling window calculations.
    - `shift(1)` is crucial BEFORE `rolling()` to ensure the current day's price/volume is NOT 
      included in the historical baseline estimation (which would constitute forward-looking leakage).
    - `news_sentiment` is assumed to be point-in-time correct (i.e., known before the market close 
      or aligned appropriately so no future news leaks into the current day's row).
    - To optimize memory, we avoid full `df.copy()`. We create necessary series instead of appending 
      to the entire dataframe until the final output step.
    """
    
    # 1. Validation & Preparation
    if nan_strategy not in ('ffill', 'drop'):
        raise ValueError("nan_strategy must be either 'ffill' or 'drop'")
        
    # We only need the relevant columns
    cols_to_keep = ['symbol', 'date', 'low', 'close', 'volume']
    work_df = df[cols_to_keep].copy() # Essential copy to avoid SettingWithCopyWarning
    
    # Strictly sort by symbol then date to ensure chronological integrity within each asset
    # MUST be done before ffill so we don't leak future dates into past NaN gaps
    work_df = work_df.sort_values(['symbol', 'date'])

    if nan_strategy == 'ffill':
        # Groupby fill to prevent cross-asset leakage during fill
        # Save the sorted symbol column to restore it after grouped ffill
        sorted_symbols = work_df['symbol'].copy()
        work_df = work_df.groupby('symbol').ffill()
        # Restore the sorted symbols array (df['symbol'] would map to the unsorted original!)
        work_df['symbol'] = sorted_symbols
    else:
        work_df = work_df.dropna(subset=['low', 'close', 'volume'])
    
    # 2. Grouped Rolling Calculations (Prevent Cross-Asset Leakage)
    grouped = work_df.groupby('symbol')
    
    # CRITICAL LEAKAGE PREVENTION: shift(1) is applied inside the groupby BEFORE rolling.
    # This guarantees that the rolling window evaluates data strictly from T-1 backwards.
    rolling_min = grouped['low'].shift(1).rolling(window=lookback_window).min()
    rolling_vol_mean = grouped['volume'].shift(1).rolling(window=lookback_window).mean()
    
    # 3. Vectorized Sweep Logic
    cond_sweep_low = work_df['low'] < rolling_min
    cond_revert_close = work_df['close'] > rolling_min
    cond_vol_spike = work_df['volume'] > (rolling_vol_mean * volume_multiplier)
    
    # ls_binary calculation (1 if sweep conditions met, else 0)
    ls_binary = (cond_sweep_low & cond_revert_close & cond_vol_spike).astype(np.int32)
    
    # 4. Performance Optimized Continuous Signal
    # Compute base metrics as pandas series first
    sweep_depth = (rolling_min - work_df['low']) / work_df['close']
    vol_ratio = work_df['volume'] / (rolling_vol_mean + 1e-9)
    
    # Use np.where to calculate strength ONLY where binary is 1, default 0.0 otherwise.
    # This is highly efficient vectorization.
    ls_strength = np.where(
        ls_binary == 1,
        sweep_depth * vol_ratio,
        0.0
    )
    
    # 5. Output Construction
    # Return strictly the requested columns, leaving the original dataframe untouched
    out_df = pd.DataFrame({
        'symbol': work_df['symbol'],
        'date': work_df['date'],
        'ls_binary': ls_binary,
        'ls_strength': ls_strength
    })
    
    return out_df


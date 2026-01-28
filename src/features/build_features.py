"""
Feature engineering module for stock price data.
This module calculates technical indicators and features for stock analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Calculate price returns for multiple periods.
    
    Args:
        df: DataFrame with 'close' column
        periods: List of periods to calculate returns (default: [1, 5, 10, 20])
    
    Returns:
        DataFrame with added return columns
    
    Example:
        >>> df = calculate_returns(df, periods=[1, 5, 10])
        >>> print(df.columns)
        # ['date', 'close', 'return_1d', 'return_5d', 'return_10d']
    """
    for period in periods:
        col_name = f'return_{period}d'
        df[col_name] = df['close'].pct_change(periods=period) * 100  # Convert to percentage
        logger.debug(f"Calculated {col_name}")
    
    return df


def calculate_moving_averages(
    df: pd.DataFrame,
    column: str = 'close',
    windows: List[int] = [5, 10, 20, 50, 200]
) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA) for multiple windows.
    
    Args:
        df: DataFrame with price data
        column: Column to calculate MA on (default: 'close')
        windows: List of window sizes (default: [5, 10, 20, 50, 200])
    
    Returns:
        DataFrame with added MA columns
    
    Example:
        >>> df = calculate_moving_averages(df, windows=[5, 10, 20])
        >>> print(df[['close', 'ma_5', 'ma_10', 'ma_20']].head())
    """
    for window in windows:
        col_name = f'ma_{window}'
        df[col_name] = df[column].rolling(window=window).mean()
        logger.debug(f"Calculated {col_name}")
    
    return df


def calculate_ema(
    df: pd.DataFrame,
    column: str = 'close',
    spans: List[int] = [12, 26]
) -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages (EMA).
    
    Args:
        df: DataFrame with price data
        column: Column to calculate EMA on (default: 'close')
        spans: List of span values (default: [12, 26])
    
    Returns:
        DataFrame with added EMA columns
    
    Example:
        >>> df = calculate_ema(df, spans=[12, 26])
        >>> print(df[['close', 'ema_12', 'ema_26']].head())
    """
    for span in spans:
        col_name = f'ema_{span}'
        df[col_name] = df[column].ewm(span=span, adjust=False).mean()
        logger.debug(f"Calculated {col_name}")
    
    return df


def calculate_volatility(
    df: pd.DataFrame,
    column: str = 'close',
    windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        df: DataFrame with price data
        column: Column to calculate volatility on (default: 'close')
        windows: List of window sizes (default: [5, 10, 20])
    
    Returns:
        DataFrame with added volatility columns
    
    Example:
        >>> df = calculate_volatility(df, windows=[10, 20])
        >>> print(df[['close', 'volatility_10', 'volatility_20']].head())
    """
    for window in windows:
        col_name = f'volatility_{window}'
        # Calculate as standard deviation of percentage returns
        returns = df[column].pct_change()
        df[col_name] = returns.rolling(window=window).std() * 100  # Convert to percentage
        logger.debug(f"Calculated {col_name}")
    
    return df


def calculate_rsi(
    df: pd.DataFrame,
    column: str = 'close',
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI ranges from 0 to 100:
    - RSI > 70: Overbought
    - RSI < 30: Oversold
    
    Args:
        df: DataFrame with price data
        column: Column to calculate RSI on (default: 'close')
        period: RSI period (default: 14)
    
    Returns:
        DataFrame with added RSI column
    
    Example:
        >>> df = calculate_rsi(df, period=14)
        >>> print(df[['close', 'rsi_14']].tail())
    """
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    col_name = f'rsi_{period}'
    df[col_name] = rsi
    logger.debug(f"Calculated {col_name}")
    
    return df


def calculate_macd(
    df: pd.DataFrame,
    column: str = 'close',
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal
    
    Args:
        df: DataFrame with price data
        column: Column to calculate MACD on (default: 'close')
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
    
    Returns:
        DataFrame with MACD, signal, and histogram columns
    
    Example:
        >>> df = calculate_macd(df)
        >>> print(df[['close', 'macd', 'macd_signal', 'macd_hist']].tail())
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    logger.debug("Calculated MACD, signal, and histogram")
    
    return df


def calculate_bollinger_bands(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    num_std: float = 2
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Middle Band = SMA(window)
    Upper Band = Middle Band + (std * num_std)
    Lower Band = Middle Band - (std * num_std)
    
    Args:
        df: DataFrame with price data
        column: Column to calculate on (default: 'close')
        window: Window size (default: 20)
        num_std: Number of standard deviations (default: 2)
    
    Returns:
        DataFrame with Bollinger Bands columns
    
    Example:
        >>> df = calculate_bollinger_bands(df, window=20)
        >>> print(df[['close', 'bb_middle', 'bb_upper', 'bb_lower']].tail())
    """
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    
    df['bb_middle'] = rolling_mean
    df['bb_upper'] = rolling_mean + (rolling_std * num_std)
    df['bb_lower'] = rolling_mean - (rolling_std * num_std)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    logger.debug(f"Calculated Bollinger Bands (window={window}, std={num_std})")
    
    return df


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume-based features.
    
    Args:
        df: DataFrame with 'volume' column
    
    Returns:
        DataFrame with added volume features
    
    Example:
        >>> df = calculate_volume_features(df)
        >>> print(df[['volume', 'volume_ma_20', 'volume_ratio']].tail())
    """
    if 'volume' not in df.columns:
        logger.warning("No 'volume' column found, skipping volume features")
        return df
    
    # Volume moving average
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    # Volume ratio (current volume / average volume)
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # Volume change
    df['volume_change'] = df['volume'].pct_change() * 100
    
    logger.debug("Calculated volume features")
    
    return df


def calculate_price_momentum(
    df: pd.DataFrame,
    periods: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate price momentum indicators.
    
    Momentum = Current Price - Price N periods ago
    
    Args:
        df: DataFrame with 'close' column
        periods: List of periods (default: [5, 10, 20])
    
    Returns:
        DataFrame with momentum columns
    
    Example:
        >>> df = calculate_price_momentum(df, periods=[5, 10, 20])
        >>> print(df[['close', 'momentum_5', 'momentum_10']].tail())
    """
    for period in periods:
        col_name = f'momentum_{period}'
        df[col_name] = df['close'] - df['close'].shift(period)
        logger.debug(f"Calculated {col_name}")
    
    return df


def calculate_price_range(
    df: pd.DataFrame,
    windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate price range features.
    
    Features calculated:
    - Daily range: high - low
    - Daily range percentage: (high - low) / close * 100
    - Rolling range: max(high, window) - min(low, window)
    - Average True Range (ATR): average of true range over window
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        windows: List of window sizes for rolling calculations (default: [5, 10, 20])
    
    Returns:
        DataFrame with price range columns
    
    Example:
        >>> df = calculate_price_range(df, windows=[5, 10, 20])
        >>> print(df[['close', 'daily_range', 'daily_range_pct', 'atr_14']].tail())
    """
    # Validate required columns
    required_cols = ['high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns {missing_cols}, skipping price range features")
        return df
    
    # Daily range (intraday volatility)
    df['daily_range'] = df['high'] - df['low']
    
    # Daily range as percentage of close price
    df['daily_range_pct'] = (df['daily_range'] / df['close']) * 100
    
    # Calculate rolling price ranges
    for window in windows:
        # Rolling high-low range
        rolling_high = df['high'].rolling(window=window).max()
        rolling_low = df['low'].rolling(window=window).min()
        col_name = f'price_range_{window}'
        df[col_name] = rolling_high - rolling_low
        
        # Price range as percentage
        df[f'{col_name}_pct'] = (df[col_name] / df['close']) * 100
        
        logger.debug(f"Calculated {col_name}")
    
    # Average True Range (ATR) - 14-period standard
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    
    # High-Low ratio
    df['hl_ratio'] = df['high'] / df['low']
    
    # Position in daily range (where close is relative to high-low)
    # 1 = close at high, 0 = close at low
    df['close_position'] = (df['close'] - df['low']) / df['daily_range']
    df['close_position'] = df['close_position'].fillna(0.5)  # Fill NaN with midpoint
    
    logger.debug("Calculated price range features including ATR")
    
    return df


def calculate_all_features(
    df: pd.DataFrame,
    feature_sets: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate all technical indicators and features.
    
    Args:
        df: DataFrame with OHLCV data
        feature_sets: List of feature sets to calculate (default: all)
                     Options: ['returns', 'ma', 'ema', 'volatility', 'rsi', 
                              'macd', 'bollinger', 'volume', 'momentum', 'price_range']
    
    Returns:
        DataFrame with all calculated features
    
    Example:
        >>> df = calculate_all_features(df)
        >>> print(f"Added {len(df.columns)} features")
    """
    if feature_sets is None:
        feature_sets = ['returns', 'ma', 'ema', 'volatility', 'rsi', 
                       'macd', 'bollinger', 'volume', 'momentum', 'price_range']
    
    initial_cols = len(df.columns)
    
    # Validate data
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate each feature set
    if 'returns' in feature_sets:
        df = calculate_returns(df, periods=[1, 5, 10, 20])
    
    if 'ma' in feature_sets:
        df = calculate_moving_averages(df, windows=[5, 10, 20, 50])
    
    if 'ema' in feature_sets:
        df = calculate_ema(df, spans=[12, 26])
    
    if 'volatility' in feature_sets:
        df = calculate_volatility(df, windows=[5, 10, 20])
    
    if 'rsi' in feature_sets:
        df = calculate_rsi(df, period=14)
    
    if 'macd' in feature_sets:
        df = calculate_macd(df)
    
    if 'bollinger' in feature_sets:
        df = calculate_bollinger_bands(df, window=20)
    
    if 'volume' in feature_sets:
        df = calculate_volume_features(df)
    
    if 'momentum' in feature_sets:
        df = calculate_price_momentum(df, periods=[5, 10, 20])
    
    if 'price_range' in feature_sets:
        df = calculate_price_range(df, windows=[5, 10, 20])
    
    final_cols = len(df.columns)
    features_added = final_cols - initial_cols
    
    logger.info(f"Added {features_added} features (total columns: {final_cols})")
    
    return df


def build_features_single(
    filename: str,
    clean_dir: str = 'data/clean',
    features_dir: str = 'data/features',
    feature_sets: Optional[List[str]] = None,
    drop_na: bool = True,
    save_file: bool = True
) -> Optional[pd.DataFrame]:
    """
    Build features for a single stock file.
    
    Args:
        filename: Name of file to process (e.g., 'FPT.csv')
        clean_dir: Directory with clean data
        features_dir: Directory to save features
        feature_sets: Which features to calculate (default: all)
        drop_na: Drop rows with NaN after feature calculation (default: True)
        save_file: Whether to save to file (default: True)
    
    Returns:
        DataFrame with features, or None if failed
    
    Example:
        >>> df = build_features_single('FPT.csv')
        >>> if df is not None:
        >>>     print(df.columns)
    """
    logger.info(f"Building features for: {filename}")
    
    input_path = Path(clean_dir) / filename
    
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        initial_rows = len(df)
        logger.info(f"Loaded {initial_rows} rows")
        
        # Calculate features
        df = calculate_all_features(df, feature_sets=feature_sets)
        
        # Drop NaN rows
        if drop_na:
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows with NaN values")
        
        final_rows = len(df)
        logger.info(f"Final dataset: {final_rows} rows, {len(df.columns)} columns")
        
        # Save to file
        if save_file:
            output_path = Path(features_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / filename
            df.to_csv(output_file, index=False)
            logger.info(f"Saved to: {output_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return None


def build_features(
    clean_dir: str = 'data/clean',
    features_dir: str = 'data/features',
    pattern: str = '*.csv',
    feature_sets: Optional[List[str]] = None,
    drop_na: bool = True,
    skip_on_error: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Build features for multiple stock files.
    
    Args:
        clean_dir: Directory with clean data
        features_dir: Directory to save features
        pattern: File pattern to match (default: '*.csv')
        feature_sets: Which features to calculate (default: all)
        drop_na: Drop rows with NaN (default: True)
        skip_on_error: Continue on error (default: True)
    
    Returns:
        Dictionary mapping filenames to DataFrames
    
    Example:
        >>> results = build_features('data/clean', 'data/features')
        >>> print(f"Built features for {len(results)} files")
    """
    # Validate input directory
    clean_path = Path(clean_dir)
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
    
    # Create output directory
    features_path = Path(features_dir)
    features_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {features_path.absolute()}")
    
    # Find all CSV files
    csv_files = list(clean_path.glob(pattern))
    
    if not csv_files:
        logger.warning(f"No files found matching pattern '{pattern}' in {clean_dir}")
        return {}
    
    # Track results
    results = {}
    success_count = 0
    failed_files = []
    
    # Log session info
    logger.info(f"Starting feature building for {len(csv_files)} files")
    start_time = datetime.now()
    
    # Process each file
    for idx, file_path in enumerate(csv_files, 1):
        filename = file_path.name
        logger.info(f"[{idx}/{len(csv_files)}] Processing {filename}...")
        
        # Skip combined files
        if 'combined' in filename.lower():
            logger.info(f"Skipping combined file: {filename}")
            continue
        
        try:
            df = build_features_single(
                filename=filename,
                clean_dir=clean_dir,
                features_dir=features_dir,
                feature_sets=feature_sets,
                drop_na=drop_na,
                save_file=True
            )
            
            if df is not None:
                results[filename] = df
                success_count += 1
                logger.info(f"{filename}: Success ({len(df)} rows, {len(df.columns)} features)")
            else:
                failed_files.append(filename)
                
        except Exception as e:
            logger.error(f"{filename}: Failed with error: {e}")
            failed_files.append(filename)
            
            if not skip_on_error:
                logger.error("skip_on_error=False, stopping execution")
                raise
    
    # Calculate statistics
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # Log summary
    logger.info("=" * 70)
    logger.info(f"Feature building completed in {elapsed_time:.2f} seconds")
    logger.info(f"Success: {success_count}/{len(csv_files)} files")
    
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")
    else:
        logger.info("All files processed successfully! 🎉")
    
    logger.info("=" * 70)
    
    return results


def get_feature_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get summary statistics of calculated features.
    
    Args:
        df: DataFrame with features
    
    Returns:
        Dictionary with feature statistics
    
    Example:
        >>> summary = get_feature_summary(df)
        >>> print(f"Total features: {summary['total_features']}")
    """
    feature_cols = [col for col in df.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
    
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'total_features': len(feature_cols),
        'feature_columns': feature_cols,
        'null_count': df[feature_cols].isnull().sum().sum(),
        'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A'
    }
    
    return summary


if __name__ == "__main__":
    """
    Example usage when running this file directly.
    """
    print("\n" + "=" * 70)
    print("Example 1: Build Features for Single File")
    print("=" * 70)
    
    # Example 1: Build features for one file
    df = build_features_single('FPT.csv', clean_dir='data/clean', features_dir='data/features')
    
    if df is not None:
        print(f"\nFeatures built:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"\nSample data:")
        print(df.tail(5))
        
        # Show summary
        summary = get_feature_summary(df)
        print(f"\nFeature summary:")
        print(f"  Total features added: {summary['total_features']}")
        print(f"  Date range: {summary['date_range']}")
    
    print("\n" + "=" * 70)
    print("Example 2: Build Features for All Files")
    print("=" * 70)
    
    # Example 2: Build features for all files
    results = build_features(
        clean_dir='data/clean',
        features_dir='data/features'
    )
    
    print(f"\nProcessed {len(results)} files:")
    for filename, df in results.items():
        print(f"  {filename}: {len(df)} rows, {len(df.columns)} columns")
    
    print("\n" + "=" * 70)
    print("Example 3: Custom Feature Sets")
    print("=" * 70)
    
    # Example 3: Build only specific features
    df = build_features_single(
        'FPT.csv',
        feature_sets=['returns', 'ma', 'rsi'],  # Only these features
        drop_na=True
    )
    
    if df is not None:
        print(f"\nCustom features built: {len(df.columns)} columns")
        print("Columns:", df.columns.tolist())
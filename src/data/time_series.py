import pandas as pd
import numpy as np
from typing import Tuple, List

def create_windows(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: str, 
    window_size: int, 
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies time-series windowing over the dataframe.
    
    Args:
        df: DataFrame sorted chronologically.
        feature_cols: List of column names used as features.
        target_col: Name of the target column (e.g., forward return or trend direction).
        window_size: Number of past timesteps (T).
        horizon: Step ahead for target prediction (e.g., T+1).
        
    Returns:
        X: numpy array of shape (samples, window_size, features)
        y: numpy array of shape (samples,) representing targets
        dates: numpy array of dates associated with the target prediction
    """
    X, y, dates_list = [], [], []
    
    # Drop rows that have NaN in the features or target (from rolling windows and shift computations)
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    
    features_data = df_clean[feature_cols].values
    target_data = df_clean[target_col].values
    
    if 'date' in df_clean.columns:
        date_data = df_clean['date'].values
    else:
        date_data = np.arange(len(df_clean))
    
    n_samples = len(df_clean) - window_size - horizon + 1
    
    for i in range(n_samples):
        # Window of features [i : i + window_size]
        X.append(features_data[i : i + window_size])
        
        # Target at the end of the horizon [i + window_size + horizon - 1]
        target_idx = i + window_size + horizon - 1
        y.append(target_data[target_idx])
        dates_list.append(date_data[target_idx])
        
    return np.array(X), np.array(y), np.array(dates_list)


def chronological_split(
    X: np.ndarray, 
    y: np.ndarray, 
    dates: np.ndarray, 
    train_ratio: float = 0.8, 
    purge_gap: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs strict chronological train-validation split avoiding point-in-time leakage.
    Leverages a 'purge_gap' to prevent overlapping bounds in forward horizons.
    
    Args:
        X: Feature tensor / array.
        y: Target label tensor / array.
        dates: Date tensor array.
        train_ratio: Float representing the proportion of training samples (0 to 1).
        purge_gap: Number of steps to purge to eliminate horizon leakage.
        
    Returns:
        X_train, y_train, dates_train, X_val, y_val, dates_val
    """
    n_samples = len(X)
    train_end_idx = int(n_samples * train_ratio)
    
    val_start_idx = train_end_idx + purge_gap
    
    if val_start_idx >= n_samples:
        raise ValueError("purge_gap is too large for the provided data and train_ratio.")
    
    X_train = X[:train_end_idx]
    y_train = y[:train_end_idx]
    dates_train = dates[:train_end_idx]
    
    X_val = X[val_start_idx:]
    y_val = y[val_start_idx:]
    dates_val = dates[val_start_idx:]
    
    return X_train, y_train, dates_train, X_val, y_val, dates_val

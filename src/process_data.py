import pandas as pd
import glob
import os
import sys

# Ensure the src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.smc import compute_smc_features

RAW_DIR = 'd:/TechPulse/data/raw/*.csv'
PROCESSED_DIR = 'd:/TechPulse/data/processed'
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'smc_features.csv')

def process_data():
    files = glob.glob(RAW_DIR)
    if not files:
        print(f"No CSV files found in {RAW_DIR}")
        return
    
    # Read and concatenate all raw data files
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        
    master_df = pd.concat(dfs, ignore_index=True)
    
    # Check if necessary columns exist
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
         if col not in master_df.columns:
              print(f"Error: Missing column '{col}' in raw data.")
              if col == 'symbol':
                  # Attempt to infer symbol from filename if it's missing (common in some raw data)
                  print("Attempting to infer symbol from filename...")
                  master_df = pd.DataFrame() # reset
                  for f in files:
                      temp_df = pd.read_csv(f)
                      symbol_name = os.path.basename(f).split('.')[0]
                      temp_df['symbol'] = symbol_name
                      master_df = pd.concat([master_df, temp_df], ignore_index=True)
              else:
                  return

    # Mock 'news_sentiment' if it doesn't exist in raw data since it's required by the schema
    if 'news_sentiment' not in master_df.columns:
        print("Warning: 'news_sentiment' column missing. Mocking zero sentiment for pipeline processing.")
        master_df['news_sentiment'] = 0.0

    # Ensure date is datetime
    master_df['date'] = pd.to_datetime(master_df['date'])
    
    # Run SMC feature engineering
    try:
        print("Computing SMC Features...")
        processed_df = compute_smc_features(master_df, lookback_window=20, volume_multiplier=1.5, nan_strategy='ffill')
        
        # Save output
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        processed_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! Processed data saved to: {OUTPUT_FILE}")
        print(f"Processed DataFrame Shape: {processed_df.shape}")
    except Exception as e:
        print(f"Error during feature processing: {e}")

if __name__ == "__main__":
    process_data()

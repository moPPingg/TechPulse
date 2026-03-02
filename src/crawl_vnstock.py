import os
import pandas as pd
from datetime import datetime

try:
    from vnstock import stock_historical_data
except ImportError:
    import sys
    import subprocess
    print("vnstock not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "vnstock"])
    from vnstock import stock_historical_data

RAW_DIR = "d:/TechPulse/data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# 30 prominent tickers from the VN30 basket
VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]

START_DATE = "2016-01-01"
END_DATE = "2026-01-01"

def crawl_vn30_data():
    # Remove mock data if present
    mock_file = os.path.join(RAW_DIR, "AAPL.csv")
    if os.path.exists(mock_file):
        os.remove(mock_file)
        print("Removed mock AAPL.csv data.")

    success_count = 0
    for symbol in VN30_TICKERS:
        print(f"Fetching data for {symbol}...")
        try:
            # Resolution '1D' for daily data. Type 'stock'
            df = stock_historical_data(symbol=symbol, start_date=START_DATE, end_date=END_DATE, resolution="1D", type="stock")
            
            if df is None or df.empty:
                print(f"  [!] No data returned for {symbol}")
                continue
                
            # vnstock historical data usually returns columns like:
            # ['time', 'open', 'high', 'low', 'close', 'volume', 'ticker']
            # We map this strictly to our system requirements: ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            
            # Map column names explicitly to avoid version discrepancy issues
            col_mapping = {
                'time': 'date',
                'ticker': 'symbol'
            }
            df.rename(columns=col_mapping, inplace=True)
            
            # Ensure proper casing
            df.columns = [c.lower() for c in df.columns]
            
            # If symbol column is missing, add it
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
                
            # Filter strictly to the required schema
            cols_to_keep = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            df = df[cols_to_keep]
            
            output_path = os.path.join(RAW_DIR, f"{symbol}.csv")
            df.to_csv(output_path, index=False)
            print(f"  [+] Saved {len(df)} records to {output_path}")
            success_count += 1
            
        except Exception as e:
            print(f"  [!] Failed to crawl {symbol}: {e}")

    print(f"\nCrawling complete. {success_count}/{len(VN30_TICKERS)} tickers successfully extracted to {RAW_DIR}.")

if __name__ == "__main__":
    crawl_vn30_data()

"""
Pipeline module for crawling multiple stock prices from CafeF.
This module provides functions to fetch and save historical price data for multiple stocks.
"""

from src.crawl.cafef_scraper import fetch_price_cafef
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crawl_many(
    symbols: List[str],
    start_date: str,
    end_date: str,
    save_dir: str = 'data/raw',
    combine: bool = False,
    skip_on_error: bool = True,
    page_size: int = 3000
) -> List[pd.DataFrame]:
    """
    Crawl historical price data for multiple stock symbols.
    
    Args:
        symbols: List of stock ticker symbols (e.g., ['FPT', 'VNM', 'HPG'])
        start_date: Start date in format 'DD/MM/YYYY'
        end_date: End date in format 'DD/MM/YYYY'
        save_dir: Directory to save CSV files (default: 'data/raw')
        combine: If True, also save a combined CSV with all stocks (default: False)
        skip_on_error: If True, continue on error; if False, raise error (default: True)
    
    Returns:
        List of DataFrames, one for each successfully crawled symbol
    
    Example:
        >>> symbols = ['FPT', 'VNM', 'HPG']
        >>> data = crawl_many(symbols, '01/01/2024', '31/01/2024')
        >>> print(f"Successfully crawled {len(data)} stocks")
    """
    # Validate inputs
    if not symbols:
        logger.warning("No symbols provided. Returning empty list.")
        return []
    
    if not isinstance(symbols, list):
        raise TypeError(f"symbols must be a list, got {type(symbols)}")
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Save directory: {save_path.absolute()}")
    
    # Track results
    all_data = []
    success_count = 0
    failed_symbols = []
    
    # Log crawling session info
    logger.info(f"Starting crawl for {len(symbols)} symbols from {start_date} to {end_date}")
    start_time = datetime.now()
    
    # Crawl each symbol
    for idx, symbol in enumerate(symbols, 1):
        symbol = symbol.strip().upper()  # Clean up symbol
        logger.info(f"[{idx}/{len(symbols)}] Crawling {symbol}...")
        
        try:
            # Fetch data
            df = fetch_price_cafef(symbol, start_date, end_date, page_size=page_size)
            
            # Check if data is empty
            if df.empty:
                logger.warning(f"{symbol}: No data returned (possibly invalid symbol or date range)")
                failed_symbols.append(symbol)
                continue
            
            # Save individual CSV
            output_file = save_path / f"{symbol}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"{symbol}: Saved {len(df)} records to {output_file}")
            
            # Append to results
            all_data.append(df)
            success_count += 1
            
        except Exception as e:
            logger.error(f"{symbol}: Failed with error: {e}")
            failed_symbols.append(symbol)
            
            if not skip_on_error:
                logger.error("skip_on_error=False, stopping execution")
                raise
            
            continue
    
    # Save combined file if requested
    if combine and all_data:
        try:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_file = save_path / f"combined_{start_date.replace('/', '')}_{end_date.replace('/', '')}.csv"
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"Saved combined file with {len(combined_df)} total records to {combined_file}")
        except Exception as e:
            logger.error(f"Failed to save combined file: {e}")
    
    # Log summary
    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info(f"Crawling completed in {elapsed_time:.2f} seconds")
    logger.info(f"Success: {success_count}/{len(symbols)} symbols")
    
    if failed_symbols:
        logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")
    else:
        logger.info("All symbols crawled successfully! 🎉")
    
    logger.info("=" * 70)
    
    return all_data


def crawl_single(
    symbol: str,
    start_date: str,
    end_date: str,
    save_dir: str = 'data/raw',
    save_file: bool = True
) -> Optional[pd.DataFrame]:
    """
    Crawl historical price data for a single stock symbol.
    
    This is a convenience function for crawling just one symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'FPT')
        start_date: Start date in format 'DD/MM/YYYY'
        end_date: End date in format 'DD/MM/YYYY'
        save_dir: Directory to save CSV file (default: 'data/raw')
        save_file: Whether to save to CSV (default: True)
    
    Returns:
        DataFrame with price data, or None if failed
    
    Example:
        >>> df = crawl_single('FPT', '01/01/2024', '31/01/2024')
        >>> if df is not None:
        >>>     print(df.head())
    """
    symbol = symbol.strip().upper()
    logger.info(f"Crawling single symbol: {symbol}")
    
    try:
        df = fetch_price_cafef(symbol, start_date, end_date)
        
        if df.empty:
            logger.warning(f"{symbol}: No data returned")
            return None
        
        if save_file:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / f"{symbol}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"{symbol}: Saved {len(df)} records to {output_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"{symbol}: Failed with error: {e}")
        return None


if __name__ == "__main__":
    """
    Example usage when running this file directly.
    You can modify the symbols and dates below for testing.
    """
    # Example 1: Crawl multiple stocks
    print("\n" + "=" * 70)
    print("Example 1: Crawling Multiple Stocks")
    print("=" * 70)
    
    symbols = ['FPT', 'VNM', 'HPG']
    results = crawl_many(
        symbols=symbols,
        start_date='01/01/2016',
        end_date='31/01/2026',
        save_dir='data/raw',
        combine=True
    )
    
    print(f"\nSuccessfully crawled {len(results)} out of {len(symbols)} stocks")
    
    # Example 2: Crawl single stock
    print("\n" + "=" * 70)
    print("Example 2: Crawling Single Stock")
    print("=" * 70)
    
    df = crawl_single(
        symbol='VCB',
        start_date='01/06/2024',
        end_date='30/06/2024'
    )
    
    if df is not None:
        print(f"\nVCB data preview:")
        print(df.head())
        print(f"\nTotal records: {len(df)}")
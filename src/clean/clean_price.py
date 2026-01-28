"""
Data cleaning module for stock price data.
This module provides functions to clean and validate stock price CSV files.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, List, Dict
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_price(
    input_path: str,
    output_path: Optional[str] = None,
    expected_columns: Optional[List[str]] = None,
    remove_duplicates: bool = True,
    remove_nulls: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Clean a single stock price CSV file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV (if None, won't save to file)
        expected_columns: List of expected column names (default: standard price columns)
        remove_duplicates: Remove duplicate rows (default: True)
        remove_nulls: Remove rows with null values (default: True)
        validate: Validate data quality (default: True)
    
    Returns:
        Cleaned DataFrame
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data validation fails
    
    Example:
        >>> df = clean_price('data/raw/FPT.csv', 'data/clean/FPT.csv')
        >>> print(f"Cleaned {len(df)} records")
    """
    # Validate input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_file.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    
    logger.info(f"Cleaning file: {input_path}")
    
    try:
        # Read CSV
        df = pd.read_csv(input_path)
        initial_rows = len(df)
        logger.info(f"Loaded {initial_rows} rows")
        
        # Check if empty
        if df.empty:
            logger.warning(f"File is empty: {input_path}")
            if output_path:
                df.to_csv(output_path, index=False)
            return df
        
        # Set expected columns
        if expected_columns is None:
            expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        
        # Store original columns for reference
        original_columns = df.columns.tolist()
        logger.debug(f"Original columns: {original_columns}")
        
        # Rename columns if needed (handle different formats)
        if len(df.columns) == len(expected_columns):
            df.columns = expected_columns
            logger.info(f"Renamed columns to: {expected_columns}")
        else:
            logger.warning(f"Column count mismatch: expected {len(expected_columns)}, got {len(df.columns)}")
        
        # Convert date column
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                invalid_dates = df['date'].isna().sum()
                if invalid_dates > 0:
                    logger.warning(f"Found {invalid_dates} invalid dates")
            except Exception as e:
                logger.error(f"Error converting dates: {e}")
                raise ValueError(f"Failed to convert date column: {e}")
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Count conversions
                null_count = df[col].isna().sum()
                if null_count > 0:
                    logger.warning(f"Column '{col}': {null_count} invalid values converted to NaN")
        
        # Remove duplicates
        if remove_duplicates:
            duplicates_before = df.duplicated().sum()
            if duplicates_before > 0:
                df = df.drop_duplicates()
                logger.info(f"Removed {duplicates_before} duplicate rows")
        
        # Remove rows with null values
        if remove_nulls:
            nulls_before = df.isnull().any(axis=1).sum()
            if nulls_before > 0:
                df = df.dropna()
                logger.info(f"Removed {nulls_before} rows with null values")
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            logger.debug("Sorted by date")
        
        # Validate data quality
        if validate:
            validation_issues = validate_price_data(df)
            if validation_issues:
                for issue in validation_issues:
                    logger.warning(f"Validation: {issue}")
        
        # Calculate cleaning summary
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        retention_rate = (final_rows / initial_rows * 100) if initial_rows > 0 else 0
        
        logger.info(f"Cleaning complete: {final_rows} rows (removed {rows_removed}, retention: {retention_rate:.1f}%)")
        
        # Save to output file if specified
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved to: {output_path}")
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty or invalid CSV: {input_path}")
        raise ValueError(f"Invalid CSV file: {input_path}")
    except Exception as e:
        logger.error(f"Error cleaning file: {e}")
        raise


def validate_price_data(df: pd.DataFrame) -> List[str]:
    """
    Validate stock price data for common issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        List of validation issue messages (empty if no issues)
    
    Example:
        >>> issues = validate_price_data(df)
        >>> if issues:
        >>>     print("Issues found:", issues)
    """
    issues = []
    
    # Check required columns
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        return issues  # Can't validate further without columns
    
    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (df[col] < 0).any():
            negative_count = (df[col] < 0).sum()
            issues.append(f"Found {negative_count} negative values in '{col}'")
    
    # Check for negative volume
    if (df['volume'] < 0).any():
        negative_volume = (df['volume'] < 0).sum()
        issues.append(f"Found {negative_volume} negative volume values")
    
    # Check OHLC logic: high >= open, close, low and low <= open, close, high
    if 'high' in df.columns and 'low' in df.columns:
        # High should be highest
        high_issues = ((df['high'] < df['open']) | 
                       (df['high'] < df['close']) | 
                       (df['high'] < df['low'])).sum()
        if high_issues > 0:
            issues.append(f"Found {high_issues} rows where high is not the highest price")
        
        # Low should be lowest
        low_issues = ((df['low'] > df['open']) | 
                      (df['low'] > df['close']) | 
                      (df['low'] > df['high'])).sum()
        if low_issues > 0:
            issues.append(f"Found {low_issues} rows where low is not the lowest price")
    
    # Check for zero prices (suspicious)
    for col in price_cols:
        if (df[col] == 0).any():
            zero_count = (df[col] == 0).sum()
            issues.append(f"Found {zero_count} zero values in '{col}' (suspicious)")
    
    # Check date continuity (are there big gaps?)
    if 'date' in df.columns and len(df) > 1:
        df_sorted = df.sort_values('date')
        date_diff = df_sorted['date'].diff()
        max_gap = date_diff.max()
        if pd.notna(max_gap) and max_gap.days > 30:
            issues.append(f"Maximum date gap: {max_gap.days} days (potential missing data)")
    
    return issues


def clean_many(
    raw_dir: str = 'data/raw',
    clean_dir: str = 'data/clean',
    pattern: str = '*.csv',
    skip_on_error: bool = True,
    remove_duplicates: bool = True,
    remove_nulls: bool = True,
    validate: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Clean multiple stock price CSV files from a directory.
    
    Args:
        raw_dir: Directory containing raw CSV files
        clean_dir: Directory to save cleaned CSV files
        pattern: File pattern to match (default: '*.csv')
        skip_on_error: Continue on error (default: True)
        remove_duplicates: Remove duplicate rows (default: True)
        remove_nulls: Remove rows with null values (default: True)
        validate: Validate data quality (default: True)
    
    Returns:
        Dictionary mapping filenames to cleaned DataFrames
    
    Example:
        >>> results = clean_many('data/raw', 'data/clean')
        >>> print(f"Cleaned {len(results)} files")
    """
    # Validate input directory
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    
    if not raw_path.is_dir():
        raise ValueError(f"Raw path is not a directory: {raw_dir}")
    
    # Create output directory
    clean_path = Path(clean_dir)
    clean_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {clean_path.absolute()}")
    
    # Find all CSV files
    csv_files = list(raw_path.glob(pattern))
    
    if not csv_files:
        logger.warning(f"No files found matching pattern '{pattern}' in {raw_dir}")
        return {}
    
    # Track results
    cleaned_data = {}
    success_count = 0
    failed_files = []
    
    # Log session info
    logger.info(f"Starting cleaning for {len(csv_files)} files")
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
            # Define output path
            output_path = clean_path / filename
            
            # Clean the file
            df = clean_price(
                input_path=str(file_path),
                output_path=str(output_path),
                remove_duplicates=remove_duplicates,
                remove_nulls=remove_nulls,
                validate=validate
            )
            
            # Store result
            cleaned_data[filename] = df
            success_count += 1
            logger.info(f"{filename}: Success ({len(df)} records)")
            
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
    logger.info(f"Cleaning completed in {elapsed_time:.2f} seconds")
    logger.info(f"Success: {success_count}/{len(csv_files)} files")
    
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")
    else:
        logger.info("All files cleaned successfully! 🎉")
    
    logger.info("=" * 70)
    
    return cleaned_data


def clean_single(
    filename: str,
    raw_dir: str = 'data/raw',
    clean_dir: str = 'data/clean',
    save_file: bool = True
) -> Optional[pd.DataFrame]:
    """
    Clean a single stock price file by name.
    
    Convenience function for cleaning one specific file.
    
    Args:
        filename: Name of file to clean (e.g., 'FPT.csv')
        raw_dir: Directory containing raw files
        clean_dir: Directory to save cleaned file
        save_file: Whether to save cleaned file (default: True)
    
    Returns:
        Cleaned DataFrame, or None if failed
    
    Example:
        >>> df = clean_single('FPT.csv')
        >>> if df is not None:
        >>>     print(df.head())
    """
    logger.info(f"Cleaning single file: {filename}")
    
    input_path = Path(raw_dir) / filename
    
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return None
    
    try:
        output_path = str(Path(clean_dir) / filename) if save_file else None
        
        df = clean_price(
            input_path=str(input_path),
            output_path=output_path
        )
        
        logger.info(f"Successfully cleaned {filename}: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Failed to clean {filename}: {e}")
        return None


def get_cleaning_stats(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate statistics about the cleaning process.
    
    Args:
        df_before: DataFrame before cleaning
        df_after: DataFrame after cleaning
    
    Returns:
        Dictionary with cleaning statistics
    
    Example:
        >>> df_raw = pd.read_csv('data/raw/FPT.csv')
        >>> df_clean = clean_price('data/raw/FPT.csv')
        >>> stats = get_cleaning_stats(df_raw, df_clean)
        >>> print(f"Retention rate: {stats['retention_rate']:.1f}%")
    """
    stats = {
        'rows_before': len(df_before),
        'rows_after': len(df_after),
        'rows_removed': len(df_before) - len(df_after),
        'retention_rate': (len(df_after) / len(df_before) * 100) if len(df_before) > 0 else 0,
        'null_count_before': df_before.isnull().sum().sum(),
        'null_count_after': df_after.isnull().sum().sum(),
        'duplicate_count_before': df_before.duplicated().sum(),
        'duplicate_count_after': df_after.duplicated().sum()
    }
    
    return stats


if __name__ == "__main__":
    """
    Example usage when running this file directly.
    """
    print("\n" + "=" * 70)
    print("Example 1: Clean Single File")
    print("=" * 70)
    
    # Example 1: Clean a single file
    df = clean_single('FPT.csv', raw_dir='data/raw', clean_dir='data/clean')
    
    if df is not None:
        print(f"\nCleaned data preview:")
        print(df.head())
        print(f"\nTotal records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print("\n" + "=" * 70)
    print("Example 2: Clean All Files")
    print("=" * 70)
    
    # Example 2: Clean all files in directory
    results = clean_many(
        raw_dir='data/raw',
        clean_dir='data/clean',
        validate=True
    )
    
    print(f"\nSuccessfully cleaned {len(results)} files")
    
    # Show summary for each file
    for filename, df in results.items():
        print(f"  {filename}: {len(df)} records")
    
    print("\n" + "=" * 70)
    print("Example 3: Validation Check")
    print("=" * 70)
    
    # Example 3: Validate a cleaned file
    if results:
        first_file = list(results.keys())[0]
        df = results[first_file]
        issues = validate_price_data(df)
        
        if issues:
            print(f"\nValidation issues in {first_file}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n{first_file}: All validation checks passed! ✅")
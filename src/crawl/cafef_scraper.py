import requests
import pandas as pd
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_price_cafef(
    symbol: str, 
    start_date: str, 
    end_date: str,
    page_size: int = 3000,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch historical price data from CafeF API.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'FPT', 'VNM')
        start_date: Start date in format 'DD/MM/YYYY' (e.g., '01/01/2023')
        end_date: End date in format 'DD/MM/YYYY' (e.g., '31/12/2023')
        page_size: Number of records to fetch per request (default: 1000)
        timeout: Request timeout in seconds (default: 30)
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume, ticker
    
    Raises:
        requests.RequestException: If network request fails
        ValueError: If API response is invalid or data is missing
    """
    url = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    params = {
        "Symbol": symbol.upper(),
        "StartDate": start_date,
        "EndDate": end_date,
        "PageIndex": 1,
        "PageSize": page_size
    }

    try:
        # Make request with timeout
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()  # Raise HTTPError for bad status codes
        
        # Parse JSON response
        try:
            data = response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from API: {e}")
        
        # Validate response structure
        if not isinstance(data, dict):
            raise ValueError("API response is not a dictionary")
        
        if "Data" not in data or not isinstance(data["Data"], dict):
            raise ValueError("Missing or invalid 'Data' field in API response")
        
        if "Data" not in data["Data"]:
            raise ValueError("Missing nested 'Data' field in API response")
        
        records = data["Data"]["Data"]
        
        # Handle empty data
        if not records:
            logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Expected columns mapping (CafeF API returns Vietnamese column names)
        column_mapping = {
            "Ngay": "date",
            "GiaMoCua": "open",
            "GiaCaoNhat": "high",
            "GiaThapNhat": "low",
            "GiaDongCua": "close",
            "KhoiLuongKhopLenh": "volume"
        }
        
        # Validate that expected columns exist
        missing_cols = set(column_mapping.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns in API response: {missing_cols}")
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Convert date column
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            raise ValueError(f"Failed to parse dates: {e}")
        
        # Add ticker symbol
        df["ticker"] = symbol.upper()
        
        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Check for NaN values after conversion
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count} invalid values converted to NaN")
        
        # Sort by date and reset index
        df = df.sort_values("date").reset_index(drop=True)
        
        # Select and reorder columns
        df = df[["date", "open", "high", "low", "close", "volume", "ticker"]]
        
        logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        return df
        
    except requests.Timeout:
        raise requests.RequestException(f"Request timeout after {timeout} seconds")
    except requests.RequestException as e:
        raise requests.RequestException(f"Network request failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {symbol}: {e}")
        raise

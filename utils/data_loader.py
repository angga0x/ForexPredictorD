import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

def get_available_pairs():
    """
    Return a list of available forex pairs.
    
    Returns:
        list: List of forex pair symbols
    """
    # Common forex pairs
    forex_pairs = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
        "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X",
        "AUDNZD=X", "AUDCAD=X", "AUDCHF=X", "AUDEUR=X", "AUDJPY=X",
        "CADCHF=X", "CADJPY=X", "CHFJPY=X", "EURAUD=X", "EURCAD=X",
        "EURCHF=X", "EURNZD=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X",
        "GBPNZD=X", "NZDCAD=X", "NZDCHF=X", "NZDJPY=X"
    ]
    return forex_pairs

def get_forex_data(symbol, start_date=None, end_date=None, interval="1d", period=None):
    """
    Fetch historical forex data using yfinance.
    
    Args:
        symbol (str): Forex pair symbol
        start_date (datetime, optional): Start date for data
        end_date (datetime, optional): End date for data
        interval (str, optional): Data interval. Default is "1d".
        period (str, optional): Period for data. Default is None.
        
    Returns:
        pd.DataFrame: DataFrame with forex data or None if error occurs
    """
    try:
        logger.info(f"Fetching data for {symbol} with interval {interval}")
        
        # Use either date range or period
        if start_date is not None and end_date is not None:
            # Convert to string format if datetime objects
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
        else:
            # If no dates provided, use period
            if period is None:
                period = "1mo"  # Default to 1 month if neither dates nor period provided
            data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        # Check if data was successfully retrieved
        if data is None:
            logger.warning(f"No data returned for {symbol}")
            return None
            
        # Check if data is empty
        if len(data) == 0:
            logger.warning(f"Empty data returned for {symbol}")
            return None
        
        # Handle MultiIndex columns in the dataframe (yfinance returns multi-level columns for forex data)
        if isinstance(data.columns, pd.MultiIndex):
            logger.info(f"Detected MultiIndex columns: {data.columns}")
            # Convert multi-level columns to single level with just the first level
            data.columns = [col[0] for col in data.columns]
        
        # Check if data has the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Data is missing required columns: {missing_columns}")
            # For forex data, volume might be missing, so create a dummy column
            if 'Volume' in missing_columns:
                data['Volume'] = 0
        
        # Rename columns for consistency
        data.columns = [str(col).lower() for col in data.columns]
        
        # Handle duplicate indices if any
        if data.index.duplicated().any():
            logger.warning(f"Duplicate timestamps found in data for {symbol}. Keeping the first occurrence.")
            data = data[~data.index.duplicated(keep='first')]
        
        # Handle NaN values
        if data.isna().any().any():
            logger.warning(f"NaN values found in data for {symbol}. Filling with forward fill method.")
            data = data.fillna(method='ffill')
            # If there are still NaN values (e.g., at the beginning), use backward fill
            data = data.fillna(method='bfill')
        
        # Add a column for the date in a more accessible format
        data['date'] = data.index.date
        
        logger.info(f"Successfully fetched {len(data)} rows of data for {symbol}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def save_forex_data(data, symbol, path="data"):
    """
    Save forex data to a CSV file.
    
    Args:
        data (pd.DataFrame): Forex data DataFrame
        symbol (str): Forex pair symbol
        path (str, optional): Directory to save the file. Default is "data".
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import os
        
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Clean symbol for filename
        clean_symbol = symbol.replace('=', '_')
        
        # Create filename with current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{path}/{clean_symbol}_{timestamp}.csv"
        
        # Save to CSV
        data.to_csv(filename)
        logger.info(f"Data saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data for {symbol}: {str(e)}")
        return False

def load_forex_data(filename):
    """
    Load forex data from a CSV file.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with forex data or None if error occurs
    """
    try:
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        logger.info(f"Data loaded from {filename}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {str(e)}")
        return None

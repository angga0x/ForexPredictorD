import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
import os
from utils.twelvedata_api import (
    get_forex_data_twelvedata, get_available_forex_pairs,
    map_yfinance_to_twelvedata_symbol, map_yfinance_to_twelvedata_interval
)

# Configure logging
logger = logging.getLogger(__name__)

# Set data source preference
# Options: 'yfinance', 'twelvedata', 'both' (tries twelvedata first, then yfinance as fallback)
DATA_SOURCE_PREFERENCE = 'both'

def get_available_pairs():
    """
    Return a list of available forex pairs.
    
    Returns:
        list: List of forex pair symbols
    """
    if DATA_SOURCE_PREFERENCE == 'twelvedata' or DATA_SOURCE_PREFERENCE == 'both':
        # Try to get pairs from TwelveData (in TwelveData format)
        td_pairs = get_available_forex_pairs()
        
        # Convert to YFinance format if needed
        if DATA_SOURCE_PREFERENCE == 'both':
            # Create pairs in YFinance format (EUR/USD -> EURUSD=X)
            yf_pairs = []
            for pair in td_pairs:
                if '/' in pair:
                    base = pair.split('/')[0]
                    quote = pair.split('/')[1]
                    yf_pairs.append(f"{base}{quote}=X")
                else:
                    yf_pairs.append(pair)
            return yf_pairs
        return td_pairs
    else:
        # Return pairs in YFinance format
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
    Fetch historical forex data using the configured data source(s).
    
    Args:
        symbol (str): Forex pair symbol (YFinance format like "EURUSD=X")
        start_date (datetime, optional): Start date for data
        end_date (datetime, optional): End date for data
        interval (str, optional): Data interval. Default is "1d".
        period (str, optional): Period for data. Default is None.
        
    Returns:
        pd.DataFrame: DataFrame with forex data or None if error occurs
    """
    logger.info(f"Fetching data for {symbol} with interval {interval}")
    
    # Choose data source based on preference
    if DATA_SOURCE_PREFERENCE == 'twelvedata' or DATA_SOURCE_PREFERENCE == 'both':
        # Try TwelveData first
        try:
            # Convert YFinance symbol to TwelveData format
            td_symbol = map_yfinance_to_twelvedata_symbol(symbol)
            
            # Convert YFinance interval to TwelveData interval format
            td_interval = map_yfinance_to_twelvedata_interval(interval)
            
            # Calculate outputsize (number of data points) based on period if provided
            outputsize = 200  # Increased default for better ML performance
            if period:
                if period == "1d":
                    outputsize = 50  # Increased from 1
                elif period == "5d":
                    outputsize = 100  # Increased from 5
                elif period == "1mo":
                    outputsize = 200  # Increased from 30
                elif period == "3mo":
                    outputsize = 300  # Increased from 90
                elif period == "6mo":
                    outputsize = 500  # Increased from 180
                elif period == "1y":
                    outputsize = 700  # Increased from 365
                elif period == "2y":
                    outputsize = 1000  # Increased from 730
                elif period == "5y":
                    outputsize = 2000  # Increased from 1825
                elif period == "10y":
                    outputsize = 4000  # Increased from 3650
                elif period == "ytd":
                    # Calculate days from start of year to today
                    today = datetime.now()
                    start_of_year = datetime(today.year, 1, 1)
                    days = (today - start_of_year).days
                    outputsize = max(days * 2, 100)  # Increased
            
            # Fetch data from TwelveData
            data = get_forex_data_twelvedata(
                symbol=td_symbol,
                interval=td_interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and not data.empty:
                # Add a column for the date in a more accessible format
                try:
                    data['date'] = [pd.Timestamp(idx).strftime('%Y-%m-%d') for idx in data.index]
                except Exception as e:
                    logger.warning(f"Could not create date column: {str(e)}")
                    data['date'] = data.index.astype(str)
                
                # Add data source attribute
                data.attrs['data_source'] = 'TwelveData'
                logger.info(f"Successfully fetched {len(data)} rows of data for {symbol} from TwelveData")
                return data
            
            # If TwelveData failed and preference is 'both', continue to YFinance
            if DATA_SOURCE_PREFERENCE == 'twelvedata':
                logger.warning(f"Failed to get data from TwelveData for {symbol}")
                return None
                
            logger.info(f"Trying YFinance as fallback for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching data from TwelveData: {str(e)}")
            if DATA_SOURCE_PREFERENCE == 'twelvedata':
                return None
            logger.info(f"Trying YFinance as fallback for {symbol}")
    
    # Use YFinance if preference is 'yfinance' or if TwelveData failed
    try:
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
            logger.warning(f"No data returned from YFinance for {symbol}")
            return None
            
        # Check if data is empty
        if len(data) == 0:
            logger.warning(f"Empty data returned from YFinance for {symbol}")
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
        if np.any(data.index.duplicated()):
            logger.warning(f"Duplicate timestamps found in data for {symbol}. Keeping the first occurrence.")
            data = data[~data.index.duplicated(keep='first')]
        
        # Handle NaN values
        if np.any(np.any(data.isna())):
            logger.warning(f"NaN values found in data for {symbol}. Filling with forward fill method.")
            data = data.fillna(method='ffill')
            # If there are still NaN values (e.g., at the beginning), use backward fill
            data = data.fillna(method='bfill')
        
        # Add a column for the date in a more accessible format
        # Convert index to datetime if needed and create a date column
        try:
            data['date'] = [pd.Timestamp(idx).strftime('%Y-%m-%d') for idx in data.index]
        except Exception as e:
            logger.warning(f"Could not create date column: {str(e)}")
            data['date'] = data.index.astype(str)
        
        # Add data source attribute
        data.attrs['data_source'] = 'YFinance'
        logger.info(f"Successfully fetched {len(data)} rows of data for {symbol} from YFinance")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data from YFinance for {symbol}: {str(e)}")
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

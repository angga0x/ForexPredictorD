"""
TwelveData API integration module for Forex Analysis System.
Provides functionality to interact with the TwelveData API.
"""

import pandas as pd
import numpy as np
import logging
from twelvedata import TDClient
from datetime import datetime, timedelta
import time
from config import TWELVE_DATA_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

def initialize_client():
    """
    Initialize the TwelveData API client.
    
    Returns:
        TDClient: TwelveData API client
    """
    try:
        return TDClient(apikey=TWELVE_DATA_API_KEY)
    except Exception as e:
        logger.error(f"Error initializing TwelveData client: {str(e)}")
        return None

def get_available_forex_pairs():
    """
    Get a list of available forex pairs for TwelveData API.
    
    Returns:
        list: List of available forex pair symbols
    """
    try:
        # Common forex pairs - hardcoded since TDClient doesn't have a direct method to get all forex pairs
        common_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
            "USD/CHF", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP",
            "AUD/NZD", "AUD/CAD", "AUD/CHF", "AUD/JPY", "CAD/CHF",
            "CAD/JPY", "CHF/JPY", "EUR/AUD", "EUR/CAD", "EUR/CHF",
            "EUR/NZD", "GBP/AUD", "GBP/CAD", "GBP/CHF", "GBP/NZD",
            "NZD/CAD", "NZD/CHF", "NZD/JPY"
        ]
        
        return common_pairs
    except Exception as e:
        logger.error(f"Error getting available forex pairs for TwelveData: {str(e)}")
        return []

def get_forex_data_twelvedata(symbol, interval="1day", outputsize=100, start_date=None, end_date=None):
    """
    Get forex data from TwelveData API.
    
    Args:
        symbol (str): Forex pair symbol (e.g., "EUR/USD")
        interval (str, optional): Time interval. Default is "1day".
        outputsize (int, optional): Number of data points to retrieve. Default is 30.
        start_date (datetime, optional): Start date for time series. Default is None.
        end_date (datetime, optional): End date for time series. Default is None.
        
    Returns:
        pd.DataFrame: DataFrame with forex data
    """
    try:
        td_client = initialize_client()
        if td_client is None:
            return None
        
        # Configure time series parameters
        ts = td_client.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=outputsize
        )
        
        # Get the data as pandas DataFrame
        data = ts.as_pandas()
        
        # Handle empty data
        if data is None or data.empty:
            logger.warning(f"No data returned from TwelveData for {symbol}")
            return None
            
        # Ensure the index is datetime type
        data.index = pd.to_datetime(data.index)
        
        # Filter by date range if provided
        if start_date is not None and end_date is not None:
            data = data[(data.index >= pd.Timestamp(start_date)) & 
                        (data.index <= pd.Timestamp(end_date))]
        
        # Rename columns for consistency with the rest of the application
        data.columns = [col.lower() for col in data.columns]
        
        # Create a new DataFrame with renamed columns to avoid the rename issue
        renamed_data = pd.DataFrame()
        
        # Map columns to standard format
        if 'close' in data.columns:
            renamed_data['close'] = data['close']
        if 'high' in data.columns:
            renamed_data['high'] = data['high']
        if 'low' in data.columns:
            renamed_data['low'] = data['low']
        if 'open' in data.columns:
            renamed_data['open'] = data['open']
        if 'volume' in data.columns:
            renamed_data['volume'] = data['volume']
            
        # Use the original index
        renamed_data.index = data.index
        
        # Add volume column if missing (some forex data doesn't include volume)
        if 'volume' not in renamed_data.columns:
            renamed_data['volume'] = 0
            
        # Sort the data by date (ascending)
        renamed_data = renamed_data.sort_index()
        
        logger.info(f"Successfully fetched {len(renamed_data)} rows of data for {symbol} from TwelveData")
        return renamed_data
        
    except Exception as e:
        logger.error(f"Error fetching data from TwelveData: {str(e)}")
        return None

def map_yfinance_to_twelvedata_interval(yf_interval):
    """
    Map YFinance interval to TwelveData interval format.
    
    Args:
        yf_interval (str): YFinance interval format
        
    Returns:
        str: TwelveData interval format
    """
    interval_map = {
        '1m': '1min',
        '2m': '2min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '60m': '1h',
        '90m': '90min',
        '1h': '1h',
        '1d': '1day',
        '5d': '5day',
        '1wk': '1week',
        '1mo': '1month',
        '3mo': '3month'
    }
    
    return interval_map.get(yf_interval, '1day')  # Default to 1day if not found

def map_yfinance_to_twelvedata_symbol(yf_symbol):
    """
    Map YFinance symbol format to TwelveData symbol format.
    
    Args:
        yf_symbol (str): YFinance symbol format (e.g., "EURUSD=X")
        
    Returns:
        str: TwelveData symbol format (e.g., "EUR/USD")
    """
    # If already in TwelveData format, return as is
    if '/' in yf_symbol:
        return yf_symbol
        
    # Handle YFinance forex pair format (e.g., "EURUSD=X")
    if '=' in yf_symbol and yf_symbol.endswith('=X'):
        # Extract the currencies from YFinance format
        pair = yf_symbol.replace('=X', '')
        
        # Common pattern is XXXYYY=X -> XXX/YYY
        if len(pair) >= 6:
            base = pair[:3]
            quote = pair[3:6]
            return f"{base}/{quote}"
    
    # For other formats, return the original symbol
    return yf_symbol

def get_individual_indicator_twelvedata(symbol, indicator_name, interval="1day", outputsize=30, **params):
    """
    Get a single technical indicator from TwelveData API.
    
    Args:
        symbol (str): Forex pair symbol
        indicator_name (str): Name of the indicator (e.g., "sma", "rsi")
        interval (str, optional): Time interval. Default is "1day".
        outputsize (int, optional): Number of data points to retrieve. Default is 30.
        **params: Additional parameters for the indicator
        
    Returns:
        pd.DataFrame: DataFrame with indicator data
    """
    try:
        td_client = initialize_client()
        if td_client is None:
            return None
        
        # Create a time series object
        ts = td_client.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=outputsize
        )
        
        # Use the correct method for the indicator
        if indicator_name.lower() == "sma":
            indicator_ts = ts.with_sma(**params)
        elif indicator_name.lower() == "ema":
            indicator_ts = ts.with_ema(**params)
        elif indicator_name.lower() == "rsi":
            indicator_ts = ts.with_rsi(**params)
        elif indicator_name.lower() == "macd":
            indicator_ts = ts.with_macd(**params)
        elif indicator_name.lower() == "bbands" or indicator_name.lower() == "bollinger_bands":
            indicator_ts = ts.with_bbands(**params)
        elif indicator_name.lower() == "atr":
            indicator_ts = ts.with_atr(**params)
        elif indicator_name.lower() == "stoch":
            indicator_ts = ts.with_stoch(**params)
        else:
            logger.warning(f"Indicator {indicator_name} not implemented for TwelveData API")
            return None
        
        # Get the data as pandas DataFrame
        data = indicator_ts.as_pandas()
        
        if data is None or data.empty:
            logger.warning(f"No {indicator_name} data returned from TwelveData for {symbol}")
            return None
            
        # Ensure datetime index
        data.index = pd.to_datetime(data.index)
        
        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Successfully fetched {indicator_name} data for {symbol} from TwelveData")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {indicator_name} data from TwelveData: {str(e)}")
        return None

def get_price_with_indicators(symbol, indicators=None, interval="1day", outputsize=30):
    """
    Get price data with indicators from TwelveData API.
    
    Args:
        symbol (str): Forex pair symbol
        indicators (list, optional): List of dictionaries with indicator configurations.
                                    Each dict should have 'name' and optionally 'params' keys.
        interval (str, optional): Time interval. Default is "1day".
        outputsize (int, optional): Number of data points to retrieve. Default is 30.
        
    Returns:
        pd.DataFrame: DataFrame with price data and indicators
    """
    try:
        td_client = initialize_client()
        if td_client is None:
            return None
            
        # Get base price data
        ts = td_client.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=outputsize
        )
        
        # Create time series with indicators
        if indicators:
            for ind in indicators:
                name = ind['name']
                params = ind.get('params', {})
                
                # Add indicator to time series
                if name.lower() == "sma":
                    ts = ts.with_sma(**params)
                elif name.lower() == "ema":
                    ts = ts.with_ema(**params)
                elif name.lower() == "rsi":
                    ts = ts.with_rsi(**params)
                elif name.lower() == "macd":
                    ts = ts.with_macd(**params)
                elif name.lower() == "bbands" or name.lower() == "bollinger_bands":
                    ts = ts.with_bbands(**params)
                elif name.lower() == "atr":
                    ts = ts.with_atr(**params)
                elif name.lower() == "stoch":
                    ts = ts.with_stoch(**params)
        
        # Get the data as pandas DataFrame
        data = ts.as_pandas()
        
        if data is None or data.empty:
            logger.warning(f"No data returned from TwelveData for {symbol}")
            return None
            
        # Ensure datetime index
        data.index = pd.to_datetime(data.index)
        
        # Lowercase column names
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Successfully fetched price data with indicators for {symbol} from TwelveData")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data with indicators from TwelveData: {str(e)}")
        return None
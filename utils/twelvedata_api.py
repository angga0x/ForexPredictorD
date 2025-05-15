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

def get_forex_data_twelvedata(symbol, interval="1day", outputsize=30, start_date=None, end_date=None):
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
        
        # Handle specific column renames if needed
        column_mapping = {
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'open': 'open',
            'volume': 'volume'
        }
        
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        # Add volume column if missing (some forex data doesn't include volume)
        if 'volume' not in data.columns:
            data['volume'] = 0
            
        # Sort the data by date (ascending)
        data = data.sort_index()
        
        logger.info(f"Successfully fetched {len(data)} rows of data for {symbol} from TwelveData")
        return data
        
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

def get_technical_indicators_twelvedata(symbol, indicator, interval="1day", outputsize=30, params=None):
    """
    Get technical indicator data from TwelveData API.
    
    Args:
        symbol (str): Forex pair symbol
        indicator (str): Technical indicator name
        interval (str, optional): Time interval. Default is "1day".
        outputsize (int, optional): Number of data points to retrieve. Default is 30.
        params (dict, optional): Additional parameters for the indicator. Default is None.
        
    Returns:
        pd.DataFrame: DataFrame with indicator data
    """
    try:
        td_client = initialize_client()
        if td_client is None:
            return None
            
        # Set default parameters if None provided
        if params is None:
            params = {}
            
        # Get indicator data
        indicator_data = td_client.get_indicator(
            symbol=symbol,
            interval=interval,
            indicator=indicator,
            outputsize=outputsize,
            **params
        )
        
        # Convert to pandas DataFrame
        data = indicator_data.as_pandas()
        
        if data is None or data.empty:
            logger.warning(f"No {indicator} data returned from TwelveData for {symbol}")
            return None
            
        # Ensure datetime index
        data.index = pd.to_datetime(data.index)
        
        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        logger.info(f"Successfully fetched {indicator} data for {symbol} from TwelveData")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {indicator} data from TwelveData: {str(e)}")
        return None

def get_multiple_indicators_twelvedata(symbol, indicators, interval="1day", outputsize=30):
    """
    Get multiple technical indicators from TwelveData API.
    
    Args:
        symbol (str): Forex pair symbol
        indicators (list): List of dictionaries with indicator configurations
                          Each dict should have 'name' and optionally 'params' keys
        interval (str, optional): Time interval. Default is "1day".
        outputsize (int, optional): Number of data points to retrieve. Default is 30.
        
    Returns:
        pd.DataFrame: DataFrame with all indicators
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
        
        # Initialize batch request with price data
        batch = td_client.batch()
        batch.add(ts)
        
        # Add all indicators to the batch
        for ind in indicators:
            name = ind['name']
            params = ind.get('params', {})
            
            indicator = td_client.get_indicator(
                symbol=symbol,
                interval=interval,
                indicator=name,
                outputsize=outputsize,
                **params
            )
            batch.add(indicator)
            
        # Execute batch request
        data = batch.execute()
        
        # Process results
        if not data or len(data) < 2:
            logger.warning(f"Failed to get indicator data for {symbol}")
            return None
            
        # Convert to pandas DataFrames
        price_data = data[0].as_pandas()
        
        # Merge all indicator DataFrames
        for i, ind in enumerate(indicators):
            indicator_df = data[i+1].as_pandas()
            
            # Skip if empty
            if indicator_df is None or indicator_df.empty:
                continue
                
            # Rename columns to avoid duplicates and add indicator prefix
            indicator_df.columns = [f"{ind['name'].lower()}_{col.lower()}" 
                                   for col in indicator_df.columns]
            
            # Merge with price data
            price_data = price_data.join(indicator_df)
            
        # Ensure lowercase column names for consistency
        price_data.columns = [col.lower() for col in price_data.columns]
        
        logger.info(f"Successfully fetched multiple indicators for {symbol} from TwelveData")
        return price_data
        
    except Exception as e:
        logger.error(f"Error fetching multiple indicators from TwelveData: {str(e)}")
        return None
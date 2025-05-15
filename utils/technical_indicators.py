import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def add_moving_averages(df, sma_period=20, ema_period=20):
    """
    Add Simple Moving Average (SMA) and Exponential Moving Average (EMA) to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        sma_period (int, optional): Period for SMA calculation. Default is 20.
        ema_period (int, optional): Period for EMA calculation. Default is 20.
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate SMA
        result[f'sma_{sma_period}'] = result['close'].rolling(window=sma_period).mean()
        
        # Calculate EMA
        result[f'ema_{ema_period}'] = result['close'].ewm(span=ema_period, adjust=False).mean()
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating moving averages: {str(e)}")
        return df

def add_rsi(df, period=14):
    """
    Add Relative Strength Index (RSI) to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int, optional): Period for RSI calculation. Default is 14.
        
    Returns:
        pd.DataFrame: DataFrame with added indicator
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate price changes
        delta = result['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return df

def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Add Moving Average Convergence Divergence (MACD) to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        fast_period (int, optional): Fast EMA period. Default is 12.
        slow_period (int, optional): Slow EMA period. Default is 26.
        signal_period (int, optional): Signal EMA period. Default is 9.
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate EMAs
        ema_fast = result['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = result['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        result['macd_line'] = ema_fast - ema_slow
        
        # Calculate MACD signal line
        result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        result['macd_histogram'] = result['macd_line'] - result['macd_signal']
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return df

def add_bollinger_bands(df, period=20, std_dev=2.0):
    """
    Add Bollinger Bands to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int, optional): Period for SMA calculation. Default is 20.
        std_dev (float, optional): Number of standard deviations. Default is 2.0.
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate SMA
        sma = result['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = result['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        result[f'bb_upper_{period}'] = sma + (std * std_dev)
        result[f'bb_middle_{period}'] = sma
        result[f'bb_lower_{period}'] = sma - (std * std_dev)
        
        # Calculate bandwidth and %B
        result[f'bb_bandwidth_{period}'] = (result[f'bb_upper_{period}'] - result[f'bb_lower_{period}']) / result[f'bb_middle_{period}']
        result[f'bb_percent_b_{period}'] = (result['close'] - result[f'bb_lower_{period}']) / (result[f'bb_upper_{period}'] - result[f'bb_lower_{period}'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return df

def add_atr(df, period=14):
    """
    Add Average True Range (ATR) to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int, optional): Period for ATR calculation. Default is 14.
        
    Returns:
        pd.DataFrame: DataFrame with added indicator
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate True Range
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                abs(result['high'] - result['close'].shift())
            ),
            abs(result['low'] - result['close'].shift())
        )
        
        # Calculate ATR
        result[f'atr_{period}'] = result['tr'].rolling(window=period).mean()
        
        # Drop temporary column
        result = result.drop('tr', axis=1)
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        return df

def add_stoch_rsi(df, period=14, smooth_k=3, smooth_d=3):
    """
    Add Stochastic RSI to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int, optional): Period for RSI calculation. Default is 14.
        smooth_k (int, optional): K period for smoothing. Default is 3.
        smooth_d (int, optional): D period for smoothing. Default is 3.
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate RSI if not already present
        if f'rsi_{period}' not in result.columns:
            result = add_rsi(result, period)
        
        # Get RSI values
        rsi_values = result[f'rsi_{period}']
        
        # Calculate Stochastic RSI
        stoch_rsi = (rsi_values - rsi_values.rolling(period).min()) / (rsi_values.rolling(period).max() - rsi_values.rolling(period).min())
        
        # Apply smoothing
        result[f'stoch_rsi_k_{period}'] = stoch_rsi.rolling(window=smooth_k).mean()
        result[f'stoch_rsi_d_{period}'] = result[f'stoch_rsi_k_{period}'].rolling(window=smooth_d).mean()
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic RSI: {str(e)}")
        return df

def add_stochastic(df, k_period=14, d_period=3):
    """
    Add Stochastic Oscillator indicator to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        k_period (int, optional): K period. Default is 14.
        d_period (int, optional): D period. Default is 3.
        
    Returns:
        pd.DataFrame: DataFrame with added indicator
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate %K
        lowest_low = result['low'].rolling(window=k_period).min()
        highest_high = result['high'].rolling(window=k_period).max()
        result['stoch_k'] = 100 * ((result['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (moving average of %K)
        result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
        
        return result
    except Exception as e:
        logger.error(f"Error adding Stochastic Oscillator: {str(e)}")
        return df

def add_adx(df, period=14):
    """
    Add Average Directional Index (ADX) indicator to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int, optional): Period for ADX calculation. Default is 14.
        
    Returns:
        pd.DataFrame: DataFrame with added indicator
    """
    try:
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Calculate True Range
        result['tr1'] = abs(result['high'] - result['low'])
        result['tr2'] = abs(result['high'] - result['close'].shift(1))
        result['tr3'] = abs(result['low'] - result['close'].shift(1))
        result['tr'] = result[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Directional Movement (+DM and -DM)
        result['up_move'] = result['high'] - result['high'].shift(1)
        result['down_move'] = result['low'].shift(1) - result['low']
        
        result['plus_dm'] = 0
        result.loc[(result['up_move'] > result['down_move']) & (result['up_move'] > 0), 'plus_dm'] = result['up_move']
        
        result['minus_dm'] = 0
        result.loc[(result['down_move'] > result['up_move']) & (result['down_move'] > 0), 'minus_dm'] = result['down_move']
        
        # Calculate smoothed averages
        result['atr_adx'] = result['tr'].rolling(window=period).mean()
        result['plus_di'] = 100 * (result['plus_dm'].rolling(window=period).mean() / result['atr_adx'])
        result['minus_di'] = 100 * (result['minus_dm'].rolling(window=period).mean() / result['atr_adx'])
        
        # Calculate ADX
        result['dx'] = 100 * (abs(result['plus_di'] - result['minus_di']) / (result['plus_di'] + result['minus_di']))
        result['adx'] = result['dx'].rolling(window=period).mean()
        
        # Drop intermediate columns
        drop_cols = ['tr1', 'tr2', 'tr3', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'dx']
        result = result.drop([col for col in drop_cols if col in result.columns], axis=1)
        
        return result
    except Exception as e:
        logger.error(f"Error adding ADX: {str(e)}")
        return df

def add_all_indicators(df, sma_period=20, ema_period=20, rsi_period=14, 
                       macd_fast=12, macd_slow=26, macd_signal=9,
                       bb_period=20, bb_std=2.0, atr_period=14,
                       stoch_rsi_period=14, stoch_rsi_smooth_k=3, stoch_rsi_smooth_d=3,
                       adx_period=14, stoch_period=14):
    """
    Add all technical indicators to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        Various indicator parameters
        
    Returns:
        pd.DataFrame: DataFrame with all indicators added
    """
    try:
        logger.info("Calculating all technical indicators")
        
        # Apply all indicators
        result = df.copy()
        result = add_moving_averages(result, sma_period, ema_period)
        result = add_rsi(result, rsi_period)
        result = add_macd(result, macd_fast, macd_slow, macd_signal)
        result = add_bollinger_bands(result, bb_period, bb_std)
        result = add_atr(result, atr_period)
        result = add_stoch_rsi(result, stoch_rsi_period, stoch_rsi_smooth_k, stoch_rsi_smooth_d)
        
        # Add new indicators for consensus strategy
        result = add_stochastic(result, stoch_period)
        result = add_adx(result, adx_period)
        
        # Add SMA with different periods for crossover analysis
        # Add short term SMA for crossover signal
        if f'sma_{sma_period}' in result.columns and 'close' in result.columns:
            result['sma_50'] = result['close'].rolling(window=50).mean()
            result['sma_9'] = result['close'].rolling(window=9).mean()
        
        # Add volatility
        result['volatility'] = result['close'].pct_change().rolling(window=20).std() * (252 ** 0.5)  # Annualized
        
        # Generate trading signals based on indicators
        result = add_trading_signals(result, sma_period, rsi_period)
        
        # Clean up any NaN values that might have been introduced
        # result.fillna(method='bfill', inplace=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating all indicators: {str(e)}")
        return df

def add_trading_signals(df, sma_period=20, rsi_period=14):
    """
    Add basic trading signals based on technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        sma_period (int, optional): SMA period used. Default is 20.
        rsi_period (int, optional): RSI period used. Default is 14.
        
    Returns:
        pd.DataFrame: DataFrame with added signals
    """
    try:
        result = df.copy()
        
        # Moving Average Crossover Signal
        # Buy when price crosses above SMA, Sell when price crosses below SMA
        result['sma_signal'] = 0
        result.loc[result['close'] > result[f'sma_{sma_period}'], 'sma_signal'] = 1
        result.loc[result['close'] < result[f'sma_{sma_period}'], 'sma_signal'] = -1
        
        # RSI Signal
        # Buy when RSI crosses above 30 (oversold), Sell when RSI crosses below 70 (overbought)
        result['rsi_signal'] = 0
        result.loc[result[f'rsi_{rsi_period}'] < 30, 'rsi_signal'] = 1
        result.loc[result[f'rsi_{rsi_period}'] > 70, 'rsi_signal'] = -1
        
        # MACD Signal
        # Buy when MACD line crosses above signal line, Sell when MACD line crosses below signal line
        result['macd_signal_crossover'] = 0
        result.loc[result['macd_line'] > result['macd_signal'], 'macd_signal_crossover'] = 1
        result.loc[result['macd_line'] < result['macd_signal'], 'macd_signal_crossover'] = -1
        
        # Bollinger Bands Signal
        # Buy when price touches lower band, Sell when price touches upper band
        result['bb_signal'] = 0
        result.loc[result['close'] <= result[f'bb_lower_{sma_period}'], 'bb_signal'] = 1
        result.loc[result['close'] >= result[f'bb_upper_{sma_period}'], 'bb_signal'] = -1
        
        # Combined Signal (simple average of all signals)
        result['combined_signal'] = (
            result['sma_signal'] + 
            result['rsi_signal'] + 
            result['macd_signal_crossover'] + 
            result['bb_signal']
        ) / 4
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating trading signals: {str(e)}")
        return df

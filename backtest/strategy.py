import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def sma_crossover_strategy(df, sma_period=20):
    """
    Implement SMA crossover strategy for backtesting.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        sma_period (int, optional): SMA period. Default is 20.
        
    Returns:
        pd.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
    """
    try:
        signals = pd.Series(0, index=df.index)
        
        # Buy signal: price crosses above SMA
        buy_signal = (df['close'] > df[f'sma_{sma_period}']) & (df['close'].shift(1) <= df[f'sma_{sma_period}'].shift(1))
        signals[buy_signal] = 1
        
        # Sell signal: price crosses below SMA
        sell_signal = (df['close'] < df[f'sma_{sma_period}']) & (df['close'].shift(1) >= df[f'sma_{sma_period}'].shift(1))
        signals[sell_signal] = -1
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in SMA crossover strategy: {str(e)}")
        return pd.Series(0, index=df.index)

def rsi_strategy(df, rsi_period=14, oversold=30, overbought=70):
    """
    Implement RSI overbought/oversold strategy for backtesting.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        rsi_period (int, optional): RSI period. Default is 14.
        oversold (int, optional): Oversold threshold. Default is 30.
        overbought (int, optional): Overbought threshold. Default is 70.
        
    Returns:
        pd.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
    """
    try:
        signals = pd.Series(0, index=df.index)
        
        # Buy signal: RSI crosses above oversold level
        buy_signal = (df[f'rsi_{rsi_period}'] > oversold) & (df[f'rsi_{rsi_period}'].shift(1) <= oversold)
        signals[buy_signal] = 1
        
        # Sell signal: RSI crosses below overbought level
        sell_signal = (df[f'rsi_{rsi_period}'] < overbought) & (df[f'rsi_{rsi_period}'].shift(1) >= overbought)
        signals[sell_signal] = -1
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in RSI strategy: {str(e)}")
        return pd.Series(0, index=df.index)

def macd_strategy(df):
    """
    Implement MACD signal line crossover strategy for backtesting.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        
    Returns:
        pd.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
    """
    try:
        signals = pd.Series(0, index=df.index)
        
        # Buy signal: MACD line crosses above signal line
        buy_signal = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        signals[buy_signal] = 1
        
        # Sell signal: MACD line crosses below signal line
        sell_signal = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
        signals[sell_signal] = -1
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in MACD strategy: {str(e)}")
        return pd.Series(0, index=df.index)

def bollinger_bands_strategy(df, bb_period=20):
    """
    Implement Bollinger Bands mean reversion strategy for backtesting.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        bb_period (int, optional): Bollinger Bands period. Default is 20.
        
    Returns:
        pd.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
    """
    try:
        signals = pd.Series(0, index=df.index)
        
        # Buy signal: price touches lower band
        buy_signal = df['close'] <= df[f'bb_lower_{bb_period}']
        signals[buy_signal] = 1
        
        # Sell signal: price touches upper band
        sell_signal = df['close'] >= df[f'bb_upper_{bb_period}']
        signals[sell_signal] = -1
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in Bollinger Bands strategy: {str(e)}")
        return pd.Series(0, index=df.index)

def ml_signal_strategy(prediction_data):
    """
    Use machine learning predictions as trading signals.
    
    Args:
        prediction_data (dict): Dictionary with prediction data
        
    Returns:
        pd.Series: Series of signals (1 for buy, -1 for sell, 0 for hold)
    """
    try:
        if prediction_data is None or 'predictions' not in prediction_data:
            logger.error("No prediction data provided for ML signal strategy")
            return None
        
        predictions = prediction_data['predictions']
        
        # Convert predictions to signals (1 for predicted price increase, -1 for predicted price decrease)
        signals = pd.Series(0, index=range(len(predictions)))
        signals[predictions == 1] = 1
        signals[predictions == 0] = -1
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in ML signal strategy: {str(e)}")
        return None

def run_backtest(df, strategy_type='SMA Crossover', initial_capital=10000, prediction_data=None):
    """
    Run backtest on the selected strategy.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        strategy_type (str, optional): Type of strategy to use. Default is 'SMA Crossover'.
        initial_capital (float, optional): Initial capital for backtesting. Default is 10000.
        prediction_data (dict, optional): Dictionary with prediction data for ML/DL strategy. Default is None.
        
    Returns:
        dict: Backtest results
    """
    try:
        logger.info(f"Running backtest with strategy: {strategy_type}")
        
        # Generate signals based on selected strategy
        if strategy_type == 'SMA Crossover':
            signals = sma_crossover_strategy(df)
        elif strategy_type == 'RSI Overbought/Oversold':
            signals = rsi_strategy(df)
        elif strategy_type == 'MACD Signal':
            signals = macd_strategy(df)
        elif strategy_type == 'Bollinger Bands':
            signals = bollinger_bands_strategy(df)
        elif strategy_type == 'ML/DL Signal':
            if prediction_data is None:
                logger.error("No prediction data provided for ML/DL signal strategy")
                signals = pd.Series(0, index=df.index)
            else:
                signals = ml_signal_strategy(prediction_data)
                if signals is None or len(signals) != len(df):
                    logger.error(f"Invalid ML/DL signals: length mismatch. signals: {len(signals) if signals is not None else 'None'}, df: {len(df)}")
                    signals = pd.Series(0, index=df.index)
                else:
                    signals.index = df.index  # Set the correct index
        else:
            logger.error(f"Unknown strategy type: {strategy_type}")
            signals = pd.Series(0, index=df.index)
        
        # Create a copy of the dataframe with signals
        backtest_df = df.copy()
        backtest_df['signal'] = signals
        
        # Initialize position (0 for no position, 1 for long)
        backtest_df['position'] = 0
        
        # Fill position column based on signals
        position = 0
        for i in range(len(backtest_df)):
            if backtest_df['signal'].iloc[i] == 1:  # Buy signal
                position = 1
            elif backtest_df['signal'].iloc[i] == -1:  # Sell signal
                position = 0
            
            backtest_df['position'].iloc[i] = position
        
        # Calculate returns
        backtest_df['returns'] = backtest_df['close'].pct_change()
        
        # Calculate strategy returns
        backtest_df['strategy_returns'] = backtest_df['position'].shift(1) * backtest_df['returns']
        
        # Calculate cumulative returns
        backtest_df['cumulative_returns'] = (1 + backtest_df['returns']).cumprod()
        backtest_df['strategy_cumulative_returns'] = (1 + backtest_df['strategy_returns']).cumprod()
        
        # Calculate equity curve
        backtest_df['equity_curve'] = initial_capital * backtest_df['strategy_cumulative_returns']
        
        # Find all trades
        trades = []
        position_changes = backtest_df['position'].diff()
        
        for i in range(1, len(backtest_df)):
            if position_changes.iloc[i] == 1:  # Enter long position
                trades.append({
                    'type': 'buy',
                    'date': backtest_df.index[i],
                    'price': backtest_df['close'].iloc[i],
                    'position_size': 1  # For simplicity, we're using a fixed position size
                })
            elif position_changes.iloc[i] == -1:  # Exit long position
                trades.append({
                    'type': 'sell',
                    'date': backtest_df.index[i],
                    'price': backtest_df['close'].iloc[i],
                    'position_size': 1  # For simplicity, we're using a fixed position size
                })
        
        # Calculate performance metrics
        total_return = backtest_df['strategy_cumulative_returns'].iloc[-1] - 1 if not backtest_df.empty else 0
        
        # Calculate annualized return
        days = (backtest_df.index[-1] - backtest_df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Calculate Sharpe ratio (using 0% as risk-free rate for simplicity)
        returns_std = backtest_df['strategy_returns'].std() * (252 ** 0.5)  # Annualized
        sharpe_ratio = annual_return / returns_std if returns_std > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_max = backtest_df['equity_curve'].cummax()
        drawdown = (backtest_df['equity_curve'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate and profit factor
        winning_trades = [t for i, t in enumerate(trades[:-1]) if 
                        t['type'] == 'buy' and trades[i+1]['type'] == 'sell' and 
                        trades[i+1]['price'] > t['price']]
        
        losing_trades = [t for i, t in enumerate(trades[:-1]) if 
                        t['type'] == 'buy' and trades[i+1]['type'] == 'sell' and 
                        trades[i+1]['price'] <= t['price']]
        
        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum([trades[i+1]['price'] - t['price'] for i, t in enumerate(trades[:-1]) if 
                        t['type'] == 'buy' and trades[i+1]['type'] == 'sell' and 
                        trades[i+1]['price'] > t['price']])
        
        gross_loss = sum([t['price'] - trades[i+1]['price'] for i, t in enumerate(trades[:-1]) if 
                        t['type'] == 'buy' and trades[i+1]['type'] == 'sell' and 
                        trades[i+1]['price'] <= t['price']])
        
        profit_factor = gross_profit / abs(gross_loss) if abs(gross_loss) > 0 else float('inf')
        
        # Log results
        logger.info(f"Backtest results: Total Return: {total_return:.2%}, Annual Return: {annual_return:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return {
            'equity_curve': backtest_df['equity_curve'],
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'trades': trades
        }
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        # Return a minimal result set in case of error
        return {
            'equity_curve': pd.Series([initial_capital]),
            'total_return': 0,
            'annual_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'trades': []
        }

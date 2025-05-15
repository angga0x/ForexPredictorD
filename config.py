"""
Configuration file for Forex Analysis and Prediction System.
This file contains default parameters for the application.
"""

# Data parameters
DEFAULT_FOREX_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"
]
DEFAULT_PERIOD = "1mo"  # 1 month
DEFAULT_INTERVAL = "1d"  # 1 day

# Technical indicator parameters
DEFAULT_SMA_PERIOD = 20
DEFAULT_EMA_PERIOD = 20
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_ATR_PERIOD = 14
DEFAULT_STOCH_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_SMOOTH_K = 3
DEFAULT_STOCH_RSI_SMOOTH_D = 3

# Prediction parameters
DEFAULT_PREDICTION_HORIZON = 5  # 5 days ahead
DEFAULT_TRAIN_TEST_SPLIT = 80  # 80% training, 20% testing
DEFAULT_FEATURE_SELECTION_METHOD = "Correlation"
DEFAULT_N_FEATURES = 10

# LSTM parameters
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_LSTM_UNITS = 64
DEFAULT_LSTM_DROPOUT = 0.2
DEFAULT_LSTM_EPOCHS = 50
DEFAULT_LSTM_BATCH_SIZE = 32

# Backtesting parameters
DEFAULT_INITIAL_CAPITAL = 10000
DEFAULT_STRATEGY = "SMA Crossover"

# Visualization parameters
DEFAULT_THEME = "plotly_white"
DEFAULT_CANDLESTICK_COLORS = {
    "increasing": "#26a69a",
    "decreasing": "#ef5350"
}

# API keys (should be loaded from environment variables in production)
TWELVE_DATA_API_KEY = ""  # os.getenv("TWELVE_DATA_API_KEY", "")

# Logging parameters
LOG_LEVEL = "INFO"

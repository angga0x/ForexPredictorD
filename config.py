"""
Configuration file for Forex Analysis and Prediction System.
This file contains default parameters for the application.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Loading configuration from .env file if available")

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
# Flag to enable/disable LSTM functionality (from environment variable or default to False)
ENABLE_LSTM = os.getenv("ENABLE_LSTM", "False").lower() == "true"

# Telegram notification parameters
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# Flag to enable/disable Telegram notifications (from environment variable or default to False)
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "False").lower() == "true"
DEFAULT_NOTIFICATION_TYPES = ["LSTM Predictions", "Trading Signals", "Price Alerts"]

# Check if Telegram is properly configured
if ENABLE_TELEGRAM and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
    logger.warning("Telegram notifications enabled but missing bot token or chat ID. Check your .env file.")
    ENABLE_TELEGRAM = False

# Backtesting parameters
DEFAULT_INITIAL_CAPITAL = 10000
DEFAULT_STRATEGY = "SMA Crossover"

# Visualization parameters
DEFAULT_THEME = "plotly_white"
DEFAULT_CANDLESTICK_COLORS = {
    "increasing": "#26a69a",
    "decreasing": "#ef5350"
}

# API keys (loaded from environment variables)
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Prediction thresholds (customizable via environment variables)
PREDICTION_THRESHOLD_UP = float(os.getenv("PREDICTION_THRESHOLD_UP", "0.52"))
PREDICTION_THRESHOLD_DOWN = float(os.getenv("PREDICTION_THRESHOLD_DOWN", "0.48"))
BACKTEST_SIGNAL_THRESHOLD = float(os.getenv("BACKTEST_SIGNAL_THRESHOLD", "0.5"))

# Logging parameters
LOG_LEVEL = "INFO"

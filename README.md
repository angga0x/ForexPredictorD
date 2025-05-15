# Forex Analysis & Prediction System

A comprehensive Python application for forex market analysis with technical indicators, ML/DL models, and interactive visualizations.

## Features

- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Stochastic Oscillator, ADX, and more
- **Machine Learning Models**: Random Forest, XGBoost, and Gradient Boosting for price prediction
- **Deep Learning**: LSTM neural network implementation (requires TensorFlow)
- **Technical Consensus Strategy**: Combined signal analysis from multiple indicators 
- **Interactive UI**: Built with Streamlit for easy visualization and analysis
- **Backtesting**: Test trading strategies with historical data
- **Telegram Notifications**: Automated alerts for predictions and signals
- **Economic Calendar**: Data on major economic events affecting forex markets
- **News Analysis**: Financial news sentiment analysis

## Installation

### Dependencies

This project requires Python 3.9+ and the following packages:
- streamlit
- yfinance
- numpy
- pandas
- scikit-learn
- plotly
- python-telegram-bot
- tensorflow (optional, for LSTM functionality)
- trafilatura
- xgboost
- beautifulsoup4
- newsapi-python
- twelvedata
- python-dotenv

The full list is available in `dependencies.txt`.

### Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install the required dependencies
4. Run the application: `streamlit run app.py`

## Configuration

### Environment Variables

The following environment variables can be set in the `.env` file:

```
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# TwelveData API (if using TwelveData as a data source)
TWELVEDATA_API_KEY=your_twelvedata_api_key_here

# NewsAPI (for news sentiment analysis)
NEWS_API_KEY=your_newsapi_key_here

# App Configuration
ENABLE_TELEGRAM=False
ENABLE_LSTM=False

# Prediction thresholds
PREDICTION_THRESHOLD_UP=0.52
PREDICTION_THRESHOLD_DOWN=0.48
BACKTEST_SIGNAL_THRESHOLD=0.5
```

## Usage

1. Select the forex pair to analyze from the dropdown
2. Choose the time period and interval
3. Configure model parameters in the sidebar
4. Click "Run Analysis" to generate predictions and visualizations
5. View the prediction summary with target levels at the bottom

## Prediction Summary

The prediction summary provides the following information:
- Currency pair
- Analysis period
- Prediction direction (UP/DOWN)
- Confidence percentage
- Current price
- Target profit level
- Stop loss level
- Risk/reward ratio
- Processing time

## Telegram Notifications

To enable Telegram notifications:
1. Create a Telegram bot using BotFather
2. Get your chat ID
3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the .env file
4. Enable notifications in the app's sidebar

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
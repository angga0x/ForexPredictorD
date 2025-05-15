import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os
import importlib
import time

# Import custom modules
from utils.data_loader import get_forex_data, get_available_pairs
from utils.technical_indicators import add_all_indicators
from utils.preprocessing import prepare_data_for_training, feature_selection
from utils.visualization import plot_candlestick_with_indicators, plot_model_performance, plot_feature_importance
from utils.news_api import get_news_sentiment_for_pair
from utils.economic_calendar import get_economic_calendar
from utils.prediction_summary import format_prediction_summary, get_prediction_summary_html
from models.machine_learning import train_evaluate_ml_models
from backtest.strategy import run_backtest

# Try to import telegram notification module
try:
    from utils.telegram_notification import (
        send_message, send_price_alert, send_ml_prediction, 
        send_trading_signal, send_lstm_prediction, is_telegram_configured
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("Telegram notification module could not be imported")

# Set LSTM availability to false by default
LSTM_AVAILABLE = False
# Do not attempt to import TensorFlow at module level due to compatibility issues
logging.warning("TensorFlow/LSTM modules disabled due to compatibility issues")
    
# Import configuration
from config import ENABLE_TELEGRAM

# Override ENABLE_LSTM based on actual availability
ENABLE_LSTM = LSTM_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setting page config
st.set_page_config(
    page_title="Forex Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Forex Market Analysis & Prediction System")
st.markdown("""
This application provides comprehensive analysis and prediction tools for forex market data.
Features include technical indicator visualization, machine learning predictions, and backtesting.
""")

# Sidebar for inputs
st.sidebar.header("Data Source Configuration")

# Data source preference
data_source_options = ["both", "yfinance", "twelvedata"]
data_source = st.sidebar.radio(
    "Data Source Preference",
    data_source_options,
    index=0,
    help="Choose which data source to use for forex data. 'both' will try TwelveData first, then fall back to YFinance if needed."
)

st.sidebar.header("Data Parameters")

# Select forex pair
forex_pairs = get_available_pairs()
selected_pair = st.sidebar.selectbox("Select Forex Pair", forex_pairs, index=0)

# Time period inputs
period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
selected_period = st.sidebar.selectbox("Select Time Period", period_options, index=2)

# Or use date range picker
use_date_range = st.sidebar.checkbox("Use Custom Date Range")

# Initialize date variables with default values
start_date = None
end_date = None

if use_date_range:
    end_date_default = datetime.now()
    start_date_default = end_date_default - timedelta(days=90)
    start_date = st.sidebar.date_input("Start Date", start_date_default)
    end_date = st.sidebar.date_input("End Date", end_date_default)
    if start_date >= end_date:
        st.sidebar.error("End date must be after start date.")

# Interval selection
interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
selected_interval = st.sidebar.selectbox("Select Interval", interval_options, index=8)

# Display info about the data sources
st.sidebar.info("""
**Data Sources**:
- **TwelveData**: Professional API with high-quality forex data
- **YFinance**: Yahoo Finance data API
""")

# Update the data source preference in the data loader module
import utils.data_loader as data_loader
data_loader.DATA_SOURCE_PREFERENCE = data_source

# Technical indicator parameters
st.sidebar.header("Technical Indicators Parameters")
sma_period = st.sidebar.slider("SMA Period", 5, 200, 20)
ema_period = st.sidebar.slider("EMA Period", 5, 200, 20)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
macd_fast = st.sidebar.slider("MACD Fast Period", 5, 30, 12)
macd_slow = st.sidebar.slider("MACD Slow Period", 15, 50, 26)
macd_signal = st.sidebar.slider("MACD Signal Period", 5, 20, 9)
bb_period = st.sidebar.slider("Bollinger Bands Period", 5, 30, 20)
bb_std = st.sidebar.slider("Bollinger Bands Std", 1.0, 4.0, 2.0)
atr_period = st.sidebar.slider("ATR Period", 5, 30, 14)
stoch_rsi_period = st.sidebar.slider("Stochastic RSI Period", 5, 30, 14)
stoch_rsi_smooth_k = st.sidebar.slider("Stochastic RSI Smooth K", 1, 10, 3)
stoch_rsi_smooth_d = st.sidebar.slider("Stochastic RSI Smooth D", 1, 10, 3)

# Prediction parameters
st.sidebar.header("Prediction Parameters")
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 5)
train_test_split = st.sidebar.slider("Train/Test Split (%)", 50, 90, 80)
feature_selection_method = st.sidebar.selectbox(
    "Feature Selection Method", 
    ["None", "Correlation", "Mutual Information", "Recursive Feature Elimination"]
)
n_features = st.sidebar.slider("Number of Features to Select", 5, 30, 10)

# Model selection
st.sidebar.header("Model Selection")
use_random_forest = st.sidebar.checkbox("Random Forest", value=True)
use_xgboost = st.sidebar.checkbox("XGBoost", value=True)
use_gradient_boosting = st.sidebar.checkbox("Gradient Boosting", value=False)

# LSTM (deep learning) option - only enable if actually available
if LSTM_AVAILABLE:
    use_lstm = st.sidebar.checkbox("LSTM (Deep Learning)", value=True,
                                help="Use Long Short-Term Memory neural networks for predictions")
    
    # LSTM parameters if enabled
    if use_lstm:
        st.sidebar.subheader("LSTM Parameters")
        lstm_units = st.sidebar.slider("LSTM Units", 16, 128, 64, step=16)
        lstm_dropout = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2, step=0.1)
        lstm_epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, step=10)
        sequence_length = st.sidebar.slider("Sequence Length", 5, 30, 10, step=1)
else:
    # If LSTM is not available, show a disabled checkbox with explanation
    use_lstm = st.sidebar.checkbox("LSTM (Deep Learning)", value=False, disabled=True,
                                help="LSTM requires TensorFlow, which is not available in this environment")
    st.sidebar.info("💡 LSTM functionality is disabled because TensorFlow could not be imported. This might be due to compatibility issues with the current environment.")
    
# Telegram notification options
if TELEGRAM_AVAILABLE and ENABLE_TELEGRAM:
    st.sidebar.header("Telegram Notifications")
    use_telegram = st.sidebar.checkbox("Enable Telegram Notifications", value=True,
                                    help="Send trading signals and alerts to Telegram")
    
    # Notification types
    if use_telegram:
        notification_types = st.sidebar.multiselect(
            "Notification Types",
            ["ML Predictions", "Trading Signals", "Price Alerts"],
            default=["Trading Signals", "Price Alerts"]
        )
        
        # Add LSTM predictions option only if LSTM is available
        if LSTM_AVAILABLE and use_lstm:
            if "LSTM Predictions" not in notification_types:
                notification_types.append("LSTM Predictions")
        
        # Telegram status indicator
        try:
            telegram_configured = is_telegram_configured()
            if telegram_configured:
                st.sidebar.success("✅ Telegram bot configured")
            else:
                st.sidebar.warning("⚠️ Telegram bot not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment variables.")
        except:
            st.sidebar.warning("⚠️ Unable to check Telegram configuration status")
else:
    use_telegram = False

# Add a note about data requirements for ML models
st.sidebar.info("Note: Machine learning models require sufficient historical data. For daily data, select a period of at least '1mo' or longer.")

# Backtesting parameters
st.sidebar.header("Backtesting Parameters")
strategy_type = st.sidebar.selectbox(
    "Strategy Type", 
    ["SMA Crossover", "RSI Overbought/Oversold", "MACD Signal", "Bollinger Bands", "ML/DL Signal"]
)
initial_capital = st.sidebar.number_input("Initial Capital", 1000, 1000000, 10000)

# Add a button to trigger analysis
analyze_button = st.sidebar.button("Run Analysis")

# Main content
try:
    # Load data
    with st.spinner("Loading forex data..."):
        # Initialize variables to avoid "possibly unbound" errors
        start_date_param = None
        end_date_param = None
        
        if use_date_range:
            start_date_param = start_date
            end_date_param = end_date
            data = get_forex_data(selected_pair, start_date_param, end_date_param, selected_interval)
        else:
            data = get_forex_data(selected_pair, period=selected_period, interval=selected_interval)
        
        if data is None or data.empty:
            st.error(f"No data available for {selected_pair} with the selected parameters.")
            st.stop()

    # Display raw data
    with st.expander("Raw Price Data"):
        st.dataframe(data.head())
        st.write(f"Data Shape: {data.shape}")
        st.write(f"Date Range: {data.index.min()} to {data.index.max()}")
        
        # Display data source information
        if 'data_source' in data.attrs:
            st.success(f"Data retrieved from: **{data.attrs['data_source']}**")
        else:
            st.info("Data source information not available")
            
    # Fetch and display news sentiment analysis
    st.header("Market News Sentiment Analysis")
    
    # Add news lookup period selection in sidebar
    st.sidebar.header("News & Economic Calendar")
    news_days = st.sidebar.slider("News & Events Lookup Period (Days)", 1, 30, 7)
    
    # Add economic calendar sources selection
    calendar_sources = st.sidebar.multiselect(
        "Economic Calendar Sources",
        ["investing", "forexfactory"],
        default=["forexfactory"]
    )
    
    with st.spinner("Fetching and analyzing forex news..."):
        # Get news sentiment for the selected pair
        news_sentiment = get_news_sentiment_for_pair(selected_pair, days=news_days)
        
        # Display sentiment overview
        col1, col2, col3 = st.columns(3)
        
        overall_sentiment = news_sentiment.get('overall_sentiment', 0)
        sentiment_label = "Neutral"
        sentiment_color = "gray"
        
        if overall_sentiment > 0.1:
            sentiment_label = "Positive"
            sentiment_color = "green"
        elif overall_sentiment < -0.1:
            sentiment_label = "Negative"
            sentiment_color = "red"
        
        col1.metric("Overall Sentiment", f"{sentiment_label} ({overall_sentiment:.2f})")
        col2.metric("Articles Analyzed", news_sentiment.get('article_count', 0))
        col3.metric("Lookback Period", f"{news_days} days")
        
        # Display sentiment explanation
        st.markdown(f"""
        <div style="padding:10px;border-radius:5px;background-color:{sentiment_color if sentiment_color != 'gray' else '#f0f0f0'};color:{'white' if sentiment_color != 'gray' else 'black'};margin:10px 0px;">
            <h4>News Sentiment: {sentiment_label} ({overall_sentiment:.2f})</h4>
            <p>The overall market news sentiment for {selected_pair} is <strong>{sentiment_label.lower()}</strong> based on news articles from the past {news_days} days.</p>
            <p>Score ranges from -1 (very negative) to +1 (very positive), with 0 being neutral.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display news articles if available
        articles_df = news_sentiment.get('articles_df', pd.DataFrame())
        if not articles_df.empty:
            st.subheader("Recent News Articles")
            
            # Sort by sentiment and then by date
            articles_df = articles_df.sort_values(by=['sentiment', 'published_at'], ascending=[False, False])
            
            with st.expander("View News Articles"):
                for idx, row in articles_df.iterrows():
                    sentiment_color = "gray"
                    if row['sentiment'] > 0.1:
                        sentiment_color = "green"
                    elif row['sentiment'] < -0.1:
                        sentiment_color = "red"
                    
                    st.markdown(f"""
                    <div style="padding:10px;border-radius:5px;border-left:5px solid {sentiment_color};margin:10px 0px;">
                        <h5>{row['title']}</h5>
                        <p><small>Source: {row['source']} | Published: {row['published_at']}</small></p>
                        <p><strong>Sentiment Score:</strong> <span style="color:{sentiment_color};">{row['sentiment']:.2f}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Fetch and display economic calendar data
    st.header("Economic Calendar")
    
    with st.spinner("Fetching economic calendar data..."):
        # Get economic calendar data from selected sources
        if calendar_sources:
            calendar_data = get_economic_calendar(days=news_days, sources=calendar_sources)
            
            if calendar_data:
                # Display the calendar data
                for source, df in calendar_data.items():
                    if source == 'text_content':
                        # Handle text content fallback
                        st.subheader(f"Raw Calendar Data (Fallback Mode)")
                        if not df.empty:
                            with st.expander("View raw calendar content"):
                                st.text(df.iloc[0]['raw_content'][:2000] + "...")
                    elif source == 'sample_data':
                        # Handle sample data mode
                        st.subheader(f"Economic Calendar (Generated Sample)")
                        st.info("📊 This is generated sample data based on common economic events. External data sources are currently unavailable.")
                    else:
                        # Handle structured calendar data
                        st.subheader(f"Economic Events from {source.capitalize()}")
                        
                        if not df.empty:
                            # Filter for events matching the currency pair if possible
                            filtered_df = df
                            
                            # Extract currency info from selected pair
                            if "=" in selected_pair:
                                base_currency = selected_pair[:3]
                                quote_currency = selected_pair[3:6]
                            else:
                                base_currency = selected_pair[:3]
                                quote_currency = selected_pair[3:]
                                
                            # Try to filter events for the selected currency pair
                            try:
                                if 'currency' in df.columns:
                                    filtered_df = df[df['currency'].str.contains(base_currency, case=False) | 
                                                    df['currency'].str.contains(quote_currency, case=False)]
                                elif 'country' in df.columns:
                                    filtered_df = df[df['country'].str.contains(base_currency, case=False) | 
                                                   df['country'].str.contains(quote_currency, case=False)]
                            except:
                                # If filtering fails, use original dataframe
                                filtered_df = df
                            
                            # Display as a table if structured data is available
                            if len(filtered_df) > 0:
                                # Sort by impact and time
                                try:
                                    filtered_df = filtered_df.sort_values(by=['impact', 'time'], ascending=[False, True])
                                except:
                                    pass
                                
                                # Create high/medium/low impact event tables
                                if 'impact' in filtered_df.columns:
                                    # High impact events
                                    high_impact = filtered_df[filtered_df['impact'] >= 3]
                                    if not high_impact.empty:
                                        st.warning("⚠️ High Impact Events")
                                        st.dataframe(high_impact)
                                    
                                    # Medium impact events
                                    medium_impact = filtered_df[(filtered_df['impact'] >= 2) & (filtered_df['impact'] < 3)]
                                    if not medium_impact.empty:
                                        st.info("ℹ️ Medium Impact Events")
                                        st.dataframe(medium_impact)
                                    
                                    # Low impact events
                                    low_impact = filtered_df[filtered_df['impact'] < 2]
                                    if not low_impact.empty:
                                        with st.expander("View Low Impact Events"):
                                            st.dataframe(low_impact)
                                else:
                                    st.dataframe(filtered_df)
                            else:
                                st.info(f"No economic events found for {selected_pair} in the next {news_days} days.")
                        else:
                            st.error(f"No data available from {source}.")
            else:
                st.error("Failed to retrieve economic calendar data from any source.")
        else:
            st.info("Please select at least one economic calendar source in the sidebar.")

    # Calculate technical indicators
    with st.spinner("Calculating technical indicators..."):
        data_with_indicators = add_all_indicators(
            data, 
            sma_period=sma_period,
            ema_period=ema_period,
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_period=bb_period,
            bb_std=bb_std,
            atr_period=atr_period,
            stoch_rsi_period=stoch_rsi_period,
            stoch_rsi_smooth_k=stoch_rsi_smooth_k,
            stoch_rsi_smooth_d=stoch_rsi_smooth_d
        )

    # Display technical indicators
    with st.expander("Technical Indicators Data"):
        st.dataframe(data_with_indicators.tail())
    
    # Visualization
    st.header("Price Chart with Technical Indicators")
    with st.spinner("Generating visualization..."):
        fig = plot_candlestick_with_indicators(data_with_indicators, selected_pair)
        st.plotly_chart(fig, use_container_width=True)

    # Start tracking time when analysis button is clicked
    analysis_start_time = None
    if analyze_button:
        analysis_start_time = time.time()
        
        # Model training and evaluation
        st.header("Prediction Models")
        
        # Initialize variables to avoid "possibly unbound" errors
        ml_prediction_data = None
        
        # Prepare data for training
        with st.spinner("Preparing data for modeling..."):
            X_train, X_test, y_train, y_test, features = prepare_data_for_training(
                data_with_indicators, 
                prediction_horizon=prediction_horizon,
                train_size=train_test_split/100.0
            )
            
            # Apply feature selection if selected
            if feature_selection_method != "None":
                X_train, X_test, selected_features = feature_selection(
                    X_train, y_train, X_test,
                    method=feature_selection_method.lower(),
                    n_features=n_features
                )
                st.write(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
            else:
                selected_features = features

        # Train and evaluate ML models
        if use_random_forest or use_xgboost or use_gradient_boosting:
            with st.spinner("Training and evaluating machine learning models..."):
                models_to_train = []
                if use_random_forest:
                    models_to_train.append("random_forest")
                if use_xgboost:
                    models_to_train.append("xgboost")
                if use_gradient_boosting:
                    models_to_train.append("gradient_boosting")
                
                model_results, best_model, feature_importances, ml_prediction_data = train_evaluate_ml_models(
                    X_train, y_train, X_test, y_test, models=models_to_train
                )
                
                # Display model performance
                st.subheader("Machine Learning Model Performance")
                st.dataframe(model_results)
                
                # Plot model performance
                fig_perf = plot_model_performance(model_results)
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Plot feature importance
                if feature_importances is not None and len(feature_importances) > 0:
                    st.subheader("Feature Importance")
                    fig_imp = plot_feature_importance(feature_importances, selected_features)
                    st.plotly_chart(fig_imp, use_container_width=True)

        # Train and evaluate LSTM model
        if use_lstm and LSTM_AVAILABLE:
            try:
                with st.spinner("Training and evaluating LSTM model..."):
                    # Prepare sequence data for LSTM
                    from utils.preprocessing import prepare_sequence_data
                    
                    # Create a new subheader for LSTM section
                    st.subheader("LSTM Model Performance")
                    
                    # Prepare data in sequence format
                    X_train_seq, X_test_seq, y_train_seq, y_test_seq = prepare_sequence_data(
                        data_with_indicators, 
                        seq_length=sequence_length,
                        prediction_horizon=prediction_horizon,
                        train_size=train_test_split/100.0
                    )
                    
                    st.write(f"Prepared sequence data with {sequence_length} time steps for LSTM model")
                    
                    # Train and evaluate LSTM model
                    lstm_results = train_evaluate_lstm(
                        X_train_seq, y_train_seq, 
                        X_test_seq, y_test_seq,
                        n_features=X_train_seq.shape[2],
                        epochs=lstm_epochs,
                        batch_size=32
                    )
                    
                    # Display LSTM results
                    lstm_model = lstm_results.get('model')
                    lstm_accuracy = lstm_results.get('accuracy', 0)
                    lstm_f1 = lstm_results.get('f1_score', 0)
                    
                    # Display performance metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{lstm_accuracy:.4f}")
                    col2.metric("F1 Score", f"{lstm_f1:.4f}")
                    col3.metric("Sequence Length", sequence_length)
                    
                    # Plot training history if available
                    history = lstm_results.get('history')
                    if history and hasattr(history, 'history'):
                        hist = history.history
                        epochs_range = range(1, len(hist['loss']) + 1)
                        
                        fig_lstm = go.Figure()
                        fig_lstm.add_trace(go.Scatter(
                            x=epochs_range, y=hist['loss'],
                            mode='lines',
                            name='Training Loss'
                        ))
                        fig_lstm.add_trace(go.Scatter(
                            x=epochs_range, y=hist['val_loss'],
                            mode='lines',
                            name='Validation Loss'
                        ))
                        fig_lstm.update_layout(
                            title='LSTM Training History',
                            xaxis_title='Epochs',
                            yaxis_title='Loss',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_lstm, use_container_width=True)
                    
                    # Display prediction results
                    predictions = lstm_results.get('predictions', [])
                    if len(predictions) > 0:
                        st.subheader("LSTM Predictions")
                        
                        # Create a DataFrame for predictions vs actual
                        prediction_df = pd.DataFrame({
                            'Actual': y_test_seq,
                            'Predicted': predictions
                        })
                        
                        # Plot predictions vs actual
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(
                            y=prediction_df['Actual'],
                            mode='lines',
                            name='Actual'
                        ))
                        fig_pred.add_trace(go.Scatter(
                            y=prediction_df['Predicted'],
                            mode='lines',
                            name='Predicted'
                        ))
                        fig_pred.update_layout(
                            title='LSTM Predictions vs Actual Values',
                            xaxis_title='Time',
                            yaxis_title='Value',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Make a prediction for the future
                        st.subheader("Future Prediction")
                        
                        # Get the latest sequence of data
                        latest_data = X_test_seq[-1:] if len(X_test_seq) > 0 else X_train_seq[-1:]
                        
                        # Make prediction
                        future_pred = lstm_model.predict(latest_data)
                        pred_value = future_pred[0][0]
                        
                        # Show prediction direction
                        direction = "UP" if pred_value > 0.5 else "DOWN"
                        confidence = abs(pred_value - 0.5) * 2  # Scale to 0-1 range
                        
                        # Display prediction
                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Direction", direction, 
                                   delta=f"{confidence:.2%} confidence",
                                   delta_color="normal")
                        col2.metric("Latest Price", f"{data_with_indicators['close'].iloc[-1]:.5f}")
                        
                        # Send Telegram notification if enabled
                        if use_telegram and TELEGRAM_AVAILABLE and "LSTM Predictions" in notification_types:
                            try:
                                symbol_display = selected_pair.replace("=X", "")
                                horizon_display = f"{prediction_horizon} periods ahead"
                                
                                # Prepare signal value (-1 to 1 range)
                                signal_value = (pred_value - 0.5) * 2  # Convert 0-1 to -1 to 1
                                
                                # Send prediction notification
                                send_lstm_prediction(
                                    symbol=symbol_display,
                                    prediction=signal_value,
                                    actual=data_with_indicators['close'].iloc[-1],
                                    confidence=confidence,
                                    horizon=horizon_display
                                )
                                st.success("✅ LSTM prediction sent to Telegram")
                            except Exception as e:
                                st.error(f"Failed to send Telegram notification: {str(e)}")
                    
                    # Save the last LSTM result for trading signals
                    if 'predictions' in lstm_results and len(lstm_results['predictions']) > 0:
                        st.session_state['lstm_prediction'] = {
                            'value': lstm_results['predictions'][-1],
                            'confidence': abs(lstm_results['predictions'][-1] - 0.5) * 2
                        }
            except Exception as e:
                st.error(f"Error in LSTM model training: {str(e)}")
                st.info("Falling back to traditional ML models only")
                
                # Create a placeholder figure to show the error
                fig_lstm = go.Figure()
                fig_lstm.add_annotation(
                    text=f"LSTM Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig_lstm.update_layout(
                    title='LSTM Training (Error)',
                    xaxis_title='Epochs',
                    yaxis_title='Loss',
                    template='plotly_white'
                )
                st.plotly_chart(fig_lstm, use_container_width=True)
        elif use_lstm and not LSTM_AVAILABLE:
            # Show explanation about LSTM being unavailable
            st.subheader("LSTM Model (Unavailable)")
            st.warning("⚠️ LSTM functionality is currently unavailable due to TensorFlow compatibility issues.")
            st.info("💡 The application will use traditional machine learning models instead.")
            
            # Create a placeholder figure
            fig_lstm = go.Figure()
            fig_lstm.add_annotation(
                text="LSTM requires TensorFlow, which is not available in this environment",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig_lstm.update_layout(
                title='LSTM Training (Unavailable)',
                xaxis_title='Epochs',
                yaxis_title='Loss',
                template='plotly_white'
            )
            st.plotly_chart(fig_lstm, use_container_width=True)

        # Backtesting
        st.header("Strategy Backtesting")
        with st.spinner("Running backtesting..."):
            # Initialize prediction data for backtesting
            backtesting_prediction_data = None
            
            # Only use ML predictions if models were trained
            if use_random_forest or use_xgboost or use_gradient_boosting:
                if 'ml_prediction_data' in locals() and ml_prediction_data is not None:
                    backtesting_prediction_data = ml_prediction_data
                    
            # Use LSTM predictions if available (override ML predictions)
            if use_lstm and LSTM_AVAILABLE and 'lstm_prediction' in st.session_state:
                # Create/update prediction data with LSTM results
                if backtesting_prediction_data is None:
                    backtesting_prediction_data = {}
                
                # Add LSTM prediction info
                backtesting_prediction_data['lstm'] = st.session_state['lstm_prediction']
                backtesting_prediction_data['model_type'] = 'LSTM'
            
            # Run backtest with the appropriate prediction data
            backtest_results = run_backtest(
                data_with_indicators, 
                strategy_type=strategy_type,
                initial_capital=initial_capital,
                prediction_data=backtesting_prediction_data
            )
            
            # Send trading signal to Telegram if enabled
            if use_telegram and TELEGRAM_AVAILABLE and "Trading Signals" in notification_types:
                try:
                    # Get the last trading signal
                    if 'signals' in backtest_results and len(backtest_results['signals']) > 0:
                        # Get the last signal
                        last_signal = backtest_results['signals'].iloc[-1]
                        last_price = data_with_indicators['close'].iloc[-1]
                        
                        # Determine signal type
                        signal_type = "NEUTRAL"
                        if last_signal > 0:
                            signal_type = "BUY"
                        elif last_signal < 0:
                            signal_type = "SELL"
                        
                        # Only send non-neutral signals
                        if signal_type != "NEUTRAL":
                            # Format symbol for display
                            symbol_display = selected_pair.replace("=X", "")
                            
                            # Prepare additional parameters
                            strategy_params = {
                                "Strategy Type": strategy_type,
                                "Timeframe": selected_interval
                            }
                            
                            # Add model info if ML/LSTM was used
                            if strategy_type == "ML/DL Signal" and backtesting_prediction_data is not None:
                                model_type = backtesting_prediction_data.get('model_type', "Unknown")
                                strategy_params["Model"] = model_type
                                
                                # Add confidence if available
                                if 'confidence' in backtesting_prediction_data.get('lstm', {}):
                                    confidence = backtesting_prediction_data['lstm']['confidence']
                                    strategy_params["Confidence"] = f"{confidence:.2%}"
                            
                            # Send the trading signal
                            send_trading_signal(
                                symbol=symbol_display,
                                signal_type=signal_type,
                                price=last_price,
                                strategy=strategy_type,
                                params=strategy_params
                            )
                            st.success(f"✅ {signal_type} signal for {symbol_display} sent to Telegram")
                except Exception as e:
                    st.error(f"Failed to send trading signal to Telegram: {str(e)}")
            
            # Display backtest results
            st.subheader("Backtest Performance")
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Total Return (%)', 
                    'Annual Return (%)', 
                    'Sharpe Ratio', 
                    'Max Drawdown (%)',
                    'Win Rate (%)',
                    'Profit Factor',
                    'Total Trades'
                ],
                'Value': [
                    f"{backtest_results['total_return']*100:.2f}",
                    f"{backtest_results['annual_return']*100:.2f}",
                    f"{backtest_results['sharpe_ratio']:.2f}",
                    f"{backtest_results['max_drawdown']*100:.2f}",
                    f"{backtest_results['win_rate']*100:.2f}",
                    f"{backtest_results['profit_factor']:.2f}",
                    f"{backtest_results['total_trades']}"
                ]
            })
            st.dataframe(metrics_df)
            
            # Plot equity curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=backtest_results['equity_curve'].index,
                y=backtest_results['equity_curve'],
                mode='lines',
                name='Equity Curve'
            ))
            fig_equity.update_layout(
                title='Strategy Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value',
                template='plotly_white'
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Plot trades on price chart
            if 'trades' in backtest_results and len(backtest_results['trades']) > 0:
                fig_trades = plot_candlestick_with_indicators(
                    data_with_indicators, 
                    selected_pair,
                    trades=backtest_results['trades']
                )
                st.subheader("Strategy Trades")
                st.plotly_chart(fig_trades, use_container_width=True)
    
    # Generate prediction summary
    if analyze_button:
        st.header("Ringkasan Prediksi")
        
        # Calculate processing time if available
        processing_time = None
        if analysis_start_time is not None:
            processing_time = time.time() - analysis_start_time
        
        # Determine latest prediction and confidence
        prediction_direction = "NEUTRAL"
        confidence = 0.5
        model_used = "ML"
        
        # Implement new technical indicators consensus strategy
        # This will check multiple indicators to get a stronger overall signal
        try:
            # Get latest data point with indicators
            latest_data = data_with_indicators.iloc[-1]
            
            # Initialize signal counters
            bullish_signals = 0
            bearish_signals = 0
            
            # Check RSI (below 30 = oversold/bullish, above 70 = overbought/bearish)
            rsi_col = [col for col in latest_data.index if 'rsi_' in col and 'stoch' not in col]
            if rsi_col:
                rsi_value = latest_data[rsi_col[0]]
                if rsi_value < 30:
                    bullish_signals += 1
                elif rsi_value > 70:
                    bearish_signals += 1
            
            # Check MACD (MACD above signal = bullish, below = bearish)
            if 'macd_line' in latest_data and 'macd_signal' in latest_data:
                if latest_data['macd_line'] > latest_data['macd_signal']:
                    bullish_signals += 1
                elif latest_data['macd_line'] < latest_data['macd_signal']:
                    bearish_signals += 1
                
                # Add another signal if MACD is strongly trending
                if latest_data['macd_line'] > latest_data['macd_signal'] * 1.1:  # 10% above
                    bullish_signals += 0.5
                elif latest_data['macd_line'] < latest_data['macd_signal'] * 0.9:  # 10% below
                    bearish_signals += 0.5
            
            # Check Bollinger Bands
            bb_upper_col = [col for col in latest_data.index if 'bb_upper_' in col]
            bb_lower_col = [col for col in latest_data.index if 'bb_lower_' in col]
            
            if 'close' in latest_data and bb_upper_col and bb_lower_col:
                if latest_data['close'] < latest_data[bb_lower_col[0]]:
                    bullish_signals += 1  # Price below lower band = potential bounce (bullish)
                elif latest_data['close'] > latest_data[bb_upper_col[0]]:
                    bearish_signals += 1  # Price above upper band = potential reversal (bearish)
            
            # Check Moving Average Crossovers
            if 'sma_20' in latest_data and 'sma_50' in latest_data:
                if latest_data['sma_20'] > latest_data['sma_50']:
                    bullish_signals += 1  # Short term MA above long term = bullish
                elif latest_data['sma_20'] < latest_data['sma_50']:
                    bearish_signals += 1  # Short term MA below long term = bearish
            
            # Check ADX - strong trend confirmation
            if 'adx' in latest_data and 'plus_di' in latest_data and 'minus_di' in latest_data:
                # ADX above 25 indicates a strong trend
                if latest_data['adx'] > 25:
                    # If +DI > -DI, bullish trend is strong
                    if latest_data['plus_di'] > latest_data['minus_di']:
                        bullish_signals += 1
                    # If -DI > +DI, bearish trend is strong
                    elif latest_data['minus_di'] > latest_data['plus_di']:
                        bearish_signals += 1
            
            # Check Stochastic oscillator
            if 'stoch_k' in latest_data and 'stoch_d' in latest_data:
                # Bullish if K crosses above D in oversold territory
                if latest_data['stoch_k'] > latest_data['stoch_d'] and latest_data['stoch_k'] < 20:
                    bullish_signals += 1
                # Bearish if K crosses below D in overbought territory
                elif latest_data['stoch_k'] < latest_data['stoch_d'] and latest_data['stoch_k'] > 80:
                    bearish_signals += 1
            
            # Check short-term momentum with 9-day MA
            if 'sma_9' in latest_data and 'close' in latest_data:
                if latest_data['close'] > latest_data['sma_9']:
                    bullish_signals += 0.5  # Price above short MA = bullish momentum
                elif latest_data['close'] < latest_data['sma_9']:
                    bearish_signals += 0.5  # Price below short MA = bearish momentum
            
            # Make prediction based on signal count
            total_signals = bullish_signals + bearish_signals
            if total_signals > 0:
                if bullish_signals > bearish_signals:
                    prediction_direction = "UP"
                    # Calculate confidence based on signal dominance
                    confidence = min(0.5 + (bullish_signals / total_signals) * 0.4, 0.95)
                elif bearish_signals > bullish_signals:
                    prediction_direction = "DOWN"
                    # Calculate confidence based on signal dominance
                    confidence = min(0.5 + (bearish_signals / total_signals) * 0.4, 0.95)
                model_used = "Technical Consensus"
        except Exception as e:
            st.error(f"Error in technical indicator consensus check: {str(e)}")
            # Will fall back to ML prediction
        
        # Check if we have ML predictions - use only if no consensus reached
        if prediction_direction == "NEUTRAL" and 'ml_prediction_data' in locals() and ml_prediction_data is not None and 'prediction' in ml_prediction_data:
            pred_value = ml_prediction_data['prediction']
            # Very sensitive thresholds (0.51/0.49 instead of 0.52/0.48)
            if pred_value > 0.51:
                prediction_direction = "UP"
                confidence = pred_value 
                # Scale confidence to make it more definitive for display
                confidence = min(0.5 + (pred_value - 0.5) * 1.5, 0.95)
            elif pred_value < 0.49:
                prediction_direction = "DOWN"
                confidence = 1 - pred_value
                # Scale confidence to make it more definitive for display
                confidence = min(0.5 + (0.5 - pred_value) * 1.5, 0.95)
            model_used = ml_prediction_data.get('model_name', 'ML Model')
        
        # Check if we have LSTM predictions (override ML if available)
        if 'lstm_prediction' in st.session_state:
            lstm_pred = st.session_state['lstm_prediction']
            pred_value = lstm_pred.get('value', 0.5)
            
            if pred_value > 0.55:
                prediction_direction = "UP"
                confidence = pred_value
            elif pred_value < 0.45:
                prediction_direction = "DOWN"
                confidence = 1 - pred_value
            model_used = "LSTM"
        
        # Get last trading signal from backtest (override ML/LSTM if clear signal)
        if 'backtest_results' in locals() and backtest_results is not None:
            if 'signals' in backtest_results and len(backtest_results['signals']) > 0:
                last_signal = backtest_results['signals'].iloc[-1]
                # Lower threshold from 0.7 to 0.5 to be more sensitive to signals
                if abs(last_signal) > 0.5:  # More sensitive threshold
                    if last_signal > 0:
                        prediction_direction = "UP"
                        # Scale confidence to make it more definitive
                        confidence = min(0.55 + abs(last_signal) * 0.45, 0.95)  # Cap at 0.95
                    elif last_signal < 0:
                        prediction_direction = "DOWN"
                        # Scale confidence to make it more definitive
                        confidence = min(0.55 + abs(last_signal) * 0.45, 0.95)  # Cap at 0.95
                    model_used = strategy_type
        
        # Get ATR value if available for target calculation
        atr_value = None
        if 'atr' in data_with_indicators.columns:
            atr_value = data_with_indicators['atr'].iloc[-1]
        
        # Get period in days
        period_days = None
        if not use_date_range and selected_period != "max":
            # Extract number from period string like "1mo", "5d", etc.
            import re
            period_match = re.match(r'(\d+)([dmy])', selected_period)
            if period_match:
                num = int(period_match.group(1))
                unit = period_match.group(2)
                if unit == 'd':
                    period_days = num
                elif unit == 'm':
                    period_days = num * 30
                elif unit == 'y':
                    period_days = num * 365
        elif use_date_range and start_date and end_date:
            # Calculate days between dates
            period_days = (end_date - start_date).days
        
        # Create prediction summary
        if prediction_direction != "NEUTRAL":
            # HTML formatted summary
            summary_html = get_prediction_summary_html(
                symbol=selected_pair,
                direction=prediction_direction,
                confidence=confidence,
                current_price=data_with_indicators['close'].iloc[-1],
                model_type=model_used,
                timeframe=selected_interval,
                atr_value=atr_value,
                processing_time=processing_time,
                period_days=period_days
            )
            
            # Display summary
            st.markdown(summary_html, unsafe_allow_html=True)
            
            # Plain text format for Telegram
            summary_text = format_prediction_summary(
                symbol=selected_pair,
                direction=prediction_direction,
                confidence=confidence,
                current_price=data_with_indicators['close'].iloc[-1],
                model_type=model_used,
                timeframe=selected_interval,
                atr_value=atr_value,
                processing_time=processing_time,
                period_days=period_days
            )
            
            # Send prediction summary to Telegram if enabled
            if 'use_telegram' in locals() and use_telegram and TELEGRAM_AVAILABLE:
                try:
                    # Get notification types if available
                    notification_types_list = []
                    if 'notification_types' in locals() and notification_types is not None:
                        notification_types_list = notification_types
                    
                    # Only send if in notification types or if ML Predictions selected
                    if not notification_types_list or "ML Predictions" in notification_types_list or "LSTM Predictions" in notification_types_list:
                        # Use the imported send_message function directly
                        if 'send_message' in globals():
                            send_message(summary_text)
                            st.success("✅ Prediction summary sent to Telegram")
                except Exception as e:
                    st.error(f"Failed to send prediction summary to Telegram: {str(e)}")
        else:
            st.info("No clear prediction direction identified. Consider adjusting model parameters or using a different strategy.")
            
            # Create a neutral summary
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background-color:#f8f9fa; border-left:5px solid gray; margin-bottom:20px;">
                <h3 style="color:gray; margin:0 0 15px 0;">Neutral Market Outlook</h3>
                <p>The current analysis does not show a clear directional bias for {selected_pair.replace('=X', '')}.</p>
                <p>Consider adjusting your analysis parameters or trying a different strategy.</p>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.info(
        "**Disclaimer**: This tool is for educational and research purposes only. "
        "It is not intended to provide investment advice. "
        "Past performance is not indicative of future results. "
        "Trading forex involves significant risk and may not be suitable for all investors."
    )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    logger.exception("Application error")

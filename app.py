import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os

# Import custom modules
from utils.data_loader import get_forex_data, get_available_pairs
from utils.technical_indicators import add_all_indicators
from utils.preprocessing import prepare_data_for_training, feature_selection
from utils.visualization import plot_candlestick_with_indicators, plot_model_performance, plot_feature_importance
from utils.news_api import get_news_sentiment_for_pair
from utils.economic_calendar import get_economic_calendar
from models.machine_learning import train_evaluate_ml_models
# Temporarily disable deep learning module due to compatibility issues
# from models.deep_learning import train_evaluate_lstm
from backtest.strategy import run_backtest

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
# Disable LSTM with explanation
use_lstm = st.sidebar.checkbox("LSTM (Deep Learning)", value=False, disabled=True, 
                             help="Temporarily disabled due to compatibility issues with TensorFlow and NumPy")

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

    # Only run analysis when the button is clicked
    if analyze_button:
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
        if use_lstm:
            with st.spinner("Training and evaluating LSTM model..."):
                st.info("Deep learning functionality is temporarily disabled due to compatibility issues with TensorFlow and NumPy.")
                
                # Placeholder for LSTM results
                st.subheader("LSTM Model Performance")
                st.write("LSTM model is not available in this version.")
                
                # Create a placeholder figure
                fig_lstm = go.Figure()
                fig_lstm.add_annotation(
                    text="LSTM training visualization is not available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig_lstm.update_layout(
                    title='LSTM Training (Not Available)',
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
            
            backtest_results = run_backtest(
                data_with_indicators, 
                strategy_type=strategy_type,
                initial_capital=initial_capital,
                prediction_data=backtesting_prediction_data
            )
            
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

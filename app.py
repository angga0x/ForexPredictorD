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
use_lstm = st.sidebar.checkbox("LSTM (Deep Learning)", value=False)

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

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def plot_candlestick_with_indicators(df, symbol, n_rows=30, trades=None):
    """
    Create an interactive candlestick chart with technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with price and indicator data
        symbol (str): Symbol to display in the chart title
        n_rows (int, optional): Number of rows for the subplot grid. Default is 30.
        trades (list, optional): List of trade dictionaries to plot. Default is None.
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure
    """
    try:
        # Use the most recent data for better visualization
        if len(df) > 300:
            plot_df = df.tail(300).copy()
        else:
            plot_df = df.copy()
        
        # Create subplots with 3 rows
        fig = make_subplots(
            rows=3, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("Price & Indicators", "Volume", "Oscillators")
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df['open'],
                high=plot_df['high'],
                low=plot_df['low'],
                close=plot_df['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add SMA
        if 'sma_20' in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df['sma_20'],
                    name="SMA(20)",
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Add EMA
        if 'ema_20' in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df['ema_20'],
                    name="EMA(20)",
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands
        if 'bb_upper_20' in plot_df.columns and 'bb_lower_20' in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df['bb_upper_20'],
                    name="Upper BB",
                    line=dict(color='rgba(173, 204, 255, 0.7)')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df['bb_lower_20'],
                    name="Lower BB",
                    line=dict(color='rgba(173, 204, 255, 0.7)'),
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # Add Volume
        fig.add_trace(
            go.Bar(
                x=plot_df.index,
                y=plot_df['volume'],
                name="Volume",
                marker=dict(color='rgba(100, 100, 100, 0.5)')
            ),
            row=2, col=1
        )
        
        # Add RSI
        if 'rsi_14' in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df['rsi_14'],
                    name="RSI(14)",
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            
            # Add RSI overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=[plot_df.index[0], plot_df.index[-1]],
                    y=[70, 70],
                    name="Overbought",
                    line=dict(color='red', dash='dash')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[plot_df.index[0], plot_df.index[-1]],
                    y=[30, 30],
                    name="Oversold",
                    line=dict(color='green', dash='dash')
                ),
                row=3, col=1
            )
            
        # Add MACD
        if 'macd_line' in plot_df.columns and 'macd_signal' in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df['macd_line'],
                    name="MACD",
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df['macd_signal'],
                    name="Signal",
                    line=dict(color='red')
                ),
                row=3, col=1
            )
            
            # Add MACD histogram
            if 'macd_histogram' in plot_df.columns:
                colors = ['green' if val >= 0 else 'red' for val in plot_df['macd_histogram']]
                fig.add_trace(
                    go.Bar(
                        x=plot_df.index,
                        y=plot_df['macd_histogram'],
                        name="Histogram",
                        marker=dict(color=colors, opacity=0.5)
                    ),
                    row=3, col=1
                )
        
        # Add trades if provided
        if trades:
            for trade in trades:
                # Add buy markers
                if trade['type'] == 'buy':
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['date']],
                            y=[trade['price']],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color='green',
                                line=dict(width=2, color='darkgreen')
                            ),
                            name='Buy',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                # Add sell markers
                elif trade['type'] == 'sell':
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['date']],
                            y=[trade['price']],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='red',
                                line=dict(width=2, color='darkred')
                            ),
                            name='Sell',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        
        # Update layout for better visualization
        fig.update_layout(
            title=f"{symbol} Price Chart with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=800,
            margin=dict(l=10, r=10, b=10, t=50)
        )
        
        # Update Y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Oscillators", row=3, col=1)
        
        # Update X-axis
        fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # Hide weekends
                dict(bounds=["sat", "mon"])
            ]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_model_performance(results_df):
    """
    Create a bar chart of model performance metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame with model performance metrics
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure
    """
    try:
        # Create a figure with subplots
        fig = make_subplots(
            rows=1, 
            cols=4, 
            subplot_titles=("Accuracy", "Precision", "Recall", "F1-Score")
        )
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['blue', 'green', 'orange', 'red']
        
        # Add a bar chart for each metric
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    x=results_df['model'],
                    y=results_df[metric],
                    marker_color=colors[i],
                    name=metric.capitalize()
                ),
                row=1, col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title="Model Performance Comparison",
            template="plotly_white",
            showlegend=False,
            height=400
        )
        
        # Set y-axis range from 0 to 1 for all metrics
        for i in range(1, 5):
            fig.update_yaxes(range=[0, 1], row=1, col=i)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating model performance chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_feature_importance(feature_importances, feature_names):
    """
    Create a horizontal bar chart of feature importances.
    
    Args:
        feature_importances (dict or list or np.array): Feature importance values
            If dict, keys are model names and values are arrays of feature importances
            If list/array, it contains feature importance values directly
        feature_names (list): List of feature names
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure
    """
    try:
        # Handle the case where feature_importances is a dictionary
        if isinstance(feature_importances, dict):
            # If it's a dictionary, use the first model's feature importances
            if not feature_importances:
                # Empty dictionary
                logger.warning("Empty feature_importances dictionary")
                raise ValueError("No feature importance data available")
                
            # Get the first model's feature importances
            model_name = list(feature_importances.keys())[0]
            importance_array = feature_importances[model_name]
            
            # Convert to numpy array if not already
            if not isinstance(importance_array, np.ndarray):
                importance_array = np.array(importance_array)
        else:
            # If feature_importances is already an array or list
            if not isinstance(feature_importances, np.ndarray):
                importance_array = np.array(feature_importances)
            else:
                importance_array = feature_importances
        
        # Convert feature names to numpy array if not already
        if not isinstance(feature_names, np.ndarray):
            feature_names_array = np.array(feature_names)
        else:
            feature_names_array = feature_names
            
        # Check if we have valid data
        if len(importance_array) == 0 or len(feature_names_array) == 0:
            raise ValueError("Empty feature importances or feature names")
            
        # Check if lengths match
        if len(importance_array) != len(feature_names_array):
            logger.warning(f"Feature importance length ({len(importance_array)}) does not match feature names length ({len(feature_names_array)})")
            # Use shorter length
            min_len = min(len(importance_array), len(feature_names_array))
            importance_array = importance_array[:min_len]
            feature_names_array = feature_names_array[:min_len]
        
        # Sort by importance
        sorted_idx = importance_array.argsort()
        
        # Limit to top 20 features for readability
        if len(sorted_idx) > 20:
            sorted_idx = sorted_idx[-20:]
        
        # Create bar chart
        fig = go.Figure(
            go.Bar(
                y=feature_names_array[sorted_idx],
                x=importance_array[sorted_idx],
                orientation='h',
                marker=dict(
                    color=importance_array[sorted_idx],
                    colorscale='Viridis'
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white",
            height=600
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_correlation_matrix(df, n_features=20):
    """
    Create a heatmap of feature correlations.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        n_features (int, optional): Number of features to include. Default is 20.
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure
    """
    try:
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Select top features by correlation with 'close'
        if 'close' in corr_matrix.columns:
            top_features = corr_matrix['close'].abs().sort_values(ascending=False).head(n_features).index
            corr_subset = corr_matrix.loc[top_features, top_features]
        else:
            # If 'close' not in columns, take top n features
            top_features = corr_matrix.columns[:n_features]
            corr_subset = corr_matrix.loc[top_features, top_features]
        
        # Create heatmap
        fig = go.Figure(
            go.Heatmap(
                z=corr_subset.values,
                x=corr_subset.columns,
                y=corr_subset.index,
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation")
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Feature Correlation Matrix",
            template="plotly_white",
            height=700,
            width=700
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_learning_curve(train_sizes, train_scores, test_scores, title="Learning Curve"):
    """
    Plot learning curve showing model performance as training size increases.
    
    Args:
        train_sizes (np.array): Array of training set sizes
        train_scores (np.array): Array of training scores for each size
        test_scores (np.array): Array of test scores for each size
        title (str, optional): Chart title. Default is "Learning Curve".
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure
    """
    try:
        # Calculate mean and std for train and test scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Create figure
        fig = go.Figure()
        
        # Add training scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines+markers',
                name="Training Score",
                line=dict(color='blue'),
                marker=dict(color='blue', size=8)
            )
        )
        
        # Add training score confidence interval
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                showlegend=False
            )
        )
        
        # Add test scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=test_mean,
                mode='lines+markers',
                name="Validation Score",
                line=dict(color='red'),
                marker=dict(color='red', size=8)
            )
        )
        
        # Add test score confidence interval
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                showlegend=False
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Training Examples",
            yaxis_title="Score",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating learning curve: {str(e)}")
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

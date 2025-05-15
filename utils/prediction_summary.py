"""
Module for generating prediction summaries in human-readable formats.
This module provides functionality to create clear and concise summaries of forex predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_target_levels(current_price, direction, confidence, atr_value=None):
    """
    Calculate target profit and stop loss levels based on prediction direction and ATR.
    
    Args:
        current_price (float): Current price
        direction (str): "UP" or "DOWN"
        confidence (float): Confidence level (0-1)
        atr_value (float, optional): Average True Range for volatility-based targets
        
    Returns:
        dict: Dictionary with target and stop loss levels
    """
    # Default pip movement (0.05% of price if ATR not available)
    default_movement = current_price * 0.0005
    
    # Use ATR if available, otherwise use default
    pip_movement = atr_value if atr_value else default_movement
    
    # Scale movement by confidence (higher confidence = wider targets)
    scaled_movement = pip_movement * (0.5 + confidence)
    
    if direction == "UP":
        target = current_price + scaled_movement
        stop_loss = current_price - (scaled_movement / 1.5)  # Risk:reward ratio of 1.5
    else:  # direction == "DOWN"
        target = current_price - scaled_movement
        stop_loss = current_price + (scaled_movement / 1.5)  # Risk:reward ratio of 1.5
        
    # Calculate risk-reward ratio
    if direction == "UP":
        risk = current_price - stop_loss
        reward = target - current_price
    else:
        risk = stop_loss - current_price
        reward = current_price - target
        
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    return {
        "target": target,
        "stop_loss": stop_loss,
        "risk_reward_ratio": risk_reward_ratio
    }

def format_prediction_summary(
    symbol, 
    direction, 
    confidence, 
    current_price, 
    model_type="ML/LSTM", 
    timeframe="daily",
    atr_value=None,
    processing_time=None,
    period_days=None
):
    """
    Create a formatted prediction summary string.
    
    Args:
        symbol (str): Forex pair symbol
        direction (str): "UP" or "DOWN"
        confidence (float): Confidence level (0-1)
        current_price (float): Current price
        model_type (str, optional): Type of model used for prediction
        timeframe (str, optional): Timeframe of the prediction
        atr_value (float, optional): ATR value for target calculation
        processing_time (float, optional): Time taken to process the prediction in seconds
        period_days (int, optional): Number of days in the analysis period
        
    Returns:
        str: Formatted prediction summary
    """
    # Clean up symbol
    symbol_display = symbol.replace("=X", "")
    if len(symbol_display) == 6:
        # Format as XXX/YYY
        symbol_display = f"{symbol_display[:3]}/{symbol_display[3:]}"
    
    # Calculate targets
    targets = calculate_target_levels(current_price, direction, confidence, atr_value)
    target_price = targets["target"]
    stop_loss = targets["stop_loss"]
    risk_reward = targets["risk_reward_ratio"]
    
    # Determine emoji
    direction_emoji = "⬆️" if direction == "UP" else "⬇️"
    
    # Format confidence
    confidence_pct = confidence * 100
    
    # Create summary
    summary = f"""
📈 Pasangan Mata Uang: {symbol_display}
"""

    if period_days:
        summary += f"⏱️ Periode Analisis: {period_days} hari\n"
        
    summary += f"""🔮 Prediksi: {direction} {direction_emoji} dengan keyakinan {confidence_pct:.2f}%
💰 Harga Saat Ini: {current_price:.5f}
🎯 Target Profit: {target_price:.5f}
🛡️ Stop Loss: {stop_loss:.5f}
⚖️ Risk/Reward Ratio: {risk_reward:.2f}
"""

    if processing_time:
        summary += f"\n⏱️ Total Waktu Proses: {processing_time:.2f} detik"
        
    return summary

def get_prediction_summary_html(
    symbol, 
    direction, 
    confidence, 
    current_price, 
    model_type="ML/LSTM", 
    timeframe="daily",
    atr_value=None,
    processing_time=None,
    period_days=None
):
    """
    Create an HTML formatted prediction summary for Streamlit.
    
    Args:
        Same as format_prediction_summary
        
    Returns:
        str: HTML formatted prediction summary
    """
    # Clean up symbol
    symbol_display = symbol.replace("=X", "")
    if len(symbol_display) == 6:
        # Format as XXX/YYY
        symbol_display = f"{symbol_display[:3]}/{symbol_display[3:]}"
    
    # Calculate targets
    targets = calculate_target_levels(current_price, direction, confidence, atr_value)
    target_price = targets["target"]
    stop_loss = targets["stop_loss"]
    risk_reward = targets["risk_reward_ratio"]
    
    # Determine color and emoji based on direction
    direction_color = "#4CAF50" if direction == "UP" else "#F44336"  # Green or Red
    direction_emoji = "⬆️" if direction == "UP" else "⬇️"
    
    # Format confidence
    confidence_pct = confidence * 100
    
    # Create HTML summary
    html = f"""
    <div style="padding:20px; border-radius:10px; background-color:#f8f9fa; border-left:5px solid {direction_color}; margin-bottom:20px;">
        <h3 style="color:{direction_color}; margin:0 0 15px 0;">Ringkasan Prediksi Forex</h3>
        <div style="display:grid; grid-template-columns:auto 1fr; gap:10px; align-items:start;">
            <div style="font-weight:bold;">📈 Pasangan:</div>
            <div>{symbol_display}</div>
    """
    
    if period_days:
        html += f"""
            <div style="font-weight:bold;">⏱️ Periode:</div>
            <div>{period_days} hari</div>
        """
    
    html += f"""
            <div style="font-weight:bold;">🔮 Prediksi:</div>
            <div style="color:{direction_color}; font-weight:bold;">{direction} {direction_emoji} ({confidence_pct:.2f}%)</div>
            
            <div style="font-weight:bold;">💰 Harga Saat Ini:</div>
            <div>{current_price:.5f}</div>
            
            <div style="font-weight:bold;">🎯 Target Profit:</div>
            <div>{target_price:.5f}</div>
            
            <div style="font-weight:bold;">🛡️ Stop Loss:</div>
            <div>{stop_loss:.5f}</div>
            
            <div style="font-weight:bold;">⚖️ Risk/Reward:</div>
            <div>{risk_reward:.2f}</div>
    """
    
    if processing_time:
        html += f"""
            <div style="font-weight:bold;">⏱️ Waktu Proses:</div>
            <div>{processing_time:.2f} detik</div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html
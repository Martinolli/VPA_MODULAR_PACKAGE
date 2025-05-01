"""
VPA Utils Module

This module provides common utilities for the VPA algorithm.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from .vpa_processor import DataProcessor
import os
import numpy as np
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime

def ensure_datetime_index(df):
    """
    Ensure DataFrame has a datetime index
    
    Parameters:
    - df: DataFrame to check
    
    Returns:
    - DataFrame with datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        # Check if there's a datetime column
        datetime_cols = [col for col in df.columns if col.lower() in ['date', 'datetime', 'time']]
        
        if datetime_cols:
            # Use the first datetime column as index
            df = df.set_index(datetime_cols[0])
            df.index = pd.to_datetime(df.index)
        else:
            # Try to convert the existing index
            try:
                df.index = pd.to_datetime(df.index)
            except:
                raise ValueError("Could not convert index to datetime and no datetime column found")
    
    return df

def calculate_relative_volume(volume, lookback_period=50):
    """
    Calculate relative volume compared to average
    
    Parameters:
    - volume: Series with volume data
    - lookback_period: Number of periods to look back for average
    
    Returns:
    - Series with relative volume
    """
    avg_volume = volume.rolling(window=lookback_period).mean()
    relative_volume = volume / avg_volume
    
    return relative_volume

def identify_swing_points(price_data, min_swing=3):
    """
    Identify swing high and low points
    
    Parameters:
    - price_data: DataFrame with OHLC data
    - min_swing: Minimum number of candles for a swing
    
    Returns:
    - Dictionary with swing highs and lows
    """
    highs = []
    lows = []
    
    # Find swing highs
    for i in range(min_swing, len(price_data) - min_swing):
        # Check if this is a local high
        is_high = True
        for j in range(1, min_swing + 1):
            if price_data["high"].iloc[i] <= price_data["high"].iloc[i-j] or \
               price_data["high"].iloc[i] <= price_data["high"].iloc[i+j]:
                is_high = False
                break
        
        if is_high:
            highs.append({
                "index": price_data.index[i],
                "price": price_data["high"].iloc[i]
            })
    
    # Find swing lows
    for i in range(min_swing, len(price_data) - min_swing):
        # Check if this is a local low
        is_low = True
        for j in range(1, min_swing + 1):
            if price_data["low"].iloc[i] >= price_data["low"].iloc[i-j] or \
               price_data["low"].iloc[i] >= price_data["low"].iloc[i+j]:
                is_low = False
                break
        
        if is_low:
            lows.append({
                "index": price_data.index[i],
                "price": price_data["low"].iloc[i]
            })
    
    return {
        "highs": highs,
        "lows": lows
    }

def plot_candlestick(ax, price_data, volume_data=None, title=None):
    """
    Plot candlestick chart with improved readability
    
    Parameters:
    - ax: Matplotlib axis
    - price_data: DataFrame with OHLC data
    - volume_data: Optional Series with volume data
    - title: Optional chart title
    
    Returns:
    - Updated axis
    """
    # Ensure datetime index
    price_data = ensure_datetime_index(price_data)
    
    # Plot candlesticks
    width2 = 0.6
    width = 0.2  # Increased wick width for better visibility
    
    up = price_data[price_data.close >= price_data.open]
    down = price_data[price_data.close < price_data.open]
    
    # Plot up candles
    ax.bar(up.index, up.close - up.open, width, bottom=up.open, color='green', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.vlines(up.index, up.low, up.high, color='green', linewidth=1.5, alpha=0.8)
    
    # Plot down candles
    ax.bar(down.index, down.open - down.close, width, bottom=down.close, color='red', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.vlines(down.index, down.low, down.high, color='red', linewidth=1.5, alpha=0.8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    
    # Add gridlines for better readability
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Add volume if provided
    if volume_data is not None:
        # Create twin axis for volume
        ax2 = ax.twinx()
        
        # Plot volume bars
        ax2.bar(volume_data.index, volume_data, width, color='blue', alpha=0.3, label='Volume')
        
        # Set volume axis label
        ax2.set_ylabel('Volume', fontsize=10)
        
        # Make volume axis less prominent
        ax2.tick_params(axis='y', colors='blue', labelcolor='blue')
        ax2.grid(False)  # Disable gridlines for volume axis
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Set labels
    ax.set_ylabel('Price', fontsize=10)
    
    return ax

def plot_vpa_signals(ax, price_data, processed_data, signals=None, support_resistance=None):
    """
    Plot VPA signals on a candlestick chart
    
    Parameters:
    - ax: Matplotlib axis with candlestick chart
    - price_data: DataFrame with OHLC data
    - processed_data: Dictionary with processed data
    - signals: Optional dictionary with signal information
    - support_resistance: Optional dictionary with support and resistance levels
    
    Returns:
    - Updated axis
    """
    # Plot volume classifications
    if "volume_class" in processed_data:
        for i, (idx, vol_class) in enumerate(processed_data["volume_class"].items()):
            if vol_class == "VERY_HIGH":
                ax.text(idx, price_data.loc[idx, "low"] * 0.99, "VH", 
                        color='blue', fontsize=8, ha='center')
            elif vol_class == "HIGH":
                ax.text(idx, price_data.loc[idx, "low"] * 0.99, "H", 
                        color='blue', fontsize=8, ha='center')
            elif vol_class == "VERY_LOW":
                ax.text(idx, price_data.loc[idx, "low"] * 0.99, "VL", 
                        color='gray', fontsize=8, ha='center')
    
    # Plot candle classifications
    if "candle_class" in processed_data:
        for i, (idx, candle_class) in enumerate(processed_data["candle_class"].items()):
            if "WIDE" in candle_class:
                ax.text(idx, price_data.loc[idx, "high"] * 1.01, "W", 
                        color='black', fontsize=8, ha='center')
            elif "NARROW" in candle_class:
                ax.text(idx, price_data.loc[idx, "high"] * 1.01, "N", 
                        color='black', fontsize=8, ha='center')
    
    # Plot support and resistance levels
    if support_resistance:
        # Plot support levels
        for level in support_resistance.get("support", []):
            ax.axhline(y=level["price"], color='green', linestyle='--', alpha=0.5)
            ax.text(price_data.index[-1], level["price"], f"S ({level['strength']:.1f})", 
                    color='green', fontsize=10)
        
        # Plot resistance levels
        for level in support_resistance.get("resistance", []):
            ax.axhline(y=level["price"], color='red', linestyle='--', alpha=0.5)
            ax.text(price_data.index[-1], level["price"], f"R ({level['strength']:.1f})", 
                    color='red', fontsize=10)
    
    # Plot signals
    if signals:
        signal_type = signals.get("type")
        signal_strength = signals.get("strength")
        
        if signal_type == "BUY":
            color = 'green'
            marker = '^'
            y_pos = price_data["low"].iloc[-1] * 0.98
        elif signal_type == "SELL":
            color = 'red'
            marker = 'v'
            y_pos = price_data["high"].iloc[-1] * 1.02
        else:
            return ax
        
        # Plot signal marker
        ax.plot(price_data.index[-1], y_pos, marker=marker, markersize=10, color=color)
        
        # Add signal text
        ax.text(price_data.index[-1], y_pos, f" {signal_type} ({signal_strength})", 
                color=color, fontsize=10, va='center')
    
    return ax

def plot_pattern_detection(price_data, patterns, output_file=None):
    """
    Plot pattern detection results
    
    Parameters:
    - price_data: DataFrame with OHLC data
    - patterns: Dictionary with pattern detection results
    - output_file: Optional file path to save the plot
    
    Returns:
    - Figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot candlestick chart
    plot_candlestick(ax, price_data)

    # Highlight significant patterns
    if patterns.get("accumulation", {}).get("detected", False):
        ax.axhspan(price_data["low"].min(), price_data["low"].min() + (price_data["high"].max() - price_data["low"].min()) * 0.2, 
                   color='green', alpha=0.2, label="Accumulation Zone")
    if patterns.get("distribution", {}).get("detected", False):
        ax.axhspan(price_data["high"].max() - (price_data["high"].max() - price_data["low"].min()) * 0.2, price_data["high"].max(), 
                   color='red', alpha=0.2, label="Distribution Zone")
    
    # Add legend for better clarity
    ax.legend(loc='upper right', fontsize=8)
    
    # Plot accumulation zones
    if patterns.get("accumulation", {}).get("detected", False):
        # Highlight accumulation zones
        ax.axhspan(price_data["low"].min(), price_data["low"].min() + (price_data["high"].max() - price_data["low"].min()) * 0.2, 
                   color='green', alpha=0.2, label="Accumulation Zone")
    
    # Plot distribution zones
    if patterns.get("distribution", {}).get("detected", False):
        # Highlight distribution zones
        ax.axhspan(price_data["high"].max() - (price_data["high"].max() - price_data["low"].min()) * 0.2, price_data["high"].max(), 
                   color='red', alpha=0.2, label="Distribution Zone")
    
    # Plot buying climax
    if patterns.get("buying_climax", {}).get("detected", False):
        # Mark buying climax
        ax.plot(price_data.index[-1], price_data["high"].iloc[-1], 'ro', markersize=10, label="Buying Climax")
    
    # Plot selling climax
    if patterns.get("selling_climax", {}).get("detected", False):
        # Mark selling climax
        ax.plot(price_data.index[-1], price_data["low"].iloc[-1], 'go', markersize=10, label="Selling Climax")
    
    # Plot testing patterns
    if patterns.get("testing", {}).get("detected", False):
        for test in patterns["testing"].get("tests", []):
            if test["type"] == "SUPPORT_TEST":
                ax.plot(test["index"], test["price"], 'g^', markersize=8, label="Support Test")
            elif test["type"] == "RESISTANCE_TEST":
                ax.plot(test["index"], test["price"], 'rv', markersize=8, label="Resistance Test")
    
    # Set title
    ax.set_title("VPA Pattern Detection", fontsize=12, fontweight='bold')
    
    # Add volatility text
    price_range = price_data["high"].max() - price_data["low"].min()
    price_volatility = price_range / price_data["close"].mean()
    ax.text(0.02, 0.95, f"Volatility: {price_volatility:.2f}", transform=ax.transAxes, fontsize=10, color='black', va='top')

    # Add grid for better readability
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Save if output file provided
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    
    return fig

def plot_multi_timeframe_analysis(timeframe_analyses, output_file=None):
    """
    Plot multi-timeframe analysis
    
    Parameters:
    - timeframe_analyses: Dictionary with analysis results for each timeframe
    - output_file: Optional file path to save the plot
    
    Returns:
    - Figure object
    """
    # Determine number of timeframes
    num_timeframes = len(timeframe_analyses)
    
    # Create figure and axes
    fig, axes = plt.subplots(num_timeframes, 2, figsize=(15, 5 * num_timeframes))
    
    # If only one timeframe, wrap axes in a 2D list
    if num_timeframes == 1:
        axes = [[axes[0], axes[1]]]  # Wrap axes in a 2D list for consistency
    else:
        axes = axes.reshape(num_timeframes, 2)  # Ensure axes is always 2D
    
    # Plot each timeframe
    for i, (timeframe, analysis) in enumerate(timeframe_analyses.items()):
        # Get data
        processed_data = analysis["processed_data"]
        price_data = processed_data["price"]
        volume_data = processed_data["volume"]
        
        # Plot price chart
        latest_date = price_data.index[-1].strftime('%Y-%m-%d %H:%M')
        ax = plot_candlestick(axes[i][0], price_data, title=f"{timeframe.upper()} – {latest_date}")
        ax.title.set_fontsize(11)
        ax.title.set_fontweight("bold")

        
        # Add signals
        plot_vpa_signals(axes[i][0], price_data, processed_data, 
                        analysis["candle_analysis"], analysis["support_resistance"])
                       
        # Plot volume
        axes[i][1].bar(volume_data.index, volume_data, color='blue', alpha=0.5)
        axes[i][1].set_title(f"{timeframe} Volume", fontsize=9, fontweight="bold")
        
        # Format x-axis
        axes[i][1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(axes[i][1].xaxis.get_majorticklabels(), rotation=45)
        
        # Add analysis information
        candle_analysis = analysis["candle_analysis"]
        trend_analysis = analysis["trend_analysis"]
        
        info_text = f"Candle: {candle_analysis['candle_class']}\n"
        info_text += f"Volume: {candle_analysis['volume_class']}\n"
        info_text += f"Signal: {candle_analysis['signal_type']} ({candle_analysis['signal_strength']})\n"
        info_text += f"Trend: {trend_analysis['trend_direction']} ({trend_analysis['volume_trend']})"
        
        axes[i][1].text(0.05, 0.95, info_text, transform=axes[i][1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=9,
                        color="green" if 'BUY' in info_text else 'red',
                        fontweight="bold"
                        )
        
        axes[i][0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(axes[i][0].xaxis.get_majorticklabels(), rotation=45)
        axes[i][0].grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

        for label in (axes[i][0].get_xticklabels() + axes[i][1].get_xticklabels()):
            label.set_fontsize(8)

    # Adjust layout
    plt.tight_layout()
    
    # Save if output file provided
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    
    return fig

def create_vpa_report(analysis_results, output_dir="vpa_reports"):
    """
    Create a comprehensive VPA analysis report
    
    Parameters:
    - analysis_results: Dictionary with analysis results
    - output_dir: Directory to save the report
    
    Returns:
    - Dictionary with report file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    ticker = analysis_results["ticker"]
    timeframe_analyses = analysis_results["timeframe_analyses"]
    signal = analysis_results["signal"]
    risk_assessment = analysis_results["risk_assessment"]

    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create report files
    report_files = {}
    
    # Create price chart with signals
    primary_tf = list(timeframe_analyses.keys())[0]
    price_data = timeframe_analyses[primary_tf]["processed_data"]["price"]
    processed_data = timeframe_analyses[primary_tf]["processed_data"]
    support_resistance = timeframe_analyses[primary_tf]["support_resistance"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_candlestick(ax, price_data, processed_data["volume"], title=f"{ticker} - VPA Analysis ({current_datetime})")
    plot_vpa_signals(ax, price_data, processed_data, signal, support_resistance)
    
    price_chart_file = os.path.join(output_dir, f"{ticker}_price_chart_{current_datetime}.png")
    plt.savefig(price_chart_file, bbox_inches='tight')
    plt.close(fig)
    report_files["price_chart"] = price_chart_file
    
    # Create multi-timeframe analysis chart
    multi_tf_file = os.path.join(output_dir, f"{ticker}_multi_timeframe_{current_datetime}.png")
    fig = plot_multi_timeframe_analysis(timeframe_analyses, multi_tf_file)
    plt.close(fig)
    report_files["multi_timeframe"] = multi_tf_file
    
    # Create pattern detection chart
    patterns = timeframe_analyses[primary_tf]["pattern_analysis"]
    pattern_file = os.path.join(output_dir, f"{ticker}_patterns_{current_datetime}.png")
    fig = plot_pattern_detection(price_data, patterns, pattern_file)
    plt.close(fig)
    report_files["patterns"] = pattern_file
    
    # Create text report
    report_text = f"VPA Analysis Report for {ticker}\n"
    report_text += f"Generated on: {current_datetime}\n"
    report_text += "=" * 50 + "\n\n"
    
    # Add signal information
    report_text += "Signal Information:\n"
    report_text += "-" * 30 + "\n"
    report_text += f"Type: {signal['type']}\n"
    report_text += f"Strength: {signal['strength']}\n"
    report_text += f"Details: {signal['details']}\n\n"
    
    # Add risk assessment
    report_text += "Risk Assessment:\n"
    report_text += "-" * 30 + "\n"
    report_text += f"Current Price: ${analysis_results['current_price']:.2f}\n"
    report_text += f"Stop Loss: ${risk_assessment['stop_loss']:.2f}\n"
    report_text += f"Take Profit: ${risk_assessment['take_profit']:.2f}\n"
    report_text += f"Risk-Reward Ratio: {risk_assessment['risk_reward_ratio']:.2f}\n"
    report_text += f"Position Size: {risk_assessment['position_size']:.2f} shares\n\n"
    
    # Add timeframe analysis
    report_text += "Timeframe Analysis:\n"
    report_text += "-" * 30 + "\n"
    
    for timeframe, analysis in timeframe_analyses.items():
        report_text += f"{timeframe}:\n"
        report_text += f"  Candle Analysis: {analysis['candle_analysis']['signal_type']} ({analysis['candle_analysis']['signal_strength']})\n"
        report_text += f"  Trend Analysis: {analysis['trend_analysis']['signal_type']} ({analysis['trend_analysis']['signal_strength']})\n"
        
        # Add pattern information
        patterns = analysis["pattern_analysis"]
        report_text += "  Patterns:\n"
        
        for pattern, data in patterns.items():
            if pattern != "testing":
                report_text += f"    {pattern.capitalize()}: {'Detected' if data['detected'] else 'Not Detected'}\n"
            else:
                report_text += f"    Testing: {'Detected' if data['detected'] else 'Not Detected'} ({len(data.get('tests', []))} tests)\n"
        
        report_text += "\n"
    
    # Write report to file
    report_file = os.path.join(output_dir, f"{ticker}_vpa_report_{current_datetime}.txt")
    with open(report_file, "w") as f:
        f.write(report_text)
    
    report_files["text_report"] = report_file
    
    return report_files

"""
Batch Reporting Function for VPA Utils Module

This function creates consolidated reports for multiple tickers analyzed with VPA.
"""

def create_batch_report(facade, tickers, output_dir="vpa_batch_reports", timeframes=None):
    """
    Create a consolidated report for multiple tickers
    
    Parameters:
    - facade: VPAFacade instance
    - tickers: List of stock symbols to analyze
    - output_dir: Directory to save the report
    - timeframes: Optional list of timeframe dictionaries with 'interval' and 'period' keys
    
    Returns:
    - Dictionary with report file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize results storage
    all_results = {}
    signals_summary = []
    
    # Analyze each ticker
    print(f"Analyzing {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers):
        try:
            print(f"[{i+1}/{len(tickers)}] Analyzing {ticker}...")
            
            # Get full analysis for the ticker
            results = facade.analyze_ticker(ticker, timeframes)
            all_results[ticker] = results
            
            # Extract key information for summary
            primary_tf = list(results["timeframe_analyses"].keys())[0]
            signal_type = results["signal"]["type"]
            signal_strength = results["signal"]["strength"]
            current_price = results["current_price"]
            stop_loss = results["risk_assessment"]["stop_loss"]
            take_profit = results["risk_assessment"]["take_profit"]
            risk_reward = results["risk_assessment"]["risk_reward_ratio"]
            
            # Get trend information
            trend_direction = results["timeframe_analyses"][primary_tf]["trend_analysis"]["trend_direction"]
            volume_trend = results["timeframe_analyses"][primary_tf]["trend_analysis"]["volume_trend"]
            
            # Get pattern information
            patterns = results["timeframe_analyses"][primary_tf]["pattern_analysis"]
            detected_patterns = [p for p, data in patterns.items() if data.get("detected", False)]
            
            # Add to summary
            signals_summary.append({
                "ticker": ticker,
                "signal_type": signal_type,
                "signal_strength": signal_strength,
                "current_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward": risk_reward,
                "trend_direction": trend_direction,
                "volume_trend": volume_trend,
                "detected_patterns": ", ".join(detected_patterns) if detected_patterns else "None"
            })
            
            # Create individual report for this ticker
            ticker_dir = os.path.join(output_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            create_vpa_report(results, ticker_dir)
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            signals_summary.append({
                "ticker": ticker,
                "signal_type": "ERROR",
                "signal_strength": "N/A",
                "current_price": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "risk_reward": 0,
                "trend_direction": "N/A",
                "volume_trend": "N/A",
                "detected_patterns": f"Error: {str(e)}"
            })
    
    # Create summary report files
    report_files = {}
    
    # Create signals summary table
    signals_df = pd.DataFrame(signals_summary)
    
    # Sort by signal strength and type
    signal_priority = {"STRONG": 3, "MODERATE": 2, "WEAK": 1, "NEUTRAL": 0, "ERROR": -1, "N/A": -1}
    type_priority = {"BUY": 2, "SELL": 1, "NO_ACTION": 0, "ERROR": -1, "N/A": -1}
    
    signals_df["signal_priority"] = signals_df["signal_strength"].map(signal_priority)
    signals_df["type_priority"] = signals_df["signal_type"].map(type_priority)
    
    signals_df = signals_df.sort_values(["signal_priority", "type_priority", "risk_reward"], 
                                        ascending=[False, False, False])
    
    # Drop priority columns used for sorting
    signals_df = signals_df.drop(columns=["signal_priority", "type_priority"])
    
    # Save summary table to CSV
    summary_file = os.path.join(output_dir, "vpa_signals_summary.csv")
    signals_df.to_csv(summary_file, index=False)
    report_files["summary_csv"] = summary_file
    
    # Create summary visualizations
    
    # 1. Signal distribution chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Signal type distribution
    signal_counts = signals_df["signal_type"].value_counts()
    axes[0].pie(signal_counts, labels=signal_counts.index, autopct='%1.1f%%', 
               colors=['green', 'red', 'gray', 'lightgray'])
    axes[0].set_title("Signal Type Distribution")
    
    # Signal strength distribution
    strength_counts = signals_df["signal_strength"].value_counts()
    axes[1].pie(strength_counts, labels=strength_counts.index, autopct='%1.1f%%',
               colors=['darkgreen', 'green', 'lightgreen', 'gray', 'lightgray'])
    axes[1].set_title("Signal Strength Distribution")
    
    plt.tight_layout()
    signal_dist_file = os.path.join(output_dir, "signal_distribution.png")
    plt.savefig(signal_dist_file)
    plt.close(fig)
    report_files["signal_distribution"] = signal_dist_file
    
    # 2. Top signals chart
    # Filter for actionable signals (BUY or SELL)
    actionable = signals_df[signals_df["signal_type"].isin(["BUY", "SELL"])]
    
    if not actionable.empty:
        # Take top 5 or fewer
        top_signals = actionable.head(min(5, len(actionable)))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart of risk-reward ratios
        bars = ax.bar(top_signals["ticker"], top_signals["risk_reward"])
        
        # Color bars by signal type
        for i, signal_type in enumerate(top_signals["signal_type"]):
            bars[i].set_color('green' if signal_type == 'BUY' else 'red')
        
        # Add labels
        for i, bar in enumerate(bars):
            signal_type = top_signals.iloc[i]["signal_type"]
            signal_strength = top_signals.iloc[i]["signal_strength"]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f"{signal_type}\n({signal_strength})",
                   ha='center', va='bottom', rotation=0, fontsize=9)
        
        ax.set_ylabel('Risk-Reward Ratio')
        ax.set_title('Top VPA Signals by Risk-Reward Ratio')
        
        plt.tight_layout()
        top_signals_file = os.path.join(output_dir, "top_signals.png")
        plt.savefig(top_signals_file)
        plt.close(fig)
        report_files["top_signals"] = top_signals_file
    
    # 3. Comparative price chart for top tickers
    if not actionable.empty:
        # Take top 3 or fewer
        top_tickers = actionable.head(min(3, len(actionable)))["ticker"].tolist()
        
        # Create comparative chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot normalized prices for comparison
        for ticker in top_tickers:
            if ticker in all_results:
                primary_tf = list(all_results[ticker]["timeframe_analyses"].keys())[0]
                price_data = all_results[ticker]["timeframe_analyses"][primary_tf]["processed_data"]["price"]
                
                # Normalize to percentage change from start
                normalized = price_data["close"] / price_data["close"].iloc[0] * 100
                
                # Plot with ticker as label
                ax.plot(normalized.index, normalized, label=ticker)
        
        ax.set_ylabel('Price (normalized %)')
        ax.set_title('Comparative Price Movement')
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        comparative_file = os.path.join(output_dir, "comparative_prices.png")
        plt.savefig(comparative_file)
        plt.close(fig)
        report_files["comparative_prices"] = comparative_file
    
    # 4. Create consolidated dashboard for top signals
    if not actionable.empty:
        # Take top 3 or fewer
        top_tickers = actionable.head(min(3, len(actionable)))["ticker"].tolist()
        
        # Create dashboard
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Main title
        fig.suptitle('VPA Analysis Dashboard - Top Signals', fontsize=16)
        
        # For each top ticker
        for i, ticker in enumerate(top_tickers):
            if ticker in all_results and i < 3:  # Limit to 3 tickers
                results = all_results[ticker]
                primary_tf = list(results["timeframe_analyses"].keys())[0]
                price_data = results["timeframe_analyses"][primary_tf]["processed_data"]["price"]
                volume_data = results["timeframe_analyses"][primary_tf]["processed_data"]["volume"]
                processed_data = results["timeframe_analyses"][primary_tf]["processed_data"]
                
                # Create subplot
                ax = fig.add_subplot(gs[i//2, i%2])
                
                # Plot candlestick
                plot_candlestick(ax, price_data, title=f"{ticker} - {results['signal']['type']} ({results['signal']['strength']})")
                plot_vpa_signals(ax, price_data, processed_data, results["signal"], 
                                results["timeframe_analyses"][primary_tf]["support_resistance"])
        
        # Add summary table in the last cell
        ax_table = fig.add_subplot(gs[:, 2])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        # Create table data
        table_data = []
        columns = ['Ticker', 'Signal', 'R/R', 'Price', 'Stop', 'Target']
        
        for ticker in top_tickers:
            if ticker in all_results:
                results = all_results[ticker]
                table_data.append([
                    ticker,
                    f"{results['signal']['type']} ({results['signal']['strength']})",
                    f"{results['risk_assessment']['risk_reward_ratio']:.2f}",
                    f"${results['current_price']:.2f}",
                    f"${results['risk_assessment']['stop_loss']:.2f}",
                    f"${results['risk_assessment']['take_profit']:.2f}"
                ])
        
        # Create table
        table = ax_table.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color rows based on signal type
        for i, ticker in enumerate(top_tickers):
            if ticker in all_results:
                signal_type = all_results[ticker]['signal']['type']
                for j in range(len(columns)):
                    cell = table[(i+1, j)]
                    cell.set_facecolor('lightgreen' if signal_type == 'BUY' else 'lightcoral')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        dashboard_file = os.path.join(output_dir, "vpa_dashboard.png")
        plt.savefig(dashboard_file, bbox_inches='tight')
        plt.close(fig)
        report_files["dashboard"] = dashboard_file
    
    # Create HTML report
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VPA Batch Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
            h1, h2 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; background-color: white; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .buy {{ background-color: rgba(0, 255, 0, 0.1); }}
            .sell {{ background-color: rgba(255, 0, 0, 0.1); }}
            .dashboard {{ width: 100%; max-width: 1200px; margin: 20px 0; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); }}
            .chart {{ width: 100%; max-width: 600px; margin: 20px 0; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); }}
            .ticker-link {{ color: #0066cc; text-decoration: none; }}
            .ticker-link:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>VPA Batch Analysis Report</h1>
        <p>Generated on: {generation_date}</p>
        <p>Tickers analyzed: {tickers_analyzed}</p>
        
        <h2>Dashboard</h2>
        <img src="vpa_dashboard.png" class="dashboard" alt="VPA Dashboard">
        
        <h2>Signal Distribution</h2>
        <img src="signal_distribution.png" class="chart" alt="Signal Distribution">
        
        <h2>Top Signals</h2>
        <img src="top_signals.png" class="chart" alt="Top Signals">
        
        <h2>Comparative Price Movement</h2>
        <img src="comparative_prices.png" class="chart" alt="Comparative Prices">
        
        <h2>Signals Summary</h2>
        <table>
            <tr>
                <th>Ticker</th>
                <th>Signal</th>
                <th>Strength</th>
                <th>Price</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Risk/Reward</th>
                <th>Trend</th>
                <th>Volume Trend</th>
                <th>Patterns</th>
            </tr>
            {table_rows}
        </table>
        
        <script>
            // Add JavaScript to make ticker links work
            document.querySelectorAll('.ticker-link').forEach(link => {{
                link.addEventListener('click', function(e) {{
                    e.preventDefault();
                    window.open(this.getAttribute('href'), '_blank');
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Add rows for each ticker
    table_rows = ""
    for _, row in signals_df.iterrows():
        signal_class = "buy" if row["signal_type"] == "BUY" else "sell" if row["signal_type"] == "SELL" else ""
        
        table_rows += f"""
        <tr class="{signal_class}">
            <td><a href="{row['ticker']}/{row['ticker']}_vpa_report_{current_datetime}.txt" class="ticker-link">{row['ticker']}</a></td>
            <td>{row['signal_type']}</td>
            <td>{row['signal_strength']}</td>
            <td>${row['current_price']:.2f}</td>
            <td>${row['stop_loss']:.2f}</td>
            <td>${row['take_profit']:.2f}</td>
            <td>{row['risk_reward']:.2f}</td>
            <td>{row['trend_direction']}</td>
            <td>{row['volume_trend']}</td>
            <td>{row['detected_patterns']}</td>
        </tr>
    """
    
    # Generate HTML content using the template
    html_content = HTML_TEMPLATE.format(
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        tickers_analyzed=len(tickers),
        table_rows=table_rows
    )
    
    # Save HTML report
    html_file = os.path.join(output_dir, "vpa_batch_report.html")
    with open(html_file, "w") as f:
        f.write(html_content)
    
    report_files["html_report"] = html_file
    
    # Create text summary report
    text_report = f"VPA Batch Analysis Report\n"
    text_report += "=" * 50 + "\n\n"
    text_report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    text_report += f"Tickers analyzed: {len(tickers)}\n\n"
    
    text_report += "Signal Summary:\n"
    text_report += "-" * 30 + "\n"
    
    # Count signal types
    buy_signals = len(signals_df[signals_df["signal_type"] == "BUY"])
    sell_signals = len(signals_df[signals_df["signal_type"] == "SELL"])
    no_action = len(signals_df[signals_df["signal_type"] == "NO_ACTION"])
    errors = len(signals_df[signals_df["signal_type"] == "ERROR"])
    
    text_report += f"BUY signals: {buy_signals}\n"
    text_report += f"SELL signals: {sell_signals}\n"
    text_report += f"NO_ACTION signals: {no_action}\n"
    text_report += f"Errors: {errors}\n\n"
    
    # Top BUY signals
    text_report += "Top BUY Signals:\n"
    text_report += "-" * 30 + "\n"
    
    buy_signals = signals_df[signals_df["signal_type"] == "BUY"].head(5)
    for _, row in buy_signals.iterrows():
        text_report += f"{row['ticker']}: {row['signal_strength']} - Risk/Reward: {row['risk_reward']:.2f}\n"
        text_report += f"  Price: ${row['current_price']:.2f}, Stop: ${row['stop_loss']:.2f}, Target: ${row['take_profit']:.2f}\n"
        text_report += f"  Patterns: {row['detected_patterns']}\n\n"
    
    # Top SELL signals
    text_report += "Top SELL Signals:\n"
    text_report += "-" * 30 + "\n"
    
    sell_signals = signals_df[signals_df["signal_type"] == "SELL"].head(5)
    for _, row in sell_signals.iterrows():
        text_report += f"{row['ticker']}: {row['signal_strength']} - Risk/Reward: {row['risk_reward']:.2f}\n"
        text_report += f"  Price: ${row['current_price']:.2f}, Stop: ${row['stop_loss']:.2f}, Target: ${row['take_profit']:.2f}\n"
        text_report += f"  Patterns: {row['detected_patterns']}\n\n"
    
    # Save text report
    text_file = os.path.join(output_dir, "vpa_batch_summary.txt")
    with open(text_file, "w") as f:
        f.write(text_report)
    
    report_files["text_summary"] = text_file
    
    print(f"Batch report created in {output_dir}")
    print(f"HTML report: {html_file}")
    
    return report_files

# vpa_visualizer.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import os
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from typing import Dict, List, Any

def plot_price_volume_chart(df: pd.DataFrame, ticker: str, timeframe: str, output_path: str = None):
    """
    Plot candlestick chart with volume for a specific ticker and timeframe.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', and optionally 'volume'
        ticker (str): Stock symbol
        timeframe (str): Timeframe label (e.g., '1d', '1h', '15m')
        output_path (str, optional): Path to save the figure. If None, just show.
    """
    fig, ax_price = plt.subplots(figsize=(12, 6), dpi=100)

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Price plot (candlestick-like using line + vertical bars)
    ax_price.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.2)
    ax_price.vlines(df.index, df['low'], df['high'], color='gray', linewidth=0.5, alpha=0.7)

    # Volume plot (if volume data is available)
    if 'volume' in df.columns:
        ax_volume = ax_price.twinx()
        ax_volume.bar(df.index, df['volume'], width=0.005, color='blue', alpha=0.2, label='Volume')
        ax_volume.set_ylim(0, df['volume'].max() * 4)
        ax_volume.set_ylabel("Volume")
        ax_volume.legend(loc='upper right')

    # Formatting
    ax_price.set_title(f"{ticker} - {timeframe.upper()} Price" + (" + Volume" if 'volume' in df.columns else ""))
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle='--', alpha=0.3)

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_price.xaxis.set_major_locator(mticker.MaxNLocator(10))
    fig.autofmt_xdate()

    ax_price.legend(loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✅ Chart saved to {output_path}")
    else:
        plt.show()

    plt.close()

def plot_pattern_analysis(df: pd.DataFrame, pattern_analysis: dict, ticker: str, timeframe: str, output_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Plot close price in blue for better visibility
    ax.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=1.2)
    
    # Define a color map for different patterns
    color_map = {
        'accumulation': 'green',
        'distribution': 'red',
        'testing': 'purple',
        'other': 'orange'  # For any other patterns
    }
    
    # Create a list to store legend handles
    legend_elements = [plt.Line2D([0], [0], color='blue', lw=2, label='Close Price')]
    
    for pattern, data in pattern_analysis.items():
        if data.get('detected', False):
            color = color_map.get(pattern.lower(), color_map['other'])
            # If specific dates are provided, use them; otherwise, use the last date
            pattern_date = data.get('date', df.index[-1])
            pattern_price = data.get('price', df['close'].iloc[-1])  # Use provided price or last close price
            ax.hlines(y=pattern_price, xmin=df.index[0], xmax=df.index[-1], color=color, alpha=0.5, linestyle='--')
            ax.text(df.index[-1], pattern_price, pattern, horizontalalignment='right', verticalalignment='center', color=color)
            
            # Add a horizontal line to the legend
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=pattern))
    
    ax.set_title(f"{ticker} - {timeframe.upper()} Pattern Analysis")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Pattern analysis chart saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_support_resistance(df: pd.DataFrame, support_resistance: dict, ticker: str, timeframe: str, output_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    ax.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.2)
    
    if isinstance(support_resistance, dict):
        support_levels = support_resistance.get('support_levels', {})
        resistance_levels = support_resistance.get('resistance_levels', {})
        
        for level, value in support_levels.items():
            ax.axhline(y=value, color='green', linestyle='--', alpha=0.7, label=f'Support {level}')
        
        for level, value in resistance_levels.items():
            ax.axhline(y=value, color='red', linestyle='--', alpha=0.7, label=f'Resistance {level}')
    else:
        print(f"Warning: Unexpected support_resistance format for {ticker} - {timeframe}")
    
    ax.set_title(f"{ticker} - {timeframe.upper()} Support and Resistance Levels")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Support and resistance chart saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def create_summary_report(extractor, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "vpa_summary_report.txt")
    
    with open(report_path, 'w') as f:
        for ticker in extractor.get_tickers():
            f.write(f"Analysis for {ticker}:\n")
            f.write(f"Current price: {extractor.get_ticker_data(ticker)['current_price']:.2f}\n\n")

            # Signal
            signal = extractor.get_signal(ticker)
            f.write("Signal:\n")
            f.write(f"  Type: {signal.get('type', 'N/A')}\n")
            f.write(f"  Strength: {signal.get('strength', 'N/A')}\n")
            f.write(f"  Details: {signal.get('details', 'N/A')}\n\n")

            # Evidence
            evidence = signal.get('evidence', {})
            f.write("Evidence:\n")
            for evidence_type, items in evidence.items():
                f.write(f"  {evidence_type.capitalize()}:\n")
                for item in items:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            f.write(f"    {key}: {value}\n")
                    else:
                        f.write(f"    {item}\n")
                f.write("\n")

            # Risk Assessment
            risk = extractor.get_risk_assessment(ticker)
            f.write("Risk Assessment:\n")
            for key, value in risk.items():
                f.write(f"  {key.replace('_', ' ').capitalize()}: {value:.2f}\n")
            f.write("\n")

            # Timeframe Analysis
            for timeframe in extractor.get_timeframes(ticker):
                f.write(f"Timeframe: {timeframe}\n")
                
                # Candle Analysis
                candle = extractor.get_candle_analysis(ticker, timeframe)
                f.write("  Candle Analysis:\n")
                for key, value in candle.items():
                    f.write(f"    {key.replace('_', ' ').capitalize()}: {value}\n")
                f.write("\n")
                
                # Trend Analysis
                trend = extractor.get_trend_analysis(ticker, timeframe)
                f.write("  Trend Analysis:\n")
                for key, value in trend.items():
                    f.write(f"    {key.replace('_', ' ').capitalize()}: {value}\n")
                f.write("\n")
                
                # Pattern Analysis
                pattern = extractor.get_pattern_analysis(ticker, timeframe)
                f.write("  Pattern Analysis:\n")
                for pat_type, pat_data in pattern.items():
                    f.write(f"    {pat_type.capitalize()}:\n")
                    for key, value in pat_data.items():
                        f.write(f"      {key.capitalize()}: {value}\n")
                f.write("\n")
                
                # Support and Resistance
                sr = extractor.get_support_resistance(ticker, timeframe)
                f.write("  Support and Resistance:\n")
                for sr_type in ['support', 'resistance']:
                    f.write(f"    {sr_type.capitalize()}:\n")
                    for level in sr.get(sr_type, []):
                        f.write(f"      Price: {level['price']:.2f}, Strength: {level['strength']:.1f}, Tests: {level.get('tests', 'N/A')}\n")
                f.write("\n")

            f.write("\n")

    print(f"✅ Summary report saved to {report_path}")

def create_dashboard(extractor, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    dashboard_path = os.path.join(output_dir, "vpa_dashboard.png")
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 16), dpi=100)
    fig.suptitle("VPA Analysis Dashboard", fontsize=16)
    
    # Plot 1: Comparative price chart
    for ticker in extractor.get_tickers():
        df = extractor.get_price_data(ticker, extractor.get_timeframes(ticker)[0])
        axs[0, 0].plot(df.index, df['close'], label=ticker)
    
    axs[0, 0].set_title("Comparative Price Chart")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Signal distribution
    signals = [extractor.get_signal(ticker)['type'] for ticker in extractor.get_tickers()]
    sns.countplot(x=signals, ax=axs[0, 1])
    axs[0, 1].set_title("Signal Distribution")
    
    # Plot 3: Risk-Reward Ratio
    risk_rewards = [extractor.get_risk_assessment(ticker)['risk_reward_ratio'] for ticker in extractor.get_tickers()]
    axs[1, 0].bar(extractor.get_tickers(), risk_rewards)
    axs[1, 0].set_title("Risk-Reward Ratio")
    axs[1, 0].set_ylim(0, max(risk_rewards) * 1.2)
    
    # Plot 4: Top Patterns
    all_patterns = []
    for ticker in extractor.get_tickers():
        for timeframe in extractor.get_timeframes(ticker):
            patterns = extractor.get_pattern_analysis(ticker, timeframe)
            all_patterns.extend([p for p, data in patterns.items() if data['detected']])
    
    pattern_counts = pd.Series(all_patterns).value_counts()
    pattern_counts.plot(kind='bar', ax=axs[1, 1])
    axs[1, 1].set_title("Top Detected Patterns")
    
    plt.tight_layout()
    plt.savefig(dashboard_path)
    plt.close()
    
    print(f"✅ Dashboard saved to {dashboard_path}")

def create_signal_dashboard(signal: Dict[str, Any], ticker: str, output_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
    # Signal type and strength
    ax1.text(0.5, 0.5, f"{signal['type']}\n{signal['strength']}", 
            ha='center', va='center', fontsize=24, 
            color='green' if signal['type'] == 'BUY' else 'red')
    ax1.axis('off')
        
    # Evidence breakdown
    evidence = signal['evidence']
    categories = list(evidence.keys())
    counts = [len(evidence[cat]) for cat in categories]
        
    ax2.bar(categories, counts)
    ax2.set_title('Evidence Breakdown')
    ax2.set_ylabel('Count')
        
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()

def plot_multi_timeframe_trends(evidence: Dict[str, List], ticker: str, output_path: str = None):
    trend_signals = evidence.get('trend_signals', [])
    
    if not trend_signals:
        print(f"Warning: No trend signals found for {ticker}")
        return
    
    timeframes = [signal['timeframe'] for signal in trend_signals]
    
    fig, axs = plt.subplots(len(timeframes), 1, figsize=(10, 5*len(timeframes)), squeeze=False)
    
    for i, tf in enumerate(timeframes):
        signal = next(s for s in trend_signals if s['timeframe'] == tf)
        axs[i, 0].text(0.5, 0.5, signal['details'], ha='center', va='center', wrap=True)
        axs[i, 0].set_title(f"{tf} Trend")
        axs[i, 0].axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Multi-timeframe trends chart saved to {output_path}")
    plt.close()

def create_pattern_signal_heatmap(evidence: Dict[str, List], ticker: str, output_path: str = None):
    pattern_signals = evidence['pattern_signals']
    timeframes = list(set(signal['timeframe'] for signal in pattern_signals))
    patterns = list(set(signal['pattern'] for signal in pattern_signals))
    
    data = np.zeros((len(timeframes), len(patterns)))
    
    for signal in pattern_signals:
        i = timeframes.index(signal['timeframe'])
        j = patterns.index(signal['pattern'])
        data[i, j] = 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='YlOrRd')
    
    ax.set_xticks(np.arange(len(patterns)))
    ax.set_yticks(np.arange(len(timeframes)))
    ax.set_xticklabels(patterns)
    ax.set_yticklabels(timeframes)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_title(f"Pattern Signals Heatmap for {ticker}")
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.close()

def plot_risk_management(risk_assessment: Dict[str, float], current_price: float, ticker: str, output_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prices = [risk_assessment['stop_loss'], current_price, risk_assessment['take_profit']]
    labels = ['Stop Loss', 'Current Price', 'Take Profit']
    colors = ['red', 'blue', 'green']
    
    ax.bar(labels, prices, color=colors)
    ax.set_title(f"Risk Management for {ticker}")
    ax.set_ylabel("Price")
    
    for i, price in enumerate(prices):
        ax.text(i, price, f'${price:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()

def visualize_risk_reward_ratio(risk_assessment: Dict[str, float], ticker: str, output_path: str = None):
    risk = risk_assessment['risk_per_share']
    reward = risk_assessment['take_profit'] - risk_assessment['stop_loss']
    ratio = risk_assessment['risk_reward_ratio']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.bar(['Risk', 'Reward'], [risk, reward], color=['red', 'green'])
    ax.set_title(f"Risk-Reward Ratio for {ticker}")
    ax.set_ylabel("Amount")
    
    ax.text(0, risk/2, f'${risk:.2f}', ha='center', va='center', color='white')
    ax.text(1, reward/2, f'${reward:.2f}', ha='center', va='center', color='white')
    
    plt.text(0.5, 1.05, f"Ratio: {ratio:.2f}", transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()

def update_price_chart_with_risk_levels(price_data: pd.DataFrame, risk_assessment: Dict[str, float], current_price: float, ticker: str, output_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(price_data.index, price_data['close'], label='Close Price')
    
    ax.axhline(y=risk_assessment['stop_loss'], color='r', linestyle='--', label='Stop Loss')
    ax.axhline(y=risk_assessment['take_profit'], color='g', linestyle='--', label='Take Profit')
    ax.axhline(y=current_price, color='b', linestyle='-', label='Current Price')
    
    ax.set_title(f"{ticker} Price Chart with Risk Levels")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
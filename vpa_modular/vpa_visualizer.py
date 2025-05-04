# vpa_visualizer.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import os

def plot_price_volume_chart(df: pd.DataFrame, ticker: str, timeframe: str, output_path: str = None):
    """
    Plot candlestick chart with volume for a specific ticker and timeframe.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume'
        ticker (str): Stock symbol
        timeframe (str): Timeframe label (e.g., '1d', '1h', '15m')
        output_path (str, optional): Path to save the figure. If None, just show.
    """
    fig, ax_price = plt.subplots(figsize=(12, 6), dpi=100)
    ax_volume = ax_price.twinx()

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Price plot (candlestick-like using line + vertical bars)
    ax_price.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.2)
    ax_price.vlines(df.index, df['low'], df['high'], color='gray', linewidth=0.5, alpha=0.7)

    # Volume plot (as transparent bars behind the price line)
    ax_volume.bar(df.index, df['volume'], width=0.005, color='blue', alpha=0.2, label='Volume')
    ax_volume.set_ylim(0, df['volume'].max() * 4)

    # Formatting
    ax_price.set_title(f"{ticker} - {timeframe.upper()} Price + Volume")
    ax_price.set_ylabel("Price")
    ax_volume.set_ylabel("Volume")
    ax_price.grid(True, linestyle='--', alpha=0.3)

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_price.xaxis.set_major_locator(mticker.MaxNLocator(10))
    fig.autofmt_xdate()

    ax_price.legend(loc='upper left')
    ax_volume.legend(loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✅ Chart saved to {output_path}")
    else:
        plt.show()

    plt.close()

def plot_pattern_analysis(df: pd.DataFrame, pattern_analysis: dict, ticker: str, timeframe: str, output_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    ax.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.2)
    
    for pattern, data in pattern_analysis.items():
        if data.get('detected', False):
            # If specific dates are not provided, use the last date in the DataFrame
            pattern_date = df.index[-1]
            ax.axvline(x=pattern_date, color='yellow', alpha=0.5, linestyle='--', label=pattern)
            ax.text(pattern_date, df['close'].iloc[-1], pattern, rotation=90, verticalalignment='bottom')
    
    ax.set_title(f"{ticker} - {timeframe.upper()} Pattern Analysis")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
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
            f.write(f"Current price: {extractor.get_ticker_data(ticker)['current_price']}\n")
            f.write(f"Signal: {extractor.get_signal(ticker)}\n")
            f.write(f"Risk assessment: {extractor.get_risk_assessment(ticker)}\n\n")
            
            for timeframe in extractor.get_timeframes(ticker):
                f.write(f"Timeframe: {timeframe}\n")
                f.write(f"Candle analysis: {extractor.get_candle_analysis(ticker, timeframe)}\n")
                f.write(f"Trend analysis: {extractor.get_trend_analysis(ticker, timeframe)}\n")
                f.write(f"Pattern analysis: {extractor.get_pattern_analysis(ticker, timeframe)}\n")
                f.write(f"Support and Resistance: {extractor.get_support_resistance(ticker, timeframe)}\n\n")
    
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
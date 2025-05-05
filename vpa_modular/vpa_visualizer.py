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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf

def plot_price_volume_chart(df: pd.DataFrame, ticker: str, timeframe: str, output_path: str = None):
    """
    Plot close price and volume as two separate subplots, renaming volume column if needed.
    """
    print("Columns in df:", df.columns.tolist())

    # Step 1: Try to detect and rename volume column if needed
    if 'volume' not in df.columns:
        volume_cols = [col for col in df.columns if col.startswith("volume")]
        if volume_cols:
            df = df.rename(columns={volume_cols[0]: 'volume'})
            print(f"🔁 Renamed column '{volume_cols[0]}' to 'volume'")
        else:
            print(f"⚠️ No recognizable volume column found for {ticker} - {timeframe}. Skipping volume plot.")
            return

    # Step 2: Ensure timezone-naive datetime index
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    df['volume_avg'] = df['volume'].rolling(window=5, min_periods=1).mean()

    fig, (ax_price, ax_volume) = plt.subplots(
        2, 1, figsize=(12, 8), dpi=100, sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # --- Price plot ---
    ax_price.plot(df.index, df['close'], color='black', linewidth=1.5, label='Close Price')
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"{ticker} - {timeframe.upper()} Price and Volume")
    ax_price.grid(True, linestyle='--', alpha=0.3)
    ax_price.legend(loc='upper left')

    # --- Volume plot ---
    bar_width = max(0.3, 100 / len(df))  # dynamic width
    ax_volume.bar(df.index, df['volume'], width=bar_width, color='skyblue', alpha=0.6, label='Volume')
    ax_volume.plot(df.index, df['volume_avg'], color='orange', linestyle='--', linewidth=1, label='Avg Volume')
    ax_volume.set_ylabel("Volume")
    ax_volume.grid(True, linestyle='--', alpha=0.3)
    ax_volume.legend(loc='upper left')
    ax_volume.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✅ Price and volume chart saved to {output_path}")
    else:
        plt.show()

    plt.close()

def plot_pattern_analysis(df: pd.DataFrame, pattern_analysis: dict, ticker: str, timeframe: str, output_dir: str = "charts", max_test_labels: int = 5):
    """
    Plots pattern analysis for a ticker on a specific timeframe.
    
    Parameters:
    - df: Price DataFrame with datetime index and 'close' column.
    - pattern_analysis: Dict with pattern analysis data.
    - ticker: Ticker symbol.
    - timeframe: Timeframe string (e.g., "1d", "1h").
    - output_dir: Folder to save the chart.
    - max_test_labels: Max number of test labels to display for clarity.
    """

    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{ticker}_{timeframe}_pattern_analysis.png"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=1.2)

    # Pattern colors and markers
    color_map = {
        'accumulation': ('green', 'o'),
        'distribution': ('red', '^'),
        'testing': ('purple', 's'),
        'other': ('orange', 'x'),
    }

    legend_elements = [plt.Line2D([0], [0], color='blue', lw=2, label='Close Price')]

    for pattern, data in pattern_analysis.items():
        if not data.get('detected', False):
            continue

        color, marker = color_map.get(pattern.lower(), color_map['other'])

        if pattern == 'testing' and 'tests' in data:
            test_points = data['tests'][-max_test_labels:]  # Limit to recent tests
            for test in test_points:
                idx = pd.to_datetime(test["index"])
                price = test["price"]
                test_type = test["type"]
                vertical_align = "top" if "RESISTANCE" in test_type else "bottom"
                ax.scatter(idx, price, color=color, marker=marker, s=60, label=None)
                ax.text(idx, price, test_type, fontsize=8, color=color, va=vertical_align, ha='right')
            legend_elements.append(plt.Line2D([0], [0], color=color, marker=marker, linestyle='', label='testing'))

        elif 'date' in data and 'price' in data:
            idx = pd.to_datetime(data['date'])
            price = data['price']
            ax.scatter(idx, price, color=color, marker=marker, s=60)
            ax.text(idx, price, pattern, fontsize=8, color=color, va='center', ha='right')
            legend_elements.append(plt.Line2D([0], [0], color=color, marker=marker, linestyle='', label=pattern))

    ax.set_title(f"{ticker} - {timeframe.upper()} Pattern Analysis")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Pattern analysis chart saved to {output_path}")
    plt.close()

def plot_support_resistance(df: pd.DataFrame, support_resistance: dict, ticker: str, timeframe: str, output_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    ax.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.2)
    
    # Plot support levels
    for level in support_resistance.get('support', []):
        price = float(level.get('price', 0))
        ax.axhline(y=price, color='green', linestyle='--', alpha=0.7, label='Support')

    # Plot resistance levels
    for level in support_resistance.get('resistance', []):
        price = float(level.get('price', 0))
        ax.axhline(y=price, color='red', linestyle='--', alpha=0.7, label='Resistance')
    
    ax.set_title(f"{ticker} - {timeframe.upper()} Support and Resistance Levels")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    
    # Avoid legend duplication
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Support and resistance chart saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def create_summary_report(extractor, output_dir: str):
    import os

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "vpa_summary_report.txt")

    with open(report_path, 'w', encoding="utf-8") as f:
        for ticker in extractor.get_tickers():
            f.write(f"{'=' * 60}\n")
            f.write(f"📊 Analysis for {ticker}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Current Price: {extractor.get_ticker_data(ticker)['current_price']:.2f}\n\n")

            # Signal
            signal = extractor.get_signal(ticker)
            f.write("🔔 Signal:\n")
            f.write(f"  • Type: {signal.get('type', 'N/A')}\n")
            f.write(f"  • Strength: {signal.get('strength', 'N/A')}\n")
            f.write(f"  • Details: {signal.get('details', 'N/A')}\n\n")

            # Evidence
            evidence = signal.get('evidence', {})
            if evidence:
                f.write("📌 Evidence:\n")
                for evidence_type, items in evidence.items():
                    f.write(f"  • {evidence_type.capitalize()}:\n")
                    for item in items:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                f.write(f"      - {key}: {value}\n")
                        else:
                            f.write(f"      - {item}\n")
                f.write("\n")

            # Risk Assessment
            risk = extractor.get_risk_assessment(ticker)
            if risk:
                f.write("⚠️ Risk Assessment:\n")
                for key, value in risk.items():
                    f.write(f"  • {key.replace('_', ' ').capitalize()}: {value:.2f}\n")
                f.write("\n")

            # Timeframe-Specific Analysis
            for timeframe in extractor.get_timeframes(ticker):
                f.write(f"⏱️ Timeframe: {timeframe}\n")

                # Candle Analysis
                candle = extractor.get_candle_analysis(ticker, timeframe)
                if candle:
                    f.write("  🕯️ Candle Analysis:\n")
                    for key, value in candle.items():
                        f.write(f"    • {key.replace('_', ' ').capitalize()}: {value}\n")

                # Trend Analysis
                trend = extractor.get_trend_analysis(ticker, timeframe)
                if trend:
                    f.write("  📈 Trend Analysis:\n")
                    for key, value in trend.items():
                        f.write(f"    • {key.replace('_', ' ').capitalize()}: {value}\n")

                # Pattern Analysis
                pattern = extractor.get_pattern_analysis(ticker, timeframe)
                if pattern:
                    f.write("  🔍 Pattern Analysis:\n")
                    for pat_type, pat_data in pattern.items():
                        f.write(f"    • {pat_type.replace('_', ' ').capitalize()}:\n")
                        for key, value in pat_data.items():
                            if key.lower() == "tests" and isinstance(value, list):
                                f.write(f"        - Tests:\n")
                                for test in value:
                                    test_type = test.get("type", "N/A")
                                    index = test.get("index", "N/A")
                                    price = test.get("price", "N/A")
                                    f.write(f"            - Type: {test_type}, Time: {index}, Price: {price:.2f}\n")
                            else:
                                f.write(f"        - {key.replace('_', ' ').capitalize()}: {value}\n")

                # Support and Resistance
                sr = extractor.get_support_resistance(ticker, timeframe)
                if sr:
                    f.write("  📉 Support and Resistance:\n")
                    for sr_type in ['support', 'resistance']:
                        levels = sr.get(sr_type, [])
                        if levels:
                            f.write(f"    • {sr_type.capitalize()} Levels:\n")
                            for level in levels:
                                f.write(f"        - Price: {level['price']:.2f}, Strength: {level['strength']:.1f}, Tests: {level.get('tests', 'N/A')}\n")

                f.write("\n")  # Space between timeframes

            f.write("\n\n")  # Space between tickers

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

def visualize_risk_reward_ratio(risk_assessment: Dict[str, float], current_price: float, ticker: str, output_path: str = None):
    stop_loss = risk_assessment['stop_loss']
    take_profit = risk_assessment['take_profit']
    risk = current_price - stop_loss
    reward = take_profit - current_price
    ratio = risk_assessment['risk_reward_ratio']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"{ticker} - Risk-Reward Overview (Ratio: {ratio:.2f})")

    # Plot risk area (red)
    ax.barh(0, risk, left=stop_loss, color='red', edgecolor='black', height=0.4, label='Risk')

    # Plot reward area (green)
    ax.barh(0, reward, left=current_price, color='green', edgecolor='black', height=0.4, label='Reward')

    # Draw vertical lines
    ax.axvline(stop_loss, color='red', linestyle='--')
    ax.axvline(current_price, color='blue', linestyle='--')
    ax.axvline(take_profit, color='green', linestyle='--')

    # Add text annotations
    ax.text(stop_loss, 0.1, f"Stop\n${stop_loss:.2f}", color='black', ha='center', va='bottom', fontsize=9)
    ax.text(current_price, 0.1, f"Current\n${current_price:.2f}", color='blue', ha='center', va='bottom', fontsize=9)
    ax.text(take_profit, 0.1, f"Profit\n${take_profit:.2f}", color='black', ha='center', va='bottom', fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("Price")
    ax.legend()

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

def plot_vpa_signals_candlestick(df: pd.DataFrame, signals: dict, ticker: str, output_path: str = None):
    """
    Plot a candlestick chart with VPA signal annotations.
    """
    # Filtrar apenas sinais detectados
    evidence = signals.get("evidence", {})
    signal_markers = []

    for timeframe, patterns in evidence.items():
        for pattern in patterns:
            index = pattern.get("index")
            label = pattern.get("type", "signal")
            if index in df.index:
                signal_markers.append((index, label))

    # Criar lista de anotações
    addplots = []
    for date, label in signal_markers:
        idx_pos = df.index.get_loc(date)
        price = df.iloc[idx_pos]["high"]
        addplots.append(mpf.make_addplot(
            [None if i != idx_pos else price * 1.01 for i in range(len(df))],
            type='scatter',
            markersize=100,
            marker='v',
            color='red'
        ))

    # Plot do candlestick com as anotações
    fig, axlist = mpf.plot(
        df,
        type='candle',
        style='yahoo',
        title=f'{ticker} - VPA Candlestick Signals',
        ylabel='Price',
        volume=False,
        addplot=addplots,
        returnfig=True,
        figratio=(12,6)
    )

    # Adicionar os rótulos das anotações
    if signal_markers:
        ax = axlist[0]
        for date, label in signal_markers:
            if date in df.index:
                idx = df.index.get_loc(date)
                price = df.iloc[idx]["high"]
                ax.annotate(label, xy=(mdates.date2num(date), price * 1.01),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=8, color='red', rotation=45)

    if output_path:
        fig.savefig(output_path)
        print(f"✅ Candlestick VPA chart saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)

def plot_relative_volume(df: pd.DataFrame, ticker: str, timeframe: str, output_path: str = None):
    """
    Plot the relative volume (volume compared to its moving average) over time.
    """
    # Detect volume column (flexível)
    volume_col = next((col for col in df.columns if 'volume' in col.lower()), None)

    if volume_col is None:
        print(f"⚠️ Volume column not found for {ticker} - {timeframe}. Skipping relative volume plot.")
        print("Columns in df:", df.columns.tolist())
        return

    df = df.copy()
    df['volume_avg'] = df[volume_col].rolling(window=20, min_periods=1).mean()
    df['relative_volume'] = df[volume_col] / df['volume_avg']

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['relative_volume'], label='Relative Volume', color='purple')
    plt.axhline(1, color='gray', linestyle='--', linewidth=1, label='Average Volume')
    plt.title(f"{ticker} - {timeframe.upper()} Relative Volume")
    plt.xlabel("Date")
    plt.ylabel("Relative Volume")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✅ Relative volume chart saved to {output_path}")
    else:
        plt.show()

    plt.close()

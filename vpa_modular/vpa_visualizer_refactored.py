# vpa_visualizer_refactored.py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
import matplotlib.dates as mdates
import pandas as pd
import os
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# VPAResultExtractor would be imported if this was part of the package
from vpa_modular.vpa_result_extractor import VPAResultExtractor 

class VPAVisualizerRefactored:
    """
    Refactored VPA Visualizer to generate charts and reports from VPAResultExtractor data.
    Ensures outputs are saved to specified directories and logs operations.
    """
    def __init__(self, result_extractor, output_base_dir="vpa_reports", log_base_dir="vpa_modular_package/logs"):
        """
        Initializes the VPAVisualizerRefactored.

        Args:
            result_extractor: An instance of VPAResultExtractor (or a similar class providing the same interface).
            output_base_dir (str): The base directory to save reports and charts.
            log_base_dir (str): The base directory to save log files.
        """
        self.extractor = result_extractor
        self.output_base_dir = output_base_dir
        self.log_base_dir = log_base_dir

        # Create base directories if they don't exist
        os.makedirs(self.output_base_dir, exist_ok=True)
        os.makedirs(self.log_base_dir, exist_ok=True)

        # Setup logger
        self.logger = self._setup_logger()
        self.logger.info(f"VPAVisualizerRefactored initialized. Output base: 	{self.output_base_dir}, Log base: {self.log_base_dir}")

    def _setup_logger(self):
        """Sets up the logger for the visualizer."""
        logger = logging.getLogger(f"VPAVisualizerRefactored_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file_path = os.path.join(self.log_base_dir, "vpa_visualizer.log")
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        
        # Create console handler (optional, for debugging)
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(fh)
        # logger.addHandler(ch)
        
        return logger

    def generate_all_outputs_for_ticker(self, ticker: str):
        """
        Generates all reports and charts for a specific ticker.
        """
        self.logger.info(f"Starting output generation for ticker: {ticker}")
        ticker_output_dir = os.path.join(self.output_base_dir, ticker)
        os.makedirs(ticker_output_dir, exist_ok=True)
        self.logger.info(f"Output directory for {ticker}: {ticker_output_dir}")

        # Generate reports
        self.generate_text_report(ticker, ticker_output_dir)
        self.generate_json_report(ticker, ticker_output_dir)

        # Generate charts
        self.plot_candlestick_with_signals(ticker, ticker_output_dir)
        self.plot_pattern_analysis_chart(ticker, ticker_output_dir)
        self.plot_multi_timeframe_dashboard(ticker, ticker_output_dir)
        self.plot_support_resistance_chart(ticker, ticker_output_dir)
        
        self.logger.info(f"Completed output generation for ticker: {ticker}")

    def generate_text_report(self, ticker: str, ticker_output_dir: str):
        """
        Generates a comprehensive text report for the given ticker.
        """
        report_path = os.path.join(ticker_output_dir, f"{ticker}_report.txt")
        self.logger.info(f"Generating TXT report for {ticker} at {report_path}")
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"VPA Analysis Report for: {ticker}\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*70 + "\n\n")

                ticker_data = self.extractor.get_ticker_data(ticker)
                if not ticker_data:
                    f.write("No data found for this ticker.\n")
                    self.logger.warning(f"No data extracted for ticker {ticker} for TXT report.")
                    return

                f.write(f"Current Price: {ticker_data.get('current_price', 'N/A')}\n\n")

                # Signal Information
                signal_info = ticker_data.get('signal', {})
                f.write("Overall Signal:\n")
                f.write(f"  Type: {signal_info.get('type', 'N/A')}\n")
                f.write(f"  Strength: {signal_info.get('strength', 'N/A')}\n")
                f.write(f"  Details: {signal_info.get('details', 'N/A')}\n")
                if 'evidence' in signal_info:
                    f.write("  Evidence:\n")
                    for ev_type, ev_list in signal_info['evidence'].items():
                        f.write(f"    {ev_type.replace('_', ' ').title()}:\n")
                        for item_idx, item_detail in enumerate(ev_list):
                            if isinstance(item_detail, dict):
                                for k, v_item in item_detail.items():
                                     f.write(f"      - {k.replace('_', ' ').title()}: {v_item}\n")
                            else:
                                f.write(f"      - {item_detail}\n")
                f.write("\n")

                # Risk Assessment
                risk_assessment = ticker_data.get('risk_assessment', {})
                if risk_assessment:
                    f.write("Risk Assessment:\n")
                    for k, v in risk_assessment.items():
                        f.write(f"  {k.replace('_', ' ').title()}: {v}\n")
                    f.write("\n")

                # Timeframe Analyses
                f.write("Timeframe Specific Analysis:\n")
                timeframes_data = ticker_data.get('timeframes', {})
                if not timeframes_data:
                    f.write("  No timeframe-specific data available.\n")
                
                for tf, tf_data in timeframes_data.items():
                    f.write(f"  Timeframe: {tf.upper()}\n")
                    f.write(f"  {'-'*20}\n")
                    
                    # Candle Analysis
                    candle = tf_data.get('candle_analysis', {})
                    if candle:
                        f.write("    Candle Analysis:\n")
                        for k, v in candle.items():
                            f.write(f"      {k.replace('_', ' ').title()}: {v}\n")
                    
                    # Trend Analysis
                    trend = tf_data.get('trend_analysis', {})
                    if trend:
                        f.write("    Trend Analysis:\n")
                        for k, v in trend.items():
                            f.write(f"      {k.replace('_', ' ').title()}: {v}\n")

                    # Pattern Analysis
                    pattern = tf_data.get('pattern_analysis', {})
                    if pattern:
                        f.write("    Pattern Analysis:\n")
                        for pat_type, pat_detail in pattern.items():
                            f.write(f"      {pat_type.replace('_', ' ').title()}:")
                            if isinstance(pat_detail, dict) and 'detected' in pat_detail:
                                f.write(f" Detected: {pat_detail['detected']}\n")
                                if pat_detail.get('detected') and 'tests' in pat_detail:
                                    f.write("        Tests:\n")
                                    for test_item in pat_detail['tests']:
                                        f.write(f"          - Type: {test_item.get('type')}, Price: {test_item.get('price')}, Index: {test_item.get('index')}\n")
                                elif pat_detail.get('detected'):
                                     f.write(f"        Details: {pat_detail}\n") # Fallback for other pattern structures
                            else:
                                f.write(f" {pat_detail}\n")

                    # Support/Resistance
                    sr = tf_data.get('support_resistance', {})
                    if sr:
                        f.write("    Support/Resistance:\n")
                        if sr.get('support'):
                            f.write("      Support Levels:\n")
                            for level in sr['support']:
                                f.write(f"        - Price: {level.get('price')}, Strength: {level.get('strength')}, Tests: {level.get('tests')}\n")
                        if sr.get('resistance'):
                            f.write("      Resistance Levels:\n")
                            for level in sr['resistance']:
                                f.write(f"        - Price: {level.get('price')}, Strength: {level.get('strength')}, Tests: {level.get('tests')}\n")
                    f.write("\n") # Spacer after each timeframe
                f.write("="*70 + "\n")
            self.logger.info(f"TXT report for {ticker} generated successfully.")
        except Exception as e:
            self.logger.error(f"Error generating TXT report for {ticker}: {e}", exc_info=True)

    def generate_json_report(self, ticker: str, ticker_output_dir: str):
        """
        Generates a JSON report for the given ticker.
        """
        report_path = os.path.join(ticker_output_dir, f"{ticker}_report.json")
        self.logger.info(f"Generating JSON report for {ticker} at {report_path}")
        
        try:
            ticker_data_full = self.extractor.get_ticker_data(ticker)
            
            # Convert pandas DataFrames to dict for JSON serialization if they exist
            # The VPAResultExtractor already converts price_data and volume_data to DataFrames
            # We need to ensure they are converted back to a JSON-serializable format here.
            serializable_data = {}
            if ticker_data_full:
                serializable_data = copy.deepcopy(ticker_data_full) # DEEP COPY to prevent modifying original mock data
                if 'timeframes' in serializable_data:
                    for tf_key, tf_content in serializable_data['timeframes'].items():
                        if 'price_data' in tf_content and isinstance(tf_content['price_data'], pd.DataFrame):
                            tf_content['price_data'] = tf_content['price_data'].to_dict(orient='records')
                        if 'volume_data' in tf_content and isinstance(tf_content['volume_data'], pd.DataFrame):
                            # Assuming volume_data might be a Series or DataFrame from extractor
                            if isinstance(tf_content['volume_data'], pd.Series):
                                 tf_content['volume_data'] = tf_content['volume_data'].to_dict()
                            else: # DataFrame
                                tf_content['volume_data'] = tf_content['volume_data'].to_dict(orient='records')
            else:
                self.logger.warning(f"No data extracted for ticker {ticker} for JSON report.")

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=4, default=str) # default=str for any other non-serializable types
            self.logger.info(f"JSON report for {ticker} generated successfully.")
        except Exception as e:
            self.logger.error(f"Error generating JSON report for {ticker}: {e}", exc_info=True)
    def _ensure_datetime_index(self, df: pd.DataFrame, df_name: str = "DataFrame") -> pd.DataFrame:
        """Ensure DataFrame has a datetime index."""
        if df is None or df.empty:
            self.logger.warning(f"{df_name} is None or empty, cannot ensure datetime index.")
            return pd.DataFrame()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                self.logger.info(f"Successfully converted index of {df_name} to DatetimeIndex.")
            except Exception as e1:
                self.logger.warning(f"Could not convert existing index of {df_name} to DatetimeIndex: {e1}. Checking for date columns.")
                datetime_cols = [col for col in df.columns if col.lower() in ["date", "datetime", "time", "timestamp"]]
                if datetime_cols:
                    date_col_to_use = datetime_cols[0]
                    try:
                        df.index = pd.to_datetime(df[date_col_to_use])
                        df = df.drop(columns=[date_col_to_use]) 
                        self.logger.info(f"Successfully set and converted 	{date_col_to_use}	 column as DatetimeIndex for {df_name}.".replace("\t", "'")) # Ensure tabs are replaced here too
                    except Exception as e2:
                        self.logger.error(f"Failed to convert 	{date_col_to_use}	 column to DatetimeIndex for {df_name}: {e2}".replace("\t", "'"))
                        return pd.DataFrame() 
                else:
                    self.logger.error(f"No suitable date/datetime column found to set as index for {df_name}.")
                    return pd.DataFrame() 
        return df

    def _plot_base_candlestick_chart(self, ax, price_data: pd.DataFrame, volume_data: pd.Series = None, chart_title: str = "Candlestick Chart"):
        """Plots a base candlestick chart with volume on the given axis."""
        if price_data.empty or not all(col in price_data.columns for col in ["open", "high", "low", "close"]):
            self.logger.warning(f"Price data is empty or missing OHLC columns for chart: {chart_title}. Skipping plot.")
            ax.text(0.5, 0.5, "Price data unavailable or incomplete", ha='center', va='center')
            return

        width = 0.8 
        width2 = 0.2 

        up = price_data[price_data.close >= price_data.open]
        down = price_data[price_data.close < price_data.open]

        ax.bar(up.index, up.close - up.open, width, bottom=up.open, color='green', edgecolor='black', linewidth=0.7)
        ax.vlines(up.index, up.low, up.high, color='green', linewidth=width2*2) 

        ax.bar(down.index, down.close - down.open, width, bottom=down.open, color='red', edgecolor='black', linewidth=0.7)
        ax.vlines(down.index, down.low, down.high, color='red', linewidth=width2*2)

        date_format_str = '%Y-%m-%d %H:%M' if len(price_data) < 100 else '%Y-%m-%d'
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format_str))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        
        ax.set_title(chart_title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        if volume_data is not None and not volume_data.empty:
            ax2 = ax.twinx()
            ax2.bar(volume_data.index, volume_data, width, color='blue', alpha=0.3, label='Volume')
            ax2.set_ylabel('Volume', fontsize=10, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.grid(False)
        
        ax.autoscale_view()

    def plot_candlestick_with_signals(self, ticker: str, ticker_output_dir: str):
        """
        Plots candlestick charts with VPA signals (Support/Resistance and Overall Signal)
        for each timeframe of the given ticker.
        """
        self.logger.info(f"Generating candlestick charts with signals for {ticker} in {ticker_output_dir}")
        
        timeframes = self.extractor.get_timeframes(ticker)
        if not timeframes:
            self.logger.warning(f"No timeframes found for ticker {ticker}. Skipping candlestick charts.")
            return

        overall_signal_data = self.extractor.get_signal(ticker)

        for timeframe in timeframes:
            self.logger.info(f"Plotting candlestick for {ticker} - Timeframe: {timeframe}")
            
            price_df = self.extractor.get_price_data(ticker, timeframe)
            volume_series = self.extractor.get_volume_data(ticker, timeframe) 
            sr_data = self.extractor.get_support_resistance(ticker, timeframe)

            price_df = self._ensure_datetime_index(price_df, f"{ticker}-{timeframe} price_df")
            
            # Ensure volume_series is a Series with a DatetimeIndex
            if isinstance(volume_series, pd.DataFrame):
                if 'volume' in volume_series.columns:
                    volume_series = volume_series['volume'] # Select the volume column if it's a DataFrame
                else:
                    self.logger.warning(f"'volume' column not found in volume DataFrame for {ticker}-{timeframe}. Skipping volume plot.")
                    volume_series = pd.Series(dtype=float)
            
            if isinstance(volume_series, pd.Series):
                temp_df_vol = pd.DataFrame(volume_series)
                temp_df_vol = self._ensure_datetime_index(temp_df_vol, f"{ticker}-{timeframe} volume_data_conversion")
                if not temp_df_vol.empty:
                    # Use the original series name if available, otherwise the first column name of the converted DataFrame
                    col_name = volume_series.name if volume_series.name is not None else temp_df_vol.columns[0]
                    volume_series = temp_df_vol[col_name]
                else: 
                    volume_series = pd.Series(dtype=float)
            elif volume_series is not None: # Not a Series or DataFrame, but not None
                self.logger.warning(f"Volume data for {ticker}-{timeframe} is not a Series or DataFrame. Type: {type(volume_series)}. Skipping volume plot.")
                volume_series = pd.Series(dtype=float) 
            else: # volume_series is None
                 volume_series = pd.Series(dtype=float) 

            if price_df.empty:
                self.logger.warning(f"Price data is empty for {ticker} - {timeframe} after processing. Skipping chart.")
                continue

            fig, ax = plt.subplots(figsize=(15, 7))
            chart_title = f"{ticker} - {timeframe.upper()} - VPA Candlestick Analysis"
            
            self._plot_base_candlestick_chart(ax, price_df, volume_series, chart_title)

            if sr_data:
                for level_type, levels in sr_data.items():
                    if level_type == "support" and levels:
                        for level_info in levels:
                            price = level_info.get('price')
                            strength = level_info.get('strength', '')
                            if price is not None:
                                ax.axhline(y=price, color='green', linestyle='--', alpha=0.6, linewidth=1.2)
                                ax.text(price_df.index[-1] + pd.Timedelta(days=1), price, f" S ({strength})", color='green', va='center', fontsize=8)
                    elif level_type == "resistance" and levels:
                        for level_info in levels:
                            price = level_info.get('price')
                            strength = level_info.get('strength', '')
                            if price is not None:
                                ax.axhline(y=price, color='red', linestyle='--', alpha=0.6, linewidth=1.2)
                                ax.text(price_df.index[-1] + pd.Timedelta(days=1), price, f" R ({strength})", color='red', va='center', fontsize=8)
            
            if overall_signal_data and not price_df.empty:
                signal_type = overall_signal_data.get("type")
                signal_strength = overall_signal_data.get("strength", "")
                
                plot_signal = False
                y_pos = 0 
                if signal_type == "BUY":
                    color = 'green'
                    marker = '^'
                    y_pos = price_df['low'].iloc[-1] * 0.98 if not price_df['low'].empty else price_df['close'].iloc[-1] * 0.98
                    plot_signal = True
                elif signal_type == "SELL":
                    color = 'red'
                    marker = 'v'
                    y_pos = price_df['high'].iloc[-1] * 1.02 if not price_df['high'].empty else price_df['close'].iloc[-1] * 1.02
                    plot_signal = True
                
                if plot_signal:
                    ax.plot(price_df.index[-1], y_pos, marker=marker, markersize=12, color=color, markeredgecolor='black')
                    ax.text(price_df.index[-1], y_pos, f" {signal_type} ({signal_strength})", 
                            color=color, fontsize=10, va='center', ha='left', fontweight='bold')

            plt.tight_layout()
            save_path = os.path.join(ticker_output_dir, f"{ticker}_{timeframe}_candlestick_signals.png")
            try:
                fig.savefig(save_path, bbox_inches='tight')
                self.logger.info(f"Successfully saved candlestick chart to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save candlestick chart for {ticker}-{timeframe} to {save_path}: {e}")
            finally:
                plt.close(fig) 



    # This file contains the Python code for the new plot_pattern_analysis_chart method
    
    
    def plot_pattern_analysis_chart(self, ticker: str, ticker_output_dir: str):
        """
        Plots candlestick charts with VPA pattern analysis overlays (e.g., accumulation/distribution tests)
        for each timeframe of the given ticker.
        """
        self.logger.info(f"Generating pattern analysis charts for {ticker} in {ticker_output_dir}")
        
        timeframes = self.extractor.get_timeframes(ticker)
        if not timeframes:
            self.logger.warning(f"No timeframes found for ticker {ticker}. Skipping pattern analysis charts.")
            return
    
        for timeframe in timeframes:
            self.logger.info(f"Plotting pattern analysis for {ticker} - Timeframe: {timeframe}")
            
            price_df = self.extractor.get_price_data(ticker, timeframe)
            volume_series = self.extractor.get_volume_data(ticker, timeframe)
            pattern_data = self.extractor.get_pattern_analysis(ticker, timeframe)
            # sr_data = self.extractor.get_support_resistance(ticker, timeframe) # May need for context
    
            if price_df.empty:
                self.logger.warning(f"Price data for {ticker} - {timeframe} is empty. Skipping pattern chart.")
                continue
            
            price_df = self._ensure_datetime_index(price_df, f"{ticker}-{timeframe} price_df for pattern chart")
            if price_df.empty: # Check again after ensure_datetime_index
                self.logger.warning(f"Price data for {ticker} - {timeframe} became empty after index check. Skipping pattern chart.")
                continue
                
            if volume_series is not None and not volume_series.empty:
                volume_series.index = pd.to_datetime(volume_series.index)
                # Align volume data with price data if necessary, though base plot handles it
                price_df, volume_series = price_df.align(volume_series, join="inner", axis=0)
                if price_df.empty:
                    self.logger.warning(f"Price data for {ticker} - {timeframe} became empty after aligning with volume. Skipping pattern chart.")
                    continue
    
            fig, ax = plt.subplots(figsize=(15, 7))
            chart_title = f"{ticker} - {timeframe.upper()} - VPA Pattern Analysis"
            self._plot_base_candlestick_chart(ax, price_df, volume_data=volume_series, chart_title=chart_title)
    
            # Overlay Pattern Analysis Details
            if pattern_data:
                for pattern_type, details in pattern_data.items(): # e.g., pattern_type = "accumulation"
                    if isinstance(details, dict) and details.get("detected") and details.get("tests"):
                        self.logger.info(f"Plotting 	{pattern_type}	 test markers for {ticker} - {timeframe}")
                        for test_idx, test_info in enumerate(details["tests"]):
                            try:
                                test_date_str = test_info.get("index")
                                test_price = test_info.get("price")
                                test_type_label = test_info.get("type", pattern_type) # Default to pattern_type if specific test type is missing
                                
                                if test_date_str is None or test_price is None:
                                    self.logger.warning(f"Skipping a test marker for {ticker}-{timeframe} due to missing date or price: {test_info}")
                                    continue
    
                                test_date = pd.to_datetime(test_date_str)
                                
                                # Ensure the test_date is within the plotted price_df index
                                if test_date not in price_df.index:
                                    self.logger.warning(f"Test date {test_date} for {ticker}-{timeframe} not in price data index. Skipping marker.")
                                    continue
    
                                # Determine marker style based on pattern type or test type
                                marker_color = "blue"
                                marker_shape = "o" # Default circle
                                if "accumulation" in pattern_type.lower():
                                    marker_color = "green"
                                    marker_shape = "^" # Triangle up for accumulation
                                elif "distribution" in pattern_type.lower():
                                    marker_color = "orange"
                                    marker_shape = "v" # Triangle down for distribution
                                
                                ax.scatter(test_date, test_price, marker=marker_shape, color=marker_color, s=100, zorder=5, label=f"{pattern_type.capitalize()} Test" if test_idx == 0 else None)
                                ax.annotate(f"{test_type_label.capitalize()}\n@ {test_price:.2f}", 
                                            (mdates.date2num(test_date), test_price),
                                            textcoords="offset points", 
                                            xytext=(0,10 if marker_shape == "^" else -20), # Adjust offset based on marker
                                            ha="center", fontsize=8, color=marker_color,
                                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
                            except Exception as e_test:
                                self.logger.error(f"Error plotting a test marker for {ticker}-{timeframe}: {test_info}, Error: {e_test}")
            else:
                self.logger.info(f"No specific pattern data to overlay for {ticker} - {timeframe}")
    
            # Add legend if there are labeled scatter points
            handles, labels = ax.get_legend_handles_labels()
            if handles: # Only show legend if there are items to show
                # Filter out duplicate labels for scatter plots if any
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=9)
            
            plt.tight_layout()
            plot_filename = os.path.join(ticker_output_dir, f"{ticker}_{timeframe}_pattern_analysis.png")
            try:
                plt.savefig(plot_filename)
                self.logger.info(f"Pattern analysis chart saved to {plot_filename}")
            except Exception as e_save:
                self.logger.error(f"Error saving pattern analysis chart {plot_filename}: {e_save}")
            plt.close(fig)
    


    # This file contains the Python code for the new plot_multi_timeframe_dashboard method
    
    
    def plot_multi_timeframe_dashboard(self, ticker: str, ticker_output_dir: str):
        """
        Plots a multi-timeframe dashboard showing price and volume for each available timeframe.
        """
        self.logger.info(f"Generating multi-timeframe dashboard for {ticker} in {ticker_output_dir}")
        
        timeframes = self.extractor.get_timeframes(ticker)
        if not timeframes:
            self.logger.warning(f"No timeframes found for ticker {ticker}. Skipping multi-timeframe dashboard.")
            return
    
        num_timeframes = len(timeframes)
        if num_timeframes == 0:
            self.logger.warning(f"Zero timeframes available for {ticker} dashboard.")
            return
    
        # Determine layout: prefer 2 columns, adjust rows accordingly
        # Max 3 rows to keep it somewhat compact, if more than 6 TFs, it might get crowded or need scrolling/multiple images
        ncols = 2 if num_timeframes > 1 else 1
        nrows = (num_timeframes + ncols - 1) // ncols # Ceiling division
        
        fig = plt.figure(figsize=(8 * ncols, 6 * nrows))
        gs = gridspec.GridSpec(nrows, ncols, figure=fig)
        fig.suptitle(f"{ticker} - Multi-Timeframe VPA Dashboard", fontsize=16, fontweight="bold")
    
        for i, timeframe in enumerate(timeframes):
            self.logger.info(f"Adding {timeframe} to dashboard for {ticker}")
            price_df = self.extractor.get_price_data(ticker, timeframe)
            volume_series = self.extractor.get_volume_data(ticker, timeframe)
    
            ax_price = fig.add_subplot(gs[i // ncols, i % ncols])
    
            if price_df.empty:
                self.logger.warning(f"Price data for {ticker} - {timeframe} is empty. Skipping this subplot.")
                ax_price.text(0.5, 0.5, f"{timeframe.upper()}\nPrice data unavailable", ha="center", va="center", fontsize=10)
                ax_price.set_title(f"{timeframe.upper()} Analysis", fontsize=12)
                continue
                
            price_df = self._ensure_datetime_index(price_df, f"{ticker}-{timeframe} price_df for dashboard")
            if price_df.empty:
                self.logger.warning(f"Price data for {ticker} - {timeframe} became empty after index check. Skipping subplot.")
                ax_price.text(0.5, 0.5, f"{timeframe.upper()}\nPrice data invalid", ha="center", va="center", fontsize=10)
                ax_price.set_title(f"{timeframe.upper()} Analysis", fontsize=12)
                continue
    
            if volume_series is not None and not volume_series.empty:
                volume_series.index = pd.to_datetime(volume_series.index)
                price_df, volume_series = price_df.align(volume_series, join="inner", axis=0)
                if price_df.empty:
                    self.logger.warning(f"Price data for {ticker} - {timeframe} became empty after aligning with volume. Skipping subplot.")
                    ax_price.text(0.5, 0.5, f"{timeframe.upper()}\nData alignment failed", ha="center", va="center", fontsize=10)
                    ax_price.set_title(f"{timeframe.upper()} Analysis", fontsize=12)
                    continue
            
            # Use the base candlestick plot for consistency
            subplot_title = f"{timeframe.upper()} Analysis"
            self._plot_base_candlestick_chart(ax_price, price_df, volume_data=volume_series, chart_title=subplot_title)
            ax_price.tick_params(axis="x", labelsize=8)
            ax_price.tick_params(axis="y", labelsize=8)
            if ax_price.get_legend() is not None: # Remove individual legends if base plot adds one
                ax_price.get_legend().remove()
            
            # You could add more specific annotations per timeframe here if needed
            # For example, overall signal for that timeframe if available from extractor
            # tf_signal = self.extractor.get_signal_for_timeframe(ticker, timeframe) # Hypothetical method
            # if tf_signal:
            #     ax_price.text(0.02, 0.95, f"Signal: {tf_signal.get(	text	, 	N/A	)}", transform=ax_price.transAxes, fontsize=9, va=	top	, bbox=dict(boxstyle=	round,pad=0.3	, fc=	yellow	, alpha=0.5))
    
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plot_filename = os.path.join(ticker_output_dir, f"{ticker}_multi_timeframe_dashboard.png")
        try:
            plt.savefig(plot_filename)
            self.logger.info(f"Multi-timeframe dashboard saved to {plot_filename}")
        except Exception as e_save:
            self.logger.error(f"Error saving multi-timeframe dashboard {plot_filename}: {e_save}")
        plt.close(fig)
    


    # This file contains the Python code for the new plot_support_resistance_chart method
    
    
    def plot_support_resistance_chart(self, ticker: str, ticker_output_dir: str):
        """
        Plots candlestick charts with VPA support and resistance levels for each timeframe.
        """
        self.logger.info(f"Generating support/resistance charts for {ticker} in {ticker_output_dir}")
        
        timeframes = self.extractor.get_timeframes(ticker)
        if not timeframes:
            self.logger.warning(f"No timeframes found for ticker {ticker}. Skipping S/R charts.")
            return
    
        for timeframe in timeframes:
            self.logger.info(f"Plotting S/R for {ticker} - Timeframe: {timeframe}")
            
            price_df = self.extractor.get_price_data(ticker, timeframe)
            volume_series = self.extractor.get_volume_data(ticker, timeframe) # For base chart context
            sr_data = self.extractor.get_support_resistance(ticker, timeframe)
    
            if price_df.empty:
                self.logger.warning(f"Price data for {ticker} - {timeframe} is empty. Skipping S/R chart.")
                continue
            
            price_df = self._ensure_datetime_index(price_df, f"{ticker}-{timeframe} price_df for S/R chart")
            if price_df.empty:
                self.logger.warning(f"Price data for {ticker} - {timeframe} became empty after index check. Skipping S/R chart.")
                continue
                
            if volume_series is not None and not volume_series.empty:
                volume_series.index = pd.to_datetime(volume_series.index)
                price_df, volume_series = price_df.align(volume_series, join="inner", axis=0)
                if price_df.empty:
                    self.logger.warning(f"Price data for {ticker} - {timeframe} became empty after aligning with volume. Skipping S/R chart.")
                    continue
    
            fig, ax = plt.subplots(figsize=(15, 7))
            chart_title = f"{ticker} - {timeframe.upper()} - Support & Resistance Analysis"
            self._plot_base_candlestick_chart(ax, price_df, volume_data=volume_series, chart_title=chart_title)
    
            # Overlay Support and Resistance Levels
            if sr_data:
                min_price = price_df["low"].min()
                max_price = price_df["high"].max()
                plot_start_date = price_df.index.min()
                plot_end_date = price_df.index.max()
    
                if sr_data.get("support"):
                    for i, level_info in enumerate(sr_data["support"]):
                        price = level_info.get("price")
                        strength = level_info.get("strength", "N/A")
                        if price is not None:
                            ax.axhline(y=price, color="green", linestyle="--", linewidth=1.2, label=f"Support ({strength})" if i == 0 else None)
                            ax.text(plot_end_date, price, f" S: {price:.2f} ({strength})", color="green", va="center", ha="left", fontsize=8, backgroundcolor="white")
                
                if sr_data.get("resistance"):
                    for i, level_info in enumerate(sr_data["resistance"]):
                        price = level_info.get("price")
                        strength = level_info.get("strength", "N/A")
                        if price is not None:
                            ax.axhline(y=price, color="red", linestyle="--", linewidth=1.2, label=f"Resistance ({strength})" if i == 0 else None)
                            ax.text(plot_end_date, price, f" R: {price:.2f} ({strength})", color="red", va="center", ha="left", fontsize=8, backgroundcolor="white")
            else:
                self.logger.info(f"No S/R data to overlay for {ticker} - {timeframe}")
    
            # Add legend if there are labeled lines
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=9)
            
            plt.tight_layout()
            plot_filename = os.path.join(ticker_output_dir, f"{ticker}_{timeframe}_support_resistance.png")
            try:
                plt.savefig(plot_filename)
                self.logger.info(f"Support/Resistance chart saved to {plot_filename}")
            except Exception as e_save:
                self.logger.error(f"Error saving S/R chart {plot_filename}: {e_save}")
            plt.close(fig)
    

if __name__ == '__main__':
    # Import the enhanced mock extractor
    # from enhanced_mock_extractor import EnhancedMockVPAResultExtractor # Make sure this file is in the same directory or PYTHONPATH

    # Initialize the enhanced mock extractor
    extractor = VPAResultExtractor()
    
    # Initialize the visualizer with the mock extractor
    visualizer = VPAVisualizerRefactored(result_extractor=extractor)
    
    # Generate outputs for a ticker
    print("--- Generating outputs for AAPL ---")
    visualizer.generate_all_outputs_for_ticker("AAPL")
    print("--- Generating outputs for MSFT ---")
    visualizer.generate_all_outputs_for_ticker("MSFT") # Test with another ticker
    print("--- Generating outputs for NONEXISTENT ---")
    visualizer.generate_all_outputs_for_ticker("NONEXISTENT") # Test with a non-existent ticker
    
    print(f"Check the '{visualizer.output_base_dir}' and '{visualizer.log_base_dir}' directories for outputs.")


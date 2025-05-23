"""
VPA LLM Training Data Generator Module

This module generates training data suitable for fine-tuning Large Language Models (LLMs)
to analyze Volume Price Analysis (VPA) results.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import traceback

# Assuming these modules are in the same parent directory or installed
from .vpa_facade import VPAFacade
from .vpa_data import PolygonIOProvider
from .vpa_logger import VPALogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class VPATrainingDataGenerator:
    """Generates VPA training data for LLMs in JSONL format."""

    def __init__(self, vpa_facade, output_dir="llm_training_data", log_level="INFO", log_file=None):
        """
        Initialize the generator.

        Parameters:
        - vpa_facade: An instance of VPAFacade to perform analysis.
        - output_dir: Directory to save the generated JSONL files.
        """
        self.vpa = vpa_facade
        self.output_dir = output_dir
        self.data_provider = PolygonIOProvider()
        os.makedirs(self.output_dir, exist_ok=True)

        if log_file is None:
            log_file = os.path.join(self.output_dir, "vpa_training_data_generator.log")
        
        self.logger = VPALogger(log_level, log_file)

    def _load_historical_data(self, ticker, start_date, end_date, timeframes):
        """
        Loads historical data for the specified ticker and timeframes.
        """
        self.logger.info(f"Loading historical data for {ticker} from {start_date} to {end_date}")
        all_data = {}
        try:
            for tf in timeframes:
                result = self.data_provider.get_data(ticker, interval=tf, start_date=start_date, end_date=end_date)
                
                # Polygon.io returns a tuple of (price_data, volume_data)
                price_data, volume_data = result
                
                if price_data is not None and not price_data.empty:
                    # Combine price and volume data
                    df = price_data.copy()
                    df['volume'] = volume_data
                    
                    # Ensure index is datetime
                    df.index = pd.to_datetime(df.index)
                    # Ensure timezone naive for consistency
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    all_data[tf] = df
                    self.logger.info(f"Loaded {len(df)} rows for {tf}")
                else:
                    self.logger.warning(f"No data loaded for timeframe {tf}")
            return all_data
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            self.logger.debug(traceback.format_exc())
            return {}

    def _extract_input_features(self, analysis_results, timestamp, primary_timeframe='1d', num_recent_candles=10):
        """
        Extracts features from VPA analysis results to form the LLM input.
        """
        if not analysis_results or primary_timeframe not in analysis_results.get("signals", {}):
            self.logger.warning(f"Analysis results missing or incomplete for {timestamp}")
            return None

        ticker = analysis_results.get("ticker", "UNKNOWN")
        tf_analysis = analysis_results["signals"][primary_timeframe]
        
        if not tf_analysis:
            self.logger.warning(f"No analysis data for {primary_timeframe} at {timestamp}")
            return None

        # Ensure timestamp is timezone-naive for consistency
        try:
            if isinstance(timestamp, str):
                current_ts = pd.to_datetime(timestamp)
            else:
                current_ts = timestamp
                
            if getattr(current_ts, 'tz', None) is not None:
                current_ts = current_ts.tz_localize(None)
                
            # Format timestamp for output
            timestamp_str = current_ts.strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            self.logger.error(f"Error processing timestamp {timestamp}: {e}")
            timestamp_str = str(timestamp)

        # Extract VPA metrics from candle analysis
        candle_analysis = tf_analysis.get("candle", {})
        vpa_metrics = {
            "volume_class": candle_analysis.get("volume_class", "UNKNOWN"),
            "candle_class": candle_analysis.get("candle_class", "UNKNOWN"),
            "signal_type": candle_analysis.get("signal_type", "UNKNOWN"),
            "signal_strength": candle_analysis.get("signal_strength", "UNKNOWN")
        }

        # Extract Trend Analysis
        trend_analysis = tf_analysis.get("trend", {})
        trend_input = {
            "trend_direction": trend_analysis.get("trend_direction", "UNKNOWN"),
            "price_change_percent": trend_analysis.get("price_change_percent", 0),
            "volume_trend": trend_analysis.get("volume_trend", "UNKNOWN"),
            "signal_type": trend_analysis.get("signal_type", "NONE"),
            "signal_strength": trend_analysis.get("signal_strength", "NONE")
        }

        # Extract Pattern Analysis
        pattern_analysis = tf_analysis.get("patterns", {})
        pattern_input = {}
        for p_name, p_data in pattern_analysis.items():
            pattern_input[p_name] = {
                "detected": p_data.get("detected", False),
                "strength": p_data.get("strength", 0),
                "details": p_data.get("details", "")
            }

        # Extract Support/Resistance
        sr_analysis = tf_analysis.get("support_resistance", {})
        support_levels = []
        resistance_levels = []
        
        if "support" in sr_analysis:
            for level in sr_analysis["support"]:
                if isinstance(level, dict) and "price" in level:
                    support_levels.append(level["price"])
                else:
                    support_levels.append(level)
                    
        if "resistance" in sr_analysis:
            for level in sr_analysis["resistance"]:
                if isinstance(level, dict) and "price" in level:
                    resistance_levels.append(level["price"])
                else:
                    resistance_levels.append(level)
        
        sr_input = {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels
        }
        
        # Extract Multi-Timeframe Context
        mttf_context = {}
        for tf, tf_data in analysis_results.get("signals", {}).items():
            if tf != primary_timeframe:
                mttf_context[tf] = {
                    "trend_direction": tf_data.get("trend", {}).get("trend_direction", "UNKNOWN"),
                    "signal_strength": tf_data.get("candle", {}).get("signal_strength", "UNKNOWN")
                }

        # Extract risk-reward information
        risk_reward = analysis_results.get("risk_reward", {})
        
        input_features = {
            "timestamp": timestamp_str,
            "ticker": ticker,
            "primary_timeframe": primary_timeframe,
            "vpa_metrics": vpa_metrics,
            "trend_analysis": trend_input,
            "pattern_analysis": pattern_input,
            "support_resistance": sr_input,
            "multi_timeframe_context": mttf_context,
            "risk_reward": {
                "risk_reward_ratio": risk_reward.get("risk_reward_ratio", 0),
                "stop_loss": risk_reward.get("stop_loss", 0),
                "take_profit": risk_reward.get("take_profit", 0)
            },
            "volatility": analysis_results.get("volatility", {}).get("volatility_percent", 0),
            "confidence_score": analysis_results.get("confidence_score", 0)
        }
        return input_features

    def _generate_explanation(self, input_features, analysis_results):
        """
        Generates a comprehensive textual explanation based on the input features and analysis.
        """
        if not input_features or not analysis_results:
            return "Error: Insufficient data for explanation."

        ticker = input_features["ticker"]
        timestamp = input_features["timestamp"]
        primary_tf = input_features["primary_timeframe"]
        trend = input_features["trend_analysis"]
        metrics = input_features["vpa_metrics"]
        patterns = input_features["pattern_analysis"]
        sr = input_features["support_resistance"]
        confidence_score = input_features.get("confidence_score", 0)
        
        # Determine overall signal type and strength
        signal_type = metrics["signal_type"]
        signal_strength = metrics["signal_strength"]
        
        if signal_type == "VALIDATION" and signal_strength == "BULLISH":
            overall_signal = "BUY"
        elif signal_type == "VALIDATION" and signal_strength == "BEARISH":
            overall_signal = "SELL"
        elif signal_type == "ANOMALY":
            if signal_strength == "BULLISH":
                overall_signal = "POTENTIAL_BUY"
            elif signal_strength == "BEARISH":
                overall_signal = "POTENTIAL_SELL"
            else:
                overall_signal = "NEUTRAL"
        else:
            overall_signal = "NEUTRAL"

        explanation = f"VPA Analysis for {ticker} on {timestamp} ({primary_tf}):\n\n" \
                      f"**Signal:** {overall_signal} ({signal_strength})\n\n" \
                      f"**Trend Context:** The primary trend is currently {trend['trend_direction']} " \
                      f"(Price change: {trend['price_change_percent']}%, Volume trend: {trend['volume_trend']}). "
        
        if trend['signal_type'] != 'NONE':
            explanation += f"Recent price action shows a {trend['signal_type']} ({trend['signal_strength']}). "
        explanation += "\n\n"

        explanation += f"**Current Candle Analysis:**\n" \
                       f"- Candle Type: {metrics['candle_class']}\n" \
                       f"- Volume: {metrics['volume_class']}\n" \
                       f"- Signal: {metrics['signal_type']} ({metrics['signal_strength']})\n\n"

        detected_patterns = [name for name, data in patterns.items() if data.get("detected")]
        if detected_patterns:
            explanation += f"**Detected Patterns:**\n"
            for name in detected_patterns:
                details = patterns[name].get("details", "No specific details.")
                strength = patterns[name].get("strength", 0)
                explanation += f"- {name.replace('_', ' ').title()} (Strength: {strength}): {details}\n"
            explanation += "\n"
        else:
            explanation += "**Detected Patterns:** None\n\n"

        explanation += f"**Support/Resistance:**\n"
        if sr['support_levels']:
            explanation += f"- Support Levels: {', '.join([f'{s:.2f}' for s in sr['support_levels']])}\n"
        if sr['resistance_levels']:
            explanation += f"- Resistance Levels: {', '.join([f'{r:.2f}' for r in sr['resistance_levels']])}\n"
        explanation += "\n"
        
        # Add risk-reward information
        risk_reward = input_features.get("risk_reward", {})
        explanation += f"**Risk-Reward Analysis:**\n" \
                       f"- Risk-Reward Ratio: {risk_reward.get('risk_reward_ratio', 0):.2f}\n" \
                       f"- Suggested Stop Loss: {risk_reward.get('stop_loss', 0):.2f}\n" \
                       f"- Suggested Take Profit: {risk_reward.get('take_profit', 0):.2f}\n\n"
        
        # Add multi-timeframe context
        mttf_context = input_features.get("multi_timeframe_context", {})
        if mttf_context:
            explanation += f"**Multi-Timeframe Context:**\n"
            for tf, data in mttf_context.items():
                explanation += f"- {tf}: {data.get('trend_direction', 'UNKNOWN')} trend, " \
                               f"{data.get('signal_strength', 'UNKNOWN')} signal\n"
            explanation += "\n"
        
        # Add confidence score
        explanation += f"**Confidence Score:** {confidence_score:.1f}/100\n\n"

        explanation += f"**Conclusion:** Based on the {trend['trend_direction']} trend, " \
                       f"the {metrics['volume_class']} volume on the {metrics['candle_class']} candle"
        if detected_patterns:
            explanation += f", and the presence of {', '.join(detected_patterns)}"
        explanation += f", a {signal_strength} {overall_signal} signal is generated with " \
                       f"a confidence score of {confidence_score:.1f}/100."

        return explanation.strip()

    def generate_training_data(self, ticker, start_date, end_date, primary_timeframe='1d', other_timeframes=None, min_lookback=50):
        """
        Generates and saves training data for a given ticker and period.

        Parameters:
        - ticker: Stock symbol.
        - start_date: Start date for data loading and analysis.
        - end_date: End date for data loading and analysis.
        - primary_timeframe: The main timeframe for iteration and analysis (e.g., '1d').
        - other_timeframes: List of secondary timeframes for context (e.g., ['1h', '15m']).
        - min_lookback: Minimum number of candles required before starting generation.
        """
        self.logger.info(f"Generating training data for {ticker}")
        
        if other_timeframes is None:
            other_timeframes = ['1h', '15m']
        all_timeframes = [primary_timeframe] + other_timeframes

        output_file = os.path.join(self.output_dir, f"{ticker}_vpa_training_data.jsonl")
        
        # Load all necessary historical data upfront
        historical_data = self._load_historical_data(ticker, start_date, end_date, all_timeframes)
        if not historical_data or primary_timeframe not in historical_data:
            self.logger.error(f"Failed to load sufficient historical data for {ticker}. Aborting.")
            return

        primary_data = historical_data[primary_timeframe]
        if len(primary_data) < min_lookback:
            self.logger.warning(f"Insufficient primary data ({len(primary_data)} rows) for {ticker}, need at least {min_lookback}. Proceeding with available data.")
            min_lookback = len(primary_data)  # Adjust min_lookback to available data

        self.logger.info(f"Starting training data generation for {ticker} into {output_file}")
        count = 0
        # Iterate through the primary timeframe index, starting after the lookback period
        for i in range(min_lookback - 1, len(primary_data)):
            current_timestamp = primary_data.index[i]
            self.logger.debug(f"Processing timestamp: {current_timestamp}")

            # Prepare data slices for point-in-time analysis
            data_up_to_t = {}
            valid_slice = True
            for tf, df in historical_data.items():
                # Filter data up to the current primary timestamp
                # Add a small buffer (e.g., 1 min) to include the exact timestamp for finer TFs
                slice_df = df[df.index <= current_timestamp + timedelta(minutes=1)] 
                if len(slice_df) < min_lookback // (2 if tf != primary_timeframe else 1): # Heuristic for min data on other TFs
                    self.logger.debug(f"Skipping {current_timestamp}: Insufficient data for {tf} ({len(slice_df)} rows)")
                    valid_slice = False
                    break
                data_up_to_t[tf] = slice_df
            
            if not valid_slice:
                continue

            try:
                # Point-in-time analysis (sliced data already prepared as 'data_up_to_t')
                analysis_results = self.vpa.analyze_ticker_at_point(ticker, data_up_to_t)
                
                if not analysis_results:
                    self.logger.warning(f"No analysis results for {ticker} at {current_timestamp}")
                    continue
                
                # Extract input features
                input_features = self._extract_input_features(analysis_results, current_timestamp, primary_timeframe)
                if not input_features:
                    self.logger.warning(f"Could not extract input features for {ticker} at {current_timestamp}")
                    continue

                # Generate explanation and output structure
                explanation = self._generate_explanation(input_features, analysis_results)
                
                # Get signal information from the primary timeframe
                signal_info = analysis_results["signals"].get(primary_timeframe, {}).get("candle", {})
                signal_type = signal_info.get("signal_type", "UNKNOWN")
                signal_strength = signal_info.get("signal_strength", "UNKNOWN")
                
                output_data = {
                    "signal": {
                        "type": signal_type,
                        "strength": signal_strength
                    },
                    "explanation": explanation,
                    "confidence_score": analysis_results.get("confidence_score", 0)
                }

                # Create the final JSON object for the line
                training_example = {
                    "input": input_features,
                    "output": output_data
                }

                # Append to JSONL file
                with open(output_file, 'a') as f:
                    json.dump(training_example, f)
                    f.write('\n')
                count += 1
                if count % 100 == 0:
                    self.logger.info(f"Generated {count} examples for {ticker}...")

            except Exception as e:
                self.logger.error(f"Error processing timestamp {current_timestamp} for {ticker}: {e}")
                self.logger.debug(traceback.format_exc())
                continue # Skip this data point

        self.logger.info(f"Finished generating {count} training examples for {ticker} in {output_file}")

# Example Usage
if __name__ == '__main__':
    try:
        # Initialize VPAFacade with default settings
        vpa_facade_instance = VPAFacade()
        
        generator = VPATrainingDataGenerator(vpa_facade_instance)
        
        # Define parameters
        ticker_to_generate = "AAPL"
        start_date_str = "2023-01-01"
        end_date_str = "2023-12-31"
        primary_tf = '1d'
        secondary_tfs = ['1h'] # Keep it simple for example
        
        generator.generate_training_data(
            ticker_to_generate, 
            start_date_str, 
            end_date_str, 
            primary_timeframe=primary_tf,
            other_timeframes=secondary_tfs
        )
    except Exception as main_e:
        print(f"Error in example usage: {main_e}")
        traceback.print_exc()

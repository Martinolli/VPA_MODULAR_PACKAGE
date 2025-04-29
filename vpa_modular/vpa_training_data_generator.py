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
# We might need a data loading mechanism, potentially reusing parts of the backtester's fetcher/validator
# For now, let's assume data is pre-loaded or fetched ad-hoc
from .vpa_data import YFinanceProvider # Example, might need adjustment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VPATrainingDataGenerator")

class VPATrainingDataGenerator:
    """Generates VPA training data for LLMs in JSONL format."""

    def __init__(self, vpa_facade: VPAFacade, output_dir="llm_training_data"):
        """
        Initialize the generator.

        Parameters:
        - vpa_facade: An instance of VPAFacade to perform analysis.
        - output_dir: Directory to save the generated JSONL files.
        """
        self.vpa = vpa_facade
        self.output_dir = output_dir
        self.data_provider = YFinanceProvider() # Example instantiation
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_historical_data(self, ticker, start_date, end_date, timeframes):
        """
        Loads historical data for the specified ticker and timeframes.
        (This is a placeholder - needs robust implementation, potentially reusing
         existing data fetching/loading logic from backtester or vpa_data).
        """
        logger.info(f"Loading historical data for {ticker} from {start_date} to {end_date}")
        all_data = {}
        try:
            for tf in timeframes:
                # Determine appropriate period based on start/end date for yfinance
                # This is simplified; yfinance period/interval logic needs care
                df = self.data_provider.get_data(ticker, interval=tf, start=start_date, end=end_date)
                if df is not None and not df.empty:
                    # Ensure standard column names
                    df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }, inplace=True)
                    # Ensure index is datetime
                    df.index = pd.to_datetime(df.index)
                    # Ensure timezone naive for consistency
                    if getattr(df.index, 'tz', None) is not None:
                         df.index = df.index.tz_convert(None)
                    all_data[tf] = df
                    logger.info(f"Loaded {len(df)} rows for {tf}")
                else:
                    logger.warning(f"No data loaded for timeframe {tf}")
            return all_data
        except Exception as e:
            logger.error(f"Error loading historical data for {ticker}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _extract_input_features(self, analysis_results, timestamp, primary_timeframe='1d', num_recent_candles=10):
        """
        Extracts features from VPA analysis results to form the LLM input.
        """
        if not analysis_results or primary_timeframe not in analysis_results.get("timeframe_analyses", {}):
            logger.warning(f"Analysis results missing or incomplete for {timestamp}")
            return None

        ticker = analysis_results.get("ticker", "UNKNOWN")
        tf_analysis = analysis_results["timeframe_analyses"][primary_timeframe]
        processed_data = tf_analysis.get("processed_data")
        
        if processed_data is None or processed_data.empty:
             logger.warning(f"Processed data missing for {primary_timeframe} at {timestamp}")
             return None

        # Find the index corresponding to the current timestamp or just before it
        try:
            # Ensure timestamp is timezone-naive if processed_data.index is
            if getattr(processed_data.index, 'tz', None) is None and getattr(timestamp, 'tz', None) is not None:
                 current_ts = timestamp.tz_convert(None)
            else:
                 current_ts = timestamp
            
            # Get the index location for the current timestamp
            idx_loc = processed_data.index.get_loc(current_ts, method='ffill')
            current_data_point = processed_data.iloc[idx_loc]
            actual_timestamp = processed_data.index[idx_loc]
        except KeyError:
            logger.warning(f"Timestamp {timestamp} not found in processed data index for {primary_timeframe}. Skipping.")
            return None
        except Exception as e:
            logger.error(f"Error finding data point for {timestamp}: {e}")
            return None

        # Extract current candle raw data (requires original data access)
        # This part needs refinement - how to access original OHLCV for the specific candle?
        # Let's assume 'price' and 'volume' are available in processed_data for now
        current_candle_data = {
            "open": current_data_point.get("open", None), # Placeholder - needs original data
            "high": current_data_point.get("high", None), # Placeholder
            "low": current_data_point.get("low", None),   # Placeholder
            "close": current_data_point.get("close", None), # Placeholder
            "volume": int(current_data_point.get("volume", 0)) # Placeholder
        }

        # Extract recent candles (requires original data access)
        recent_candles_data = []
        if idx_loc >= num_recent_candles -1:
             recent_slice = processed_data.iloc[idx_loc - num_recent_candles + 1 : idx_loc + 1]
             for ts, row in recent_slice.iterrows():
                  recent_candles_data.append({
                       "timestamp": ts.isoformat(),
                       "open": row.get("open", None), # Placeholder
                       "high": row.get("high", None), # Placeholder
                       "low": row.get("low", None),   # Placeholder
                       "close": row.get("close", None), # Placeholder
                       "volume": int(row.get("volume", 0)) # Placeholder
                  })

        # Extract VPA metrics
        vpa_metrics = {
            "relative_volume": current_data_point.get("volume_ratio", None),
            "volume_class": current_data_point.get("volume_class", "UNKNOWN"),
            "candle_spread_ratio": current_data_point.get("spread_ratio", None),
            "candle_class": current_data_point.get("candle_class", "UNKNOWN"),
            "upper_wick_ratio": current_data_point.get("upper_wick_ratio", None),
            "lower_wick_ratio": current_data_point.get("lower_wick_ratio", None)
        }

        # Extract Trend Analysis
        trend_analysis_data = tf_analysis.get("trend_analysis", {})
        trend_input = {
            "status": trend_analysis_data.get("status", "UNKNOWN"),
            "strength": trend_analysis_data.get("strength", "UNKNOWN"),
            "validation_signal": trend_analysis_data.get("validation_signal", "NONE")
        }

        # Extract Pattern Analysis
        pattern_analysis_data = tf_analysis.get("pattern_analysis", {})
        pattern_input = {}
        for p_name, p_data in pattern_analysis_data.items():
            pattern_input[p_name] = {
                "detected": p_data.get("detected", False),
                "details": p_data.get("details", "")
            }

        # Extract Support/Resistance
        sr_analysis_data = tf_analysis.get("support_resistance_analysis", {})
        sr_input = {
            "nearby_support": sr_analysis_data.get("support_levels", []),
            "nearby_resistance": sr_analysis_data.get("resistance_levels", [])
        }
        
        # Extract Multi-Timeframe Context (Simplified)
        mttf_context = {}
        for tf, tf_data in analysis_results.get("timeframe_analyses", {}).items():
            if tf != primary_timeframe:
                mttf_context[tf] = {
                    "trend_status": tf_data.get("trend_analysis", {}).get("status", "UNKNOWN"),
                    "signal_type": analysis_results.get("signal", {}).get("component_signals", {}).get(tf, {}).get("type", "UNKNOWN")
                }

        input_features = {
            "timestamp": actual_timestamp.isoformat(),
            "ticker": ticker,
            "primary_timeframe": primary_timeframe,
            "current_candle": current_candle_data,
            "recent_candles": recent_candles_data,
            "vpa_metrics": vpa_metrics,
            "trend_analysis": trend_input,
            "pattern_analysis": pattern_input,
            "support_resistance": sr_input,
            "multi_timeframe_context": mttf_context
        }
        return input_features

    def _generate_explanation(self, input_features, analysis_results):
        """
        Generates a comprehensive textual explanation based on the input features and analysis.
        """
        if not input_features or not analysis_results:
            return "Error: Insufficient data for explanation."

        signal_data = analysis_results.get("signal", {})
        signal_type = signal_data.get("type", "NO_ACTION")
        signal_strength = signal_data.get("strength", "WEAK")
        ticker = input_features["ticker"]
        timestamp = input_features["timestamp"]
        primary_tf = input_features["primary_timeframe"]
        trend = input_features["trend_analysis"]
        metrics = input_features["vpa_metrics"]
        patterns = input_features["pattern_analysis"]
        sr = input_features["support_resistance"]
        candle = input_features["current_candle"]

        explanation = f"VPA Analysis for {ticker} on {timestamp} ({primary_tf}):\n\n" \
                      f"**Signal:** {signal_type} ({signal_strength})\n\n" \
                      f"**Trend Context:** The primary trend is currently {trend['status']} ({trend['strength']}). "
        if trend['validation_signal'] != 'NONE':
            explanation += f"Recent price action shows a trend {trend['validation_signal']}. "
        explanation += "\n\n"

        explanation += f"**Current Candle Analysis:**\n" \
                       f"- Spread: {metrics['candle_class']} (Ratio: {metrics['candle_spread_ratio']:.2f} vs avg)\n" \
                       f"- Volume: {metrics['volume_class']} (Ratio: {metrics['relative_volume']:.2f} vs avg)\n"
        if metrics['upper_wick_ratio'] is not None and metrics['upper_wick_ratio'] > 0.5: # Example threshold
             explanation += f"- Significant upper wick present (Ratio: {metrics['upper_wick_ratio']:.2f}).\n"
        if metrics['lower_wick_ratio'] is not None and metrics['lower_wick_ratio'] > 0.5: # Example threshold
             explanation += f"- Significant lower wick present (Ratio: {metrics['lower_wick_ratio']:.2f}).\n"
        explanation += "\n"

        detected_patterns = [name for name, data in patterns.items() if data.get("detected")]
        if detected_patterns:
            explanation += f"**Detected Patterns:**\n"
            for name in detected_patterns:
                details = patterns[name].get("details", "No specific details.")
                explanation += f"- {name.replace('_', ' ').title()}: {details}\n"
            explanation += "\n"
        else:
            explanation += "**Detected Patterns:** None\n\n"

        explanation += f"**Support/Resistance:**\n"
        if sr['nearby_support']:
            explanation += f"- Nearby Support Levels: {', '.join([f'{s:.2f}' for s in sr['nearby_support']])}\n"
        if sr['nearby_resistance']:
            explanation += f"- Nearby Resistance Levels: {', '.join([f'{r:.2f}' for r in sr['nearby_resistance']])}\n"
        explanation += "\n"
        
        # Incorporate reasoning from signal details if available
        signal_details = signal_data.get("details", "")
        if signal_details:
             explanation += f"**Signal Reasoning:** {signal_details}\n\n"

        explanation += f"**Conclusion:** Based on the {trend['status']} trend, the {metrics['volume_class']} volume on the {metrics['candle_class']} spread candle" 
        if detected_patterns:
             explanation += f", and the presence of {', '.join(detected_patterns)}",
        explanation += f" a {signal_strength} {signal_type} signal is generated."

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
        if other_timeframes is None:
            other_timeframes = ['1h', '15m']
        all_timeframes = [primary_timeframe] + other_timeframes

        output_file = os.path.join(self.output_dir, f"{ticker}_vpa_training_data.jsonl")
        
        # Load all necessary historical data upfront
        historical_data = self._load_historical_data(ticker, start_date, end_date, all_timeframes)
        if not historical_data or primary_timeframe not in historical_data:
            logger.error(f"Failed to load sufficient historical data for {ticker}. Aborting.")
            return

        primary_data = historical_data[primary_timeframe]
        if len(primary_data) < min_lookback:
             logger.warning(f"Insufficient primary data ({len(primary_data)} rows) for {ticker}, need at least {min_lookback}. Aborting.")
             return

        logger.info(f"Starting training data generation for {ticker} into {output_file}")
        count = 0
        # Iterate through the primary timeframe index, starting after the lookback period
        for i in range(min_lookback -1, len(primary_data)):
            current_timestamp = primary_data.index[i]
            logger.debug(f"Processing timestamp: {current_timestamp}")

            # Prepare data slices for point-in-time analysis
            data_up_to_t = {}
            valid_slice = True
            for tf, df in historical_data.items():
                # Filter data up to the current primary timestamp
                # Add a small buffer (e.g., 1 min) to include the exact timestamp for finer TFs
                slice_df = df[df.index <= current_timestamp + timedelta(minutes=1)] 
                if len(slice_df) < min_lookback // (2 if tf != primary_timeframe else 1): # Heuristic for min data on other TFs
                     logger.debug(f"Skipping {current_timestamp}: Insufficient data for {tf} ({len(slice_df)} rows)")
                     valid_slice = False
                     break
                data_up_to_t[tf] = slice_df
            
            if not valid_slice:
                 continue

            # Perform point-in-time analysis using the facade
            # *** This requires VPAFacade to have a method like analyze_ticker_at_point ***
            # *** Let's assume it exists for now: analysis_results = self.vpa.analyze_ticker_at_point(ticker, data_up_to_t) ***
            # *** Mocking the call for structure - replace with actual call ***
            try:
                 # --- Replace this mock with actual call --- 
                 # analysis_results = self.vpa.analyze_ticker_at_point(ticker, data_up_to_t)
                 # Mock implementation detail: For now, let's call the standard analyze_ticker
                 # but it will use *all* data, not point-in-time. This needs fixing in VPAFacade.
                 logger.warning("Using full analysis instead of point-in-time. Facade needs update.")
                 analysis_results = self.vpa.analyze_ticker(ticker, config_override={'timeframes': all_timeframes})
                 # --- End Replace ---
                 
                 if not analysis_results:
                      logger.warning(f"Analysis failed for {ticker} at {current_timestamp}")
                      continue

                 # Extract input features
                 input_features = self._extract_input_features(analysis_results, current_timestamp, primary_timeframe)
                 if not input_features:
                      continue

                 # Generate explanation and output structure
                 explanation = self._generate_explanation(input_features, analysis_results)
                 output_data = {
                     "signal": analysis_results.get("signal", {}),
                     "explanation": explanation
                 }
                 # Remove potentially large/redundant details from signal for training data
                 if 'details' in output_data['signal']: del output_data['signal']['details']
                 if 'component_signals' in output_data['signal']: del output_data['signal']['component_signals']
                 if 'risk_assessment' in output_data['signal']: del output_data['signal']['risk_assessment']

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
                      logger.info(f"Generated {count} examples for {ticker}...")

            except Exception as e:
                logger.error(f"Error processing timestamp {current_timestamp} for {ticker}: {e}")
                logger.debug(traceback.format_exc())
                continue # Skip this data point

        logger.info(f"Finished generating {count} training examples for {ticker} in {output_file}")

# Example Usage (requires VPAFacade and potentially updated VPADataFetcher)
if __name__ == '__main__':
    # This example assumes VPAFacade is initialized correctly
    # You might need to adjust paths or configurations
    try:
        # Assuming VPAFacade can be initialized without args or with defaults
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
    except NameError:
         print("Skipping example usage: VPAFacade not defined or importable in this context.")
    except Exception as main_e:
         print(f"Error in example usage: {main_e}")
         traceback.print_exc()



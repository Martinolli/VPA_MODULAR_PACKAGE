"""
VPA Facade Module

This module provides a simplified API for VPA analysis.
"""

import os
import pandas as pd # Ensure pandas is imported for type hints if used
from .vpa_config import VPAConfig
from .vpa_data import PolygonIOProvider, MultiTimeframeProvider
from .vpa_processor import DataProcessor
from .vpa_analyzer import MultiTimeframeAnalyzer, PointInTimeAnalyzer, CandleAnalyzer, TrendAnalyzer, PatternRecognizer
from .vpa_signals import SignalGenerator, RiskAssessor
from .vpa_logger import VPALogger # Added VPALogger import

class VPAFacade:
    """Simplified API for VPA analysis"""
    
    def __init__(self, log_level="INFO", log_file=None, config=None, logger=None):
        """
        Initialize the VPA facade
        
        Parameters:
        - log_level: Logging level
        - log_file: Optional path to log file
        - config: Optional VPAConfig instance (if None, a new one will be created)
        - logger: Optional VPALogger instance (if None, a new one will be created)
        """
        # Initialize logger
        if logger is None:
            if log_file is None:
                log_file = "logs/vpa_analysis.log"
            self.logger = VPALogger(log_level, log_file)
        else:
            self.logger = logger
        
        # Initialize config
        self.config = config if config is not None else VPAConfig()

        # Initialize Data Provider
        self.data_provider = PolygonIOProvider()

        # Initialize processor
        self.processor = DataProcessor(self.config)
        
        # Initialize analyzers
        self.candle_analyzer = CandleAnalyzer(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config)
        self.pattern_recognizer = PatternRecognizer(self.config)
        self.analyzer = PointInTimeAnalyzer(self.config)

        # Initialize signal generator and risk assessor
        self.signal_generator = SignalGenerator(self.config)
        self.risk_assessor = RiskAssessor(self.config)

        self.multi_tf_provider = MultiTimeframeProvider(self.data_provider)
        self.multi_tf_analyzer = MultiTimeframeAnalyzer(self.config)
        
        # Initialize the PointInTimeAnalyzer for analyze_ticker_at_point method
        self.analyzer = PointInTimeAnalyzer(self.config, self.logger)

    def analyze_ticker(self, ticker, timeframes=None):
        self.logger.log_analysis_start(ticker, timeframes or [])
        """
        Analyze a ticker with VPA
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        - timeframes: Optional list of timeframe dictionaries with 'interval' and 'period' keys
        
        Returns:
        - Dictionary with analysis results
        """
        if timeframes is None:
            timeframes = self.config.get_timeframes()
        
        timeframe_data = self.multi_tf_provider.get_multi_timeframe_data(ticker, timeframes)
        
        if not timeframe_data:
            self.logger.error(f"No data fetched for {ticker} across specified timeframes. Aborting analysis.")
            return {
                "ticker": ticker,
                "error": "Failed to fetch market data for any specified timeframe.",
                "timeframe_analyses": {},
                "confirmations": [],
                "signal": {"type": "NO_SIGNAL", "strength": "NONE", "details": "Data unavailable"},
                "risk_assessment": {},
                "current_price": None
            }

        timeframe_analyses, confirmations = self.multi_tf_analyzer.analyze_multiple_timeframes(timeframe_data)
        signal = self.signal_generator.generate_signals(timeframe_analyses, confirmations)
        
        primary_tf_key = list(timeframe_analyses.keys())[0] 
        current_price = None
        support_resistance = {}

        if timeframe_analyses.get(primary_tf_key) and not timeframe_analyses[primary_tf_key]["processed_data"]["price"]["close"].empty:
            current_price = timeframe_analyses[primary_tf_key]["processed_data"]["price"]["close"].iloc[-1]
            support_resistance = timeframe_analyses[primary_tf_key]["support_resistance"]
        else:
            self.logger.warning(f"Could not determine current price or S/R for {ticker} from primary timeframe {primary_tf_key}.")
            signal["details"] += " (Current price unavailable for full risk assessment)"

        risk_assessment = self.risk_assessor.assess_trade_risk(signal, current_price, support_resistance)
        
        results = {
            "ticker": ticker,
            "timeframe_analyses": timeframe_analyses,
            "confirmations": confirmations,
            "signal": signal,
            "risk_assessment": risk_assessment,
            "current_price": current_price
        }
        self.logger.log_analysis_complete(ticker, signal)
        return results

    def analyze_ticker_at_point(self, ticker: str, sliced_data_by_timeframe: dict):
        """
        Analyze a ticker using only data up to a specific historical point in time.

        Parameters:
        - ticker: Stock symbol
        - sliced_data_by_timeframe: dict with {'1d': df, '1h': df, '15m': df} where df is a pandas DataFrame
                                     containing OHLCV data up to the point of analysis.

        Returns:
        - dict containing signal summary, score, and metadata, or None on error.
        """
        try:
            if self.analyzer is None:
                self.logger.error("CRITICAL: self.analyzer is not initialized in VPAFacade. analyze_ticker_at_point cannot function.")
                raise AttributeError("'VPAFacade' object has no attribute 'analyzer' that is properly initialized. This must be fixed by initializing self.analyzer in __init__.")

            processed_timeframe_data = {}
            if not sliced_data_by_timeframe:
                self.logger.error(f"No sliced data provided for {ticker} in analyze_ticker_at_point.")
                return None

            for tf, data_slice in sliced_data_by_timeframe.items():
                if data_slice.empty:
                    self.logger.warning(f"Empty data slice for timeframe {tf} for ticker {ticker}. Skipping timeframe.")
                    continue
                
                self.logger.info(f"Processing {ticker} data for point-in-time analysis, timeframe {tf}")
                self.logger.debug(f"Data slice shape: {data_slice.shape}, Columns: {data_slice.columns.tolist()}")
                self.logger.debug(f"Data slice index from {data_slice.index.min()} to {data_slice.index.max()}")

                start_date = data_slice.index.min()
                end_date = data_slice.index.max()

                historical_price_data, historical_volume_data = self.data_provider.get_data(
                    ticker,
                    interval=tf, 
                    start_date=start_date, 
                    end_date=end_date
                )

                price_to_process = None
                volume_to_process = None

                if historical_price_data.empty:
                    self.logger.warning(f"PolygonIOProvider returned no data for {ticker}, timeframe {tf}, range {start_date}-{end_date}. Using provided slice directly for preprocessing.")
                    if not all(col in data_slice.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                        self.logger.error(f"Data slice for {tf} is missing required OHLCV columns. Cannot proceed.")
                        continue
                    price_to_process = data_slice[['open', 'high', 'low', 'close']]
                    volume_to_process = data_slice['volume']
                else:
                    # Use axis parameter for alignment to avoid ValueError
                    aligned_price_data = historical_price_data.reindex(data_slice.index).dropna()
                    aligned_volume_data = historical_volume_data.reindex(data_slice.index).dropna()

                    if aligned_price_data.empty:
                        self.logger.warning(f"Alignment resulted in empty data for {ticker}, timeframe {tf}. Using provided slice directly.")
                        if not all(col in data_slice.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                            self.logger.error(f"Data slice for {tf} is missing required OHLCV columns. Cannot proceed.")
                            continue
                        price_to_process = data_slice[['open', 'high', 'low', 'close']]
                        volume_to_process = data_slice['volume']
                    else:
                        price_to_process = aligned_price_data
                        volume_to_process = aligned_volume_data
                
                if price_to_process is not None and volume_to_process is not None:
                    self.logger.debug(f"Data to process for {tf} - Price shape: {price_to_process.shape}, Volume shape: {volume_to_process.shape}")
                    processed_timeframe_data[tf] = self.processor.preprocess_data(price_to_process, volume_to_process)
            
            if not processed_timeframe_data:
                self.logger.error(f"No data could be processed for {ticker} in analyze_ticker_at_point after attempting all timeframes.")
                return None

            signals = self.analyzer.analyze_all(processed_timeframe_data)
            
            primary_tf_key_for_pit = self.config.get_primary_timeframe()
            if primary_tf_key_for_pit not in processed_timeframe_data:
                if processed_timeframe_data:
                    primary_tf_key_for_pit = list(processed_timeframe_data.keys())[0]
                else:
                    self.logger.error(f"No processed data available to determine primary timeframe for {ticker}.")
                    return None

            rr_info = self.analyzer.compute_risk_reward(processed_timeframe_data[primary_tf_key_for_pit], signals.get(primary_tf_key_for_pit, {}))
            volatility = self.analyzer.compute_volatility(processed_timeframe_data[primary_tf_key_for_pit])
            pattern_summary = signals[primary_tf_key_for_pit].get("pattern_summary", "")
            confidence_score = self.analyzer.compute_confidence_score(signals)

            return {
                "ticker": ticker,
                "timestamp": processed_timeframe_data[primary_tf_key_for_pit]["price"].index[-1].strftime("%Y-%m-%d %H:%M"),
                "signals": signals,
                "risk_reward": rr_info,
                "volatility": volatility,
                "pattern_summary": pattern_summary,
                "confidence_score": confidence_score,
            }

        except AttributeError as ae:
            # This specific check for 'self.analyzer' might be redundant if the initial check for self.analyzer is None is hit first.
            if 'self.analyzer' in str(ae) or "'NoneType' object has no attribute 'analyze_all'" in str(ae):
                self.logger.error(f"❌ Error in analyze_ticker_at_point for {ticker}: 'self.analyzer' is not defined or initialized in VPAFacade. This is a known issue from the original function structure. Details: {str(ae)}")
                self.logger.exception("Detailed error information:")
            else:
                self.logger.error(f"❌ AttributeError in analyze_ticker_at_point for {ticker}: {str(ae)}")
                self.logger.exception("Detailed error information:")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error analyzing {ticker} at point-in-time: {str(e)}")
            self.logger.exception("Detailed error information:")
            return None

    def get_signals(self, ticker, timeframes=None):
        results = self.analyze_ticker(ticker, timeframes)
        if "error" in results: 
            return results 
        signal_results = {
            "ticker": ticker,
            "signal": results["signal"],
            "risk_assessment": results["risk_assessment"],
            "current_price": results["current_price"]
        }
        return signal_results
    
    def explain_signal(self, ticker, timeframes=None):
        results = self.analyze_ticker(ticker, timeframes)
        if "error" in results: 
            return f"Could not generate explanation for {ticker} due to an error: {results['error']}"

        signal = results["signal"]
        evidence = signal.get("evidence", {}) 
        risk_assessment = results["risk_assessment"]
        current_price = results["current_price"]
        
        explanation = f"VPA Analysis for {ticker}:\n\n"
        explanation += f"Signal: {signal.get('type', 'N/A')} ({signal.get('strength', 'N/A')})\n"
        explanation += f"Details: {signal.get('details', 'N/A')}\n\n"
        explanation += "Supporting Evidence:\n"
        
        if evidence.get("candle_signals"):
            explanation += "- Candle Signals:\n"
            for candle_signal in evidence["candle_signals"]:
                explanation += f"  * {candle_signal.get('timeframe', 'N/A')}: {candle_signal.get('details', 'N/A')}\n"
        
        if evidence.get("trend_signals"):
            explanation += "- Trend Signals:\n"
            for trend_signal in evidence["trend_signals"]:
                explanation += f"  * {trend_signal.get('timeframe', 'N/A')}: {trend_signal.get('details', 'N/A')}\n"
        
        if evidence.get("pattern_signals"):
            explanation += "- Pattern Signals:\n"
            for pattern_signal in evidence["pattern_signals"]:
                explanation += f"  * {pattern_signal.get('timeframe', 'N/A')} - {pattern_signal.get('pattern', 'N/A')}: {pattern_signal.get('details', 'N/A')}\n"
        
        if evidence.get("timeframe_confirmations"):
            explanation += "- Timeframe Confirmations:\n"
            explanation += f"  * Confirmed in timeframes: {', '.join(evidence['timeframe_confirmations'])}\n"
        
        explanation += "\nRisk Assessment:\n"
        
        # Handle current price formatting
        if isinstance(current_price, (int, float)):
            current_price_str = f"${current_price:.2f}"
        else:
            current_price_str = "N/A"
        explanation += f"- Current Price: {current_price_str}\n"
        
        explanation += f"- Stop Loss: ${risk_assessment.get('stop_loss', 0):.2f}\n"
        explanation += f"- Take Profit: ${risk_assessment.get('take_profit', 0):.2f}\n"
        explanation += f"- Risk-Reward Ratio: {risk_assessment.get('risk_reward_ratio', 0):.2f}\n"
        explanation += f"- Recommended Position Size: {risk_assessment.get('position_size', 0):.2f} shares\n"
        
        self.logger.info(f"Generated explanation for {ticker}")
        self.logger.debug(explanation)
        return explanation
    
    def batch_analyze(self, tickers, timeframes=None):
        results = {}
        for ticker in tickers:
            try:
                ticker_results = self.get_signals(ticker, timeframes)
                results[ticker] = ticker_results
            except Exception as e:
                self.logger.error(f"Error in batch_analyze for {ticker}: {e}", exc_info=True)
                results[ticker] = {
                    "error": str(e)
                }
        return results
    
    def scan_for_signals(self, tickers, signal_type=None, signal_strength=None, timeframes=None):
        self.logger.info(f"Scanning {len(tickers)} tickers for signals. Type: {signal_type}, Strength: {signal_strength}")
        all_results = self.batch_analyze(tickers, timeframes)
        filtered_results = {}
        for ticker, result in all_results.items():
            if "error" in result:
                self.logger.warning(f"Error in results for {ticker}: {result['error']}")
                continue
            if signal_type and result.get("signal", {}).get("type") != signal_type:
                continue
            if signal_strength and result.get("signal", {}).get("strength") != signal_strength:
                continue
            filtered_results[ticker] = result
        self.logger.info(f"Found {len(filtered_results)} matching signals")
        return filtered_results

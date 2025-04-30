"""
VPA Facade Module

This module provides a simplified API for VPA analysis.
"""

import os
from .vpa_config import VPAConfig
from .vpa_data import YFinanceProvider, MultiTimeframeProvider
from .vpa_processor import DataProcessor
from .vpa_analyzer import CandleAnalyzer, TrendAnalyzer, PatternRecognizer, SupportResistanceAnalyzer, MultiTimeframeAnalyzer
from .vpa_signals import SignalGenerator, RiskAssessor
from .vpa_processor import DataProcessor
from .vpa_logger import VPALogger

class VPAFacade:
    """Simplified API for VPA analysis"""
    
    def __init__(self, config_file=None, log_level="INFO", log_file="vpa.log"):
        
        """
        Initialize the VPA facade
        
        Parameters:
        - config_file: Optional path to configuration file
        """
        self.config = VPAConfig(config_file)
        self.logger = VPALogger(log_level, log_file)
        self.data_provider = YFinanceProvider()
        self.multi_tf_provider = MultiTimeframeProvider(self.data_provider)
        self.processor = DataProcessor(self.config)
        self.multi_tf_analyzer = MultiTimeframeAnalyzer(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.risk_assessor = RiskAssessor(self.config)
    
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
        # Use default timeframes if not specified
        if timeframes is None:
            timeframes = self.config.get_timeframes()
        
        # Fetch data for multiple timeframes
        timeframe_data = self.multi_tf_provider.get_multi_timeframe_data(ticker, timeframes)
        
        # Analyze data across timeframes
        timeframe_analyses, confirmations = self.multi_tf_analyzer.analyze_multiple_timeframes(timeframe_data)
        
        # Generate signals
        signal = self.signal_generator.generate_signals(timeframe_analyses, confirmations)
        
        # Get current price
        primary_tf = list(timeframe_analyses.keys())[0]  # Assuming first timeframe is primary
        current_price = timeframe_analyses[primary_tf]["processed_data"]["price"]["close"].iloc[-1]
        
        # Get support/resistance levels
        support_resistance = timeframe_analyses[primary_tf]["support_resistance"]
        
        # Assess risk
        risk_assessment = self.risk_assessor.assess_trade_risk(signal, current_price, support_resistance)
        
        # Compile results
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
    
    def get_signals(self, ticker, timeframes=None):
        """
        Get trading signals for a ticker
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        - timeframes: Optional list of timeframe dictionaries with 'interval' and 'period' keys
        
        Returns:
        - Dictionary with signal and risk assessment
        """
        # Perform full analysis
        results = self.analyze_ticker(ticker, timeframes)
        
        # Extract signal and risk assessment
        signal_results = {
            "ticker": ticker,
            "signal": results["signal"],
            "risk_assessment": results["risk_assessment"],
            "current_price": results["current_price"]
        }
        
        return signal_results
    
    def explain_signal(self, ticker, timeframes=None):
        """
        Explain the reasoning behind a signal
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        - timeframes: Optional list of timeframe dictionaries with 'interval' and 'period' keys
        
        Returns:
        - String with natural language explanation
        """
        # Perform full analysis
        results = self.analyze_ticker(ticker, timeframes)
        
        # Extract signal and evidence
        signal = results["signal"]
        evidence = signal["evidence"]
        risk_assessment = results["risk_assessment"]
        current_price = results["current_price"]
        
        # Generate explanation
        explanation = f"VPA Analysis for {ticker}:\n\n"
        
        # Signal type and strength
        explanation += f"Signal: {signal['type']} ({signal['strength']})\n"
        explanation += f"Details: {signal['details']}\n\n"
        
        # Supporting evidence
        explanation += "Supporting Evidence:\n"
        
        if evidence["candle_signals"]:
            explanation += "- Candle Signals:\n"
            for candle_signal in evidence["candle_signals"]:
                explanation += f"  * {candle_signal['timeframe']}: {candle_signal['details']}\n"
        
        if evidence["trend_signals"]:
            explanation += "- Trend Signals:\n"
            for trend_signal in evidence["trend_signals"]:
                explanation += f"  * {trend_signal['timeframe']}: {trend_signal['details']}\n"
        
        if evidence["pattern_signals"]:
            explanation += "- Pattern Signals:\n"
            for pattern_signal in evidence["pattern_signals"]:
                explanation += f"  * {pattern_signal['timeframe']} - {pattern_signal['pattern']}: {pattern_signal['details']}\n"
        
        if evidence["timeframe_confirmations"]:
            explanation += "- Timeframe Confirmations:\n"
            explanation += f"  * Confirmed in timeframes: {', '.join(evidence['timeframe_confirmations'])}\n"
        
        # Risk assessment
        explanation += "\nRisk Assessment:\n"
        explanation += f"- Current Price: ${current_price:.2f}\n"
        explanation += f"- Stop Loss: ${risk_assessment['stop_loss']:.2f}\n"
        explanation += f"- Take Profit: ${risk_assessment['take_profit']:.2f}\n"
        explanation += f"- Risk-Reward Ratio: {risk_assessment['risk_reward_ratio']:.2f}\n"
        explanation += f"- Recommended Position Size: {risk_assessment['position_size']:.2f} shares\n"
        
        return explanation
    
    def batch_analyze(self, tickers, timeframes=None):
        """
        Analyze multiple tickers
        
        Parameters:
        - tickers: List of stock symbols
        - timeframes: Optional list of timeframe dictionaries with 'interval' and 'period' keys
        
        Returns:
        - Dictionary with analysis results for each ticker
        """
        results = {}
        
        for ticker in tickers:
            try:
                ticker_results = self.get_signals(ticker, timeframes)
                results[ticker] = ticker_results
            except Exception as e:
                results[ticker] = {
                    "error": str(e)
                }
        
        return results
    
    def scan_for_signals(self, tickers, signal_type=None, signal_strength=None, timeframes=None):
        """
        Scan multiple tickers for specific signals
        
        Parameters:
        - tickers: List of stock symbols
        - signal_type: Optional signal type to filter for ('BUY', 'SELL')
        - signal_strength: Optional signal strength to filter for ('STRONG', 'MODERATE')
        - timeframes: Optional list of timeframe dictionaries with 'interval' and 'period' keys
        
        Returns:
        - Dictionary with filtered signals
        """
        # Analyze all tickers
        all_results = self.batch_analyze(tickers, timeframes)
        
        # Filter results
        filtered_results = {}
        
        for ticker, result in all_results.items():
            if "error" in result:
                continue
            
            # Check if signal matches filters
            if signal_type and result["signal"]["type"] != signal_type:
                continue
            
            if signal_strength and result["signal"]["strength"] != signal_strength:
                continue
            
            # Include in filtered results
            filtered_results[ticker] = result
        
        return filtered_results
    def analyze_ticker_at_point(self, ticker: str, sliced_data_by_timeframe: dict):
        """
        Analyze a ticker using only data up to a specific historical point in time.

        Parameters:
        - ticker: Stock symbol
        - sliced_data_by_timeframe: dict with {'1d': df, '1h': df, '15m': df}

        Returns:
        - dict containing signal summary, score, and metadata
        """
        try:
            # 1. Preprocess data slices
            processed = self.processor.preprocess_all(sliced_data_by_timeframe)

            # 2. Run candle, volume, trend, and signal analysis (point-in-time)
            signals = self.analyzer.analyze_all(processed)

            # 3. (Optional) Compute R/R, SL/TP from recent context
            rr_info = self.analyzer.compute_risk_reward(processed["1d"], signals.get("1d", {}))

            # 4. Prepare analysis result
            return {
                "ticker": ticker,
                "timestamp": processed["1d"].index[-1].strftime("%Y-%m-%d %H:%M"),
                "signals": signals,
                "risk_reward": rr_info,
                "volatility": self.analyzer.compute_volatility(processed["1d"]),
                "pattern_summary": processed["1d"].get("pattern_summary", ""),
                "confidence_score": self.analyzer.compute_confidence_score(signals),
            }

        except Exception as e:
            self.logger.error(f"❌ Error analyzing {ticker} at point-in-time: {e}")
            return None


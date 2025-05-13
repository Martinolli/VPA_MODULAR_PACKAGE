"""
VPA Test Script

This script tests the modular VPA implementation.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import VPA modules
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_data import PolygonIOProvider, MultiTimeframeProvider
from vpa_modular.vpa_processor import DataProcessor
from vpa_modular.vpa_analyzer import CandleAnalyzer, TrendAnalyzer, PatternRecognizer, SupportResistanceAnalyzer, MultiTimeframeAnalyzer
from vpa_modular.vpa_signals import SignalGenerator, RiskAssessor
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_llm_interface import VPALLMInterface
from vpa_modular.vpa_utils import plot_candlestick, plot_vpa_signals, create_vpa_report
from vpa_modular.vpa_logger import VPALogger

# Set up logger
logger = VPALogger(log_level="INFO", log_file="/home/ubuntu/vpa_test.log")
logger.info("Starting VPA modular implementation test")

def test_config():
    """Test VPA configuration module"""
    logger.info("Testing VPA configuration module")
    
    # Create config
    config = VPAConfig()
    
    # Test getters
    volume_thresholds = config.get_volume_thresholds()
    candle_thresholds = config.get_candle_thresholds()
    trend_params = config.get_trend_parameters()
    pattern_params = config.get_pattern_parameters()
    signal_params = config.get_signal_parameters()
    risk_params = config.get_risk_parameters()
    timeframes = config.get_timeframes()
    
    # Verify config values
    # Verify config values
    assert volume_thresholds["very_high_threshold"] == 2.0, "Incorrect volume threshold"
    assert candle_thresholds["wide_threshold"] == 1.3, "Incorrect candle threshold"
    
    # Add assertions for other parameters
    assert "lookback_period" in trend_params, "Missing lookback_period in trend parameters"
    assert "min_trend_length" in trend_params, "Missing min_trend_length in trend parameters"
    
    assert "accumulation" in pattern_params, "Missing accumulation in pattern parameters"
    assert "distribution" in pattern_params, "Missing distribution in pattern parameters"
    
    assert "strong_signal_threshold" in signal_params, "Missing strong_signal_threshold in signal parameters"
    
    assert "default_risk_percent" in risk_params, "Missing default_risk_percent in risk parameters"
    assert "default_risk_reward" in risk_params, "Missing default_risk_reward in risk parameters"
    
    assert len(timeframes) > 0, "No timeframes defined"
    
    logger.info("VPA configuration module test passed")
    return True

def test_data_provider():
    """Test VPA data provider module"""
    logger.info("Testing VPA data provider module")
    
    # Create data provider
    provider = PolygonIOProvider()
    
    # Test data retrieval
    ticker = "AAPL"
    interval = "1d"
    period = "1mo"
    
    try:
        # Get data
        price_data, volume_data = provider.get_data(ticker, interval, period)
        
        # Verify data
        assert not price_data.empty, "Empty price data"
        assert not volume_data.empty, "Empty volume data"
        assert "open" in price_data.columns, "Missing open column"
        assert "high" in price_data.columns, "Missing high column"
        assert "low" in price_data.columns, "Missing low column"
        assert "close" in price_data.columns, "Missing close column"
        
        # Test multi-timeframe provider
        multi_provider = MultiTimeframeProvider(provider)
        timeframes = [
            {"interval": "1d", "period": "1mo"},
            {"interval": "1h", "period": "7d"}
        ]
        
        timeframe_data = multi_provider.get_multi_timeframe_data(ticker, timeframes)
        
        # Verify multi-timeframe data
        assert "1d" in timeframe_data, "Missing daily timeframe"
        assert "1h" in timeframe_data, "Missing hourly timeframe"
        
        logger.info("VPA data provider module test passed")
        return True
    
    except Exception as e:
        logger.error(f"Data provider test failed: {e}")
        return False

def test_processor():
    """Test VPA processor module"""
    logger.info("Testing VPA processor module")
    
    # Create data provider and processor
    provider = PolygonIOProvider()
    processor = DataProcessor()
    
    # Get data
    ticker = "AAPL"
    interval = "1d"
    period = "1mo"
    
    try:
        # Get data
        price_data, volume_data = provider.get_data(ticker, interval, period)
        
        # Process data
        processed_data = processor.preprocess_data(price_data, volume_data)
        
        # Verify processed data
        assert "price" in processed_data, "Missing price data"
        assert "volume" in processed_data, "Missing volume data"
        assert "spread" in processed_data, "Missing spread data"
        assert "body_percent" in processed_data, "Missing body_percent data"
        assert "upper_wick" in processed_data, "Missing upper_wick data"
        assert "lower_wick" in processed_data, "Missing lower_wick data"
        assert "avg_volume" in processed_data, "Missing avg_volume data"
        assert "volume_ratio" in processed_data, "Missing volume_ratio data"
        assert "volume_class" in processed_data, "Missing volume_class data"
        assert "candle_class" in processed_data, "Missing candle_class data"
        assert "price_direction" in processed_data, "Missing price_direction data"
        assert "volume_direction" in processed_data, "Missing volume_direction data"
        
        logger.info("VPA processor module test passed")
        return True
    
    except Exception as e:
        logger.error(f"Processor test failed: {e}")
        return False

def test_analyzer():
    """Test VPA analyzer module"""
    logger.info("Testing VPA analyzer module")
    
    # Create data provider, processor, and analyzers
    provider = PolygonIOProvider()
    processor = DataProcessor()
    candle_analyzer = CandleAnalyzer()
    trend_analyzer = TrendAnalyzer()
    pattern_recognizer = PatternRecognizer()
    sr_analyzer = SupportResistanceAnalyzer()
    
    # Get data
    ticker = "AAPL"
    interval = "1d"
    period = "1mo"
    
    try:
        # Get data
        price_data, volume_data = provider.get_data(ticker, interval, period)
        
        # Process data
        processed_data = processor.preprocess_data(price_data, volume_data)
        
        # Analyze candle
        current_idx = processed_data["price"].index[-1]
        candle_analysis = candle_analyzer.analyze_candle(current_idx, processed_data)
        
        # Verify candle analysis
        assert "candle_class" in candle_analysis, "Missing candle_class in analysis"
        assert "volume_class" in candle_analysis, "Missing volume_class in analysis"
        assert "is_up_candle" in candle_analysis, "Missing is_up_candle in analysis"
        assert "signal_type" in candle_analysis, "Missing signal_type in analysis"
        assert "signal_strength" in candle_analysis, "Missing signal_strength in analysis"
        
        # Analyze trend
        trend_analysis = trend_analyzer.analyze_trend(processed_data, current_idx)
        
        # Verify trend analysis
        assert "trend_direction" in trend_analysis, "Missing trend_direction in analysis"
        assert "volume_trend" in trend_analysis, "Missing volume_trend in analysis"
        assert "signal_type" in trend_analysis, "Missing signal_type in analysis"
        assert "signal_strength" in trend_analysis, "Missing signal_strength in analysis"
        
        # Identify patterns
        pattern_analysis = pattern_recognizer.identify_patterns(processed_data, current_idx)
        
        # Verify pattern analysis
        assert "accumulation" in pattern_analysis, "Missing accumulation in analysis"
        assert "distribution" in pattern_analysis, "Missing distribution in analysis"
        assert "testing" in pattern_analysis, "Missing testing in analysis"
        assert "buying_climax" in pattern_analysis, "Missing buying_climax in analysis"
        assert "selling_climax" in pattern_analysis, "Missing selling_climax in analysis"
        
        # Analyze support/resistance
        sr_analysis = sr_analyzer.analyze_support_resistance(processed_data)
        
        # Verify support/resistance analysis
        assert "support" in sr_analysis, "Missing support in analysis"
        assert "resistance" in sr_analysis, "Missing resistance in analysis"
        
        # Test multi-timeframe analyzer
        multi_analyzer = MultiTimeframeAnalyzer()
        
        # Create multi-timeframe data
        multi_provider = MultiTimeframeProvider(provider)
        timeframes = [
            {"interval": "1d", "period": "1mo"},
            {"interval": "1h", "period": "7d"}
        ]
        
        timeframe_data = multi_provider.get_multi_timeframe_data(ticker, timeframes)
        
        # Analyze multiple timeframes
        timeframe_analyses, confirmations = multi_analyzer.analyze_multiple_timeframes(timeframe_data)
        
        # Verify multi-timeframe analysis
        assert "1d" in timeframe_analyses, "Missing daily timeframe in analysis"
        assert "1h" in timeframe_analyses, "Missing hourly timeframe in analysis"
        assert "bullish" in confirmations, "Missing bullish confirmations"
        assert "bearish" in confirmations, "Missing bearish confirmations"
        
        logger.info("VPA analyzer module test passed")
        return True
    
    except Exception as e:
        logger.error(f"Analyzer test failed: {e}")
        return False

def test_signals():
    """Test VPA signals module"""
    logger.info("Testing VPA signals module")
    
    # Create data provider, processor, analyzers, and signal generator
    provider = PolygonIOProvider()
    processor = DataProcessor()
    multi_analyzer = MultiTimeframeAnalyzer()
    signal_generator = SignalGenerator()
    risk_assessor = RiskAssessor()
    
    # Get data
    ticker = "AAPL"
    
    try:
        # Create multi-timeframe data
        multi_provider = MultiTimeframeProvider(provider)
        timeframes = [
            {"interval": "1d", "period": "1mo"},
            {"interval": "1h", "period": "7d"}
        ]
        
        timeframe_data = multi_provider.get_multi_timeframe_data(ticker, timeframes)
        
        # Analyze multiple timeframes
        timeframe_analyses, confirmations = multi_analyzer.analyze_multiple_timeframes(timeframe_data)
        
        # Generate signals
        signal = signal_generator.generate_signals(timeframe_analyses, confirmations)
        
        # Verify signal
        assert "type" in signal, "Missing type in signal"
        assert "strength" in signal, "Missing strength in signal"
        assert "details" in signal, "Missing details in signal"
        assert "evidence" in signal, "Missing evidence in signal"
        
        # Assess risk
        primary_tf = list(timeframe_analyses.keys())[0]
        current_price = timeframe_analyses[primary_tf]["processed_data"]["price"]["close"].iloc[-1]
        support_resistance = timeframe_analyses[primary_tf]["support_resistance"]
        
        risk_assessment = risk_assessor.assess_trade_risk(signal, current_price, support_resistance)
        
        # Verify risk assessment
        assert "stop_loss" in risk_assessment, "Missing stop_loss in risk assessment"
        assert "take_profit" in risk_assessment, "Missing take_profit in risk assessment"
        assert "risk_reward_ratio" in risk_assessment, "Missing risk_reward_ratio in risk assessment"
        assert "position_size" in risk_assessment, "Missing position_size in risk assessment"
        
        logger.info("VPA signals module test passed")
        return True
    
    except Exception as e:
        logger.error(f"Signals test failed: {e}")
        return False

def test_facade():
    """Test VPA facade module"""
    logger.info("Testing VPA facade module")
    
    # Create facade
    facade = VPAFacade()
    
    # Test analyze_ticker
    ticker = "AAPL"
    
    try:
        # Analyze ticker
        results = facade.analyze_ticker(ticker)
        
        # Verify results
        assert "ticker" in results, "Missing ticker in results"
        assert "timeframe_analyses" in results, "Missing timeframe_analyses in results"
        assert "confirmations" in results, "Missing confirmations in results"
        assert "signal" in results, "Missing signal in results"
        assert "risk_assessment" in results, "Missing risk_assessment in results"
        assert "current_price" in results, "Missing current_price in results"
        
        # Test get_signals
        signal_results = facade.get_signals(ticker)
        
        # Verify signal results
        assert "ticker" in signal_results, "Missing ticker in signal results"
        assert "signal" in signal_results, "Missing signal in signal results"
        assert "risk_assessment" in signal_results, "Missing risk_assessment in signal results"
        
        # Test explain_signal
        explanation = facade.explain_signal(ticker)
        
        # Verify explanation
        assert isinstance(explanation, str), "Explanation is not a string"
        assert len(explanation) > 0, "Empty explanation"
        assert ticker in explanation, "Ticker not in explanation"
        
        # Test batch_analyze
        tickers = ["AAPL", "MSFT"]
        batch_results = facade.batch_analyze(tickers)
        
        # Verify batch results
        assert "AAPL" in batch_results, "Missing AAPL in batch results"
        assert "MSFT" in batch_results, "Missing MSFT in batch results"
        
        logger.info("VPA facade module test passed")
        return True
    
    except Exception as e:
        logger.error(f"Facade test failed: {e}")
        return False

def test_llm_interface():
    """Test VPA LLM interface module"""
    logger.info("Testing VPA LLM interface module")
    
    # Create LLM interface
    llm_interface = VPALLMInterface()
    
    try:
        # Test process_query
        query = "What is accumulation in VPA?"
        response = llm_interface.process_query(query)
        
        # Verify response
        assert isinstance(response, str), "Response is not a string"
        assert len(response) > 0, "Empty response"
        assert "accumulation" in response.lower(), "Response doesn't contain query topic"
        
        # Test get_ticker_analysis
        ticker = "AAPL"
        analysis = llm_interface.get_ticker_analysis(ticker)
        
        # Verify analysis
        assert "ticker" in analysis, "Missing ticker in analysis"
        assert "current_price" in analysis, "Missing current_price in analysis"
        assert "signal" in analysis, "Missing signal in analysis"
        assert "risk" in analysis, "Missing risk in analysis"
        assert "patterns" in analysis, "Missing patterns in analysis"
        assert "explanation" in analysis, "Missing explanation in analysis"
        
        # Test explain_vpa_concept
        concept = "vpa_overview"
        explanation = llm_interface.explain_vpa_concept(concept)
        
        # Verify explanation
        assert isinstance(explanation, str), "Explanation is not a string"
        assert len(explanation) > 0, "Empty explanation"
        
        # Test suggest_parameters
        parameters = llm_interface.suggest_parameters("AAPL", "day_trading")
        
        # Verify parameters
        assert "parameters" in parameters, "Missing parameters in suggestion"
        assert "explanation" in parameters, "Missing explanation in suggestion"
        
        logger.info("VPA LLM interface module test passed")
        return True
    
    except Exception as e:
        logger.error(f"LLM interface test failed: {e}")
        return False

def test_utils():
    """Test VPA utils module"""
    logger.info("Testing VPA utils module")
    
    # Create facade for data
    facade = VPAFacade()
    
    # Get analysis results
    ticker = "AAPL"
    
    try:
        # Analyze ticker
        results = facade.analyze_ticker(ticker)
        
        # Get data for visualization
        primary_tf = list(results["timeframe_analyses"].keys())[0]
        price_data = results["timeframe_analyses"][primary_tf]["processed_data"]["price"]
        volume_data = results["timeframe_analyses"][primary_tf]["processed_data"]["volume"]
        processed_data = results["timeframe_analyses"][primary_tf]["processed_data"]
        
        # Test plot_candlestick
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_candlestick(ax, price_data, volume_data, title=f"{ticker} - Candlestick Chart")
        plt.close(fig)
        
        # Test plot_vpa_signals
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_candlestick(ax, price_data, volume_data, title=f"{ticker} - VPA Signals")
        plot_vpa_signals(ax, price_data, processed_data, results["signal"], 
                        results["timeframe_analyses"][primary_tf]["support_resistance"])
        plt.close(fig)
        
        # Test create_vpa_report
        output_dir = "/home/ubuntu/vpa_test_reports"
        report_files = create_vpa_report(results, output_dir)
        
        # Verify report files
        assert "price_chart" in report_files, "Missing price chart in report"
        assert "multi_timeframe" in report_files, "Missing multi-timeframe chart in report"
        assert "patterns" in report_files, "Missing patterns chart in report"
        assert "text_report" in report_files, "Missing text report in report"
        
        # Check if files exist
        for file_path in report_files.values():
            assert os.path.exists(file_path), f"Report file {file_path} does not exist"
        
        logger.info("VPA utils module test passed")
        return True
    
    except Exception as e:
        logger.error(f"Utils test failed: {e}")
        return False

def run_all_tests():
    """Run all VPA module tests"""
    logger.info("Running all VPA module tests")
    
    # Run tests
    config_test = test_config()
    data_test = test_data_provider()
    processor_test = test_processor()
    analyzer_test = test_analyzer()
    signals_test = test_signals()
    facade_test = test_facade()
    llm_interface_test = test_llm_interface()
    utils_test = test_utils()
    
    # Print results
    print("\nVPA Modular Implementation Test Results:")
    print("=" * 50)
    print(f"Configuration Module: {'PASSED' if config_test else 'FAILED'}")
    print(f"Data Provider Module: {'PASSED' if data_test else 'FAILED'}")
    print(f"Processor Module: {'PASSED' if processor_test else 'FAILED'}")
    print(f"Analyzer Module: {'PASSED' if analyzer_test else 'FAILED'}")
    print(f"Signals Module: {'PASSED' if signals_test else 'FAILED'}")
    print(f"Facade Module: {'PASSED' if facade_test else 'FAILED'}")
    print(f"LLM Interface Module: {'PASSED' if llm_interface_test else 'FAILED'}")
    print(f"Utils Module: {'PASSED' if utils_test else 'FAILED'}")
    print("=" * 50)
    
    # Overall result
    all_passed = all([config_test, data_test, processor_test, analyzer_test, 
                     signals_test, facade_test, llm_interface_test, utils_test])
    
    print(f"\nOverall Result: {'PASSED' if all_passed else 'FAILED'}")
    
    logger.info(f"All VPA module tests completed. Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()

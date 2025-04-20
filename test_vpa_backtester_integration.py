"""
Test script for the VPA backtester integration with data fetching and validation.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import traceback

# Import our modules
from vpa_modular.vpa_data_fetcher import VPADataFetcher
from vpa_modular.vpa_data_validator import VPADataValidator
from vpa_modular.vpa_backtester_integration import VPABacktesterIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VPABacktesterTest')

def test_data_fetching():
    """Test data fetching functionality"""
    logger.info("Testing data fetching...")
    
    # Create data fetcher
    fetcher = VPADataFetcher()
    
    # Test tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Test timeframes
    timeframes = ["1d", "1h", "15m"]
    
    # Fetch data for each ticker
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}...")
        results = fetcher.fetch_data(ticker, timeframes)
        
        logger.info(f"Fetch results for {ticker}:")
        for timeframe, result in results.items():
            status = result.get("status", "unknown")
            if status == "success":
                logger.info(f"  {timeframe}: {status} - {result.get('rows', 0)} rows from {result.get('start_date', 'N/A')} to {result.get('end_date', 'N/A')}")
            else:
                logger.info(f"  {timeframe}: {status} - {result.get('message', 'No message')}")
    
    logger.info("Data fetching test completed")
    return True

def test_data_validation():
    """Test data validation functionality"""
    logger.info("Testing data validation...")
    
    # Create data validator
    validator = VPADataValidator()
    
    # Test tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Test date range
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Validate data for each ticker
    for ticker in tickers:
        logger.info(f"Validating data for {ticker}...")
        validation_results = validator.validate_ticker(ticker, start_date, end_date)
        
        logger.info(f"Validation results for {ticker}:")
        for timeframe, result in validation_results["timeframes"].items():
            status = result.get("status", "unknown")
            if status == "valid":
                logger.info(f"  {timeframe}: {status} - {result.get('rows', 0)} rows from {result.get('data_start', 'N/A')} to {result.get('data_end', 'N/A')}")
            elif status == "issues_found":
                logger.info(f"  {timeframe}: {status} - {len(result.get('issues', []))} issues found")
                for issue in result.get("issues", []):
                    logger.info(f"    - {issue.get('type', 'unknown')}: {issue.get('message', 'No message')}")
            else:
                logger.info(f"  {timeframe}: {status} - {result.get('message', 'No message')}")
        
        # Check backtesting readiness
        readiness = validator.check_backtesting_readiness(ticker, start_date, end_date)
        logger.info(f"Backtesting readiness for {ticker}: {readiness['is_ready']}")
        if not readiness["is_ready"]:
            for issue in readiness["issues"]:
                logger.info(f"  - {issue.get('timeframe', 'unknown')}: {issue.get('message', 'No message')}")
    
    logger.info("Data validation test completed")
    return True

def test_backtester_integration():
    """Test backtester integration functionality"""
    logger.info("Testing backtester integration...")
    
    # Create components
    fetcher = VPADataFetcher()
    validator = VPADataValidator()
    integration = VPABacktesterIntegration(fetcher, validator)
    
    # Test tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Test date range - use a shorter range for testing
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # Prepare data for each ticker
    for ticker in tickers:
        logger.info(f"Preparing data for {ticker}...")
        preparation_results = integration.prepare_data_for_backtest(
            ticker, start_date, end_date, force_refresh=False
        )
        
        logger.info(f"Preparation results for {ticker}: {preparation_results['is_ready']}")
        if not preparation_results["is_ready"]:
            for issue in preparation_results["readiness"]["issues"]:
                logger.info(f"  - {issue.get('timeframe', 'unknown')}: {issue.get('message', 'No message')}")
        
        # If data is ready, run a backtest
        if preparation_results["is_ready"]:
            logger.info(f"Running backtest for {ticker}...")
            try:
                results = integration.run_backtest(
                    ticker, 
                    start_date, 
                    end_date,
                    prepare_data=False,  # Skip preparation since we just did it
                    initial_capital=100000.0,
                    commission_rate=0.001,
                    slippage_percent=0.001,
                    risk_per_trade=0.02
                )
                
                if results:
                    logger.info(f"Backtest completed for {ticker}")
                    if "metrics" in results:
                        metrics = results["metrics"]
                        logger.info(f"  - Total return: {metrics.get('total_return', 0):.2%}")
                        logger.info(f"  - Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
                        logger.info(f"  - Win rate: {metrics.get('win_rate', 0):.2%}")
                        logger.info(f"  - Profit factor: {metrics.get('profit_factor', 0):.2f}")
                        logger.info(f"  - Total trades: {metrics.get('total_trades', 0)}")
                else:
                    logger.warning(f"Backtest failed for {ticker}")
            except Exception as e:
                logger.error(f"Error running backtest for {ticker}: {str(e)}")
                logger.debug(traceback.format_exc())
    
    logger.info("Backtester integration test completed")
    return True

def test_walk_forward_analysis():
    """Test walk-forward analysis functionality"""
    logger.info("Testing walk-forward analysis...")
    
    # Create components
    fetcher = VPADataFetcher()
    validator = VPADataValidator()
    integration = VPABacktesterIntegration(fetcher, validator)
    
    # Test ticker - use just one ticker for walk-forward analysis
    ticker = "AAPL"
    
    # Test date range - use a longer range for walk-forward analysis
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Run walk-forward analysis
    logger.info(f"Running walk-forward analysis for {ticker}...")
    try:
        results = integration.run_walk_forward_analysis(
            ticker, 
            start_date, 
            end_date,
            window_size=60,  # 60-day windows
            step_size=30,    # 30-day steps
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_percent=0.001,
            risk_per_trade=0.02
        )
        
        if results and results.get("status") == "success":
            logger.info(f"Walk-forward analysis completed for {ticker}")
            aggregated = results.get("aggregated_results", {})
            logger.info(f"  - Average return: {aggregated.get('avg_return', 0):.2%}")
            logger.info(f"  - Return std dev: {aggregated.get('std_return', 0):.2%}")
            logger.info(f"  - Min return: {aggregated.get('min_return', 0):.2%}")
            logger.info(f"  - Max return: {aggregated.get('max_return', 0):.2%}")
            logger.info(f"  - Average drawdown: {aggregated.get('avg_drawdown', 0):.2%}")
            logger.info(f"  - Consistency: {aggregated.get('consistency', 0):.2%}")
        else:
            logger.warning(f"Walk-forward analysis failed for {ticker}: {results.get('message', 'No message')}")
    except Exception as e:
        logger.error(f"Error running walk-forward analysis for {ticker}: {str(e)}")
        logger.debug(traceback.format_exc())
    
    logger.info("Walk-forward analysis test completed")
    return True

def run_all_tests():
    """Run all tests"""
    logger.info("Running all tests...")
    
    test_results = {
        "data_fetching": test_data_fetching(),
        "data_validation": test_data_validation(),
        "backtester_integration": test_backtester_integration(),
        "walk_forward_analysis": test_walk_forward_analysis()
    }
    
    logger.info("All tests completed")
    logger.info(f"Test results: {json.dumps(test_results, indent=2)}")
    
    return test_results

if __name__ == "__main__":
    run_all_tests()

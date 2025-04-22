#!/usr/bin/env python3
"""
Test script for the VPA backtesting framework.
This script validates the functionality of the VPA backtesting framework
with sample tickers and different testing scenarios.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add the parent directory to the path to import VPA modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import VPA modules
from vpa_modular.vpa_config import VPAConfig
from vpa_backtest.vpa_backtester import VPABacktester
from vpa_modular.vpa_logger import VPALogger

# Set up logger
logger = VPALogger()
logger.logger.setLevel(logging.INFO)

# Create output directory for test results
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_basic_backtest():
    """
    Test basic backtesting functionality with a single ticker.
    """
    logger.info("=== Testing Basic Backtest ===")
    
    # Create backtester with default parameters
    backtester = VPABacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000.0
    )
    
    # Run backtest on a single ticker
    ticker = "AAPL"
    results = backtester.run_backtest(ticker)
    
    # Validate results
    assert results is not None, "Backtest results should not be None"
    assert 'performance' in results, "Results should contain performance metrics"
    assert 'trades' in results, "Results should contain trade data"
    assert 'equity_curve' in results, "Results should contain equity curve data"
    
    # Create report
    report_dir = os.path.join(OUTPUT_DIR, "basic_backtest")
    report_files = backtester.create_backtest_report(report_dir)
    
    logger.info(f"Basic backtest completed for {ticker}")
    logger.info(f"Performance: {results['performance']['total_return_pct']:.2f}% return, {results['performance']['sharpe_ratio']:.2f} Sharpe ratio")
    logger.info(f"Report available at: {report_files.get('html')}")
    
    return results

def test_multi_ticker_backtest():
    """
    Test backtesting with multiple tickers.
    """
    logger.info("=== Testing Multi-Ticker Backtest ===")
    
    # Create backtester
    backtester = VPABacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000.0,
        max_positions=3  # Allow up to 3 positions simultaneously
    )
    
    # Run backtest on multiple tickers
    tickers = ["AAPL", "MSFT"]
    results = backtester.run_backtest(tickers)
    
    # Validate results
    assert results is not None, "Backtest results should not be None"
    assert 'performance' in results, "Results should contain performance metrics"
    assert 'trades' in results, "Results should contain trade data"
    
    # Create report
    report_dir = os.path.join(OUTPUT_DIR, "multi_ticker_backtest")
    report_files = backtester.create_backtest_report(report_dir)
    
    logger.info(f"Multi-ticker backtest completed for {tickers}")
    logger.info(f"Performance: {results['performance']['total_return_pct']:.2f}% return, {results['performance']['sharpe_ratio']:.2f} Sharpe ratio")
    logger.info(f"Report available at: {report_files.get('html')}")
    
    return results

def test_benchmark_comparison():
    """
    Test benchmark comparison functionality.
    """
    logger.info("=== Testing Benchmark Comparison ===")
    
    # Create backtester with benchmark
    backtester = VPABacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000.0,
        benchmark_ticker="SPY"
    )
    
    # Run backtest with benchmark comparison
    ticker = "AAPL"
    results = backtester.run_backtest(ticker, include_benchmark=True)
    
    # Validate benchmark results
    assert results is not None, "Backtest results should not be None"
    assert 'benchmark_performance' in results, "Results should contain benchmark performance"
    assert 'benchmark_ticker' in results, "Results should contain benchmark ticker"
    
    # Create report
    report_dir = os.path.join(OUTPUT_DIR, "benchmark_comparison")
    report_files = backtester.create_backtest_report(report_dir)
    
    # Plot equity curve with benchmark
    fig = backtester.plot_equity_curve(include_benchmark=True)
    plt.savefig(os.path.join(report_dir, "equity_with_benchmark.png"))
    plt.close(fig)
    
    # Plot drawdown with benchmark
    fig = backtester.plot_drawdown(include_benchmark=True)
    plt.savefig(os.path.join(report_dir, "drawdown_with_benchmark.png"))
    plt.close(fig)
    
    logger.info(f"Benchmark comparison completed for {ticker} vs {results['benchmark_ticker']}")
    if 'benchmark_performance' in results and results['benchmark_performance']:
        benchmark = results['benchmark_performance']
        logger.info(f"Strategy return: {results['performance']['total_return_pct']:.2f}%, Benchmark return: {benchmark['total_return_pct']:.2f}%")
        if 'alpha' in benchmark:
            logger.info(f"Alpha: {benchmark['alpha_pct']:.2f}%, Beta: {benchmark['beta']:.2f}")
    logger.info(f"Report available at: {report_files.get('html')}")
    
    return results

def test_walk_forward_analysis():
    """
    Test walk-forward analysis functionality.
    """
    logger.info("=== Testing Walk-Forward Analysis ===")
    
    # Create backtester
    backtester = VPABacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000.0
    )
    
    # Run walk-forward analysis
    ticker = "MSFT"
    walk_forward_results = backtester.run_walk_forward_analysis(
        ticker=ticker,
        window_size=180,  # 6-month windows
        step_size=60,     # 2-month steps
        lookback_days=90  # 3-month lookback for analysis
    )
    
    # Validate results
    assert walk_forward_results is not None, "Walk-forward results should not be None"
    assert 'window_results' in walk_forward_results, "Results should contain window results"
    assert 'aggregate_stats' in walk_forward_results, "Results should contain aggregate statistics"
    
    # Create output directory
    report_dir = os.path.join(OUTPUT_DIR, "walk_forward_analysis")
    os.makedirs(report_dir, exist_ok=True)
    
    # Plot walk-forward results
    fig = backtester.plot_walk_forward_results(walk_forward_results)
    plt.savefig(os.path.join(report_dir, "walk_forward_results.png"))
    plt.close(fig)
    
    # Save results to file
    results_file = os.path.join(report_dir, "walk_forward_results.json")
    import json
    with open(results_file, 'w') as f:
        # Convert non-serializable objects to strings
        serializable_results = {
            'ticker': walk_forward_results['ticker'],
            'windows': [
                {
                    'window_number': w['window_number'],
                    'start_date': w['start_date'].strftime('%Y-%m-%d'),
                    'end_date': w['end_date'].strftime('%Y-%m-%d')
                }
                for w in walk_forward_results['windows']
            ],
            'metrics_by_window': walk_forward_results['metrics_by_window'],
            'aggregate_stats': walk_forward_results['aggregate_stats'],
            'total_trades': walk_forward_results['total_trades'],
            'profitable_windows': walk_forward_results['profitable_windows'],
            'window_consistency': walk_forward_results['window_consistency'],
            'window_consistency_pct': walk_forward_results['window_consistency_pct']
        }
        json.dump(serializable_results, f, indent=4)
    
    logger.info(f"Walk-forward analysis completed for {ticker}")
    logger.info(f"Window consistency: {walk_forward_results['window_consistency_pct']:.2f}%")
    logger.info(f"Results saved to: {results_file}")
    
    return walk_forward_results

def test_monte_carlo_simulation():
    """
    Test Monte Carlo simulation functionality.
    """
    logger.info("=== Testing Monte Carlo Simulation ===")
    
    # First run a backtest to get trade data
    backtester = VPABacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000.0
    )
    
    # Run backtest
    ticker = "NVDA"
    backtest_results = backtester.run_backtest(ticker)
    
    # Validate we have enough trades for Monte Carlo
    if not backtest_results or 'trades' not in backtest_results or len(backtest_results['trades']) < 5:
        logger.warning(f"Not enough trades for Monte Carlo simulation: {len(backtest_results.get('trades', []))}")
        return None
    
    # Run Monte Carlo simulation
    monte_carlo_results = backtester.run_monte_carlo_simulation(
        num_simulations=500,
        confidence_level=0.95,
        save_path=os.path.join(OUTPUT_DIR, "monte_carlo", "monte_carlo_simulation.png")
    )
    
    # Validate results
    assert monte_carlo_results is not None, "Monte Carlo results should not be None"
    assert 'median_final_capital' in monte_carlo_results, "Results should contain median final capital"
    assert 'prob_profit_pct' in monte_carlo_results, "Results should contain probability of profit"
    
    # Save results to file
    report_dir = os.path.join(OUTPUT_DIR, "monte_carlo")
    os.makedirs(report_dir, exist_ok=True)
    
    results_file = os.path.join(report_dir, "monte_carlo_results.json")
    import json
    with open(results_file, 'w') as f:
        json.dump(monte_carlo_results, f, indent=4)
    
    logger.info(f"Monte Carlo simulation completed for {ticker}")
    logger.info(f"Median return: {monte_carlo_results['median_return_pct']:.2f}%")
    logger.info(f"Probability of profit: {monte_carlo_results['prob_profit_pct']:.2f}%")
    logger.info(f"Results saved to: {results_file}")
    
    return monte_carlo_results

def test_parameter_optimization():
    """
    Test parameter optimization functionality.
    """
    logger.info("=== Testing Parameter Optimization ===")
    
    # Create backtester
    backtester = VPABacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000.0
    )
    
    # Define parameter grid
    param_grid = {
        'risk_per_trade': [0.01, 0.02, 0.03],
        'vpa_volume_thresholds.very_high_threshold': [1.8, 2.0, 2.2],
        'vpa_signal_parameters.strong_signal_threshold': [0.6, 0.7, 0.8]
    }
    
    # Run parameter optimization
    ticker = "AAPL"
    optimization_results = backtester.optimize_parameters(
        ticker=ticker,
        param_grid=param_grid,
        metric='sharpe_ratio'
    )
    
    # Validate results
    assert optimization_results is not None, "Optimization results should not be None"
    assert 'best_params' in optimization_results, "Results should contain best parameters"
    assert 'best_value' in optimization_results, "Results should contain best value"
    
    # Save results to file
    report_dir = os.path.join(OUTPUT_DIR, "parameter_optimization")
    os.makedirs(report_dir, exist_ok=True)
    
    results_file = os.path.join(report_dir, "optimization_results.json")
    import json
    with open(results_file, 'w') as f:
        # Convert non-serializable objects to strings
        serializable_results = {
            'best_params': optimization_results['best_params'],
            'best_value': float(optimization_results['best_value']),
            'all_results': [
                {
                    'params': r['params'],
                    'value': float(r['value']),
                    'performance': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                   for k, v in r['performance'].items()}
                }
                for r in optimization_results['all_results']
            ]
        }
        json.dump(serializable_results, f, indent=4)
    
    # Create visualization of parameter optimization results
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Parameter values vs metric
    param_values = []
    metric_values = []
    param_names = []
    
    for result in optimization_results['all_results']:
        for param_name, param_value in result['params'].items():
            param_values.append(param_value)
            metric_values.append(result['value'])
            param_names.append(param_name)
    
    # Convert to categorical for plotting
    param_categories = pd.Categorical(param_names)
    param_codes = param_categories.codes
    
    scatter = axs[0].scatter(param_values, metric_values, c=param_codes, cmap='viridis', alpha=0.7)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.viridis(param_categories.categories.get_loc(cat) / len(param_categories.categories)), 
                                 markersize=10, label=cat)
                      for cat in param_categories.categories]
    
    axs[0].legend(handles=legend_elements, title="Parameters")
    axs[0].set_title('Parameter Values vs Sharpe Ratio')
    axs[0].set_xlabel('Parameter Value')
    axs[0].set_ylabel('Sharpe Ratio')
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Top 10 parameter combinations
    top_results = sorted(optimization_results['all_results'], key=lambda x: x['value'], reverse=True)[:10]
    
    combination_labels = [f"Combo {i+1}" for i in range(len(top_results))]
    sharpe_values = [r['value'] for r in top_results]
    
    axs[1].bar(combination_labels, sharpe_values, color='skyblue')
    axs[1].set_title('Top 10 Parameter Combinations')
    axs[1].set_xlabel('Combination')
    axs[1].set_ylabel('Sharpe Ratio')
    axs[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "parameter_optimization.png"))
    plt.close(fig)
    
    logger.info(f"Parameter optimization completed for {ticker}")
    logger.info(f"Best parameters: {optimization_results['best_params']}")
    logger.info(f"Best Sharpe ratio: {optimization_results['best_value']:.2f}")
    logger.info(f"Results saved to: {results_file}")
    
    return optimization_results

def run_all_tests():
    """
    Run all tests and return a summary of results.
    """
    logger.info("Starting VPA Backtester Tests")
    
    test_results = {}
    
    # Run tests
    try:
        test_results['basic_backtest'] = test_basic_backtest()
        logger.info("Basic backtest test passed")
    except Exception as e:
        logger.error(f"Basic backtest test failed: {str(e)}")
        test_results['basic_backtest'] = None
    
    try:
        test_results['multi_ticker_backtest'] = test_multi_ticker_backtest()
        logger.info("Multi-ticker backtest test passed")
    except Exception as e:
        logger.error(f"Multi-ticker backtest test failed: {str(e)}")
        test_results['multi_ticker_backtest'] = None
    
    try:
        test_results['benchmark_comparison'] = test_benchmark_comparison()
        logger.info("Benchmark comparison test passed")
    except Exception as e:
        logger.error(f"Benchmark comparison test failed: {str(e)}")
        test_results['benchmark_comparison'] = None
    
    try:
        test_results['walk_forward_analysis'] = test_walk_forward_analysis()
        logger.info("Walk-forward analysis test passed")
    except Exception as e:
        logger.error(f"Walk-forward analysis test failed: {str(e)}")
        test_results['walk_forward_analysis'] = None
    
    try:
        test_results['monte_carlo_simulation'] = test_monte_carlo_simulation()
        logger.info("Monte Carlo simulation test passed")
    except Exception as e:
        logger.error(f"Monte Carlo simulation test failed: {str(e)}")
        test_results['monte_carlo_simulation'] = None
    
    try:
        test_results['parameter_optimization'] = test_parameter_optimization()
        logger.info("Parameter optimization test passed")
    except Exception as e:
        logger.error(f"Parameter optimization test failed: {str(e)}")
        test_results['parameter_optimization'] = None
    
    # Create summary report
    summary_file = os.path.join(OUTPUT_DIR, "test_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("VPA Backtester Test Summary\n")
        f.write("==========================\n\n")
        
        for test_name, result in test_results.items():
            status = "PASSED" if result is not None else "FAILED"
            f.write(f"{test_name}: {status}\n")
            
            if result is not None and 'performance' in result:
                f.write(f"  Return: {result['performance']['total_return_pct']:.2f}%\n")
                f.write(f"  Sharpe: {result['performance']['sharpe_ratio']:.2f}\n")
                f.write(f"  Trades: {result['performance']['num_trades']}\n")
                f.write(f"  Win Rate: {result['performance']['win_rate_pct']:.2f}%\n\n")
    
    logger.info(f"Test summary saved to: {summary_file}")
    logger.info("All tests completed")
    
    return test_results

if __name__ == "__main__":
    run_all_tests()

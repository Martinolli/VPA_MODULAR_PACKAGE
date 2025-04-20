"""
VPA Backtester Integration Module

This module integrates the VPA backtester with the data fetching and validation system
to ensure data consistency and reliability in backtesting.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback
from vpa_modular.vpa_backtester import VPABacktester
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_processor import DataProcessor
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VPABacktesterIntegration')

class VPABacktesterIntegration:
    """Integration layer between VPA backtester and data fetching/validation system"""
    
    def __init__(self, data_fetcher, data_validator, base_dir="fetched_data"):
        """
        Initialize the backtester integration
        
        Parameters:
        - data_fetcher: Instance of VPADataFetcher
        - data_validator: Instance of VPADataValidator
        - base_dir: Base directory for stored data
        """
        self.data_fetcher = data_fetcher
        self.data_validator = data_validator
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "backtest_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def prepare_data_for_backtest(self, ticker, start_date, end_date, timeframes=None, force_refresh=False):
        """
        Prepare data for backtesting by fetching and validating
        
        Parameters:
        - ticker: Stock symbol
        - start_date: Start date for backtesting
        - end_date: End date for backtesting
        - timeframes: List of timeframe strings required for backtesting
        - force_refresh: Whether to force refresh data
        
        Returns:
        - Dictionary with preparation status and issues
        """
        if timeframes is None:
            timeframes = ["1d", "1h", "15m"]
        
        logger.info(f"Preparing data for backtesting {ticker} from {start_date} to {end_date}")
        
        # Step 1: Fetch data
        fetch_results = self.data_fetcher.fetch_data(ticker, timeframes, force_refresh)
        
        # Step 2: Validate data
        validation_results = self.data_validator.validate_ticker(ticker, start_date, end_date, timeframes)
        
        # Step 3: Check backtesting readiness
        readiness = self.data_validator.check_backtesting_readiness(ticker, start_date, end_date, timeframes)
        
        preparation_results = {
            "ticker": ticker,
            "start_date": start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d'),
            "end_date": end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d'),
            "fetch_results": fetch_results,
            "validation_results": validation_results,
            "readiness": readiness,
            "is_ready": readiness["is_ready"]
        }
        
        # Save preparation results
        self._save_preparation_results(ticker, preparation_results)
        
        return preparation_results
    
    def _save_preparation_results(self, ticker, preparation_results):
        """
        Save preparation results to JSON file
        
        Parameters:
        - ticker: Stock symbol
        - preparation_results: Dictionary with preparation results
        """
        file_path = os.path.join(self.base_dir, f"{ticker}_preparation.json")
        with open(file_path, 'w') as f:
            json.dump(preparation_results, f, indent=4)
        logger.info(f"Saved preparation results for {ticker} to {file_path}")
    
    def run_backtest(self, ticker, start_date, end_date, config=None, prepare_data=True, 
                    force_refresh=False, timeframes=None, **backtest_params):
        """
        Run a backtest with data validation and preparation
        
        Parameters:
        - ticker: Stock symbol
        - start_date: Start date for backtesting
        - end_date: End date for backtesting
        - config: VPAConfig instance or None to use default
        - prepare_data: Whether to prepare data before backtesting
        - force_refresh: Whether to force refresh data
        - timeframes: List of timeframe strings required for backtesting
        - backtest_params: Additional parameters for VPABacktester
        
        Returns:
        - Dictionary with backtest results or None if preparation failed
        """
        if timeframes is None:
            timeframes = ["1d", "1h", "15m"]
        
        # Step 1: Prepare data if requested
        if prepare_data:
            preparation_results = self.prepare_data_for_backtest(
                ticker, start_date, end_date, timeframes, force_refresh
            )
            
            if not preparation_results["is_ready"]:
                logger.warning(f"Data preparation failed for {ticker}. Backtest cannot proceed.")
                logger.warning(f"Issues: {json.dumps(preparation_results['readiness']['issues'], indent=2)}")
                return None
        
        # Step 2: Create backtester instance
        backtester = VPABacktester(
            start_date=start_date,
            end_date=end_date,
            **backtest_params
        )
        
        # Step 3: Patch the backtester to use our local data
        self._patch_backtester(backtester, timeframes)
        
        # Step 4: Run the backtest
        try:
            logger.info(f"Running backtest for {ticker} from {start_date} to {end_date}")
            results = backtester.run_backtest(ticker)
            
            # Step 5: Save results
            self._save_backtest_results(ticker, results, backtester)
            
            return results
        except Exception as e:
            logger.error(f"Error running backtest for {ticker}: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    
    def _patch_backtester(self, backtester, timeframes):
        """
        Patch the backtester to use our local data and fix method name issues
        
        Parameters:
        - backtester: VPABacktester instance
        - timeframes: List of timeframe strings
        """
        # Store original methods for reference
        original_get_data = backtester.data_manager.get_data
        
        # Define a new get_data method that uses our local data
        def patched_get_data(ticker):
            logger.info(f"Using local data for {ticker}")
            data = {}
            
            for timeframe in timeframes:
                file_path = os.path.join(self.base_dir, timeframe, f"{ticker}.csv")
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        if not df.empty:
                            # Filter data to the requested date range
                            if hasattr(backtester.data_manager, 'start_date') and backtester.data_manager.start_date:
                                start_date = pd.to_datetime(backtester.data_manager.start_date)
                                df = df[df.index >= start_date]
                            
                            if hasattr(backtester.data_manager, 'end_date') and backtester.data_manager.end_date:
                                end_date = pd.to_datetime(backtester.data_manager.end_date)
                                df = df[df.index <= end_date]
                            
                            data[timeframe.replace('1', '')] = df
                            logger.info(f"Loaded {len(df)} rows of {timeframe} data for {ticker}")
                        else:
                            logger.warning(f"Empty data file for {ticker} at {timeframe} timeframe")
                    except Exception as e:
                        logger.error(f"Error loading data for {ticker} at {timeframe} timeframe: {str(e)}")
                else:
                    logger.warning(f"No data file found for {ticker} at {timeframe} timeframe")
            
            if not data:
                logger.error(f"Failed to load any data for {ticker}")
                return {}
            
            return data
        
        # Replace the get_data method
        backtester.data_manager.get_data = patched_get_data
        
        # Fix the method name mismatch in analyze_at_date
        original_analyze_at_date = backtester.analyze_at_date
        
        def patched_analyze_at_date(ticker, current_date, lookback_days=None):
            """
            Patched version of analyze_at_date that uses preprocess_data instead of process_data
            """
            # Get data for this ticker
            ticker_data = backtester.data_manager.get_data(ticker)
            if not ticker_data:
                logger.warning(f"No data available for {ticker} at {current_date}")
                return None
            
            # Process each timeframe
            processed_data = {}
            for timeframe, df in ticker_data.items():
                # Filter data up to current date
                historical_data = df[df.index <= current_date].copy()
                
                if historical_data.empty:
                    logger.warning(f"No historical data for {ticker} at {timeframe} timeframe up to {current_date}")
                    continue
                
                # Apply lookback limit if specified
                if lookback_days:
                    lookback_date = current_date - pd.Timedelta(days=lookback_days)
                    historical_data = historical_data[historical_data.index >= lookback_date]
                
                # Use preprocess_data instead of process_data
                if hasattr(backtester.data_processor, 'preprocess_data'):
                    # Extract price and volume data
                    price_data = historical_data[['open', 'high', 'low', 'close']]
                    volume_data = historical_data['volume']
                    
                    # Call preprocess_data with the correct parameters
                    processed_data[timeframe] = backtester.data_processor.preprocess_data(
                        price_data, volume_data
                    )
                else:
                    logger.error("DataProcessor does not have preprocess_data method")
                    return None
            
            if not processed_data:
                logger.warning(f"No processed data available for {ticker} at {current_date}")
                return None
            
            # Generate signals
            signals = backtester.signal_generator.generate_signals(processed_data)
            
            return {
                "date": current_date,
                "processed_data": processed_data,
                "signals": signals
            }
        
        # Replace the analyze_at_date method
        backtester.analyze_at_date = patched_analyze_at_date
    
    def _save_backtest_results(self, ticker, results, backtester):
        """
        Save backtest results and generate reports
        
        Parameters:
        - ticker: Stock symbol
        - results: Backtest results
        - backtester: VPABacktester instance
        """
        # Create directory for this backtest
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backtest_dir = os.path.join(self.results_dir, f"{ticker}_{timestamp}")
        os.makedirs(backtest_dir, exist_ok=True)
        
        # Save results to JSON
        results_path = os.path.join(backtest_dir, "results.json")
        with open(results_path, 'w') as f:
            # Convert DataFrame to dict for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict(orient='records')
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=4, default=str)
        
        # Generate reports using backtester's methods
        try:
            # Create report
            report_files = backtester.create_backtest_report(backtest_dir)
            
            # Plot equity curve
            backtester.plot_equity_curve(
                include_benchmark=True, 
                save_path=os.path.join(backtest_dir, "equity_curve.png")
            )
            
            # Plot drawdown
            backtester.plot_drawdown(
                include_benchmark=True,
                save_path=os.path.join(backtest_dir, "drawdown.png")
            )
            
            # Plot trade analysis
            backtester.plot_trade_analysis(
                save_path=os.path.join(backtest_dir, "trade_analysis.png")
            )
            
            logger.info(f"Saved backtest results and reports for {ticker} to {backtest_dir}")
        except Exception as e:
            logger.error(f"Error generating backtest reports: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def run_multi_ticker_backtest(self, tickers, start_date, end_date, **kwargs):
        """
        Run backtests for multiple tickers
        
        Parameters:
        - tickers: List of stock symbols
        - start_date: Start date for backtesting
        - end_date: End date for backtesting
        - kwargs: Additional parameters for run_backtest
        
        Returns:
        - Dictionary with backtest results for each ticker
        """
        results = {}
        
        for ticker in tickers:
            logger.info(f"Running backtest for {ticker}")
            ticker_results = self.run_backtest(ticker, start_date, end_date, **kwargs)
            results[ticker] = ticker_results
        
        return results
    
    def run_walk_forward_analysis(self, ticker, start_date, end_date, window_size=90, step_size=30, **kwargs):
        """
        Run walk-forward analysis
        
        Parameters:
        - ticker: Stock symbol
        - start_date: Start date for analysis
        - end_date: End date for analysis
        - window_size: Size of each window in days
        - step_size: Step size between windows in days
        - kwargs: Additional parameters for run_backtest
        
        Returns:
        - Dictionary with walk-forward analysis results
        """
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Prepare data for the entire period
        preparation_results = self.prepare_data_for_backtest(
            ticker, start_date, end_date, **kwargs
        )
        
        if not preparation_results["is_ready"]:
            logger.warning(f"Data preparation failed for {ticker}. Walk-forward analysis cannot proceed.")
            return {"status": "error", "message": "Data preparation failed"}
        
        # Generate windows
        windows = []
        current_start = start_date
        while current_start + timedelta(days=window_size) <= end_date:
            current_end = current_start + timedelta(days=window_size)
            windows.append((current_start, current_end))
            current_start += timedelta(days=step_size)
        
        if not windows:
            logger.warning(f"No valid windows for walk-forward analysis between {start_date} and {end_date}")
            return {"status": "error", "message": "No valid windows for analysis"}
        
        # Run backtests for each window
        window_results = []
        
        for i, (window_start, window_end) in enumerate(windows):
            logger.info(f"Running backtest for window {i+1}/{len(windows)}: {window_start} to {window_end}")
            
            # Run backtest for this window
            # Set prepare_data=False since we already prepared data for the entire period
            backtest_results = self.run_backtest(
                ticker, 
                window_start.strftime('%Y-%m-%d'), 
                window_end.strftime('%Y-%m-%d'),
                prepare_data=False,
                **kwargs
            )
            
            if backtest_results:
                window_results.append({
                    "window_id": i + 1,
                    "start_date": window_start.strftime('%Y-%m-%d'),
                    "end_date": window_end.strftime('%Y-%m-%d'),
                    "results": backtest_results
                })
        
        # Aggregate results
        aggregated_results = self._aggregate_walk_forward_results(window_results)
        
        # Save walk-forward analysis results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wfa_dir = os.path.join(self.results_dir, f"{ticker}_wfa_{timestamp}")
        os.makedirs(wfa_dir, exist_ok=True)
        
        with open(os.path.join(wfa_dir, "walk_forward_results.json"), 'w') as f:
            json.dump({
                "ticker": ticker,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "window_size": window_size,
                "step_size": step_size,
                "window_results": window_results,
                "aggregated_results": aggregated_results
            }, f, indent=4, default=str)
        
        # Generate walk-forward analysis report
        self._generate_walk_forward_report(ticker, window_results, aggregated_results, wfa_dir)
        
        return {
            "status": "success",
            "window_results": window_results,
            "aggregated_results": aggregated_results
        }
    
    def _aggregate_walk_forward_results(self, window_results):
        """
        Aggregate results from walk-forward analysis
        
        Parameters:
        - window_results: List of results from each window
        
        Returns:
        - Dictionary with aggregated results
        """
        if not window_results:
            return {"status": "error", "message": "No window results to aggregate"}
        
        # Extract metrics from each window
        returns = []
        drawdowns = []
        win_rates = []
        profit_factors = []
        
        for window in window_results:
            results = window["results"]
            if isinstance(results, dict) and "metrics" in results:
                metrics = results["metrics"]
                returns.append(metrics.get("total_return", 0))
                drawdowns.append(metrics.get("max_drawdown", 0))
                win_rates.append(metrics.get("win_rate", 0))
                profit_factors.append(metrics.get("profit_factor", 0))
        
        # Calculate aggregated metrics
        aggregated = {
            "avg_return": np.mean(returns) if returns else None,
            "std_return": np.std(returns) if returns else None,
            "min_return": np.min(returns) if returns else None,
            "max_return": np.max(returns) if returns else None,
            "avg_drawdown": np.mean(drawdowns) if drawdowns else None,
            "max_drawdown": np.max(drawdowns) if drawdowns else None,
            "avg_win_rate": np.mean(win_rates) if win_rates else None,
            "avg_profit_factor": np.mean(profit_factors) if profit_factors else None,
            "consistency": np.sum([1 for r in returns if r > 0]) / len(returns) if returns else None
        }
        
        return aggregated
    
    def _generate_walk_forward_report(self, ticker, window_results, aggregated_results, report_dir):
        """
        Generate walk-forward analysis report
        
        Parameters:
        - ticker: Stock symbol
        - window_results: List of results from each window
        - aggregated_results: Dictionary with aggregated results
        - report_dir: Directory to save the report
        """
        try:
            # Extract data for plotting
            window_ids = [w["window_id"] for w in window_results]
            returns = [w["results"]["metrics"]["total_return"] if "metrics" in w["results"] else 0 for w in window_results]
            drawdowns = [w["results"]["metrics"]["max_drawdown"] if "metrics" in w["results"] else 0 for w in window_results]
            win_rates = [w["results"]["metrics"]["win_rate"] if "metrics" in w["results"] else 0 for w in window_results]
            
            # Plot returns by window
            plt.figure(figsize=(12, 6))
            plt.bar(window_ids, returns, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=aggregated_results["avg_return"], color='g', linestyle='--', alpha=0.7, 
                      label=f'Avg Return: {aggregated_results["avg_return"]:.2%}')
            plt.title(f"{ticker} Walk-Forward Analysis - Returns by Window")
            plt.xlabel("Window ID")
            plt.ylabel("Return (%)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "wfa_returns.png"))
            plt.close()
            
            # Plot drawdowns by window
            plt.figure(figsize=(12, 6))
            plt.bar(window_ids, drawdowns, alpha=0.7, color='r')
            plt.axhline(y=aggregated_results["avg_drawdown"], color='orange', linestyle='--', alpha=0.7,
                      label=f'Avg Drawdown: {aggregated_results["avg_drawdown"]:.2%}')
            plt.title(f"{ticker} Walk-Forward Analysis - Drawdowns by Window")
            plt.xlabel("Window ID")
            plt.ylabel("Drawdown (%)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "wfa_drawdowns.png"))
            plt.close()
            
            # Plot win rates by window
            plt.figure(figsize=(12, 6))
            plt.bar(window_ids, win_rates, alpha=0.7, color='g')
            plt.axhline(y=aggregated_results["avg_win_rate"], color='blue', linestyle='--', alpha=0.7,
                      label=f'Avg Win Rate: {aggregated_results["avg_win_rate"]:.2%}')
            plt.title(f"{ticker} Walk-Forward Analysis - Win Rates by Window")
            plt.xlabel("Window ID")
            plt.ylabel("Win Rate (%)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "wfa_win_rates.png"))
            plt.close()
            
            # Generate HTML report
            html_report = self._generate_walk_forward_html_report(ticker, window_results, aggregated_results)
            with open(os.path.join(report_dir, "walk_forward_report.html"), 'w') as f:
                f.write(html_report)
            
            logger.info(f"Generated walk-forward analysis report for {ticker} at {report_dir}")
        except Exception as e:
            logger.error(f"Error generating walk-forward report: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def _generate_walk_forward_html_report(self, ticker, window_results, aggregated_results):
        """
        Generate HTML report for walk-forward analysis
        
        Parameters:
        - ticker: Stock symbol
        - window_results: List of results from each window
        - aggregated_results: Dictionary with aggregated results
        
        Returns:
        - HTML report as string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Walk-Forward Analysis Report for {ticker}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .chart {{ margin: 20px 0; }}
                .chart img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Walk-Forward Analysis Report for {ticker}</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Average Return</td>
                        <td class="{self._get_color_class(aggregated_results['avg_return'])}">{self._format_percent(aggregated_results['avg_return'])}</td>
                    </tr>
                    <tr>
                        <td>Return Standard Deviation</td>
                        <td>{self._format_percent(aggregated_results['std_return'])}</td>
                    </tr>
                    <tr>
                        <td>Minimum Return</td>
                        <td class="{self._get_color_class(aggregated_results['min_return'])}">{self._format_percent(aggregated_results['min_return'])}</td>
                    </tr>
                    <tr>
                        <td>Maximum Return</td>
                        <td class="{self._get_color_class(aggregated_results['max_return'])}">{self._format_percent(aggregated_results['max_return'])}</td>
                    </tr>
                    <tr>
                        <td>Average Drawdown</td>
                        <td class="negative">{self._format_percent(aggregated_results['avg_drawdown'])}</td>
                    </tr>
                    <tr>
                        <td>Maximum Drawdown</td>
                        <td class="negative">{self._format_percent(aggregated_results['max_drawdown'])}</td>
                    </tr>
                    <tr>
                        <td>Average Win Rate</td>
                        <td>{self._format_percent(aggregated_results['avg_win_rate'])}</td>
                    </tr>
                    <tr>
                        <td>Average Profit Factor</td>
                        <td>{aggregated_results['avg_profit_factor']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Consistency (% of profitable windows)</td>
                        <td>{self._format_percent(aggregated_results['consistency'])}</td>
                    </tr>
                </table>
            </div>
            
            <h2>Window Results</h2>
            <table>
                <tr>
                    <th>Window</th>
                    <th>Date Range</th>
                    <th>Return</th>
                    <th>Drawdown</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Trades</th>
                </tr>
        """
        
        for window in window_results:
            results = window["results"]
            metrics = results.get("metrics", {})
            
            html += f"""
                <tr>
                    <td>{window['window_id']}</td>
                    <td>{window['start_date']} to {window['end_date']}</td>
                    <td class="{self._get_color_class(metrics.get('total_return', 0))}">{self._format_percent(metrics.get('total_return', 0))}</td>
                    <td class="negative">{self._format_percent(metrics.get('max_drawdown', 0))}</td>
                    <td>{self._format_percent(metrics.get('win_rate', 0))}</td>
                    <td>{metrics.get('profit_factor', 0):.2f}</td>
                    <td>{metrics.get('total_trades', 0)}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Charts</h2>
            <div class="chart">
                <h3>Returns by Window</h3>
                <img src="wfa_returns.png" alt="Returns by Window">
            </div>
            
            <div class="chart">
                <h3>Drawdowns by Window</h3>
                <img src="wfa_drawdowns.png" alt="Drawdowns by Window">
            </div>
            
            <div class="chart">
                <h3>Win Rates by Window</h3>
                <img src="wfa_win_rates.png" alt="Win Rates by Window">
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_color_class(self, value):
        """Get CSS class based on value"""
        if value is None:
            return ""
        return "positive" if value > 0 else "negative" if value < 0 else ""
    
    def _format_percent(self, value):
        """Format value as percentage"""
        if value is None:
            return "N/A"
        return f"{value:.2%}"

# Example usage
if __name__ == "__main__":
    from vpa_data_fetcher import VPADataFetcher
    from vpa_data_validator import VPADataValidator
    
    # Create components
    fetcher = VPADataFetcher()
    validator = VPADataValidator()
    integration = VPABacktesterIntegration(fetcher, validator)
    
    # Run a backtest
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    results = integration.run_backtest(
        ticker, 
        start_date, 
        end_date,
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_percent=0.001,
        risk_per_trade=0.02
    )
    
    print(f"Backtest results for {ticker}:")
    print(json.dumps(results, indent=4, default=str))

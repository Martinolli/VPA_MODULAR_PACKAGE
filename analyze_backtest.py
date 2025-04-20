from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_backtester import VPABacktester
from vpa_modular.vpa_data_fetcher import VPADataFetcher
from vpa_modular.vpa_data_validator import VPADataValidator
from vpa_modular.vpa_data_fetcher_csv_fix import fix_all_csv_files
from vpa_modular.vpa_data_fetcher_csv_fix import fix_csv_format, enhance_data_fetcher_load_method

# Patch the VPADataFetcher.load_data method
# VPADataFetcher.load_data = enhance_data_fetcher_load_method(VPADataFetcher.load_data)

# Example usage
# fetcher = VPADataFetcher()
# results = fetcher.fetch_data("AMZN", timeframes=["1d", "1h", "15m"])
# print(results)


# Example usage
# validator = VPADataValidator()
# validation_results = validator.validate_ticker("AMZN", "2023-01-01", "2023-12-31")
# readiness = validator.check_backtesting_readiness("AAPL", "2023-01-01", "2023-12-31")

backtester = VPABacktester(
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_percent=0.001,
        risk_per_trade=0.02,
        max_positions=5
    )

# Run a backtest
results = backtester.run_backtest(['AAPL', 'MSFT', 'GOOGL'])
    
# Create a report
report_files = backtester.create_backtest_report("backtest_reports")
    
print(f"Backtest completed. Report available at: {report_files.get('html')}")
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_backtester import VPABacktester
from vpa_modular.vpa_data_fetcher import VPADataFetcher
from vpa_modular.vpa_data_validator import VPADataValidator
from vpa_modular.vpa_data_fetcher_csv_fix import fix_all_csv_files
from vpa_modular.vpa_data_fetcher_csv_fix import fix_csv_format, enhance_data_fetcher_load_method
from vpa_modular.vpa_backtester_integration import VPABacktesterIntegration

# Patch the VPADataFetcher.load_data method
# VPADataFetcher.load_data = enhance_data_fetcher_load_method(VPADataFetcher.load_data)
"""
# Example usage
# Create components
fetcher = VPADataFetcher()
validator = VPADataValidator()
integration = VPABacktesterIntegration(fetcher, validator)

# Run a backtest
ticker = "AAPL"
start_date = "2024-01-01"
end_date = "2025-01-01"

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


# Example usage
validator = VPADataValidator()
validation_results = validator.validate_ticker("AMZN", "2024-06-01", "2025-01-01")
# readiness = validator.check_backtesting_readiness("AAPL", "2023-01-01", "2023-12-31")
"""

fetcher = VPADataFetcher()
data = fetcher.fetch_data("AAPL", timeframes=['1d', '1h', '15m'])
print("Data fetched successfully.")
print(data)

validator = VPADataValidator()
validation_results = validator.validate_ticker("AAPL", "2024-01-01", "2024-12-31")
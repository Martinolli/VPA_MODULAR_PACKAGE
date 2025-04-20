from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_backtester import VPABacktester
from vpa_modular.vpa_data_fetcher import VPADataFetcher
from vpa_modular.vpa_data_validator import VPADataValidator

fetcher = VPADataFetcher()
fetcher.fetch_data("AAPL", timeframes=["1d", "1h", "15m"])

# Example usage
validator = VPADataValidator()
validation_results = validator.validate_ticker("AAPL", "2023-01-01", "2023-12-31")

# smooth_backtest.py

from vpa_modular.vpa_data_fetcher import VPADataFetcher
from vpa_modular.vpa_data_validator import VPADataValidator
from vpa_modular.vpa_backtester_integration import VPABacktesterIntegration

if __name__ == "__main__":
    ticker     = "AAPL"
    start_date = "2024-01-01"
    end_date   = "2025-01-01"

    # 1) Instantiate
    fetcher   = VPADataFetcher()
    validator = VPADataValidator()
    integration = VPABacktesterIntegration(fetcher, validator)

    # 2) (Optional) ensure CSVs exist. 
    #    Remove the comment if this is your first run or you've cleared fetched_data/
    # fetcher.fetch_data(ticker, timeframes=["1d","1h","15m"])

    # 3) Run backtest **without** prepare_data (so no double‑validate + plotting)


    results = integration.run_backtest(
        ticker,
        start_date,
        end_date,
        prepare_data=False,       # <-- skips fetch+validate
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_percent=0.001,
        risk_per_trade=0.02
    )

    print("Backtest results for", ticker)
    print(results)

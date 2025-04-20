from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_backtester import VPABacktester

# Create a backtester
backtester = VPABacktester(
    start_date='2024-01-01',
    end_date='2025-01-01',
    initial_capital=100000.0,
    commission_rate=0.001,
    slippage_percent=0.001,
    risk_per_trade=0.02,
    max_positions=5,
    benchmark_ticker="SPY"
)

# Run a backtest on a single ticker
results = backtester.run_backtest('AAPL')

# Create a report
report_files = backtester.create_backtest_report("backtest_reports/apple")

# Plot equity curve with benchmark comparison
backtester.plot_equity_curve(include_benchmark=True, save_path="backtest_reports/apple/equity_curve.png")

# Plot drawdown
backtester.plot_drawdown(include_benchmark=True, save_path="backtest_reports/apple/drawdown.png")

# Plot trade analysis
backtester.plot_trade_analysis(save_path="backtest_reports/apple/trade_analysis.png")

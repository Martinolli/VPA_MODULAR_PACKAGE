#!/usr/bin/env python3


# Info: The contents in this file are only an example and cannot be used as a function or imported.

"""

from marketquant.strategy_simulator import TradingEngine
from strategies.macd_strategy import MACDStrategy

def main():

    # User Note: Initialize the trading core with your params for data_provider, ticker, start_date, end_date,
    # candle_aggregation, starting_balance, and shares below.
    engine = TradingEngine(
        data_provider="yahoo",
        ticker="SPY",
        start_date="2023-01-01",
        end_date="2024-08-01",
        candle_aggregation="1d",
        starting_balance=100000,
        shares=100,
        print_tradehistory=False,
        print_pnl=True,
        print_balance=True,
        print_buypower=True,
        print_unrealizedpnl=True,
        print_timecomplexity=True,
        chart=True
    )

    # Initialize the MACD strategy with the trading engine and MACD parameters
    macd_strategy = MACDStrategy(engine, macd_params={
        "short_period": 12,
        "long_period": 26,
        "signal_period": 9
    })

    # Apply the strategy
    macd_strategy.apply_strategy()

    # Example: This prints the results of the enabled outputs
    engine.print_results()

if __name__ == "__main__":
    main()

"""
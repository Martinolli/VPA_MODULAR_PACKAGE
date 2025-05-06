from marketquant.strategy_simulator.core.data import DataEngine
from marketquant.strategy_simulator.core.trade_simulator import TradeSimulator
from marketquant.strategy_simulator.core.account_manager import AccountManager
from marketquant.strategy_simulator.core.data_sources.yahoo import YahooDataSource
from marketquant.strategy_simulator.core.config import DEFAULT_CONFIG
from marketquant.strategy_simulator.core.cli.cli_output import CLIOutput
from marketquant.strategy_simulator.core.charting import TradeChart

class TradingEngine:
    def __init__(self, data_provider=None, ticker=None, start_date=None, end_date=None, candle_aggregation=None,
                 starting_balance=None, shares=None, print_tradehistory=True, print_pnl=True, print_balance=True,
                 print_buypower=True, print_unrealizedpnl=True, print_timecomplexity=True, chart=True):
        # Note: this will use the default config if parameters are not provided in strategy
        self.data_provider = data_provider or DEFAULT_CONFIG['data_provider']
        self.ticker = ticker or DEFAULT_CONFIG['ticker']
        self.start_date = start_date or DEFAULT_CONFIG['start_date']
        self.end_date = end_date or DEFAULT_CONFIG['end_date']
        self.candle_aggregation = candle_aggregation or DEFAULT_CONFIG['candle_aggregation']
        self.starting_balance = starting_balance or DEFAULT_CONFIG['starting_balance']
        self.shares = shares or DEFAULT_CONFIG['shares']
        self.chart = chart or DEFAULT_CONFIG['chart']

        # Note: setups data source
        if self.data_provider == "yahoo":
            self.data_source = YahooDataSource(self.ticker, self.start_date, self.end_date, self.candle_aggregation)
        # Future: Add more providers like Schwab here

        # Initialize components
        self.data_engine = DataEngine(self.data_source)
        self.account_manager = AccountManager(self.starting_balance)
        self.simulator = TradeSimulator(self.account_manager)

        # Print control flags
        self.print_tradehistory = print_tradehistory
        self.print_pnl = print_pnl
        self.print_balance = print_balance
        self.print_buypower = print_buypower
        self.print_unrealizedpnl = print_unrealizedpnl
        self.print_timecomplexity = print_timecomplexity
        self.chart = chart

    def calculate_time_complexity(self):
        # Dev Note: This only assumes O(n) time complexity where n is the number of data points fetched.
        # You chose may add your own logic here.
        if self.print_timecomplexity:
            data_size = len(self.data_engine.fetch_data())
            print(f"Time complexity of data fetching/processing: O(n), where n = {data_size}")

    def print_results(self):

        # Action: Prints trade history
        if self.print_tradehistory:
            print("Trade History:")
            for trade in self.simulator.get_trade_history():
                if 'Buy' in trade:
                    CLIOutput.print_buy_action(trade)
                elif 'Sell' in trade:
                    CLIOutput.print_sell_action(trade)

        # Action: Prints Final PNL and Balance
        if self.print_pnl:
            CLIOutput.print_final_pnl(f"Final PNL: ${self.account_manager.get_pnl()}")
        if self.print_balance:
            CLIOutput.print_final_balance(f"Final Balance: ${self.account_manager.get_balance()}")

        # Action: Prints final buying power
        if self.print_buypower:
            print(f"Final Buying Power: ${self.account_manager.get_buying_power()}")

        # Action: Fetches the latest market price (last closing price in data)
        current_price = self.data_engine.fetch_data()['Close'].iloc[-1]

        # Action: Calculates unrealized PNL
        unrealized_pnl = self.account_manager.get_unrealized_pnl(current_price)

        # Action: Prints unrealized PNL
        if self.print_unrealizedpnl and unrealized_pnl != 0:
            CLIOutput.print_unrealized_pnl(f"Unrealized PNL: ${unrealized_pnl}")

        # Action: Prints time complexity
        self.calculate_time_complexity()

        # Action: Builds and shows a chart with the trades and data requested. If PNL is '-' then
        # price line and area fill will be red and if '+' then price line and area fill will be green.
        if self.chart:
            pnl = self.account_manager.get_pnl()
            trade_history = self.simulator.get_trade_history()
            trade_chart = TradeChart(self.data_engine.fetch_data(), trade_history, pnl)
            trade_chart.plot_chart()
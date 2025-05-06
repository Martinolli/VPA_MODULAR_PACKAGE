from ..indicators.macd import MACDIndicator

class MACDStrategy:
    def __init__(self, trading_engine, macd_params=None):
        self.trading_engine = trading_engine
        self.macd_params = macd_params if macd_params else {}
        self.macd_indicator = MACDIndicator(**self.macd_params)

    def apply_strategy(self):
        # Fetch data from the engine
        data = self.trading_engine.data_engine.fetch_data()

        # Calculate MACD and signal using the indicator class
        data = self.macd_indicator.calculate(data)

        # Loop through the data and apply the MACD-based strategy
        for i in range(1, len(data)):
            date = data['Date'][i]
            macd = data['MACD'][i]
            signal = data['Signal'][i]
            previous_macd = data['MACD'][i - 1]
            previous_signal = data['Signal'][i - 1]

            # Check for cover signal (MACD crosses above signal line) to close the short
            if previous_macd <= previous_signal and macd > signal:
                price = data['Close'][i]
                # If in a short position, cover it
                if self.trading_engine.simulator.account_manager.positions.get('long', {}).get('quantity', 0) > 0:
                    self.trading_engine.simulator.sell(date, price, self.trading_engine.shares)

            # Check for short signal (MACD crosses below signal line)
            elif previous_macd >= previous_signal and macd < signal:
                price = data['Close'][i]
                # Short the stock
                self.trading_engine.simulator.buy(date, price, self.trading_engine.shares)

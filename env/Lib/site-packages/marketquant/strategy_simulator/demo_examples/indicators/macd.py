import pandas as pd

class MACDIndicator:
    def __init__(self, short_period=12, long_period=26, signal_period=9):
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period

    def calculate(self, data):
        data['EMA_short'] = data['Close'].ewm(span=self.short_period, adjust=False).mean()
        data['EMA_long'] = data['Close'].ewm(span=self.long_period, adjust=False).mean()

        data['MACD'] = data['EMA_short'] - data['EMA_long']
        data['Signal'] = data['MACD'].ewm(span=self.signal_period, adjust=False).mean()

        return data

class TradeSimulator:
    def __init__(self, account_manager):
        self.account_manager = account_manager
        self.trades = []

    def buy(self, date, price, quantity):
        self.trades.append(f"Buy {quantity} at {price} on {date}")
        # User Note: This will check if buying is allowed (e.g., sufficient balance)
        self.account_manager.update_position('buy', price, quantity)

    def sell(self, date, price, quantity):
        # User Note: This will ensure enough quantity is available to sell
        if 'long' in self.account_manager.positions and self.account_manager.positions['long']['quantity'] >= quantity:
            self.trades.append(f"Sell {quantity} at {price} on {date}")
            self.account_manager.update_position('sell', price, quantity)
        else:
            print(f"Warning: Not enough shares to sell on {date}. (Likely a position trying to be "
                  f"closed from before given time period)")

    def short(self, date, price, quantity):
        self.trades.append(f"Short {quantity} at {price} on {date}")
        self.account_manager.update_position('short', price, quantity)

    def cover(self, date, price, quantity):
        # User Note: This will ensure enough quantity is available to cover
        if 'short' in self.account_manager.positions and self.account_manager.positions['short']['quantity'] >= quantity:
            self.trades.append(f"Cover {quantity} at {price} on {date}")
            self.account_manager.update_position('cover', price, quantity)
        else:
            print(f"Warning: Not enough shares to cover short on {date}. (Likely a position trying to be "
                  f"closed from before given time period)")

    def get_trade_history(self):
        return self.trades

class AccountManager:
    def __init__(self, starting_balance):
        self.balance = starting_balance
        self.buying_power = starting_balance
        self.starting_balance = starting_balance
        self.positions = {}
        self.realized_pnl = 0
        self.unrealized_pnl = 0

    def update_position(self, action, price, quantity):
        # print(f"Action: {action}, Price: {price}, Quantity: {quantity}, Buying Power Before: {self.buying_power}")

        if action == 'buy':
            self._buy(price, quantity)
        elif action == 'sell':
            self._sell(price, quantity)
        elif action == 'short':
            self._short(price, quantity)
        elif action == 'cover':
            self._cover(price, quantity)

        # print(f"Buying Power After: {self.buying_power}, Balance: {self.balance}, PNL: {self.realized_pnl}")

    def _buy(self, price, quantity):
        total_cost = price * quantity

        # Action: Deduct from buying power, not balance, when buying shares
        if self.buying_power >= total_cost:
            self.buying_power -= total_cost
        else:
            raise ValueError("Not enough buying power to execute the buy order.")

        # Track long position
        if 'long' in self.positions:
            position = self.positions['long']
            new_quantity = position['quantity'] + quantity
            new_avg_price = (position['quantity'] * position['avg_price'] + total_cost) / new_quantity
            self.positions['long'] = {'quantity': new_quantity, 'avg_price': new_avg_price}
        else:
            self.positions['long'] = {'quantity': quantity, 'avg_price': price}

    def _sell(self, price, quantity):
        if 'long' in self.positions and self.positions['long']['quantity'] >= quantity:
            position = self.positions['long']
            total_sale = price * quantity
            realized_profit = (price - position['avg_price']) * quantity
            self.balance += realized_profit  # Add the realized profit to the balance
            self.realized_pnl += realized_profit

            # Action: Update or clear the position after sale
            if position['quantity'] == quantity:
                del self.positions['long']
            else:
                self.positions['long']['quantity'] -= quantity

            # Action: Restore buying power after selling shares
            self.buying_power += total_sale

    def _short(self, price, quantity):
        total_sale = price * quantity
        self.buying_power -= total_sale  # Reduce buying power when shorting

        # Track short position
        if 'short' in self.positions:
            position = self.positions['short']
            new_quantity = position['quantity'] + quantity
            new_avg_price = (position['quantity'] * position['avg_price'] + total_sale) / new_quantity
            self.positions['short'] = {'quantity': new_quantity, 'avg_price': new_avg_price}
        else:
            self.positions['short'] = {'quantity': quantity, 'avg_price': price}

    def _cover(self, price, quantity):
        if 'short' in self.positions and self.positions['short']['quantity'] >= quantity:
            position = self.positions['short']
            total_cost = price * quantity
            realized_profit = (position['avg_price'] - price) * quantity
            self.balance += realized_profit  # Add realized profit to balance
            self.realized_pnl += realized_profit

            # Action: Update or clear the position after covering
            if position['quantity'] == quantity:
                del self.positions['short']
            else:
                self.positions['short']['quantity'] -= quantity

            # Action: Restore buying power after covering shorts
            self.buying_power += total_cost

    def get_unrealized_pnl(self, current_price):
        # Action: Calculate unrealized PNL for open positions
        unrealized_pnl = 0
        if 'long' in self.positions:
            position = self.positions['long']
            unrealized_pnl += (current_price - position['avg_price']) * position['quantity']
        if 'short' in self.positions:
            position = self.positions['short']
            unrealized_pnl += (position['avg_price'] - current_price) * position['quantity']
        return unrealized_pnl

    def get_pnl(self):
        # Action: Returns the realized profit/loss
        return self.realized_pnl

    def get_balance(self):
        # Action: Returns the current balance (only affected by realized PNL)
        return self.balance

    def get_buying_power(self):
        # Action: Returns the current available buying power
        return self.buying_power

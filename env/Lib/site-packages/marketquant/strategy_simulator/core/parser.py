class StrategyParser:
    def __init__(self, strategy, simulator, shares=100):
        self.strategy = strategy
        self.simulator = simulator
        self.shares = shares

    def execute_strategy(self, data):
        # Note: this iterates over the data to execute the passed strategy
        for i in range(1, len(data)):
            date = data['Date'][i]
            price = data['Close'][i]

            # Note: this executes the passed strategy dynamically
            exec(self.strategy)

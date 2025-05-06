import yfinance as yf


class YahooDataSource:
    def __init__(self, ticker, start_date, end_date, aggregation="1d"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.aggregation = aggregation

    def get_data(self):
        # Fetch data from Yahoo Finance using yfinance and disable the progress bar
        ticker_data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval=self.aggregation,
            progress=False  # Disable the progress bar
        )

        # Reset index so that Date becomes a column
        ticker_data.reset_index(inplace=True)

        # Standardize the output
        return ticker_data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']] if 'Datetime' in ticker_data else \
        ticker_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

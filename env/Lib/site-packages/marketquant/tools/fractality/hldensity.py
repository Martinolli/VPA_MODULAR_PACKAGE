import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta


class HLdensity:
    def __init__(self, client, ticker_symbol, days=100, chart=False, printdata=False):
        """
        Initializes the HLdensity class with required parameters.
        :param client: 'yahoo' for Yahoo Finance.
        :param ticker_symbol: Stock ticker symbol (e.g., 'AAPL').
        :param days: Number of days to look back from the current date (default: 100).
        :param chart: Whether to plot the chart (default: False).
        :param printdata: Whether to print the high and low order data (default: False).
        """
        self.client = client
        self.ticker_symbol = ticker_symbol
        self.days = days
        self.chart = chart
        self.printdata = printdata

    @classmethod
    def run(cls, client, ticker_symbol, days=100, chart=False, printdata=False):
        """
        Runs the HLdensity Analysis and outputs the chart or data as required.
        :param client: 'yahoo' for Yahoo Finance.
        :param ticker_symbol: Stock ticker symbol (e.g., 'AAPL').
        :param days: Number of days to look back.
        :param chart: Whether to plot the chart (default: False).
        :param printdata: Whether to print the high and low order data (default: False).
        :return: None (prints data or plots a chart based on args).
        """
        instance = cls(client, ticker_symbol, days, chart, printdata)

        # Fetch the OHLC data
        data = instance.fetch_ohlc_data()

        # Normalize the OHLC data before plotting
        normalized_data = instance.normalize_ohlc(data)

        if instance.printdata:
            instance.print_high_low_order(data)

        if instance.chart:
            fig, ax = plt.subplots(figsize=(12, 8))
            instance.plot_ohlc_moves(normalized_data, ax)
            plt.show()
        else:
            print("To see the 'chart', set chart=True")

    def fetch_ohlc_data(self):
        """
        Fetch historical OHLC data for the given ticker using Yahoo Finance.
        :return: DataFrame with OHLC data.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.days)

        df = yf.download(self.ticker_symbol, start=start_date, end=end_date, interval='1d')

        if df.empty:
            raise ValueError(f"No data found for ticker {self.ticker_symbol}")

        df.reset_index(inplace=True)
        return df

    def normalize_ohlc(self, df):
        """
        Normalize OHLC data relative to the opening price.
        :param df: DataFrame with OHLC data.
        :return: Normalized DataFrame.
        """
        normalized_df = df.copy()
        normalized_df['Move_High'] = df['High'] - df['Open']
        normalized_df['Move_Low'] = df['Low'] - df['Open']
        normalized_df['Move_Close'] = df['Close'] - df['Open']
        normalized_df['Move_Open'] = 0
        return normalized_df

    def print_high_low_order(self, df):
        """
        Print whether the high or low was set first and second for each day,
        including the actual high and low prices.
        :param df: DataFrame with OHLC data.
        """
        for i in range(len(df)):
            date = df['Date'].iloc[i]
            high_price = df['High'].iloc[i]
            low_price = df['Low'].iloc[i]
            open_price = df['Open'].iloc[i]

            high_move = high_price - open_price
            low_move = low_price - open_price

            if low_move < high_move:
                print(f"{date}: Low was reached first (Low: {low_price}), then High (High: {high_price}).")
            else:
                print(f"{date}: High was reached first (High: {high_price}), then Low (Low: {low_price}).")

    def plot_ohlc_moves(self, df, ax):
        """
        Visualizes price movements within a trading day using a density map.
        :param df: Normalized OHLC DataFrame.
        :param ax: Matplotlib axis to plot on.
        """
        ax.clear()
        ax.set_title(f"Density Map for {self.ticker_symbol}")
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Move in Dollars')
        high_moves = df['Move_High'].values
        low_moves = df['Move_Low'].values
        open_price = df['Open'].iloc[-1]

        # Process each trading day's data
        for i in range(len(df)):
            date = df['Date'].iloc[i]
            high_move = df['Move_High'].iloc[i]
            low_move = df['Move_Low'].iloc[i]

            try:
                intraday_data = fetch_intraday_data(self.ticker_symbol, date)
                if len(intraday_data) < 2:
                    raise ValueError("Not enough intraday data to process.")

                high_idx = intraday_data['High'].idxmax()
                low_idx = intraday_data['Low'].idxmin()
                high_time = intraday_data['Datetime'][high_idx]
                low_time = intraday_data['Datetime'][low_idx]

                # Normalize time for plotting
                total_seconds = (intraday_data['Datetime'].iloc[-1] - intraday_data['Datetime'].iloc[0]).total_seconds()
                if total_seconds == 0:
                    raise ValueError("Total seconds for intraday data is zero, cannot normalize time.")

                x_high = (high_time - intraday_data['Datetime'].iloc[0]).total_seconds() / total_seconds
                x_low = (low_time - intraday_data['Datetime'].iloc[0]).total_seconds() / total_seconds

                # Plot the high and low based on time sequence
                if x_low < x_high:
                    # Low was first
                    x_coords = [0, 0.25, 0.75, 1]
                    y_coords = [0, low_move, high_move, 0]
                else:
                    # High was first
                    x_coords = [0, 0.25, 0.75, 1]
                    y_coords = [0, high_move, low_move, 0]

                # Create a cubic spline interpolation
                cs = CubicSpline(x_coords, y_coords)
                x_new = np.linspace(0, 1, 100)
                y_new = cs(x_new)

                # Plot the data
                ax.plot(x_new, y_new, color='black', alpha=0.6, linewidth=1)
                ax.fill_between([0, 1], low_move, high_move, color='#363636', alpha=0.1)
            except ValueError as ve:
                print(f"Error processing date {date}: {ve}")
                continue

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        ax.figure.canvas.draw()


# Helper functions used in the HLdensity class

def fetch_intraday_data(ticker_symbol, date, interval='1h'):
    ticker = yf.Ticker(ticker_symbol)
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError(f"No intraday data found for {ticker_symbol} on {date}")

    df.reset_index(inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df[df['Datetime'].dt.date == date.date()]
    df = df.set_index('Datetime').between_time('09:30', '16:00').reset_index()

    if df.empty:
        raise ValueError(f"No intraday data found for {ticker_symbol} on {date} within market hours")

    return df

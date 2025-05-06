import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from hurst import compute_Hc
from itertools import combinations
from joblib import Parallel, delayed


class HurstHalfLifeCointegration:
    def __init__(self, tickers, start, end, p_value_threshold=0.05, hurst_threshold=0.5,
                 min_half_life=1, max_half_life=365, min_crossings=12):
        """
        Initialize the PairsTradingAnalyzer with customizable thresholds.

        :param tickers: List of tickers to analyze.
        :param start: Start date for downloading data.
        :param end: End date for downloading data.
        :param p_value_threshold: Maximum p-value for cointegration.
        :param hurst_threshold: Maximum Hurst exponent value (less than 0.5 for mean reversion).
        :param min_half_life: Minimum half-life for the spread (in days).
        :param max_half_life: Maximum half-life for the spread (in days).
        :param min_crossings: Minimum number of mean crossings per year.
        """
        self.tickers = tickers
        self.start = start
        self.end = end
        self.p_value_threshold = p_value_threshold
        self.hurst_threshold = hurst_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_crossings = min_crossings
        self.data = None

    # Step 1: Download Historical Data from Yahoo Finance
    def download_data(self):
        try:
            self.data = yf.download(self.tickers, start=self.start, end=self.end)['Adj Close']
        except Exception as e:
            print(f"Error downloading data: {e}")
            self.data = pd.DataFrame()

    # Step 2: Cointegration Test Function (Engle-Granger)
    def calculate_cointegration(self, pair):
        series1 = self.data[pair[0]].dropna()
        series2 = self.data[pair[1]].dropna()

        min_length = min(len(series1), len(series2))
        series1 = series1[:min_length]
        series2 = series2[:min_length]

        if min_length == 0:
            return None  # Skip pair if either series is empty

        try:
            coint_result = coint(series1, series2)
            p_value = coint_result[1]
            if p_value < self.p_value_threshold:
                return (pair[0], pair[1], p_value)
        except Exception as e:
            print(f"Error with pair {pair}: {e}")
        return None

    # Step 3: Calculate Hurst Exponent (Mean Reversion Check)
    def calculate_hurst(self, series):
        series_cleaned = series.dropna()

        if len(series_cleaned) == 0 or series_cleaned.nunique() == 1:
            raise ValueError("Invalid time series for Hurst calculation. Series is either empty or constant.")

        if (series_cleaned <= 0).any():
            raise ValueError("Invalid values in series for Hurst calculation. Contains non-positive values.")

        try:
            H, _, _ = compute_Hc(series_cleaned, kind='price')
            return H
        except FloatingPointError as e:
            print(f"FloatingPointError encountered during Hurst calculation: {e}")
            return None  # Skip this pair if there is an error

    # Step 4: Calculate Half-Life of Mean Reversion
    def calculate_half_life(self, spread):
        spread_lag = spread.shift(1)
        spread_ret = spread - spread_lag
        spread_lag = spread_lag[1:]
        spread_ret = spread_ret[1:]

        beta = np.polyfit(spread_lag, spread_ret, 1)[0]
        half_life = -np.log(2) / beta
        return half_life

    # Step 5: Calculate Mean Crossings
    def calculate_mean_crossings(self, spread):
        spread_mean = spread.mean()
        crossings = ((spread.shift(1) < spread_mean) & (spread > spread_mean)).sum()
        return crossings

    # Step 6: Full Pairs Selection Process Based on Rules
    def process_pair(self, pair):
        series1 = self.data[pair[0]].dropna()
        series2 = self.data[pair[1]].dropna()

        min_length = min(len(series1), len(series2))
        series1 = series1[:min_length]
        series2 = series2[:min_length]

        if min_length == 0:
            return None  # Skip pair if either series is empty

        # 1. Cointegration Test
        try:
            cointegration_result = self.calculate_cointegration(pair)
            if not cointegration_result:
                return None
            p_value = cointegration_result[2]
        except Exception as e:
            print(f"Error with cointegration test for {pair}: {e}")
            return None

        # Calculate the spread
        spread = series1 - series2

        # 2. Hurst Exponent Check (H < 0.5)
        try:
            hurst_exponent = self.calculate_hurst(spread)
            if hurst_exponent is None or hurst_exponent >= self.hurst_threshold:
                return None
        except ValueError as e:
            print(f"Discarded the calculation of Hurst exponent for {pair}: {e}")
            return None

        # 3. Half-Life Calculation (1 day < half-life < 1 year)
        try:
            half_life = self.calculate_half_life(spread)
            if not (self.min_half_life < half_life < self.max_half_life):
                return None
        except Exception as e:
            print(f"Error calculating half-life for {pair}: {e}")
            return None

        # 4. Mean Crossings (Must cross mean at least 12 times)
        mean_crossings = self.calculate_mean_crossings(spread)
        if mean_crossings < self.min_crossings:
            return None

        return (pair[0], pair[1], p_value, hurst_exponent, half_life, mean_crossings)

    # Step 7: Parallel Processing for Pairs Selection
    def find_eligible_pairs(self):
        ticker_pairs = list(combinations(self.data.columns, 2))

        eligible_pairs = Parallel(n_jobs=-1)(delayed(self.process_pair)(pair) for pair in ticker_pairs)
        eligible_pairs = [pair for pair in eligible_pairs if pair is not None]

        return eligible_pairs

    def run_analysis(self):
        self.download_data()

        if self.data.empty:
            print("No data available.")
            return

        eligible_pairs = self.find_eligible_pairs()

        # Display the eligible pairs
        if eligible_pairs:
            print("Eligible Pairs Found:")
            for pair in eligible_pairs:
                print(f"Pair: {pair[0]} and {pair[1]} | P-Value: {pair[2]:.4f} | Hurst: {pair[3]:.4f} | Half-Life: {pair[4]:.2f} | Mean Crossings: {pair[5]}")
        else:
            print("No eligible pairs found.")

def load_tickers_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.replace("'", "").str.strip()
        print("Cleaned Columns in CSV:", df.columns)
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

if __name__ == '__main__':
    tickers_csv_path = 'tickers.csv'
    tickers = load_tickers_from_csv(tickers_csv_path)

    # Initialize the analyzer with customizable parameters
    analyzer = PairsTradingAnalyzer(
        tickers=tickers,
        start='2022-01-01',
        end='2024-01-01',
        p_value_threshold=0.05,
        hurst_threshold=0.4,
        min_half_life=5,
        max_half_life=250,
        min_crossings=10
    )

    # Run the analysis
    analyzer.run_analysis()

"""
VPA Data Module

This module provides data acquisition and management for the VPA algorithm.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import datetime, timedelta

class DataProvider:
    """Base class for data providers"""
    
    def get_price_data(self, ticker, timeframe, period):
        """Get price data for a ticker"""
        raise NotImplementedError
    
    def get_volume_data(self, ticker, timeframe, period):
        """Get volume data for a ticker"""
        raise NotImplementedError
    
    def get_data(self, ticker, timeframe, period):
        """Get both price and volume data"""
        price_data = self.get_price_data(ticker, timeframe, period)
        volume_data = self.get_volume_data(ticker, timeframe, period)
        return price_data, volume_data


class YFinanceProvider:
    """YFinance implementation of data provider"""
    
    def get_data(self, ticker, interval='1d', period='1y', start_date=None, end_date=None):
        """
        Fetch market data using yfinance
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        - interval: Data interval ('1d', '1h', etc.)
        - period: Data period ('1y', '6mo', etc.) - used if start_date and end_date are None
        - start_date: Start date for data retrieval (optional)
        - end_date: End date for data retrieval (optional)
        
        Returns:
        - price_data: DataFrame with OHLC data
        - volume_data: Series with volume data
        """
        # Download data
        try:
            if start_date and end_date:
                # Convert to string format if datetime objects
                if isinstance(start_date, (datetime, pd.Timestamp)):
                    start_date = start_date.strftime('%Y-%m-%d')
                if isinstance(end_date, (datetime, pd.Timestamp)):
                    end_date = end_date.strftime('%Y-%m-%d')
                
                data = yf.download(ticker, start=start_date, end=end_date, 
                                  interval=interval, auto_adjust=False, progress=False)
            else:
                data = yf.download(ticker, interval=interval, period=period, 
                                  auto_adjust=False, progress=False)
        except Exception as e:
            raise ValueError(f"Error downloading data for {ticker}: {str(e)}")

        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}")

        # Handle potential MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        
        # Ensure datetime index
        if 'Date' in data.columns:
            data = data.rename(columns={'Date': 'datetime'})

        # If 'datetime' column is still not present, create it from the index
        if 'datetime' not in data.columns:
            data['datetime'] = data.index

        # Convert to UTC datetime
        for col in data.columns:
            if col.startswith("date"):
                data.rename(columns={col: 'datetime'}, inplace=True)
                break
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce', utc=True)
        data = data.dropna(subset=['datetime'])
        data.set_index('datetime', inplace=True)
        
        # Ensure column names are lowercase
        data.columns = data.columns.str.lower()

        # Check for ticker suffix in column names
        suffix = ''
        for col in data.columns:
            if col.endswith(f'_{ticker.lower()}'):
                suffix = f'_{ticker.lower()}'
                break
                
        # Extract price and volume data
        try:
            if suffix:
                # Handle columns with ticker suffix
                price_data = data[[f'open{suffix}', f'high{suffix}', f'low{suffix}', f'close{suffix}']].copy()
                price_data.columns = ['open', 'high', 'low', 'close']
                volume_data = data[f'volume{suffix}'].copy()
            else:
                # Handle standard column names
                price_data = data[['open', 'high', 'low', 'close']].copy()
                volume_data = data['volume'].copy()
                
        except KeyError as e:
            raise ValueError(f"Missing expected price/volume columns for ticker {ticker}: {e}")

        return price_data, volume_data
    
    def get_price_data(self, ticker, interval='1d', period='1y', start_date=None, end_date=None):
        """Get price data using yfinance"""
        price_data, _ = self.get_data(ticker, interval, period, start_date, end_date)
        return price_data
    
    def get_volume_data(self, ticker, interval='1d', period='1y', start_date=None, end_date=None):
        """Get volume data using yfinance"""
        _, volume_data = self.get_data(ticker, interval, period, start_date, end_date)
        return volume_data

class CSVProvider(DataProvider):
    """CSV implementation of data provider"""
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    def get_data(self, ticker, interval='1d', period=None):
        """
        Get price and volume data from CSV files
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        - interval: Data interval ('1d', '1h', etc.) - used to construct filename
        - period: Not used for CSV provider, included for API compatibility
        
        Returns:
        - price_data: DataFrame with OHLC data
        - volume_data: Series with volume data
        """
        import os
        
        # Construct filename
        filename = os.path.join(self.data_path, f"{ticker}_{interval}.csv")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        # Read data
        data = pd.read_csv(filename)
        
        # Ensure datetime column exists
        datetime_cols = [col for col in data.columns if col.lower() in ['date', 'datetime', 'time']]
        if not datetime_cols:
            raise ValueError(f"No datetime column found in {filename}")
        
        # Rename datetime column
        data = data.rename(columns={datetime_cols[0]: 'datetime'})
        
        # Convert to datetime and set as index
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data = data.dropna(subset=['datetime'])
        data.set_index('datetime', inplace=True)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns and col.upper() not in data.columns:
                raise ValueError(f"Required column '{col}' not found in {filename}")
        
        # Standardize column names to lowercase
        data.columns = data.columns.str.lower()
        
        # Extract price and volume data
        price_data = data[['open', 'high', 'low', 'close']].copy()
        volume_data = data['volume'].copy()
        
        return price_data, volume_data
    
    def get_price_data(self, ticker, interval='1d', period=None):
        """Get price data from CSV files"""
        price_data, _ = self.get_data(ticker, interval, period)
        return price_data
    
    def get_volume_data(self, ticker, interval='1d', period=None):
        """Get volume data from CSV files"""
        _, volume_data = self.get_data(ticker, interval, period)
        return volume_data


class MultiTimeframeProvider:
    """Provider for multiple timeframes"""
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
    
    def get_multi_timeframe_data(self, ticker, timeframes):
        """
        Fetch market data for multiple timeframes
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        - timeframes: List of timeframe dictionaries with 'interval' and 'period' keys
        
        Returns:
        - Dictionary with timeframe data
        """
        timeframe_data = {}
        
        for tf in timeframes:
            interval = tf['interval']
            period = tf['period']
            
            # Create a key for this timeframe
            tf_key = f"{interval}"
            
            # Fetch data
            try:
                price_data, volume_data = self.data_provider.get_data(ticker, interval, period)
                
                # Store in dictionary
                timeframe_data[tf_key] = {
                    'price_data': price_data,
                    'volume_data': volume_data
                }
            except Exception as e:
                print(f"Error fetching data for {ticker} at {interval} timeframe: {e}")
        
        if not timeframe_data:
            raise ValueError(f"Could not fetch data for any timeframe for {ticker}")
        
        return timeframe_data

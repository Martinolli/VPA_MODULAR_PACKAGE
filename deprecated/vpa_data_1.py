"""
VPA Data Module

This module provides data acquisition and management for the VPA algorithm.
"""

import pandas as pd
import numpy as np
# import yfinance as yf # No longer needed
import datetime as dt # Alias datetime to dt for clarity
from datetime import datetime, timedelta # Keep for timedelta
import os
from polygon import RESTClient # For Polygon.io

class DataProvider:
    """Base class for data providers"""
    
    def get_price_data(self, ticker, interval, period=None, start_date=None, end_date=None):
        """Get price data for a ticker"""
        raise NotImplementedError
    
    def get_volume_data(self, ticker, interval, period=None, start_date=None, end_date=None):
        """Get volume data for a ticker"""
        raise NotImplementedError
    
    def get_data(self, ticker, interval, period=None, start_date=None, end_date=None):
        """Get both price and volume data"""
        price_data = self.get_price_data(ticker, interval, period, start_date, end_date)
        volume_data = self.get_volume_data(ticker, interval, period, start_date, end_date)
        return price_data, volume_data

class PolygonIOProvider(DataProvider):
    """Polygon.io implementation of data provider"""
    
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            raise ValueError("Polygon.io API key not found. Set POLYGON_API_KEY environment variable.")
        self.client = RESTClient(api_key)

    def _parse_interval(self, interval_str):
        if not isinstance(interval_str, str):
            raise ValueError(f"Interval must be a string, got {type(interval_str)}")

        if interval_str == "1d":
            return {"multiplier": 1, "timespan": "day"}
        if interval_str == "1h":
            return {"multiplier": 1, "timespan": "hour"}
        if "m" in interval_str:
            try:
                multiplier = int(interval_str.replace("m", ""))
                if multiplier > 0:
                    return {"multiplier": multiplier, "timespan": "minute"}
                else:
                    raise ValueError(f"Invalid minute multiplier: {multiplier}")
            except ValueError:
                raise ValueError(f"Invalid minute interval format: {interval_str}")
        # Add support for '1wk', '1mo' if needed by Polygon and your VPA
        if interval_str == "1wk":
            return {"multiplier": 1, "timespan": "week"}
        if interval_str == "1mo":
            return {"multiplier": 1, "timespan": "month"}
        raise ValueError(f"Unsupported interval for Polygon.io: {interval_str}. Supported: '1d', '1h', 'Xm', '1wk', '1mo'.")

    def _calculate_from_date(self, to_date_obj, period_str):
        if period_str is None:
            # Default to a reasonable period if none specified, e.g., 1 year
            # This case should ideally be handled by ensuring period or start_date is always passed
            return to_date_obj - timedelta(days=365)
        
        period_str = period_str.lower()
        num = int("".join(filter(str.isdigit, period_str))) if any(char.isdigit() for char in period_str) else 1
        
        if "y" in period_str:
            return to_date_obj - timedelta(days=num * 365)
        elif "mo" in period_str:
            # Approximate months as 30 days for simplicity, Polygon might have specific ways to handle this
            return to_date_obj - timedelta(days=num * 30) 
        elif "wk" in period_str:
            return to_date_obj - timedelta(weeks=num)
        elif "d" in period_str:
            return to_date_obj - timedelta(days=num)
        else:
            # Default if period format is not recognized, or raise error
            raise ValueError(f"Unsupported period string format: {period_str}") 

    def get_data(self, ticker, interval="1d", period="1y", start_date=None, end_date=None):
        parsed_interval = self._parse_interval(interval)
        multiplier = parsed_interval["multiplier"]
        timespan = parsed_interval["timespan"]

        if start_date and end_date:
            from_date_str = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
            to_date_str = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
        elif period:
            to_date_obj = dt.datetime.now() # Use dt alias
            from_date_obj = self._calculate_from_date(to_date_obj, period)
            to_date_str = to_date_obj.strftime("%Y-%m-%d")
            from_date_str = from_date_obj.strftime("%Y-%m-%d")
        else:
            raise ValueError("Either period or start_date/end_date must be provided.")

        try:
            aggs = self.client.get_aggs(
                ticker=ticker.upper(),
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date_str,
                to=to_date_str,
                adjusted=True,
                sort="asc", # Ensure chronological order
                limit=50000 # Max limit for Polygon.io aggregates
            )
        except Exception as e:
            # Log the error for debugging
            print(f"Polygon API Error for {ticker} ({from_date_str} to {to_date_str}, {multiplier} {timespan}): {e}")
            raise ValueError(f"Error fetching data from Polygon.io for {ticker}: {e}")

        if not aggs:
            # It's possible no data exists for the period, which might not be an error
            # depending on expectations. For now, treat as empty result.
            print(f"No data returned from Polygon.io for {ticker} ({from_date_str} to {to_date_str}, {multiplier} {timespan})")
            # Return empty DataFrames with expected structure
            price_cols = ["open", "high", "low", "close"]
            price_data = pd.DataFrame(columns=price_cols)
            price_data.index = pd.to_datetime([])
            price_data.index.name = "datetime"
            volume_data = pd.Series(dtype=np.float64, name="volume")
            volume_data.index = pd.to_datetime([])
            volume_data.index.name = "datetime"
            return price_data, volume_data

        df = pd.DataFrame([agg.__dict__ for agg in aggs]) # Convert list of objects to DataFrame
        
        # Polygon timestamps are in milliseconds, convert to UTC datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")
        
        # Rename columns to match VPA expectations (o, h, l, c, v -> open, high, low, close, volume)
        column_map = {
            "open": "open", 
            "high": "high", 
            "low": "low", 
            "close": "close", 
            "volume": "volume"
        }
        df = df.rename(columns=column_map)
        
        price_data = df[["open", "high", "low", "close"]].copy()
        volume_data = df["volume"].copy().astype(np.float64) # Ensure volume is float
        
        return price_data, volume_data

    def get_price_data(self, ticker, interval="1d", period="1y", start_date=None, end_date=None):
        price_data, _ = self.get_data(ticker, interval, period, start_date, end_date)
        return price_data
    
    def get_volume_data(self, ticker, interval="1d", period="1y", start_date=None, end_date=None):
        _, volume_data = self.get_data(ticker, interval, period, start_date, end_date)
        return volume_data

class YFinanceProvider: # Keep for reference or if user wants to switch back later
    """YFinance implementation of data provider"""
    # ... (previous YFinanceProvider code as read from file, can be kept or removed)
    # For this task, we are replacing its usage, so it's not strictly needed now.
    # To avoid confusion, I will comment it out for now.
    pass
"""
class YFinanceProvider:
    def get_data(self, ticker, interval='1d', period='1y', start_date=None, end_date=None):
        raise DeprecationWarning("YFinanceProvider is deprecated. Use PolygonIOProvider.")
    # ... (minimal stubs for other methods if needed to avoid breaking old direct calls)
"""

class CSVProvider(DataProvider):
    """CSV implementation of data provider"""
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    def get_data(self, ticker, interval="1d", period=None, start_date=None, end_date=None):
        # (CSVProvider code as read from file - unchanged)
        import os
        filename = os.path.join(self.data_path, f"{ticker}_{interval}.csv")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        data = pd.read_csv(filename)
        datetime_cols = [col for col in data.columns if col.lower() in ['date', 'datetime', 'time']]
        if not datetime_cols:
            raise ValueError(f"No datetime column found in {filename}")
        data = data.rename(columns={datetime_cols[0]: 'datetime'}) # Use dt alias
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        data = data.dropna(subset=['datetime'])
        data.set_index('datetime', inplace=True)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns and col.upper() not in data.columns:
                raise ValueError(f"Required column '{col}' not found in {filename}")
        data.columns = data.columns.str.lower()
        price_data = data[["open", "high", "low", "close"]].copy()
        volume_data = data["volume"].copy()
        return price_data, volume_data
    
    def get_price_data(self, ticker, interval="1d", period=None, start_date=None, end_date=None):
        price_data, _ = self.get_data(ticker, interval, period, start_date, end_date)
        return price_data
    
    def get_volume_data(self, ticker, interval="1d", period=None, start_date=None, end_date=None):
        _, volume_data = self.get_data(ticker, interval, period, start_date, end_date)
        return volume_data

class MultiTimeframeProvider:
    """Provider for multiple timeframes"""
    
    def __init__(self, data_provider):
        self.data_provider = data_provider # This will now be an instance of PolygonIOProvider
    
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
        
        for tf_config in timeframes: # Renamed tf to tf_config for clarity
            interval = tf_config['interval']
            # Period is now handled by PolygonIOProvider.get_data based on its logic
            # Or, if start/end dates are preferred, they should be in tf_config
            period = tf_config.get('period') # Make period optional at this level
            start_date = tf_config.get('start_date')
            end_date = tf_config.get('end_date')
            
            tf_key = f"{interval}"
            
            try:
                # Pass all relevant parameters to the data_provider
                price_data, volume_data = self.data_provider.get_data(
                    ticker, 
                    interval=interval, 
                    period=period,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if price_data.empty:
                    print(f"Warning: No price data returned for {ticker} at {interval} timeframe. Skipping.")
                    continue

                timeframe_data[tf_key] = {
                    'price_data': price_data,
                    'volume_data': volume_data
                }
            except Exception as e:
                print(f"Error fetching data for {ticker} at {interval} timeframe: {e}")
        
        if not timeframe_data:
            # This might be acceptable if some timeframes fail but others succeed.
            # Consider if an error should be raised only if ALL fail.
            print(f"Warning: Could not fetch data for any requested timeframe for {ticker}")
            # raise ValueError(f"Could not fetch data for any timeframe for {ticker}")
        
        return timeframe_data

"""
VPA Data Fetcher Module

This module provides functionality to fetch and store market data for VPA analysis,
addressing Yahoo Finance API limitations and ensuring data consistency.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import time
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VPADataFetcher')

def datetime_json_serializer(obj):
    """
    Helper function for json.dumps to serialize datetime objects.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class VPADataFetcher:
    """Fetch and store market data for VPA analysis"""
    
    def __init__(self, base_dir="fetched_data"):
        """
        Initialize the data fetcher
        
        Parameters:
        - base_dir: Base directory for storing fetched data
        """
        self.base_dir = base_dir
        self._ensure_directories()
        
        # Define timeframe configurations with appropriate periods based on API limitations
        self.timeframe_configs = {
            "1d": {
                "interval": "1d",
                "max_period": "max",  # Max available historical data
                "min_period": "1y"    # Minimum period to fetch
            },
            "1h": {
                "interval": "1h",
                "max_period": "730d", # Max available for hourly data (2 years)
                "min_period": "60d"   # Minimum period to fetch
            },
            "15m": {
                "interval": "15m",
                "max_period": "60d",  # Max available for 15-min data (60 days)
                "min_period": "7d"    # Minimum period to fetch
            }
        }
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create subdirectories for each timeframe
        for timeframe in ["1d", "1h", "15m"]:
            os.makedirs(os.path.join(self.base_dir, timeframe), exist_ok=True)
        
        # Create directory for validation reports
        os.makedirs(os.path.join(self.base_dir, "validation"), exist_ok=True)
    
    def fetch_data(self, ticker, timeframes=None, force_refresh=False):
        """
        Fetch data for a ticker across specified timeframes
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        - timeframes: List of timeframe strings (e.g., ['1d', '1h', '15m'])
                     If None, all configured timeframes will be used
        - force_refresh: Whether to force refresh data even if it exists
        
        Returns:
        - Dictionary with fetched data status for each timeframe
        """
        if timeframes is None:
            timeframes = list(self.timeframe_configs.keys())
        
        results = {}
        
        for timeframe in timeframes:
            if timeframe not in self.timeframe_configs:
                logger.warning(f"Unsupported timeframe: {timeframe}")
                continue
            
            config = self.timeframe_configs[timeframe]
            
            try:
                # Check if data already exists and is recent
                if not force_refresh and self._data_exists(ticker, timeframe):
                    last_modified = self._get_last_modified(ticker, timeframe)
                    if (datetime.now() - last_modified).total_seconds() < 86400:  # 24 hours
                        logger.info(f"Using existing data for {ticker} at {timeframe} timeframe (last updated: {last_modified})")
                        results[timeframe] = {"status": "cached", "last_modified": last_modified}
                        continue
                
                # Fetch data
                logger.info(f"Fetching {timeframe} data for {ticker}")
                
                # Use the maximum available period for the timeframe
                period = config["max_period"]
                
                # Download data with retry logic
                data = self._download_with_retry(ticker, config["interval"], period)
                
                if data.empty:
                    logger.warning(f"No data returned for {ticker} at {timeframe} timeframe")
                    results[timeframe] = {"status": "empty", "message": "No data returned"}
                    continue
                
                # Save data
                self._save_data(ticker, timeframe, data)
                
                results[timeframe] = {
                    "status": "success", 
                    "rows": len(data),
                    "start_date": data.index.min().strftime('%Y-%m-%d'),
                    "end_date": data.index.max().strftime('%Y-%m-%d')
                }
                
            except Exception as e:
                logger.error(f"Error fetching {timeframe} data for {ticker}: {str(e)}")
                logger.debug(traceback.format_exc())
                results[timeframe] = {"status": "error", "message": str(e)}
        
        # Save metadata about the fetch operation
        self._save_metadata(ticker, results)
        
        return results
    
    def _download_with_retry(self, ticker, interval, period, max_retries=3, retry_delay=5):
        """
        Download data with retry logic
        
        Parameters:
        - ticker: Stock symbol
        - interval: Data interval ('1d', '1h', etc.)
        - period: Data period ('1y', '6mo', etc.)
        - max_retries: Maximum number of retry attempts
        - retry_delay: Delay between retries in seconds
        
        Returns:
        - DataFrame with downloaded data
        """
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                data = yf.download(
                    ticker, 
                    interval=interval, 
                    period=period, 
                    auto_adjust=False, 
                    progress=False
                )
                return data
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"Retry {retries}/{max_retries} for {ticker} at {interval} timeframe: {str(e)}")
                time.sleep(retry_delay)
        
        # If we get here, all retries failed
        raise last_error or ValueError(f"Failed to download data for {ticker} at {interval} timeframe after {max_retries} retries")
    
    def _save_data(self, ticker, timeframe, data):
        """
        Save data to CSV file (overwrites existing cleanly)
        """
        ticker_dir = os.path.join(self.base_dir, timeframe)
        os.makedirs(ticker_dir, exist_ok=True)

        file_path = os.path.join(ticker_dir, f"{ticker}.csv")

        # Always overwrite old file completely (prevents header duplication)
        try:
            data.to_csv(file_path, index=True)
            logger.info(f"✅ Saved clean data: {ticker} [{timeframe}] with {len(data)} rows.")
        except Exception as e:
            logger.error(f"❌ Failed to save data for {ticker} [{timeframe}]: {e}")

    def _clean_existing_csv(self, filepath):
        """
        Clean existing CSV file by removing duplicate header rows.

        Parameters:
        - filepath: Full path to the CSV file
        """
        if not os.path.exists(filepath):
            return  # nothing to clean

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            header = lines[0]
            unique_lines = [header]
            for line in lines[1:]:
                if line.strip() != header.strip():
                    unique_lines.append(line)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(unique_lines)

            logger.info(f"🧹 Cleaned duplicate headers in: {filepath}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to clean file {filepath}: {e}")
    
    def _save_metadata(self, ticker, results):
        """
        Save metadata about the fetch operation
        
        Parameters:
        - ticker: Stock symbol
        - results: Dictionary with fetch results
        """
        metadata = {
            "ticker": ticker,
            "fetch_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "results": results
        }
        
        # Save metadata to JSON
        file_path = os.path.join(self.base_dir, f"{ticker}_metadata.json")
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=datetime_json_serializer)
            
    def _data_exists(self, ticker, timeframe):
        """
        Check if data exists for a ticker at a specific timeframe
        
        Parameters:
        - ticker: Stock symbol
        - timeframe: Timeframe string ('1d', '1h', '15m')
        
        Returns:
        - Boolean indicating whether data exists
        """
        file_path = os.path.join(self.base_dir, timeframe, f"{ticker}.csv")
        return os.path.exists(file_path)
    
    def _get_last_modified(self, ticker, timeframe):
        """
        Get last modified time for a ticker's data file
        
        Parameters:
        - ticker: Stock symbol
        - timeframe: Timeframe string ('1d', '1h', '15m')
        
        Returns:
        - Datetime object representing last modified time
        """
        file_path = os.path.join(self.base_dir, timeframe, f"{ticker}.csv")
        if os.path.exists(file_path):
            return datetime.fromtimestamp(os.path.getmtime(file_path))
        return datetime.min
    
    def load_data(self, ticker, timeframe):
        """
        Load data for a ticker at a specific timeframe
        
        Parameters:
        - ticker: Stock symbol
        - timeframe: Timeframe string ('1d', '1h', '15m')
        
        Returns:
        - DataFrame with loaded data or None if data doesn't exist
        """
        file_path = os.path.join(self.base_dir, timeframe, f"{ticker}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"No data file found for {ticker} at {timeframe} timeframe")
            return None
        
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(data)} rows of {timeframe} data for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Error loading data for {ticker} at {timeframe} timeframe: {str(e)}")
            return None
    
    def fetch_multiple_tickers(self, tickers, timeframes=None, force_refresh=False):
        """
        Fetch data for multiple tickers
        
        Parameters:
        - tickers: List of stock symbols
        - timeframes: List of timeframe strings
        - force_refresh: Whether to force refresh data
        
        Returns:
        - Dictionary with fetch results for each ticker
        """
        results = {}
        
        for ticker in tickers:
            logger.info(f"Fetching data for {ticker}")
            ticker_results = self.fetch_data(ticker, timeframes, force_refresh)
            results[ticker] = ticker_results
        
        return results
    
    def validate_data(self, ticker, start_date=None, end_date=None, timeframes=None):
        """
        Validate data for a ticker across timeframes
        
        Parameters:
        - ticker: Stock symbol
        - start_date: Start date for validation (string or datetime)
        - end_date: End date for validation (string or datetime)
        - timeframes: List of timeframe strings to validate
        
        Returns:
        - Dictionary with validation results
        """
        if timeframes is None:
            timeframes = list(self.timeframe_configs.keys())
        
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        validation_results = {
            "ticker": ticker,
            "validation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "timeframes": {}
        }
        
        for timeframe in timeframes:
            # Load data
            data = self.load_data(ticker, timeframe)
            if data is None:
                validation_results["timeframes"][timeframe] = {
                    "status": "missing",
                    "message": "No data file found"
                }
                continue
            
            # Check if data is empty
            if data.empty:
                validation_results["timeframes"][timeframe] = {
                    "status": "empty",
                    "message": "Data file is empty"
                }
                continue
            
            # Check date range
            data_start = data.index.min()
            data_end = data.index.max()
            
            timeframe_result = {
                "status": "valid",
                "rows": len(data),
                "data_start": data_start.strftime('%Y-%m-%d'),
                "data_end": data_end.strftime('%Y-%m-%d'),
                "issues": []
            }
            
            # Check if data covers the requested date range
            if start_date is not None and data_start > start_date:
                timeframe_result["issues"].append({
                    "type": "date_range",
                    "message": f"Data starts at {data_start.strftime('%Y-%m-%d')} but requested start date is {start_date.strftime('%Y-%m-%d')}"
                })
            
            if end_date is not None and data_end < end_date:
                timeframe_result["issues"].append({
                    "type": "date_range",
                    "message": f"Data ends at {data_end.strftime('%Y-%m-%d')} but requested end date is {end_date.strftime('%Y-%m-%d')}"
                })
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                timeframe_result["issues"].append({
                    "type": "missing_columns",
                    "message": f"Missing required columns: {', '.join(missing_columns)}"
                })
            
            # Check for gaps in the data
            if timeframe == "1d":
                # For daily data, check for missing trading days
                business_days = pd.date_range(start=data_start, end=data_end, freq='B')
                missing_days = business_days.difference(data.index)
                if len(missing_days) > 0:
                    timeframe_result["issues"].append({
                        "type": "data_gaps",
                        "message": f"Missing {len(missing_days)} trading days",
                        "missing_days": [day.strftime('%Y-%m-%d') for day in missing_days]
                    })
            
            # Update status based on issues
            if timeframe_result["issues"]:
                timeframe_result["status"] = "issues_found"
            
            validation_results["timeframes"][timeframe] = timeframe_result
        
        # Save validation results
        self._save_validation_results(ticker, validation_results)
        
        return validation_results
    
    def _save_validation_results(self, ticker, validation_results):
        """
        Save validation results to JSON file
        
        Parameters:
        - ticker: Stock symbol
        - validation_results: Dictionary with validation results
        """
        file_path = os.path.join(self.base_dir, "validation", f"{ticker}_validation.json")
        with open(file_path, 'w') as f:
            json.dump(validation_results, f, indent=4)
        logger.info(f"Saved validation results for {ticker} to {file_path}")

# Example usage
if __name__ == "__main__":
    # Create data fetcher
    fetcher = VPADataFetcher()
    
    # Fetch data for a ticker
    ticker = "AAPL"
    results = fetcher.fetch_data(ticker)
    print(f"Fetch results for {ticker}:")
    print(json.dumps(results, indent=4))
    
    # Validate data
    validation_results = fetcher.validate_data(ticker)
    print(f"Validation results for {ticker}:")
    print(json.dumps(validation_results, indent=4))

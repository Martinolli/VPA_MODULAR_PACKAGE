"""
VPA Data Module with Enhanced Error Handling

This module provides data access for VPA analysis with robust error handling.
"""

import os
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
from polygon import RESTClient
from polygon.rest.models import Agg

from .vpa_logger import get_logger
from .config_manager import get_config_manager

# Constants for error handling
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 30  # seconds
REQUEST_TIMEOUT = 30  # seconds

# Error categories
class ErrorCategory:
    NETWORK = "NETWORK_ERROR"
    API = "API_ERROR"
    AUTHENTICATION = "AUTHENTICATION_ERROR"
    RATE_LIMIT = "RATE_LIMIT_ERROR"
    DATA_PROCESSING = "DATA_PROCESSING_ERROR"
    UNKNOWN = "UNKNOWN_ERROR"

class PolygonIOProvider:
    """Data provider using Polygon.io API with enhanced error handling"""
    
    def __init__(self):
        """Initialize the Polygon.io data provider"""
        self.logger = get_logger(module_name="PolygonIOProvider")
        self.config_manager = get_config_manager()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Polygon.io REST client with error handling"""
        try:
            # Get API key from config manager
            api_key = self.config_manager.get_api_key('polygon')
            
            # Validate API key
            if not api_key:
                self.logger.error("Polygon.io API key not found")
                raise ValueError("Polygon.io API key not found")
            
            # Initialize client
            self.client = RESTClient(api_key)
            self.logger.debug("Polygon.io client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Polygon.io client: {e}")
            self.client = None
    
    def _validate_client(self) -> bool:
        """Validate that the client is initialized"""
        if self.client is None:
            self.logger.warning("Polygon.io client not initialized, attempting to initialize")
            self._initialize_client()
        
        return self.client is not None
    
    def _handle_error(self, error: Exception, ticker: str, interval: str, 
                      attempt: int = 0) -> Tuple[ErrorCategory, str]:
        """
        Handle errors with categorization and logging
        
        Parameters:
        - error: The exception that occurred
        - ticker: Stock symbol
        - interval: Timeframe interval
        - attempt: Current retry attempt
        
        Returns:
        - Tuple of (error_category, error_message)
        """
        error_str = str(error)
        error_category = ErrorCategory.UNKNOWN
        
        # Categorize error
        if "ConnectionError" in error_str or "timeout" in error_str.lower():
            error_category = ErrorCategory.NETWORK
            message = f"Network error fetching data for {ticker} at {interval} timeframe: {error}"
        elif "401" in error_str or "Unauthorized" in error_str:
            error_category = ErrorCategory.AUTHENTICATION
            message = f"Authentication error with Polygon.io API: {error}"
        elif "429" in error_str or "Too Many Requests" in error_str:
            error_category = ErrorCategory.RATE_LIMIT
            message = f"Rate limit exceeded for Polygon.io API: {error}"
        elif "404" in error_str or "Not Found" in error_str:
            error_category = ErrorCategory.API
            message = f"Resource not found for {ticker} at {interval} timeframe: {error}"
        elif "format" in error_str.lower() or "invalid" in error_str.lower():
            error_category = ErrorCategory.DATA_PROCESSING
            message = f"Data format error for {ticker} at {interval} timeframe: {error}"
        else:
            message = f"Unknown error fetching data for {ticker} at {interval} timeframe: {error}"
        
        # Log with appropriate level based on attempt and category
        if attempt < MAX_RETRIES - 1 and error_category in [ErrorCategory.NETWORK, ErrorCategory.RATE_LIMIT]:
            self.logger.warning(message)
        else:
            self.logger.error(message)
            if error_category == ErrorCategory.AUTHENTICATION:
                self.logger.error("Please check your Polygon.io API key")
        
        return error_category, message
    
    def _calculate_retry_delay(self, attempt: int, error_category: ErrorCategory) -> float:
        """
        Calculate retry delay with exponential backoff
        
        Parameters:
        - attempt: Current retry attempt
        - error_category: Category of the error
        
        Returns:
        - Delay in seconds
        """
        # Base exponential backoff
        delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
        
        # Add jitter to prevent thundering herd
        delay = delay * (0.5 + 0.5 * np.random.random())
        
        # Increase delay for rate limiting
        if error_category == ErrorCategory.RATE_LIMIT:
            delay = delay * 2
        
        return delay
    
    def _should_retry(self, error_category: ErrorCategory, attempt: int) -> bool:
        """
        Determine if a retry should be attempted
        
        Parameters:
        - error_category: Category of the error
        - attempt: Current retry attempt
        
        Returns:
        - True if retry should be attempted, False otherwise
        """
        # Don't retry authentication errors
        if error_category == ErrorCategory.AUTHENTICATION:
            return False
        
        # Don't retry if max attempts reached
        if attempt >= MAX_RETRIES - 1:
            return False
        
        # Always retry network and rate limit errors
        if error_category in [ErrorCategory.NETWORK, ErrorCategory.RATE_LIMIT]:
            return True
        
        # Retry API errors once
        if error_category == ErrorCategory.API and attempt == 0:
            return True
        
        return False
    
    def _process_aggregates(self, aggs: List[Agg]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process aggregates into price and volume data
        
        Parameters:
        - aggs: List of aggregate data from Polygon.io
        
        Returns:
        - Tuple of (price_df, volume_series)
        """
        if not aggs:
            return pd.DataFrame(), pd.Series()
        
        # Extract data
        data = []
        for agg in aggs:
            data.append({
                'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Extract price and volume data
        price_df = df[['open', 'high', 'low', 'close']]
        volume_series = df['volume']
        
        return price_df, volume_series
    
    def get_data(self, ticker: str, interval: str = "1d", period: str = "1y") -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Get price and volume data for a ticker with robust error handling
        
        Parameters:
        - ticker: Stock symbol
        - interval: Timeframe interval (e.g., "1d", "1h")
        - period: Period to fetch (e.g., "1d", "1m", "1y")
        
        Returns:
        - Tuple of (price_df, volume_series) or None if data could not be fetched
        """
        # Validate client
        if not self._validate_client():
            self.logger.error("Cannot fetch data: Polygon.io client not initialized")
            return None
        
        # Parse interval
        multiplier, timespan = self._parse_interval(interval)
        if multiplier is None or timespan is None:
            self.logger.error(f"Unsupported interval format: {interval}")
            return None
        
        # Calculate date range
        end_date = datetime.now()
        start_date = self._calculate_start_date(end_date, period)
        
        # Fetch data with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                self.logger.debug(f"Fetching data for {ticker} at {interval} timeframe (Attempt {attempt + 1}/{MAX_RETRIES})")
                
                # Fetch aggregates
                aggs = self.client.get_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    limit=50000
                )
                
                # Process aggregates
                price_df, volume_series = self._process_aggregates(aggs)
                
                # Validate data
                if price_df.empty:
                    self.logger.warning(f"No data returned for {ticker} at {interval} timeframe")
                    return None
                
                self.logger.debug(f"Successfully fetched {len(price_df)} data points for {ticker} at {interval} timeframe")
                return price_df, volume_series
            
            except Exception as e:
                # Handle error
                error_category, message = self._handle_error(e, ticker, interval, attempt)
                
                # Determine if retry should be attempted
                if self._should_retry(error_category, attempt):
                    # Calculate retry delay
                    delay = self._calculate_retry_delay(attempt, error_category)
                    
                    self.logger.info(f"Retrying in {delay:.2f} seconds (Attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Error fetching data for {ticker} at {interval} timeframe: {e}")
                    return None
        
        self.logger.error(f"Failed to fetch data for {ticker} at {interval} timeframe after {MAX_RETRIES} attempts")
        return None
    
    def _parse_interval(self, interval: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Parse interval string into multiplier and timespan
        
        Parameters:
        - interval: Interval string (e.g., "1d", "4h")
        
        Returns:
        - Tuple of (multiplier, timespan)
        """
        try:
            # Extract multiplier and timespan
            if interval.endswith('m'):
                return int(interval[:-1]), 'minute'
            elif interval.endswith('h'):
                return int(interval[:-1]), 'hour'
            elif interval.endswith('d'):
                return int(interval[:-1]), 'day'
            elif interval.endswith('w'):
                return int(interval[:-1]), 'week'
            else:
                self.logger.error(f"Unsupported interval format: {interval}")
                return None, None
        except Exception as e:
            self.logger.error(f"Error parsing interval {interval}: {e}")
            return None, None
    
    def _calculate_start_date(self, end_date: datetime, period: str) -> datetime:
        """
        Calculate start date based on end date and period
        
        Parameters:
        - end_date: End date
        - period: Period string (e.g., "1d", "1m", "1y")
        
        Returns:
        - Start date
        """
        try:
            # Parse period
            if period.endswith('d'):
                days = int(period[:-1])
                return end_date - timedelta(days=days)
            elif period.endswith('w'):
                weeks = int(period[:-1])
                return end_date - timedelta(weeks=weeks)
            elif period.endswith('m'):
                months = int(period[:-1])
                return end_date - timedelta(days=months * 30)
            elif period.endswith('y'):
                years = int(period[:-1])
                return end_date - timedelta(days=years * 365)
            else:
                self.logger.warning(f"Unsupported period format: {period}, defaulting to 1 year")
                return end_date - timedelta(days=365)
        except Exception as e:
            self.logger.error(f"Error calculating start date with period {period}: {e}")
            # Default to 1 year
            return end_date - timedelta(days=365)

class MultiTimeframeProvider:
    """Provider for multiple timeframes"""

    def __init__(self, data_provider):
            self.data_provider = data_provider # This will now be an instance of PolygonIOProvider

    def get_multi_timeframe_data(self, ticker, timeframes):
        timeframe_data = {}
        
        for tf_config in timeframes:
            interval = tf_config['interval']
            period = tf_config.get('period', '1y')  # Default to 1 year if not provided
            
            tf_key = f"{interval}"
            
            try:
                # Only pass the parameters that get_data accepts
                price_data, volume_data = self.data_provider.get_data(
                    ticker, 
                    interval=interval, 
                    period=period
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
            print(f"Warning: Could not fetch data for any requested timeframe for {ticker}")
        
        return timeframe_data
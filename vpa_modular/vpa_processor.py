"""
VPA Processor Module

This module provides data processing and feature calculation for the VPA algorithm.
"""

import pandas as pd
import numpy as np
from .vpa_config import VPAConfig

class DataProcessor:
    """Process raw price and volume data for VPA analysis"""
    
    def __init__(self, config=None):
        self.config = config or VPAConfig()
        self.volume_thresholds = self.config.get_volume_thresholds()
        self.candle_thresholds = self.config.get_candle_thresholds()
    
    def preprocess_data(self, price_data, volume_data, lookback_period=None):
        """
        Preprocess price and volume data for VPA analysis
        
        Parameters:
        - price_data: DataFrame with OHLC data
        - volume_data: Series with volume data
        - lookback_period: Optional override for lookback period
        
        Returns:
        - Dictionary with processed data
        """
        # Use config lookback period if not specified
        if lookback_period is None:
            lookback_period = self.volume_thresholds.get("lookback_period", 50)
        
        # Align and ensure datetime index
        price_data = price_data.copy()
        volume_data = volume_data.copy()
        
        # Align both datasets (important if there's any mismatch in timestamps)
        price_data, volume_data = price_data.align(volume_data, join='inner', axis=0)

        # Create dictionary to hold all calculated data
        processed_data = {
            "price": price_data,
            "volume": volume_data
        }

        # Calculate candle properties
        processed_data = self.calculate_candle_properties(processed_data)
        
        # Calculate volume metrics
        processed_data = self.calculate_volume_metrics(processed_data, lookback_period)
        
        # Classify candles and volume
        processed_data["volume_class"] = self.classify_volume(processed_data["volume_ratio"])
        processed_data["candle_class"] = self.classify_candles(processed_data)
        
        # Calculate trend metrics
        processed_data["price_direction"] = self.calculate_price_direction(price_data, lookback_period)
        processed_data["volume_direction"] = self.calculate_volume_direction(volume_data, lookback_period)

        return processed_data
    
    def calculate_candle_properties(self, processed_data):
        """
        Calculate candle properties
        
        Parameters:
        - processed_data: Dictionary with price and volume data
        
        Returns:
        - Updated processed_data dictionary with candle properties
        """
        price_data = processed_data["price"]
        
        # Calculate spread (body size)
        processed_data["spread"] = abs(price_data["close"] - price_data["open"])
        
        # Calculate body percentage of total range
        processed_data["body_percent"] = processed_data["spread"] / (price_data["high"] - price_data["low"])
        
        # Calculate upper and lower wicks
        processed_data["upper_wick"] = price_data["high"] - price_data[["open", "close"]].max(axis=1)
        processed_data["lower_wick"] = price_data[["open", "close"]].min(axis=1) - price_data["low"]
        
        return processed_data
    
    def calculate_volume_metrics(self, processed_data, lookback_period):
        """
        Calculate volume metrics
        
        Parameters:
        - processed_data: Dictionary with price and volume data
        - lookback_period: Number of periods to look back for average volume
        
        Returns:
        - Updated processed_data dictionary with volume metrics
        """
        volume_data = processed_data["volume"]
        
        # Calculate average volume
        processed_data["avg_volume"] = volume_data.rolling(window=lookback_period).mean()
        
        # Calculate volume ratio
        processed_data["volume_ratio"] = volume_data / processed_data["avg_volume"]
        
        return processed_data
    
    def classify_volume(self, volume_ratio):
        """
        Classify volume as VERY_HIGH, HIGH, AVERAGE, LOW, or VERY_LOW
        based on volume ratio.
        
        Parameters:
        - volume_ratio: Series with volume ratio values
        
        Returns:
        - Series with volume classifications
        """
        # Ensure the index is clean and flat
        volume_ratio = volume_ratio.copy()
        volume_ratio.index = pd.to_datetime(volume_ratio.index)
        
        # Get thresholds from config
        very_high_threshold = self.volume_thresholds.get("very_high_threshold", 2.0)
        high_threshold = self.volume_thresholds.get("high_threshold", 1.3)
        low_threshold = self.volume_thresholds.get("low_threshold", 0.6)
        very_low_threshold = self.volume_thresholds.get("very_low_threshold", 0.3)
        
        # Initialize volume class with AVERAGE
        volume_class = pd.Series("AVERAGE", index=volume_ratio.index, dtype="object")
        
        # Apply classification with thresholds from config
        volume_class.loc[volume_ratio >= very_high_threshold] = "VERY_HIGH"
        volume_class.loc[(volume_ratio >= high_threshold) & (volume_ratio < very_high_threshold)] = "HIGH"
        volume_class.loc[(volume_ratio <= low_threshold) & (volume_ratio > very_low_threshold)] = "LOW"
        volume_class.loc[volume_ratio <= very_low_threshold] = "VERY_LOW"

        return volume_class
    
    def classify_candles(self, data):
        """
        Classify candles based on spread and wicks
        
        Parameters:
        - data: Dictionary with processed data
        
        Returns:
        - Series with candle classifications
        """
        # Get thresholds from config
        wide_threshold = self.candle_thresholds.get("wide_threshold", 1.3)
        narrow_threshold = self.candle_thresholds.get("narrow_threshold", 0.6)
        wick_threshold = self.candle_thresholds.get("wick_threshold", 1.5)
        
        # Calculate average spread for relative comparison
        avg_spread = data["spread"].rolling(window=20).mean()
        spread_ratio = data["spread"] / avg_spread
        
        # Initialize with default classification
        candle_class = pd.Series(index=data["price"].index, data="NEUTRAL")
        
        # Classify based on spread with thresholds from config
        candle_class[spread_ratio >= wide_threshold] = "WIDE"
        candle_class[spread_ratio <= narrow_threshold] = "NARROW"
        
        # Refine classification based on wicks
        for idx in candle_class.index:
            if data["upper_wick"].loc[idx] > data["spread"].loc[idx] * wick_threshold:
                candle_class.loc[idx] += "_UPPER_WICK"
            if data["lower_wick"].loc[idx] > data["spread"].loc[idx] * wick_threshold:
                candle_class.loc[idx] += "_LOWER_WICK"
        
        return candle_class
    
    def calculate_price_direction(self, price_data, lookback_period=10):
        """
        Calculate the direction of price movement
        
        Parameters:
        - price_data: DataFrame with OHLC data
        - lookback_period: Number of periods to consider
        
        Returns:
        - Series with price direction classifications
        """
        # Calculate short-term price direction
        price_change = price_data["close"].diff(lookback_period)
        direction = pd.Series(index=price_data.index, data="SIDEWAYS")
        direction[price_change > 0] = "UP"
        direction[price_change < 0] = "DOWN"
        
        return direction
    
    def calculate_volume_direction(self, volume_data, lookback_period=10):
        """
        Calculate the direction of volume movement
        
        Parameters:
        - volume_data: Series with volume data
        - lookback_period: Number of periods to consider
        
        Returns:
        - Series with volume direction classifications
        """
        # Calculate short-term volume direction
        volume_change = volume_data.diff(lookback_period)
        direction = pd.Series(index=volume_data.index, data="FLAT")
        direction[volume_change > 0] = "INCREASING"
        direction[volume_change < 0] = "DECREASING"
        
        return direction

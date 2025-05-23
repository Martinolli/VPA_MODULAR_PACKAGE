"""
VPA Logger Module

This module provides logging functionality for the VPA algorithm.
"""

import logging
import os
from datetime import datetime

class VPALogger:
    """Logging framework for VPA"""
    
    def __init__(self, log_level="INFO", log_file=None):
        """
        Initialize the VPA logger
        
        Parameters:
        - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - log_file: Optional path to log file
        """
        self.logger = self._setup_logger(log_level, log_file)
    
    def _setup_logger(self, log_level, log_file):
        """
        Set up logger with appropriate configuration
        
        Parameters:
        - log_level: Logging level
        - log_file: Optional path to log file
        
        Returns:
        - Configured logger
        """
        # Create logger
        logger = logging.getLogger("VPA")
        
        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log file specified
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log exception with traceback"""
        self.logger.exception(message)
    
    def log_analysis_start(self, ticker, timeframes):
        """
        Log the start of an analysis
        
        Parameters:
        - ticker: Stock symbol
        - timeframes: List of timeframes
        """
        timeframe_str = ", ".join([f"{tf['interval']}" for tf in timeframes])
        self.info(f"Starting VPA analysis for {ticker} on timeframes: {timeframe_str}")
    
    def log_analysis_complete(self, ticker, signal):
        """
        Log the completion of an analysis
        
        Parameters:
        - ticker: Stock symbol
        - signal: Signal information
        """
        self.info(f"Completed VPA analysis for {ticker}. Signal: {signal['type']} ({signal['strength']})")
    
    def log_error(self, ticker, error):
        """
        Log an error during analysis
        
        Parameters:
        - ticker: Stock symbol
        - error: Error information
        """
        self.error(f"Error analyzing {ticker}: {error}")
    
    def log_data_retrieval(self, ticker, timeframe, success):
        """
        Log data retrieval status
        
        Parameters:
        - ticker: Stock symbol
        - timeframe: Timeframe information
        - success: Whether retrieval was successful
        """
        if success:
            self.debug(f"Successfully retrieved {timeframe} data for {ticker}")
        else:
            self.warning(f"Failed to retrieve {timeframe} data for {ticker}")
    
    def log_pattern_detection(self, ticker, pattern, detected):
        """
        Log pattern detection
        
        Parameters:
        - ticker: Stock symbol
        - pattern: Pattern name
        - detected: Whether pattern was detected
        """
        if detected:
            self.info(f"Detected {pattern} pattern for {ticker}")
        else:
            self.debug(f"No {pattern} pattern detected for {ticker}")
    
    def log_performance(self, operation, start_time, end_time=None):
        """
        Log performance metrics
        
        Parameters:
        - operation: Operation name
        - start_time: Start time
        - end_time: Optional end time (defaults to now)
        """
        if end_time is None:
            end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        self.debug(f"Performance: {operation} took {duration:.2f} seconds")

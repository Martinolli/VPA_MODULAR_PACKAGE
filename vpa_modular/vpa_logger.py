"""
VPA Logger Module

This module provides standardized logging functionality for the VPA system.
It supports console and file logging, automatic log directory creation,
log rotation, and specialized logging methods for VPA operations.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, Union

def get_project_root():
    return Path(__file__).parent.parent

# Default log directory
DEFAULT_LOG_DIR = os.path.join(get_project_root(), ".vpa", "logs")

class VPALogger:
    """Standardized logging framework for VPA system"""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None, 
                 module_name: Optional[str] = None, enable_rotation: bool = True,
                 max_bytes: int = 10485760, backup_count: int = 5):
        """
        Initialize the VPA logger
        
        Parameters:
        - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - log_file: Optional path to log file. If None, a default path will be used
        - module_name: Optional module name for logger identification
        - enable_rotation: Whether to enable log rotation (default: True)
        - max_bytes: Maximum log file size in bytes before rotation (default: 10MB)
        - backup_count: Number of backup files to keep (default: 5)
        """
        self.module_name = module_name or "VPA"
        
        # If log_file is not specified, create a default one based on module name
        if log_file is None:
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
            log_file = os.path.join(DEFAULT_LOG_DIR, f"{self.module_name.lower()}.log")
        
        self.log_file = log_file
        self.enable_rotation = enable_rotation
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.logger = self._setup_logger(log_level)
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """
        Set up logger with appropriate configuration
        
        Parameters:
        - log_level: Logging level
        
        Returns:
        - Configured logger
        """
        # Create logger
        logger = logging.getLogger(self.module_name)
        
        # Prevent adding handlers multiple times
        if logger.handlers:
            return logger
        
        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log file specified
        if self.log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Use rotating file handler if enabled
            if self.enable_rotation:
                file_handler = RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count
                )
            else:
                file_handler = logging.FileHandler(self.log_file)
                
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message: str) -> None:
        """Log exception with traceback"""
        self.logger.exception(message)
    
    def log_analysis_start(self, ticker: str, timeframes: list) -> None:
        """
        Log the start of an analysis
        
        Parameters:
        - ticker: Stock symbol
        - timeframes: List of timeframes
        """
        timeframe_str = ", ".join([f"{tf['interval']}" for tf in timeframes])
        self.info(f"Starting VPA analysis for {ticker} on timeframes: {timeframe_str}")
    
    def log_analysis_complete(self, ticker: str, signal: Dict[str, Any]) -> None:
        """
        Log the completion of an analysis
        
        Parameters:
        - ticker: Stock symbol
        - signal: Signal information
        """
        self.info(f"Completed VPA analysis for {ticker}. Signal: {signal['type']} ({signal['strength']})")
    
    def log_error(self, ticker: str, error: Union[str, Exception]) -> None:
        """
        Log an error during analysis
        
        Parameters:
        - ticker: Stock symbol
        - error: Error information
        """
        self.error(f"Error analyzing {ticker}: {error}")
    
    def log_data_retrieval(self, ticker: str, timeframe: str, success: bool) -> None:
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
    
    def log_pattern_detection(self, ticker: str, pattern: str, detected: bool) -> None:
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
    
    def log_performance(self, operation: str, start_time: datetime, end_time: Optional[datetime] = None) -> None:
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


# Create a singleton logger instance for each module
_loggers = {}

def get_logger(module_name: str = "VPA", log_level: str = "INFO", 
               log_file: Optional[str] = None, enable_rotation: bool = True) -> VPALogger:
    """
    Get a logger instance for the specified module
    
    Parameters:
    - module_name: Module name for logger identification
    - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - log_file: Optional path to log file
    - enable_rotation: Whether to enable log rotation
    
    Returns:
    - VPALogger instance
    """
    global _loggers
    
    if module_name not in _loggers:
        _loggers[module_name] = VPALogger(
            log_level=log_level,
            log_file=log_file,
            module_name=module_name,
            enable_rotation=enable_rotation
        )
    
    return _loggers[module_name]

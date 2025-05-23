#!/usr/bin/env python3
"""
VPA Facade Basic Functionality Test

This script tests the basic functionality of the VPAFacade class,
including initialization and simple analysis operations.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to sys.path to import VPA modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    """Print a formatted header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}\n")

def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_info(message):
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def test_vpa_facade_initialization():
    """Test VPAFacade initialization"""
    print_header("Testing VPAFacade Initialization")
    
    try:
        from vpa_modular.vpa_facade import VPAFacade
        from vpa_modular.vpa_config import VPAConfig
        from vpa_modular.vpa_logger import get_logger
        
        # Get logger
        logger = get_logger(module_name="TestVPAFacade")
        
        # Initialize with default config
        print_info("Initializing VPAFacade with default config...")
        facade = VPAFacade()
        print_success("VPAFacade initialized with default config")
        
        # Initialize with custom config
        print_info("Initializing VPAFacade with custom config...")
        config = VPAConfig()
        facade_custom = VPAFacade(config=config, logger=logger)
        print_success("VPAFacade initialized with custom config")
        
        return facade
    except Exception as e:
        print_error(f"Error initializing VPAFacade: {e}")
        return None

def test_basic_analysis(facade, ticker="AAPL", timeframe="1d"):
    """Test basic VPA analysis"""
    if not facade:
        print_error("VPAFacade not initialized, skipping analysis test")
        return False
    
    print_header(f"Testing Basic Analysis for {ticker} ({timeframe})")
    
    try:
        # Define timeframe
        timeframes = [{"interval": timeframe, "period": "1m"}]
        
        # Perform analysis
        print_info(f"Analyzing {ticker} on {timeframe} timeframe...")
        start_time = time.time()
        results = facade.analyze_ticker(ticker=ticker, timeframes=timeframes)
        end_time = time.time()
        
        if results:
            print_success(f"Analysis completed in {end_time - start_time:.2f} seconds")
            
            # Check for key components in results
            if "signal" in results:
                signal = results["signal"]
                print_info(f"Signal: {signal.get('type', 'N/A')} ({signal.get('strength', 'N/A')})")
            
            if "timeframe_analyses" in results:
                for tf, tf_data in results["timeframe_analyses"].items():
                    print_info(f"Timeframe {tf} analysis present")
            
            return True
        else:
            print_warning("Analysis returned no results")
            return False
    except Exception as e:
        print_error(f"Error performing analysis: {e}")
        return False

def main():
    """Main function to test VPAFacade functionality"""
    print_header("VPA Facade Basic Functionality Test")
    
    # Test initialization
    facade = test_vpa_facade_initialization()
    
    # Test basic analysis
    if facade:
        success = test_basic_analysis(facade)
    else:
        success = False
    
    # Print summary
    print_header("Test Summary")
    
    if success:
        print_success("VPAFacade basic functionality test passed!")
    else:
        print_warning("VPAFacade basic functionality test failed.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
VPA Sample Analysis Test Script

This script tests VPA analysis with sample tickers and handles potential API failures.
It demonstrates proper error handling and fallback mechanisms.
"""

import os
import sys
import time
import json
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

def save_results(results, ticker, output_dir="output"):
    """Save analysis results to a JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{ticker}_analysis_{timestamp}.json"
    
    # Save results to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_success(f"Results saved to {filename}")
    return filename

def test_sample_tickers(tickers=None, timeframes=None):
    """Test VPA analysis with sample tickers"""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    if timeframes is None:
        timeframes = [
            {"interval": "1d", "period": "1m"},
            {"interval": "1h", "period": "1w"}
        ]
    
    print_header(f"Testing VPA Analysis with Sample Tickers")
    
    try:
        from vpa_modular.vpa_facade import VPAFacade
        from vpa_modular.vpa_logger import get_logger
        from vpa_modular.config_manager import get_config_manager
        
        # Get logger and config manager
        logger = get_logger(module_name="SampleAnalysis")
        config_manager = get_config_manager()
        
        # Initialize VPAFacade
        facade = VPAFacade(logger=logger)
        
        # Test each ticker
        results = {}
        success_count = 0
        
        for ticker in tickers:
            print_header(f"Analyzing {ticker}")
            
            try:
                # Validate API key before analysis
                try:
                    polygon_key = config_manager.get_api_key('polygon')
                    if not config_manager.validate_api_key('polygon'):
                        print_warning("Polygon.io API key may be invalid, but proceeding anyway")
                except Exception as e:
                    print_error(f"Error validating Polygon.io API key: {e}")
                    print_info("Attempting analysis anyway...")
                
                # Perform analysis with retry mechanism
                max_retries = 3
                retry_delay = 2  # seconds
                
                for attempt in range(max_retries):
                    try:
                        print_info(f"Attempt {attempt + 1}/{max_retries}: Analyzing {ticker}...")
                        start_time = time.time()
                        ticker_results = facade.analyze_ticker(ticker=ticker, timeframes=timeframes)
                        end_time = time.time()
                        
                        if ticker_results:
                            print_success(f"Analysis completed in {end_time - start_time:.2f} seconds")
                            
                            # Extract signal information
                            signal = ticker_results.get("signal", {})
                            signal_type = signal.get("type", "N/A")
                            signal_strength = signal.get("strength", "N/A")
                            
                            print_info(f"Signal: {signal_type} ({signal_strength})")
                            
                            # Save results
                            results[ticker] = ticker_results
                            success_count += 1
                            
                            # Save to file
                            save_results(ticker_results, ticker)
                            
                            break  # Success, exit retry loop
                        else:
                            print_warning(f"Analysis returned no results on attempt {attempt + 1}")
                            if attempt < max_retries - 1:
                                print_info(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                    except Exception as e:
                        print_error(f"Error on attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            print_info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            print_error(f"All retry attempts failed for {ticker}")
            except Exception as e:
                print_error(f"Unrecoverable error analyzing {ticker}: {e}")
        
        # Print summary
        print_header("Analysis Summary")
        
        if success_count == len(tickers):
            print_success(f"Successfully analyzed all {len(tickers)} tickers!")
        else:
            print_warning(f"Successfully analyzed {success_count} out of {len(tickers)} tickers.")
        
        return results, success_count == len(tickers)
    
    except Exception as e:
        print_error(f"Error in test_sample_tickers: {e}")
        return {}, False

def main():
    """Main function to test VPA analysis with sample tickers"""
    print_header("VPA Sample Analysis Test")
    
    # Define sample tickers and timeframes
    tickers = ["AAPL", "MSFT", "GOOGL"]
    timeframes = [{"interval": "1d", "period": "1m"}]
    
    # Test sample tickers
    results, success = test_sample_tickers(tickers, timeframes)
    
    # Print summary
    print_header("Test Summary")
    
    if success:
        print_success("Sample analysis test passed!")
    else:
        print_warning("Sample analysis test completed with some failures.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

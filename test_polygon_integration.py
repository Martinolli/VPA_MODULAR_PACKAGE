"""
Test script for Polygon.io integration in VPA modules.

This script tests the data fetching capabilities of the new PolygonIOProvider
through the VPAFacade.

**Note:** This script requires the `POLYGON_API_KEY` environment variable to be set.
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add the project root to the Python path to allow importing vpa_modular
# This assumes the script is run from VPA_MODULAR_PACKAGE_LATEST directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_config import VPAConfig # For timeframe definitions

def run_test():
    # Load environment variables from .env file (if it exists)
    # This is where POLYGON_API_KEY should be defined by the user
    load_dotenv()

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set.")
        print("Please set it in your environment or in a .env file in the project root.")
        return

    print("POLYGON_API_KEY found. Initializing VPAFacade...")

    try:
        # Initialize VPAFacade (which in turn initializes PolygonIOProvider)
        # We can pass a config if needed, or let it use defaults
        # For this test, we might not need a full config if we specify timeframes directly
        facade = VPAFacade()
        print("VPAFacade initialized successfully.")
    except Exception as e:
        print(f"Error initializing VPAFacade: {e}")
        return

    test_ticker = "AAPL" # Apple Inc.
    
    # Define test timeframes similar to what VPAFacade might expect or use from config
    # These are examples; adjust if your VPAConfig.get_timeframes() returns a different structure
    # The PolygonIOProvider expects interval strings like '1d', '1h', '5m'
    # and a period string like '1y', '3mo', '1wk', '7d'
    # Or, start_date and end_date strings in 'YYYY-MM-DD' format.

    test_scenarios = [
        {"name": "Daily Data (1 year period)", "interval": "1d", "period": "1y", "start_date": None, "end_date": None},
        {"name": "Hourly Data (1 month period)", "interval": "1h", "period": "1mo", "start_date": None, "end_date": None},
        {"name": "5-Minute Data (7 days period)", "interval": "5m", "period": "7d", "start_date": None, "end_date": None},
        # Example with specific dates (adjust dates as needed for testing)
        # {"name": "Daily Data (Specific Dates)", "interval": "1d", "period": None, "start_date": "2023-01-01", "end_date": "2023-01-10"},
    ]

    for scenario in test_scenarios:
        print(f"\n--- Testing: {scenario['name']} for {test_ticker} ---")
        try:
            # Use the get_data method of the underlying provider for a direct test
            # The MultiTimeframeProvider expects a list of timeframe dicts
            # For a more direct test of PolygonIOProvider, we call its get_data directly.
            price_data, volume_data = facade.data_provider.get_data(
                ticker=test_ticker,
                interval=scenario["interval"],
                period=scenario["period"],
                start_date=scenario["start_date"],
                end_date=scenario["end_date"]
            )

            if not price_data.empty:
                print(f"Successfully fetched {scenario['interval']} data for {test_ticker}.")
                print(f"Price data shape: {price_data.shape}")
                print("Price data head:")
                print(price_data.head())
                print("Price data tail:")
                print(price_data.tail())
                print(f"Volume data length: {len(volume_data)}")
                print("Volume data head:")
                print(volume_data.head())
                print("Volume data tail:")
                print(volume_data.tail())
            else:
                print(f"No price data returned for {test_ticker} with interval {scenario['interval']}.")
            
            # Optional: Test through VPAFacade.analyze_ticker if you want to test the full pipeline
            # This requires VPAConfig to be set up correctly for timeframes if not passed explicitly.
            # print(f"\n--- Testing full analysis via VPAFacade for {scenario['name']} ---")
            # analysis_timeframes = [{'interval': scenario['interval'], 'period': scenario['period']}] # Example structure
            # if scenario['start_date'] and scenario['end_date']:
            #     analysis_timeframes = [{'interval': scenario['interval'], 'start_date': scenario['start_date'], 'end_date': scenario['end_date']}]
            
            # results = facade.analyze_ticker(test_ticker, timeframes=analysis_timeframes)
            # print(f"Full analysis signal for {test_ticker} ({scenario['interval']}): {results.get('signal', {}).get('type')}")

        except Exception as e:
            print(f"Error during test for {scenario['name']}: {e}")

if __name__ == "__main__":
    run_test()


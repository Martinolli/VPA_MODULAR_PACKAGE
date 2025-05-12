#--------------------------------------------------------------------------------------------------------------
# Date: 2025-05-01
# This script is designed to run the VPA analysis and generate visualizations and reports.
# It uses the VPAFacade to analyze multiple tickers and the VPAResultExtractor to extract and visualize results.
# To run analysis: "python run_analysis.py"
#--------------------------------------------------------------------------------------------------------------
# Import necessary libraries and modules

#--------------------------------------------------- Importing Libraries ---------------------------------------

import os
import json
from vpa_modular.vpa_result_extractor import VPAResultExtractor
from vpa_modular.vpa_visualizer import generate_all_visualizations, create_summary_report, create_dashboard
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_result_extractor import VPAResultExtractor, extract_testing_signals

#--------------------------------------------------- Main Function ---------------------------------------------
def main():
    print("📥 Loading VPA analysis results...")
    with open("vpa_analysis_results.json", "r") as f:
        results = json.load(f)
    
    vpa = VPAFacade() # Initialize the VPAFacade
    tickers = ["NFLX", "AAPL", "NVDA", "TSLA", "LCID", "SOUN", "QBTS", "SOFI"] # List of tickers to analyze
    results = {}

    for ticker in tickers:
        results[ticker] = vpa.analyze_ticker(ticker)
        print("\n--- Extracted Testing Patterns ---")
        testing_patterns = extract_testing_signals(results)
        for ticker, timeframes in testing_patterns.items():
            print(f"Ticker: {ticker}")
            for tf, tests in timeframes.items():
                print(f"  Timeframe: {tf}")
                for test in tests:
                    print(f"    Test: {test}")

    extractor = VPAResultExtractor(results) # Initialize the extractor with the results

    print("📊 Generating visualizations...")
    generate_all_visualizations(results, output_dir="vpa_analysis_output") 

    print("📝 Creating summary report...")
    create_summary_report(extractor, output_dir=".")  # Changed to root directory

    print("📈 Creating dashboard...")
    create_dashboard(extractor, output_dir=".")  # Changed to root directory

    print("✅ All reports and visualizations are ready.")

if __name__ == "__main__":
    main()
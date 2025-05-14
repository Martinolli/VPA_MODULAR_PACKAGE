#--------------------------------------------------------------------------------------------------------------
# Date: 2025-05-01
# This script is designed to run the VPA analysis and generate visualizations and reports.
# It uses the VPAFacade to analyze multiple tickers and the VPAResultExtractor to extract and visualize results.
# To run analysis: "python run_analysis.py"
#--------------------------------------------------------------------------------------------------------------
# Import necessary libraries and modules

#--------------------------------------------------- Importing Libraries ---------------------------------------

import json
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_result_extractor import VPAResultExtractor, extract_testing_signals
from vpa_modular.vpa_visualizer import generate_all_visualizations, create_summary_report, create_dashboard
from vpa_modular.vpa_utils import create_batch_report

#--------------------------------------------------- Main Function ---------------------------------------------
def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Results saved to {filename}")

def analyze_tickers(vpa, tickers):
    results = {}
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        results[ticker] = vpa.analyze_ticker(ticker)
    return results

def print_testing_patterns(results):
    testing_patterns = extract_testing_signals(results)
    for ticker, timeframes in testing_patterns.items():
        print(f"\nTicker: {ticker}")
        for tf, tests in timeframes.items():
            print(f"  Timeframe: {tf}")
            for test in tests:
                print(f"    Test: {test}")

def generate_reports_and_visualizations(results, vpa):
    extractor = VPAResultExtractor(results)

    print("📊 Generating visualizations...")
    generate_all_visualizations(results, output_dir="vpa_analysis_output")

    print("📝 Creating summary report...")
    create_summary_report(extractor, output_dir=".")

    print("📈 Creating dashboard...")
    create_dashboard(extractor, output_dir=".")

    print("📊 Generating batch report...")
    create_batch_report(vpa, list(results.keys()), output_dir="vpa_batch_reports")

def main():
    vpa = VPAFacade()  # Initialize the VPAFacade
    tickers = ["NFLX", "AAPL", "NVDA", "TSLA", "LCID", "SOUN", "QBTS", "SOFI", "ACHR", "INTC", "UBER", "GOLD", "BLK", "GOOG", "QCOM", "PLTR"]

    print("📊 Analyzing tickers...")
    results = analyze_tickers(vpa, tickers)

    print("\n📝 Extracting testing patterns...")
    print_testing_patterns(results)

    print("\n💾 Saving results to JSON...")
    save_results_to_json(results, "vpa_analysis_results.json")

    print("\n📈 Generating reports and visualizations...")
    generate_reports_and_visualizations(results, vpa)

    print("\n✅ Analysis complete. Check the output directories for results.")

if __name__ == "__main__":
    main()
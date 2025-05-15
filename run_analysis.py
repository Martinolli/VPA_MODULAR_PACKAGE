#--------------------------------------------------------------------------------------------------------------
# Date: 2025-05-01
# This script is designed to run the VPA analysis and generate visualizations and reports.
# It uses the VPAFacade to analyze multiple tickers and the VPAResultExtractor to extract and visualize results.
# To run analysis: "python run_analysis.py"
#--------------------------------------------------------------------------------------------------------------
# Import necessary libraries and modules

#--------------------------------------------------- Importing Libraries ---------------------------------------
import sys
import io
import os
import json
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_result_extractor import VPAResultExtractor, extract_testing_signals
from vpa_modular.vpa_visualizer import generate_all_visualizations, create_summary_report, create_dashboard
from vpa_modular.vpa_utils import create_batch_report
from vpa_modular.vpa_logger import VPALogger

#--------------------------------------------------- Main Function ---------------------------------------------
# create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Initialize logger
logger = VPALogger(log_level="INFO", log_file="logs/vpa_analysis.log")

def save_results_to_json(results, filename):
    output_path = os.path.join("output", filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Results saved to {output_path}")

def analyze_tickers(vpa, tickers):
    results = {}
    for ticker in tickers:
        logger.info(f"Analyzing {ticker}...")
        results[ticker] = vpa.analyze_ticker(ticker)
    return results

def print_testing_patterns(results):
    testing_patterns = extract_testing_signals(results)
    for ticker, timeframes in testing_patterns.items():
        logger.info(f"\nTicker: {ticker}")
        for tf, tests in timeframes.items():
            print(f"  Timeframe: {tf}")
            for test in tests:
                print(f"    Test: {test}")

def generate_reports_and_visualizations(results, vpa):
    extractor = VPAResultExtractor(results)

    logger.info(" Generating visualizations...")
    generate_all_visualizations(results, output_dir="vpa_analysis_output")

    logger.info(" Creating summary report...")
    create_summary_report(extractor, output_dir=".")

    logger.info(" Creating dashboard...")
    create_dashboard(extractor, output_dir=".")

    # logger.info(" Generating batch report...")
    # create_batch_report(vpa, list(results.keys()), output_dir="vpa_batch_reports")

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # Ensure stdout is UTF-8 encoded

    try:
        vpa = VPAFacade()  # Initialize the VPAFacade
        tickers = ["NFLX", "AAPL", "NVDA", "TSLA", "LCID", "SOUN", "QBTS", "SOFI" "ACHR", "INTC", "UBER", "GOLD", "BLK", "GOOG", "QCOM", "PLTR"]

        logger.info(" Analyzing tickers...")
        results = analyze_tickers(vpa, tickers)

        logger.info("\n Extracting testing patterns...")
        print_testing_patterns(results)

        logger.info("\n Saving results to JSON...")
        save_results_to_json(results, "vpa_analysis_results.json")

        logger.info("\n Generating reports and visualizations...")
        generate_reports_and_visualizations(results, vpa)

        logger.info("\n Analysis complete. Check the output directories for results.")

    except UnicodeEncodeError as e:
        logger.error(f"Encoding error occurred: {e}")
        logger.info("Try running the script in a console that supports UTF-8, or remove emoji characters from logging statements.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
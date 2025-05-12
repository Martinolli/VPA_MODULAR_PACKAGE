import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend
import os
import sys
import json

# Print Python path and current working directory
print("Python Path:")
for path in sys.path:
    print(path)
print("\nCurrent Working Directory:", os.getcwd())

# Append the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print("\nAppended to Python Path:", project_root)


try:
    from vpa_modular.vpa_utils import create_vpa_report, create_batch_report
    from vpa_modular.vpa_llm_interface import VPALLMInterface
    from vpa_modular.vpa_facade import VPAFacade
    from vpa_modular.vpa_logger import VPALogger
    print("\nSuccessfully imported vpa_modular modules")
except ImportError as e:
        print(f"\nFailed to import: {e}")
        sys.exit(1)

vpa = None
logger = None
llm_interface = None

def analyze_tickers(tickers):
    global vpa, logger
    logger.info("Starting VPA analysis...")
    all_results = {}
    for ticker in tickers:
        logger.info(f"Analyzing {ticker}...")
        results = vpa.analyze_ticker(ticker)
        all_results[ticker] = results
        report_files = create_vpa_report(results, "vpa_reports")
        logger.info(f"Report files created: {report_files}")
        plt.close('all')  # Close all figures
    
    # Save all results to a JSON file
    save_results_to_json(all_results, "vpa_analysis_results.json")
    
def perform_nl_analysis(tickers):
    global logger, llm_interface
    logger.info("NL Analysis:")
    nl_results = {}
    for ticker in tickers:
        logger.info(f"Analyzing {ticker}...")
        nl_analysis = llm_interface.get_ticker_analysis(ticker)
        nl_results[ticker] = nl_analysis
        for key, value in nl_analysis.items():
            logger.info(f"{key}: {value}")
        logger.info("\n")
        logger.info("--------------------------------------------------\n")
    
    # Save NL analysis results to a JSON file
    save_results_to_json(nl_results, "vpa_nl_analysis_results.json")

def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Results saved to {filename}")

def main():

    global vpa, logger, llm_interface

    # Initialize global objects
    logger = VPALogger(log_level="INFO", log_file="logs/vpa.log")
    vpa = VPAFacade()
    llm_interface = VPALLMInterface()


    tickers = ["NVDA"]
    logger.info("Starting VPA analysis...")
    logger.info("Tickers to analyze:")

    analyze_tickers(tickers)
    perform_nl_analysis(tickers)

    logger.info("Analysis complete.")

    signals = vpa.get_signals(tickers[-1])
    print("Signals:")
    for key, value in signals.items():
        print(f"{key}: {value}")
        print("\n")

    # Scanner Signal Example
    # Call the method with required and optional arguments
    tickers = ["MSFT"]  # Example list of tickers
    signal_type = "BUY"
    signal_strength = "MODERATE"
    timeframes = [
            {"interval": "30m", "period": "10d"},
            {"interval": "15m", "period": "10d"},
            {"interval": "5m", "period": "5d"}
        ]

    # Call the method

    scanner = vpa.scan_for_signals(tickers, signal_type, signal_strength, timeframes)

    if not scanner:
        print("No signals found.")
    else:
        print("Scanner Results:")
        for ticker, result in scanner.items():
            print(f"{ticker}: {result}")
            print("\n")
            for key, value in result.items():
                print(f"{key}: {value}")
                print("\n")


    """
    # Batch Report Analysis
    logger.info("Starting batch report analysis...")
    batch_report_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "META", "NFLX", "DIS", "INTC"]
    batch_report_files = create_batch_report(vpa, batch_report_tickers, "vpa_batch_reports")
    logger.info(f"Batch report files created: {batch_report_files}")

    
    # Define a list of tickers to analyze
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "META", "NFLX", "DIS", "INTC"]

    # Create a batch report
    report_files = create_batch_report(vpa, tickers, "vpa_batch_reports")

    # Print the generated report files
    print(f"Report files: {report_files}")


    config = VPAConfig()
    print(config.get_all())

    for key, value in config.get_all().items():
        print(f"{key}: {value}")


    llm_interface = VPALLMInterface()


    nl_analysis = llm_interface.analyze_ticker_nl("AAPL")
    print(f"NL Analysis: {nl_analysis}")


    print(llm_interface.generate_code_example('analyze_ticker'))
    print(llm_interface.get_ticker_analysis("AAPL"))

    from vpa_modular.vpa_llm_interface import VPALLMInterface

    # Initialize the VPA LLM interface
    vpa_llm = VPALLMInterface()

    # Example 1: Process a natural language query
    query = "What is accumulation in VPA?"
    response = vpa_llm.process_query(query)
    print("Query Response:")
    print(response)

    # Example 2: Get analysis for a specific ticker
    ticker = "AAPL"
    analysis = vpa_llm.get_ticker_analysis(ticker)
    print("\nTicker Analysis:")
    print(analysis)

    for key, value in analysis.items():
        print(f"{key}: {value}")

    goal = "swing_trading"
    parameters = vpa_llm.suggest_parameters(ticker, goal)
    print("\nSuggested Parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")

    task = "scan_market"
    code_example = vpa_llm.generate_code_example(task, ticker)
    print("\nGenerated Code Example:")
    print(code_example)


    # Example 3: Explain a specific VPA concept
    concept = "effort_vs_result"
    concept_explanation = vpa_llm.explain_vpa_concept(concept)
    print("\nConcept Explanation:")
    print(concept_explanation)

    # Example 4: Suggest parameters for a trading goal
    goal = "swing_trading"
    parameters = vpa_llm.suggest_parameters(ticker, goal)
    print("\nSuggested Parameters:")
    print(parameters)

    # Example 5: Generate a code example for a specific task
    task = "analyze_ticker"
    code_example = vpa_llm.generate_code_example(task, ticker)
    print("\nGenerated Code Example:")
    print(code_example)

    """

if __name__ == "__main__":
    main()
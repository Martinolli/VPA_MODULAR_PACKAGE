import os
import sys

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
    from vpa_modular.vpa_processor import DataProcessor
    from vpa_modular.vpa_config import VPAConfig
    from vpa_modular.vpa_facade import VPAFacade
    from vpa_modular.vpa_logger import VPALogger
    from vpa_modular.vpa_llm_interface import VPALLMInterface
    print("\nSuccessfully imported vpa_modular modules")
except ImportError as e:
    print(f"\nFailed to import: {e}")
    sys.exit(1)

# Initialize the logger
logger = VPALogger(log_level="INFO", log_file="logs/vpa.log")
llm_interface = VPALLMInterface()

config = VPAConfig()

custom_timeframes = [
        {"interval": "30m", "period": "10d"},
        {"interval": "15m", "period": "10d"},
        {"interval": "5m", "period": "5d"}
    ]

tickers = ["AAPL", "NVDA", "NFLX", "AMZN", "TSLA"]
vpa = VPAFacade()

for ticker in tickers:
    results = vpa.analyze_ticker(ticker, timeframes=custom_timeframes)  # This will now use the custom timeframes
    print(f"Analysis for {ticker}:")

for ticker in tickers:
    logger.info(f"Analyzing {ticker}...")
    nl_analysis = llm_interface.get_ticker_analysis(ticker)  # This will also use the custom timeframes
    for key, value in nl_analysis.items():
        logger.info(f"{key}: {value}")
    logger.info("\n")
    logger.info("--------------------------------------------------\n")

print("Analysis complete.")
print("LLM Analysis:")
query = "Analyze AAPL using VPA"
response = llm_interface.process_query(query)
for key, value in response.items():
    print(f"{key}: {value}")
print("\n")


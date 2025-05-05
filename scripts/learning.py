from vpa_modular.vpa_processor import DataProcessor
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_logger import VPALogger
from vpa_modular.vpa_llm_interface import VPALLMInterface

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


# Example: Scan market for VPA signals
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_utils import create_batch_report
from vpa_modular.vpa_logger import VPALogger
from vpa_modular.vpa_llm_interface import VPALLMInterface


# Initialize the VPA facade
vpa = VPAFacade()
llm_interface = VPALLMInterface()

"""
# Define a list of tickers to analyze
tickers = ["AAPL", "MSFT", "GOOGL", "NFLX", "AMZN", "TSLA","LILM"]

# Create a batch report
report_files = create_batch_report(vpa, tickers, "vpa_batch_reports")

# Print the generated report files
print(f"Report files: {report_files}")

"""
# Scanner Signal Example
# Call the method with required and optional arguments
tickers = ["MSFT", "NFLX"]  # Example list of tickers
signal_type = "SELL"
signal_strength = "MODERATE"
timeframes = [
        {"interval": "30m", "period": "10d"},
        {"interval": "15m", "period": "10d"},
        {"interval": "5m", "period": "5d"}
    ]

# Call the method

scanner = vpa.scan_for_signals(tickers, signal_type, signal_strength, timeframes)
print(type(scanner))
print(scanner)

batch = vpa.batch_analyze(tickers, timeframes)
print(type(batch))
print(batch)

for ticker, result in batch.items():
    print(f"Ticker: {ticker}")
    for key, value in result.items():
        print(f"{key}: {value}")
        print("\n")
   


for ticker, result in scanner.items():
    print(f"Ticker: {ticker}")
    print(f"Signal Type: {result['signal_type']}")
    print(f"Signal Strength: {result['signal_strength']}")
    print(f"Timeframes: {result['timeframes']}")
    print(f"Signal Details: {result['signal_details']}")
    print("--------------------------------------------------\n")

""""
if not scanner:
    print("No signals found.")
else:
    print("Scanner Results:")
    for ticker, result in scanner.items():
        nl_analysis = llm_interface.get_ticker_analysis(ticker)
        print(f"NL Analysis for {ticker}:")
        for key, value in nl_analysis.items():
            print(f"{key}: {value}")
        print("\n")
    print("--------------------------------------------------\n")

"""
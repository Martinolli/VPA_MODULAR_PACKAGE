from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_utils import create_vpa_report
from vpa_modular.vpa_utils import create_batch_report
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_llm_interface import VPALLMInterface
from vpa_modular.vpa_logger import VPALogger


# Initialize the logger
logger = VPALogger(log_level="INFO")
logger = VPALogger(log_level="INFO", log_file="logs/vpa.log")


vpa = VPAFacade()
results = vpa.analyze_ticker("BTC-USD")
timeframes = [{"interval": "1h"}, {"interval": "1d"}]
logger.log_analysis_start("BTC-USD", timeframes)

print(f"Signal: {results['signal']['type']} ({results['signal']['strength']})")

report_file = create_vpa_report(results, "vpa_reports")
print(f"Report files created: {report_file}")

"""

tickers = ["BTC-USD", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "META", "NFLX"]
llm_interface = VPALLMInterface()
nl_analyzis = [llm_interface.get_ticker_analysis(ticker) for ticker in tickers]
for analysis in nl_analyzis:
    for key, value in analysis.items():
        print(f"{key}: {value}")
    print("\n")
    print("--------------------------------------------------\n")
"""

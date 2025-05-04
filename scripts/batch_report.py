from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_utils import create_batch_report
from vpa_modular.vpa_logger import VPALogger
from vpa_modular.vpa_llm_interface import VPALLMInterface


vpa = VPAFacade()
# Define a list of tickers to analyze
tickers = ["AAPL", "MSFT", "GOOGL", "NFLX", "AMZN", "TSLA"]

# Create a batch report
report_files = create_batch_report(vpa, tickers, "vpa_batch_reports")

# Print the generated report files
print(f"Report files: {report_files}")
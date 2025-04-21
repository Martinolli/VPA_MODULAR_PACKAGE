# Example: Scan market for VPA signals
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_utils import create_batch_report
from vpa_modular.vpa_logger import VPALogger


# Initialize the VPA facade
vpa = VPAFacade()

# Define a list of tickers to analyze
tickers = ["AAPL", "MSFT", "GOOGL", "NFLX"]

# Create a batch report
report_files = create_batch_report(vpa, tickers, "vpa_batch_reports")

# Print the generated report files
print(f"Report files: {report_files}")

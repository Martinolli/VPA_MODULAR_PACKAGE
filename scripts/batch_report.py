import sys
import os

# Print Python path and current working directory
print("Python Path:")
for path in sys.path:
    print(path)
print("\nCurrent Working Directory:", os.getcwd())

# Append the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print("\nAppended to Python Path:", project_root)

# Now try to import the modules
try:
    from vpa_modular.vpa_facade import VPAFacade
    from vpa_modular.vpa_utils import create_batch_report
    from vpa_modular.vpa_logger import VPALogger
    from vpa_modular.vpa_llm_interface import VPALLMInterface
    print("\nSuccessfully imported vpa_modular modules")
except ImportError as e:
    print(f"\nFailed to import: {e}")
    sys.exit(1)

vpa = VPAFacade()
# Define a list of tickers to analyze
tickers = ["AAPL", "MSFT", "GOOGL", "NFLX", "AMZN", "TSLA"]

# Create a batch report
report_files = create_batch_report(vpa, tickers, "vpa_batch_reports")

# Print the generated report files
print(f"Report files: {report_files}")
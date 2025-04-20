from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_utils import create_vpa_report
from vpa_modular.vpa_utils import create_batch_report
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_llm_interface import VPALLMInterface


vpa = VPAFacade()
results = vpa.analyze_ticker("NVDA")
print(f"Signal: {results['signal']['type']} ({results['signal']['strength']})")

report_file = create_vpa_report(results, "vpa_reports")
print(f"Report files created: {report_file}")

# Define a list of tickers to analyze
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "META"]

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

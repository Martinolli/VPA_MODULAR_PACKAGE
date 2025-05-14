import json
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_result_extractor import VPAResultExtractor
from vpa_modular.vpa_visualizer_refactored import VPAVisualizerRefactored

def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Results saved to {filename}")

def analyze_tickers(vpa, tickers):
    results = {}
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        results[ticker] = vpa.analyze_ticker(ticker)
    return results

def generate_visualizations_with_refactored(results):
    extractor = VPAResultExtractor(results)
    visualizer = VPAVisualizerRefactored(extractor)

    print("📊 Generating visualizations with VPAVisualizerRefactored...")
    for ticker in extractor.get_tickers():
        print(f"Generating outputs for {ticker}...")
        visualizer.generate_all_outputs_for_ticker(ticker)

def main():
    vpa = VPAFacade()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # You can modify this list as needed

    print("📊 Analyzing tickers...")
    results = analyze_tickers(vpa, tickers)

    print("\n📈 Generating visualizations with refactored visualizer...")
    generate_visualizations_with_refactored(results)

    print("\n✅ Analysis and visualization complete. Check the 'vpa_refactored_output' directory for results.")

if __name__ == "__main__":
    main()
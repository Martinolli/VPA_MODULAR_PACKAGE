# test_visualizer.py

from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_result_extractor import VPAResultExtractor
from vpa_modular.vpa_visualizer import (
    plot_price_volume_chart,
    plot_pattern_analysis,
    plot_support_resistance,
    create_summary_report,
    create_dashboard
)
import os

# Initialize VPAFacade and perform analysis
vpa = VPAFacade()
tickers = ["AAPL", "MSFT", "GOOGL"]
results = {}

for ticker in tickers:
    results[ticker] = vpa.analyze_ticker(ticker)

# Extract results
extractor = VPAResultExtractor(results)

# Create output directory
output_dir = "vpa_analysis_output"
os.makedirs(output_dir, exist_ok=True)

# Generate visualizations and reports
for ticker in extractor.get_tickers():
    ticker_dir = os.path.join(output_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    for timeframe in extractor.get_timeframes(ticker):
        # Get data
        price_data = extractor.get_price_data(ticker, timeframe)
        volume_data = extractor.get_volume_data(ticker, timeframe)
        pattern_analysis = extractor.get_pattern_analysis(ticker, timeframe)
        support_resistance = extractor.get_support_resistance(ticker, timeframe)
        
        # Combine price and volume data
        combined_data = price_data.copy()
        combined_data['volume'] = volume_data

        # Generate charts
        plot_price_volume_chart(combined_data, ticker=ticker, timeframe=timeframe, 
                                output_path=os.path.join(ticker_dir, f"{ticker}_{timeframe}_price_volume.png"))
        
        print(f"Pattern analysis for {ticker} - {timeframe}:")
        print(pattern_analysis)
        plot_pattern_analysis(price_data, pattern_analysis, ticker=ticker, timeframe=timeframe,
                              output_path=os.path.join(ticker_dir, f"{ticker}_{timeframe}_patterns.png"))
        
        print(f"Support and resistance for {ticker} - {timeframe}:")
        print(support_resistance)
        plot_support_resistance(price_data, support_resistance, ticker=ticker, timeframe=timeframe,
                                output_path=os.path.join(ticker_dir, f"{ticker}_{timeframe}_support_resistance.png"))

# Create summary report and dashboard
create_summary_report(extractor, output_dir)
create_dashboard(extractor, output_dir)

print("VPA analysis and visualization complete!")
# test_visualizer.py
import pandas as pd
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_result_extractor import extract_testing_signals
from vpa_modular.vpa_result_extractor import VPAResultExtractor
from vpa_modular.vpa_visualizer import (
    plot_price_volume_chart,
    plot_pattern_analysis,
    plot_support_resistance,
    create_summary_report,
    create_dashboard,
    create_signal_dashboard,
    plot_multi_timeframe_trends,
    create_pattern_signal_heatmap,
    plot_risk_management,
    visualize_risk_reward_ratio,
    update_price_chart_with_risk_levels,
    plot_vpa_signals_candlestick,
    plot_relative_volume
)
import os


def test_visualizer():
    # Initialize VPAFacade and perform analysis
    vpa = VPAFacade()
    tickers = ["AAPL", "NFLX", "GOOGL", "AMZN", "MSFT", "TSLA", "META", "NVDA", "AMD", "INTC", "JPM", "BAC"]
    results = {}

    for ticker in tickers:
        results[ticker] = vpa.analyze_ticker(ticker)
        print("\n--- Extracted Testing Patterns ---")
        testing_patterns = extract_testing_signals(results)
        for ticker, timeframes in testing_patterns.items():
            print(f"Ticker: {ticker}")
            for tf, tests in timeframes.items():
                print(f"  Timeframe: {tf}")
                for test in tests:
                    print(f"    Test: {test}")

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
            price_data = extractor.get_price_data(ticker, timeframe)
            volume_data = extractor.get_volume_data(ticker, timeframe)  # ✅ Get volume
            full_data = pd.concat([price_data, volume_data], axis=1)    # ✅ Merge them

            pattern_analysis = extractor.get_pattern_analysis(ticker, timeframe)
            support_resistance = extractor.get_support_resistance(ticker, timeframe)

            # Test existing visualizations
            plot_price_volume_chart(full_data, ticker, timeframe, os.path.join(ticker_dir, f"{ticker}_{timeframe}_price_volume.png"))
            plot_pattern_analysis(price_data, pattern_analysis, ticker, timeframe, os.path.join(ticker_dir, f"{ticker}_{timeframe}_patterns.png"))
            plot_support_resistance(price_data, support_resistance, ticker, timeframe, os.path.join(ticker_dir, f"{ticker}_{timeframe}_support_resistance.png"))
            
        # Test new visualizations
        signal = extractor.get_signal(ticker)
        risk_assessment = extractor.get_risk_assessment(ticker)
        current_price = extractor.get_current_price(ticker)
        # Filtrar o signal para o timeframe atual:
        signal_for_tf = {
            "evidence": {
                timeframe: signal.get("evidence", {}).get(timeframe, [])
            }
        }
        plot_vpa_signals_candlestick(price_data, signal_for_tf, ticker, os.path.join(ticker_dir, f"{ticker}_{timeframe}_candlestick_signals.png"))
        create_signal_dashboard(signal, ticker, os.path.join(ticker_dir, f"{ticker}_signal_dashboard.png"))
        plot_multi_timeframe_trends(signal['evidence'], ticker, os.path.join(ticker_dir, f"{ticker}_multi_timeframe_trends.png"))
        create_pattern_signal_heatmap(signal['evidence'], ticker, os.path.join(ticker_dir, f"{ticker}_pattern_heatmap.png"))
        plot_risk_management(risk_assessment, current_price, ticker, os.path.join(ticker_dir, f"{ticker}_risk_management.png"))
        current_price = extractor.get_current_price(ticker)
        visualize_risk_reward_ratio(risk_assessment, current_price, ticker, os.path.join(ticker_dir, f"{ticker}_risk_reward_ratio.png"))
        update_price_chart_with_risk_levels(price_data, risk_assessment, current_price, ticker, os.path.join(ticker_dir, f"{ticker}_price_with_risk_levels.png"))
        plot_relative_volume(full_data, ticker, timeframe, os.path.join(ticker_dir, f"{ticker}_{timeframe}_relative_volume.png"))


    # Create summary report and dashboard
    report_path = os.path.join(os.getcwd(), "vpa_summary_report.txt")
    dashboard_path = os.path.join(os.getcwd(), "vpa_dashboard.png")

    create_summary_report(extractor, report_path=report_path)
    create_dashboard(extractor, dashboard_path=dashboard_path)


    print("VPA analysis and visualization complete!")
if __name__ == "__main__":
    test_visualizer()
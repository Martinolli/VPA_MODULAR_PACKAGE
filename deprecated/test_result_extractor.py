from vpa_modular.vpa_result_extractor import VPAResultExtractor
from vpa_modular.vpa_facade import VPAFacade


ticker = "AAPL"
vpa = VPAFacade()
results ={}
results[ticker] = vpa.analyze_ticker(ticker)
extractor = VPAResultExtractor(results)
print("Extracted Data:")
print(type(extractor))

# print(extractor.get_tickers())
# print(extractor.get_ticker_data(ticker))

# print(extractor.get_signal(ticker))

# print(extractor.get_pattern_analysis(ticker, "1d"))

# print(extractor.get_trend_analysis(ticker, "15m"))

# print(extractor.get_volume_data(ticker, "15m").head())

print(extractor.get_timeframes(ticker))
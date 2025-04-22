from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_utils import create_vpa_report
from vpa_modular.vpa_utils import create_batch_report
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_llm_interface import VPALLMInterface
from vpa_modular.vpa_data import YFinanceProvider
from vpa_modular.vpa_processor import DataProcessor
from vpa_modular.vpa_analyzer import CandleAnalyzer

config = VPAConfig()
volume_threshold = config.get_volume_thresholds()
print(f"Volume Thresholds: {volume_threshold}")

provider = YFinanceProvider()
price_data, volume_data = provider.get_data("AAPL", interval='15m', period='1mo')
print(type(price_data))
print(price_data)

processor = DataProcessor(config)
processed_data = processor.preprocess_data(price_data, volume_data)
print("Processed Data:")
print(processed_data)

volume_metrics = processor.calculate_volume_metrics(processed_data, lookback_period=20)
print("Volume Metrics:")
print(volume_metrics)

volume_info = processor.classify_volume(volume_metrics["volume_ratio"])
print("Volume Classifications:")
print(volume_info)

candle_info = processor.classify_candles(processed_data)
print("Candle Classifications:")
print(candle_info)

print(type(processed_data))
for key, value in processed_data.items():
    print(f"{key}: {key}")

candle_analyzer = CandleAnalyzer(config)
for idx in processed_data["price"].index:
    result = candle_analyzer.analyze_candle(idx, processed_data)
    print(f"Analysis for candle at {idx}: {result}")

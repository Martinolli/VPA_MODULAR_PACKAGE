from vpa_modular.vpa_processor import DataProcessor
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_facade import VPAFacade

facade = VPAFacade()
print(facade.data_provider)
print(facade.config.get_timeframes())
print(facade.config.get_all())

config = VPAConfig()
print("Timeframe Configurations:")
print(config.get_timeframes())

print()
print("All Configurations:")
for key, value in config.get_all().items():
    print(f"{key}: {value}")
print()
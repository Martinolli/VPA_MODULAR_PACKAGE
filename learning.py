from vpa_modular.vpa_processor import DataProcessor
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_facade import VPAFacade

facade = VPAFacade()
print(facade.data_provider)
print(facade.config.get_timeframes())
print(facade.config.get_all())
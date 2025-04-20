"""
VPA Modular Architecture - __init__.py

This file initializes the VPA modular package.
"""

# Import core modules
from vpa_modular.vpa_config import VPAConfig
from vpa_modular.vpa_data import DataProvider, YFinanceProvider, MultiTimeframeProvider
from vpa_modular.vpa_processor import DataProcessor
from vpa_modular.vpa_analyzer import CandleAnalyzer, TrendAnalyzer, PatternRecognizer, SupportResistanceAnalyzer, MultiTimeframeAnalyzer
from vpa_modular.vpa_signals import SignalGenerator, RiskAssessor

# Import integration modules
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_llm_interface import VPALLMInterface

# Import utility modules
from vpa_modular.vpa_logger import VPALogger

# Package metadata
__version__ = "1.0.0"
__author__ = "Manus AI"
__description__ = "Modular implementation of Volume Price Analysis algorithm"

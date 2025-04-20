# VPA Modular Architecture - README

This package contains a modular implementation of the Volume Price Analysis (VPA) algorithm.
"""

## VPA Modular Architecture

## Overview

This package provides a modular implementation of the Volume Price Analysis (VPA) algorithm based on Anna Coulling's methodology. The architecture is designed to be:

1. **Modular** - Clean separation of concerns with well-defined interfaces
2. **Extensible** - Easy to add new features or modify existing ones
3. **Maintainable** - Well-documented code with clear responsibilities
4. **LLM-friendly** - Structured for easy integration with language models

## Module Structure

The package is organized into the following modules:

### Core Modules

- **vpa_config.py** - Configuration management
- **vpa_data.py** - Data acquisition and preprocessing
- **vpa_processor.py** - Data processing and feature extraction
- **vpa_analyzer.py** - Analysis and pattern recognition
- **vpa_signals.py** - Signal generation and risk assessment

### Integration Modules

- **vpa_facade.py** - Simplified API for VPA analysis
- **vpa_llm_interface.py** - Specialized interface for LLM integration

### Utility Modules

- **vpa_utils.py** - Visualization and helper functions
- **vpa_logger.py** - Logging framework

## Getting Started

### Installation

1.Clone the repository:

```git
git clone <repository-url>
```

2.Install dependencies:

```pip
pip install pandas numpy matplotlib yfinance
```

### Basic Usage

```python
from vpa_modular.vpa_facade import VPAFacade

# Initialize the VPA facade
vpa = VPAFacade()

# Analyze a ticker
results = vpa.analyze_ticker("AAPL")

# Print the signal
print(f"Signal: {results['signal']['type']} ({results['signal']['strength']})")
print(f"Details: {results['signal']['details']}")

# Get risk assessment
print(f"Stop Loss: ${results['risk_assessment']['stop_loss']:.2f}")
print(f"Take Profit: ${results['risk_assessment']['take_profit']:.2f}")
print(f"Risk-Reward Ratio: {results['risk_assessment']['risk_reward_ratio']:.2f}")
```

### LLM Integration

```python
from vpa_modular.vpa_llm_interface import VPALLMInterface

# Initialize the VPA LLM interface
vpa_llm = VPALLMInterface()

# Process a natural language query
response = vpa_llm.process_query("What is the VPA signal for AAPL?")
print(response)

# Get structured analysis for a ticker
analysis = vpa_llm.get_ticker_analysis("MSFT")
print(analysis)
```

### Visualization

```python
from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_utils import create_vpa_report

# Initialize the VPA facade
vpa = VPAFacade()

# Analyze a ticker
results = vpa.analyze_ticker("AAPL")

# Create a comprehensive report with visualizations
report_files = create_vpa_report(results, "vpa_reports")
print(f"Report files: {report_files}")
```

## Testing

Run the test suite to verify the implementation:

```python
python vpa_modular/test_vpa.py
```

## Extending the Framework

### Adding New Data Sources

Extend the `DataProvider` class in `vpa_data.py`:

```python
class CustomDataProvider(DataProvider):
    def get_data(self, ticker, interval, period):
        # Implement custom data retrieval
        return price_data, volume_data
```

### Adding New Analysis Methods

Extend the appropriate analyzer class:

```python
class CustomPatternRecognizer(PatternRecognizer):
    def identify_patterns(self, processed_data, current_idx):
        # Call parent method
        patterns = super().identify_patterns(processed_data, current_idx)
        
        # Add custom pattern detection
        patterns["custom_pattern"] = self._detect_custom_pattern(processed_data, current_idx)
        
        return patterns
    
    def _detect_custom_pattern(self, processed_data, current_idx):
        # Implement custom pattern detection
        return {"detected": True, "details": "Custom pattern detected"}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

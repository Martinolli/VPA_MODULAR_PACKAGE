# Module Description: vpa_config.py

## Module Name: vpa_config

Purpose: This module provides a configuration management system for the VPA (Variable Pattern Analysis) algorithm. It allows for flexible setup of thresholds, parameters, and timeframes used in the analysis process.

## Classes

VPAConfig

## Attributes

config (dict): Stores the loaded configuration parameters. Initialized during instantiation.

## Methods

__init__(self, config_file=None): Constructor for the VPAConfig class. Loads the configuration from a JSON file if provided; otherwise, uses the default configuration.
_load_config(self, config_file): Loads the configuration from a JSON file. Handles potential errors during file loading and gracefully falls back to the default configuration if an error occurs.
get_volume_thresholds(self): Returns the volume classification thresholds defined in the configuration.
get_candle_thresholds(self): Returns the candle classification thresholds defined in the configuration.
get_trend_parameters(self): Returns the trend analysis parameters defined in the configuration.
get_pattern_parameters(self): Returns the pattern recognition parameters, including thresholds and parameters for accumulation, distribution, buying climax, and selling climax patterns.
get_signal_parameters(self): Returns the signal parameters, like thresholds and candle counts.
get_risk_parameters(self): Returns the risk management parameters, such as stop-loss and take-profit percentages.
get_timeframes(self): Returns the specified timeframes, each containing an interval (e.g., "1d", "1h") and a period (e.g., "1y", "60d").
update_config(self, new_config): Allows updating the configuration parameters by providing a dictionary of changes. Supports nested dictionaries for hierarchical updates.
Default Configuration: A dictionary containing default values for all configuration parameters. This allows for a baseline setup and customization. It’s organized into logical sections: “volume”, “candle”, “trend”, “pattern”, “signal”, “risk”, and “timeframes”

## Usage Examples

```python
from vpa_config import VPAConfig

# Create a VPAConfig instance using the default configuration
config = VPAConfig()

# Get the current timeframes
current_timeframes = config.get_timeframes()
print("Original Timeframes:", current_timeframes)

# Modify the timeframes – change the 15m timeframe to 30m
new_timeframes = [
    {"interval": "1d", "period": "1y"},
    {"interval": "1h", "period": "60d"},
    {"interval": "30m", "period": "5d"}
]
config.update_timeframes(new_timeframes)

# Verify the change
updated_timeframes = config.get_timeframes()
print("Updated Timeframes:", updated_timeframes)
```

```python
from vpa_config import VPAConfig

config = VPAConfig()

# Get the current signal parameters
current_signal_params = config.get_signal_parameters()
print("Original Signal Parameters:", current_signal_params)

# Change the bullish candles threshold to 3
new_signal_params = {
    "bullish_candles_threshold": 3,
    "bearish_candles_threshold": 1  # Also updating bearish to make it consistent
}

config.update_signal_params(new_signal_params)

# Verify the change
updated_signal_params = config.get_signal_parameters()
print("Updated Signal Parameters:", updated_signal_params)
```

```python
from vpa_config import VPAConfig

config = VPAConfig()

# Get the default stop loss
current_stop_loss = config.get_risk_parameters()["default_stop_loss_percent"]
print(f"Original Stop Loss: {current_stop_loss}")

# Modify the stop loss to 0.05
new_risk_params = {
    "default_stop_loss_percent": 0.05,
    "default_take_profit_percent": 0.08
}

config.update_risk_params(new_risk_params)

# Verify the change
updated_risk_params = config.get_risk_parameters()
print("Updated Risk Parameters:", updated_risk_params)
```

## Explanation and Key Points

update_timeframes(): This method takes a new list of timeframes as input. You need to create a new list containing the desired timeframe configurations.
update_signal_params() and update_risk_params(): These methods also take a dictionary of changes, where the keys are the parameter names and the values are the new values.
Data Structure: The update_* methods rely on you providing data in a dictionary format. It's a good practice to create a copy of the original dictionary and modify the copy, rather than directly modifying the internal config dictionary.
To run these examples:

Make sure you have the vpa_config.py file in the same directory as your Python script.
Run the code. The output will show the original and updated values for the specified parameters.

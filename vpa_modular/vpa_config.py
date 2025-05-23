"""
VPA Configuration Module

This module provides configuration management for the VPA algorithm.
"""

class VPAConfig:
    """Configuration for VPA analysis"""
    
    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        if config_file:
            import json
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration instead.")
                return default_config
        return default_config
    
    def get_volume_thresholds(self):
        """Get volume classification thresholds"""
        return self.config["volume"]
    
    def get_candle_thresholds(self):
        """Get candle classification thresholds"""
        return self.config["candle"]
    
    def get_trend_parameters(self):
        """Get trend analysis parameters"""
        return self.config["trend"]
    
    def get_pattern_parameters(self):
        """Get pattern recognition parameters"""
        return self.config["pattern"]
    
    def get_signal_parameters(self):
        """Get signal generation parameters"""
        return self.config["signal"]
    
    def get_risk_parameters(self):
        """Get risk assessment parameters"""
        return self.config["risk"]
    
    def get_timeframes(self):
        """Get default timeframes for analysis"""
        return self.config["timeframes"]
    
    def get_primary_timeframe(self):
        """Get primary timeframe for analysis"""
        if "timeframes" in self.config and self.config["timeframes"]:
            return self.config["timeframes"][0]["interval"]
        return "1d"  # Default to daily timeframe if none specified
    
    def get_all(self):
        """Get all configuration parameters"""
        return self.config
    
    def update_parameters(self, params):
        """
        Update configuration parameters
        
        Args:
            params: Dictionary of parameters to update
        """
        for key, value in params.items():
            # Handle nested parameters with dot notation
            if '.' in key:
                sections = key.split('.')
                config_section = self.config
                for section in sections[:-1]:
                    if section not in config_section:
                        config_section[section] = {}
                    config_section = config_section[section]
                config_section[sections[-1]] = value
            else:
                # Handle top-level parameters
                self.config[key] = value
        
        return self.config

# Default configuration
default_config = {
    "volume": {
        "very_high_threshold": 2.0, # 200% increase
        "high_threshold": 1.3, # 130% increase
        "low_threshold": 0.6,
        "very_low_threshold": 0.3,
        "lookback_period": 50
    },
    "candle": {
        "wide_threshold": 1.3,
        "narrow_threshold": 0.6,
        "wick_threshold": 1.5,
        "lookback_period": 20
    },
    "trend": {
        "lookback_period": 5,
        "sideways_threshold": 2,  # 2% change for sideways movement
        "strong_trend_threshold": 5,  # 5% change for strong trend
        "volume_change_threshold": 10,  # 10% change for significant volume change
        "min_trend_length": 3
    },
    "pattern": {
        "accumulation": {
            "price_volatility_threshold": 0.08,
            "high_volume_threshold": 2,
            "support_tests_threshold": 1
        },
        "distribution": {
            "price_volatility_threshold": 0.08,
            "high_volume_threshold": 2,
            "resistance_tests_threshold": 1
        },
        "buying_climax": {
            "near_high_threshold": 0.97,
            "wide_up_threshold": 0.6,
            "upper_wick_threshold": 0.25
        },
        "selling_climax": {
            "near_low_threshold": 1.07,
            "wide_down_threshold": 0.6,
            "lower_wick_threshold": 0.25
        }
    },
    "signal": {
        "strong_signal_threshold": 0.8,
        "bullish_confirmation_threshold": 1,
        "bearish_confirmation_threshold": 1,
        "bullish_candles_threshold": 2,
        "bearish_candles_threshold": 2
    },
    "risk": {
        "default_stop_loss_percent": 0.02,
        "default_take_profit_percent": 0.05,
        "support_resistance_buffer": 0.005,
        "default_risk_percent": 0.01,
        "default_risk_reward": 2.0
    },
    "timeframes": [
        {"interval": "1d", "period": "1y"},
        {"interval": "1h", "period": "60d"},
        {"interval": "15m", "period": "5d"}
    ]
}

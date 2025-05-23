"""
VPA Configuration Manager Module

This module provides centralized API key and configuration management for the VPA system.
It supports multiple sources for configuration values (environment variables, config files, etc.)
with appropriate fallbacks and validation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class ConfigManager:
    """Manages API keys and configuration for the VPA system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Parameters:
        - config_file: Optional path to a JSON configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_data = {}
        
        # Default config file locations to check
        self.config_file_paths = [
            config_file,  # User-provided path (if any)
            os.path.expanduser("~/.vpa/config.json"),  # User home directory
            os.path.join(os.getcwd(), "vpa_config.json"),  # Current working directory
            os.path.join(os.path.dirname(__file__), "config.json"),  # Module directory
        ]
        
        # Load configuration from file
        self._load_config_from_file()
    
    def _load_config_from_file(self) -> None:
        """Load configuration from the first available config file."""
        for file_path in self.config_file_paths:
            if file_path and os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as f:
                        self.config_data = json.load(f)
                    self.logger.info(f"Loaded configuration from {file_path}")
                    return
                except Exception as e:
                    self.logger.warning(f"Error loading config from {file_path}: {e}")
        
        self.logger.info("No configuration file found, using environment variables only")
    
    def get_api_key(self, service: str) -> str:
        """
        Get API key for the specified service with appropriate fallbacks.
        
        Parameters:
        - service: Service name (e.g., 'polygon', 'openai')
        
        Returns:
        - API key string
        
        Raises:
        - ValueError: If API key is not found in any source
        """
        # Map service names to environment variable names
        env_var_map = {
            'polygon': 'POLYGON_API_KEY',
            'openai': 'OPENAI_API_KEY',
        }
        
        # Map service names to config file keys
        config_key_map = {
            'polygon': 'polygon_api_key',
            'openai': 'openai_api_key',
        }
        
        # Check if service is supported
        if service not in env_var_map:
            raise ValueError(f"Unsupported service: {service}")
        
        # Try to get API key from environment variable
        env_var_name = env_var_map[service]
        api_key = os.getenv(env_var_name)
        
        # If not found in environment, try config file
        if not api_key and service in config_key_map:
            config_key = config_key_map[service]
            api_key = self.config_data.get(config_key)
        
        # If still not found, raise error with helpful message
        if not api_key:
            raise ValueError(
                f"API key for {service} not found. Please set the {env_var_name} environment variable "
                f"or add '{config_key_map[service]}' to your configuration file."
            )
        
        return api_key
    
    def validate_api_key(self, service: str) -> bool:
        """
        Validate that the API key for the specified service exists and has the correct format.
        
        Parameters:
        - service: Service name (e.g., 'polygon', 'openai')
        
        Returns:
        - True if API key is valid, False otherwise
        """
        try:
            api_key = self.get_api_key(service)
            
            # Basic validation based on service
            if service == 'polygon':
                # Polygon.io API keys are typically alphanumeric
                return bool(api_key and len(api_key) > 10)
            elif service == 'openai':
                # OpenAI API keys typically start with 'sk-'
                return bool(api_key and api_key.startswith('sk-'))
            else:
                # For other services, just check if key exists
                return bool(api_key)
        except ValueError:
            return False
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback to default.
        
        Parameters:
        - key: Configuration key
        - default: Default value if key is not found
        
        Returns:
        - Configuration value or default
        """
        # Try to get from environment variable (uppercase with prefix VPA_)
        env_var_name = f"VPA_{key.upper()}"
        env_value = os.getenv(env_var_name)
        if env_value is not None:
            return env_value
        
        # Try to get from config file
        return self.config_data.get(key, default)
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Parameters:
        - config_file: Optional path to save configuration file
        """
        if not config_file:
            # Use first writable path from config_file_paths
            for path in self.config_file_paths:
                if path:
                    try:
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        config_file = path
                        break
                    except Exception:
                        continue
        
        if not config_file:
            # Default to current directory if no writable path found
            config_file = os.path.join(os.getcwd(), "vpa_config.json")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config_data, f, indent=4)
            
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {config_file}: {e}")
            raise
    
    def set_config_value(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Parameters:
        - key: Configuration key
        - value: Configuration value
        """
        self.config_data[key] = value
    
    def set_api_key(self, service: str, api_key: str) -> None:
        """
        Set API key for the specified service.
        
        Parameters:
        - service: Service name (e.g., 'polygon', 'openai')
        - api_key: API key value
        """
        # Map service names to config file keys
        config_key_map = {
            'polygon': 'polygon_api_key',
            'openai': 'openai_api_key',
        }
        
        # Check if service is supported
        if service not in config_key_map:
            raise ValueError(f"Unsupported service: {service}")
        
        # Set API key in config data
        self.config_data[config_key_map[service]] = api_key


# Create a singleton instance
_config_manager = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get the singleton ConfigManager instance.
    
    Parameters:
    - config_file: Optional path to a JSON configuration file
    
    Returns:
    - ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager

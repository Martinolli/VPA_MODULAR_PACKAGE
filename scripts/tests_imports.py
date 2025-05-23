#!/usr/bin/env python3
"""
VPA Module Import Test Script

This script verifies that all VPA modules are importable and accessible.
It serves as a basic integration test to ensure the core components can work together.
"""

import os
import sys
import importlib
from pathlib import Path

# Add the parent directory to sys.path to import VPA modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    """Print a formatted header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}\n")

def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_info(message):
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def test_module_import(module_name):
    """Test if a module can be imported"""
    try:
        module = importlib.import_module(module_name)
        print_success(f"Successfully imported {module_name}")
        return module
    except ImportError as e:
        print_error(f"Failed to import {module_name}: {e}")
        return None
    except Exception as e:
        print_error(f"Error importing {module_name}: {e}")
        return None

def main():
    """Main function to test module imports"""
    print_header("VPA Module Import Test")
    
    # List of modules to test
    modules = [
        "vpa_modular.config_manager",
        "vpa_modular.vpa_logger",
        "vpa_modular.vpa_config",
        "vpa_modular.vpa_data",
        "vpa_modular.vpa_processor",
        "vpa_modular.vpa_analyzer",
        "vpa_modular.vpa_signals",
        "vpa_modular.vpa_facade",
        "vpa_modular.vpa_result_extractor",
        "vpa_modular.vpa_visualizer",
        "vpa_modular.vpa_training_data_generator",
        "vpa_modular.vpa_llm_interface",
        "vpa_modular.vpa_llm_query_engine",
        "vpa_modular.memory_manager"
    ]
    
    # Test each module
    imported_modules = {}
    for module_name in modules:
        module = test_module_import(module_name)
        if module:
            imported_modules[module_name] = module
    
    # Print summary
    print_header("Import Test Summary")
    
    success_count = len(imported_modules)
    total_count = len(modules)
    
    if success_count == total_count:
        print_success(f"All {total_count} modules imported successfully!")
    else:
        print_warning(f"{success_count} out of {total_count} modules imported successfully.")
        
        # List failed modules
        failed_modules = set(modules) - set(imported_modules.keys())
        print_info("\nFailed modules:")
        for module in failed_modules:
            print(f"  - {module}")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())

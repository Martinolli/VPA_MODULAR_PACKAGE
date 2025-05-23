#!/usr/bin/env python3
"""
VPA Environment Setup and Validation Script

This script checks and sets up the environment for the VPA Modular Package:
1. Verifies required packages are installed
2. Checks for .env file and API keys
3. Validates API keys
4. Creates necessary directories
5. Reports environment status
"""

import os
import sys
import subprocess
import importlib
import pkg_resources
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to sys.path to import VPA modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version_info = sys.version_info
    major = version_info.major
    minor = version_info.minor
    version_str = f"{major}.{minor}"
    
    if major < 3 or (major == 3 and minor < 8):
        print_error(f"Python version {version_str} is not supported. Please use Python 3.8 or higher.")
        return False
    
    print_success(f"Python version {version_str} is compatible.")
    return True

def check_packages():
    """Check if required packages are installed"""
    print_header("Checking Required Packages")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(project_root, "requirements.txt")

    if not os.path.exists(requirements_path):
        print_error(f"Requirements file not found at {requirements_path}")
        return False
    
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    all_installed = True
    missing_packages = []
    outdated_packages = []
    
    for req in requirements:
        try:
            if req.startswith('-e'):
                # Skip editable installs for now
                continue
            req_name = req.split('==')[0]
            req_version = req.split('==')[1] if '==' in req else None
            
            dist = pkg_resources.get_distribution(req_name)
            version = dist.version
            
            if req_version and pkg_resources.parse_version(version) < pkg_resources.parse_version(req_version):
                print_warning(f"{req_name} version {version} is installed, but {req_version} is required.")
                outdated_packages.append((req_name, version, req_version))
                all_installed = False
            else:
                print_success(f"{req_name} version {version} is installed.")
                
        except pkg_resources.DistributionNotFound:
            print_error(f"{req_name} is not installed.")
            missing_packages.append(req_name)
            all_installed = False
        except Exception as e:
            print_error(f"Error checking {req_name}: {e}")
            all_installed = False
    
    if missing_packages or outdated_packages:
        print_info("\nSome packages need to be installed or updated.")
        
        if missing_packages:
            print_info("\nMissing packages:")
            for package in missing_packages:
                print(f"  - {package}")
        
        if outdated_packages:
            print_info("\nOutdated packages:")
            for package, current, required in outdated_packages:
                print(f"  - {package} (current: {current}, required: {required})")
        
        print_info("\nYou can install or update these packages with:")
        print(f"  pip install -r {requirements_path}")
        
        return False
    
    print_success("All required packages are installed with compatible versions.")
    return True

def check_env_file():
    """Check if .env file exists and contains required API keys"""
    print_header("Checking .env File")
    
    # Check for .env file in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(project_root, ".env")
    
    if not os.path.exists(env_path):
        print_error(f".env file not found at {env_path}")
        print_info("Please create a .env file with the following content:")
        print("  POLYGON_API_KEY=your_polygon_api_key")
        print("  OPENAI_API_KEY=your_openai_api_key")
        return False
    
    # Load .env file
    load_dotenv(env_path)
    
    # Check for required API keys
    polygon_key = os.getenv("POLYGON_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not polygon_key:
        print_error("POLYGON_API_KEY not found in .env file")
        return False
    else:
        print_success("POLYGON_API_KEY found in .env file")
    
    if not openai_key:
        print_error("OPENAI_API_KEY not found in .env file")
        return False
    else:
        print_success("OPENAI_API_KEY found in .env file")
    
    return True

def validate_api_keys():
    """Validate API keys using the config manager"""
    print_header("Validating API Keys")
    
    try:
        from vpa_modular.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        
        # Validate Polygon.io API key
        try:
            polygon_key = config_manager.get_api_key('polygon')
            if config_manager.validate_api_key('polygon'):
                print_success("Polygon.io API key is valid")
            else:
                print_warning("Polygon.io API key format may be invalid")
        except Exception as e:
            print_error(f"Error validating Polygon.io API key: {e}")
            return False
        
        # Validate OpenAI API key
        try:
            openai_key = config_manager.get_api_key('openai')
            if config_manager.validate_api_key('openai'):
                print_success("OpenAI API key is valid")
            else:
                print_warning("OpenAI API key format may be invalid (should start with 'sk-')")
        except Exception as e:
            print_error(f"Error validating OpenAI API key: {e}")
            return False
        
        return True
    except ImportError:
        print_error("Could not import config_manager module")
        return False
    except Exception as e:
        print_error(f"Error validating API keys: {e}")
        return False

def create_directories():
    """Create necessary directories for the VPA system"""
    print_header("Creating Necessary Directories")
    
    # Define directories to create
    project_root = os.path.dirname(os.path.abspath(__file__))
    directories = [
        os.path.join(project_root, "logs"),
        os.path.join(project_root, "data"),
        os.path.join(project_root, "output"),
        os.path.join(project_root, "visualizations"),
        os.path.join(project_root, "llm_training_data"),
        os.path.join(project_root, ".vpa", "logs"),
        os.path.join(project_root, ".vpa", "config"),
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print_success(f"Created directory: {directory}")
        except Exception as e:
            print_error(f"Error creating directory {directory}: {e}")
    
    return True

def main():
    """Main function to run all checks and setup"""
    print_header("VPA Environment Setup and Validation")
    
    # Track overall status
    status = {
        "python_version": False,
        "packages": False,
        "env_file": False,
        "api_keys": False,
        "directories": False
    }
    
    # Run checks
    status["python_version"] = check_python_version()
    status["packages"] = check_packages()
    status["env_file"] = check_env_file()
    status["api_keys"] = validate_api_keys()
    status["directories"] = create_directories()
    
    # Print summary
    print_header("Setup Summary")
    
    all_passed = all(status.values())
    
    if all_passed:
        print_success("All checks passed! Your environment is ready for VPA development.")
    else:
        print_warning("Some checks failed. Please address the issues above before proceeding.")
        
        # Print specific recommendations
        if not status["packages"]:
            print_info("Run 'pip install -r requirements.txt' to install missing packages.")
        
        if not status["env_file"] or not status["api_keys"]:
            print_info("Ensure your .env file exists and contains valid API keys.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

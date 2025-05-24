#!/usr/bin/env python3
"""
VPA OpenAI Function Calling Test

This script tests the OpenAI function calling integration in the VPAQueryEngine,
verifying that queries are properly processed and functions are correctly executed.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

def test_openai_api_key():
    """Test OpenAI API key configuration"""
    print_header("Testing OpenAI API Key Configuration")
    
    # Check for API key in environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print_error("OPENAI_API_KEY not found in environment variables")
        print_info("Please set the OPENAI_API_KEY environment variable or add it to your .env file")
        return False
    
    # Basic validation (OpenAI keys typically start with "sk-")
    if not api_key.startswith("sk-"):
        print_warning("OPENAI_API_KEY format may be invalid (should start with 'sk-')")
    else:
        print_success("OPENAI_API_KEY found and format appears valid")
    
    return bool(api_key)

def test_query_engine_initialization():
    """Test VPAQueryEngine initialization with OpenAI client"""
    print_header("Testing VPAQueryEngine Initialization")
    
    try:
        from vpa_modular.vpa_llm_query_engine_enhanced import VPAQueryEngine
        
        # Initialize query engine
        print_info("Initializing VPAQueryEngine...")
        query_engine = VPAQueryEngine()
        
        # Check if OpenAI client was initialized
        if query_engine.client:
            print_success("VPAQueryEngine initialized with OpenAI client")
            return query_engine
        else:
            print_error("VPAQueryEngine initialized but OpenAI client is None")
            print_info("Please check your OpenAI API key configuration")
            return None
    except Exception as e:
        print_error(f"Error initializing VPAQueryEngine: {e}")
        return None

def test_function_definitions(query_engine):
    """Test function definitions in VPAQueryEngine"""
    if not query_engine:
        print_error("VPAQueryEngine not initialized, skipping function definitions test")
        return False
    
    print_header("Testing Function Definitions")
    
    try:
        # Get function definitions
        functions = query_engine.functions
        
        if not functions:
            print_error("No function definitions found")
            return False
        
        print_success(f"Found {len(functions)} function definitions")
        
        # Check for required functions
        required_functions = ["analyze_ticker", "batch_analyze", "explain_vpa_concepts", "get_market_summary"]
        missing_functions = [f for f in required_functions if not any(func["name"] == f for func in functions)]
        
        if missing_functions:
            print_error(f"Missing required functions: {', '.join(missing_functions)}")
            return False
        
        print_success("All required functions are defined")
        
        # Print function names
        for func in functions:
            print_info(f"Function: {func['name']} - {func['description'][:50]}...")
        
        return True
    except Exception as e:
        print_error(f"Error testing function definitions: {e}")
        return False

def test_function_execution(query_engine):
    """Test function execution in VPAQueryEngine"""
    if not query_engine:
        print_error("VPAQueryEngine not initialized, skipping function execution test")
        return False
    
    print_header("Testing Function Execution")
    
    try:
        # Get function map
        function_map = query_engine._get_function_map()
        
        if not function_map:
            print_error("No function map found")
            return False
        
        print_success(f"Found function map with {len(function_map)} functions")
        
        # Test explain_vpa_concepts function (doesn't require external API calls)
        print_info("Testing explain_vpa_concepts function...")
        
        if "explain_vpa_concepts" not in function_map:
            print_error("explain_vpa_concepts function not found in function map")
            return False
        
        # Execute function
        result = function_map["explain_vpa_concepts"]({"concept": "volume spread analysis"})
        
        if not result:
            print_error("Function returned no result")
            return False
        
        if "explanation" not in result:
            print_error("Function result does not contain 'explanation' field")
            return False
        
        print_success("Function executed successfully")
        print_info(f"Explanation: {result['explanation'][:100]}...")
        
        return True
    except Exception as e:
        print_error(f"Error testing function execution: {e}")
        return False

def test_mock_query_processing(query_engine):
    """Test query processing with mock OpenAI responses"""
    if not query_engine:
        print_error("VPAQueryEngine not initialized, skipping query processing test")
        return False
    
    print_header("Testing Query Processing (Mock)")
    
    try:
        # Create a mock OpenAI client
        class MockOpenAI:
            class MockChoice:
                def __init__(self, message):
                    self.message = message

            class MockMessage:
                def __init__(self, content=None, function_call=None):
                    self.content = content
                    self.function_call = function_call

            class MockFunctionCall:
                def __init__(self, name, arguments):
                    self.name = name
                    self.arguments = arguments

            class Completions:
                def __init__(self, parent):
                    self.parent = parent

                def create(self, model, messages, functions=None, function_call=None, temperature=None):
                    # For the first call, return a function call
                    if any(msg.get("role") == "user" for msg in messages):
                        function_call = MockOpenAI.MockFunctionCall(
                            "explain_vpa_concepts",
                            json.dumps({"concept": "volume spread analysis"})
                        )
                        message = MockOpenAI.MockMessage(content=None, function_call=function_call)
                        return type("MockResponse", (), {"choices": [MockOpenAI.MockChoice(message)]})()
                    # For the second call (after function execution), return content
                    else:
                        message = MockOpenAI.MockMessage(
                            content="Volume Spread Analysis (VSA) is a methodology that analyzes the relationship between price, spread (range), and volume.",
                            function_call=None
                        )
                        return type("MockResponse", (), {"choices": [MockOpenAI.MockChoice(message)]})()

            class Chat:
                def __init__(self, parent):
                    self.completions = MockOpenAI.Completions(parent)

            def __init__(self):
                self.chat = MockOpenAI.Chat(self)

        # Save original client
        original_client = query_engine.client
        
        # Replace with mock
        query_engine.client = MockOpenAI()
        
        # Test query processing
        print_info("Processing mock query...")
        result = query_engine.process_query("What is volume spread analysis?")
        
        # Restore original client
        query_engine.client = original_client
        
        if result is None:
            print_error("Query processing returned None")
            return False
        
        if not isinstance(result, dict):
            print_error(f"Query processing returned unexpected type: {type(result)}")
            return False
        
        if "error" in result:
            print_error(f"Query processing returned error: {result['error']}")
            return False
        
        if "response" not in result:
            print_error("Query processing result does not contain 'response' field")
            return False

        if result["response"] is None:
            print_error("Query processing result 'response' field is None")
            return False

        print_success("Query processing completed successfully")
        print_info(f"Response: {result['response'][:100]}...")
        
        if "function_called" in result:
            print_success(f"Function called: {result['function_called']}")
        
        return True
    except Exception as e:
        print_error(f"Error testing query processing: {e}")
        # Restore original client if exception occurred
        if 'original_client' in locals():
            query_engine.client = original_client
        return False

def main():
    """Main function to test OpenAI function calling integration"""
    print_header("VPA OpenAI Function Calling Test")
    
    # Test OpenAI API key
    api_key_valid = test_openai_api_key()
    
    if not api_key_valid:
        print_warning("OpenAI API key test failed, some subsequent tests may fail")
    
    # Test query engine initialization
    query_engine = test_query_engine_initialization()
    
    # Test function definitions
    function_defs_valid = test_function_definitions(query_engine)
    
    # Test function execution
    function_exec_valid = test_function_execution(query_engine)
    
    # Test mock query processing
    query_processing_valid = test_mock_query_processing(query_engine)
    
    # Print summary
    print_header("Test Summary")
    
    if api_key_valid and query_engine and function_defs_valid and function_exec_valid and query_processing_valid:
        print_success("OpenAI function calling integration test passed!")
    else:
        print_warning("OpenAI function calling integration test completed with some failures.")
        
        # Print specific recommendations
        if not api_key_valid:
            print_info("Please set the OPENAI_API_KEY environment variable or add it to your .env file.")
        
        if not query_engine:
            print_info("Check VPAQueryEngine initialization and OpenAI client configuration.")
        
        if not function_defs_valid:
            print_info("Review function definitions in VPAQueryEngine.")
        
        if not function_exec_valid:
            print_info("Check function execution logic in VPAQueryEngine.")
        
        if not query_processing_valid:
            print_info("Review query processing flow in VPAQueryEngine.")
    
    return 0 if (api_key_valid and query_engine and function_defs_valid and function_exec_valid and query_processing_valid) else 1

if __name__ == "__main__":
    sys.exit(main())

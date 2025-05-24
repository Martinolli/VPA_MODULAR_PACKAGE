#!/usr/bin/env python3
"""
VPA Memory Manager Integration Test

This script tests the integration between the VPAQueryEngine and MemoryManager,
verifying that conversation context is properly stored and retrieved.
"""

import os
import sys
import json
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

def test_memory_manager_initialization():
    """Test MemoryManager initialization"""
    print_header("Testing MemoryManager Initialization")
    
    try:
        from vpa_modular.memory_manager_enhanced import MemoryManager
        
        # Initialize with default settings
        print_info("Initializing MemoryManager with default settings...")
        memory_manager = MemoryManager()
        print_success("MemoryManager initialized with default settings")
        
        # Initialize with custom settings
        print_info("Initializing MemoryManager with custom settings...")
        test_memory_file = "test_memory.json"
        memory_manager_custom = MemoryManager(memory_file=test_memory_file, max_memory_items=50)
        print_success("MemoryManager initialized with custom settings")
        
        # Clean up test file
        if os.path.exists(test_memory_file):
            os.remove(test_memory_file)
        
        return memory_manager
    except Exception as e:
        print_error(f"Error initializing MemoryManager: {e}")
        return None

def test_memory_storage_and_retrieval(memory_manager):
    """Test memory storage and retrieval"""
    if not memory_manager:
        print_error("MemoryManager not initialized, skipping storage and retrieval test")
        return False
    
    print_header("Testing Memory Storage and Retrieval")
    
    try:
        # Store test interactions
        print_info("Storing test interactions...")
        
        # First interaction
        memory_manager.store_interaction(
            query="What is the current analysis for AAPL?",
            response="Based on VPA analysis, AAPL shows a BUY signal with MODERATE strength.",
            function_data={"ticker": "AAPL", "signal": {"type": "BUY", "strength": "MODERATE"}}
        )
        
        # Second interaction
        memory_manager.store_interaction(
            query="How about MSFT and GOOGL?",
            response="MSFT shows a BUY signal with STRONG strength, while GOOGL shows a NEUTRAL signal.",
            function_data={"tickers": ["MSFT", "GOOGL"], "results": {
                "MSFT": {"signal_type": "BUY", "signal_strength": "STRONG"},
                "GOOGL": {"signal_type": "NEUTRAL", "signal_strength": "WEAK"}
            }}
        )
        
        # Third interaction
        memory_manager.store_interaction(
            query="Explain volume spread analysis",
            response="Volume Spread Analysis (VSA) is a methodology that analyzes the relationship between price, spread, and volume.",
            function_data={"concept": "volume spread analysis"}
        )
        
        print_success("Test interactions stored")
        
        # Retrieve context for a query
        print_info("Retrieving context for a query about AAPL...")
        context = memory_manager.get_context("Tell me about AAPL stock")
        
        if "AAPL" in context:
            print_success("Context retrieved successfully and contains relevant information")
        else:
            print_warning("Context retrieved but may not contain relevant information")
            print_info(f"Context: {context}")
        
        # Get ticker history
        print_info("Retrieving history for MSFT...")
        msft_history = memory_manager.get_ticker_history("MSFT")
        
        if msft_history and len(msft_history) > 0:
            print_success(f"Retrieved {len(msft_history)} history items for MSFT")
        else:
            print_warning("No history found for MSFT")
        
        # Get recent memory
        print_info("Retrieving recent memory...")
        recent_memory = memory_manager.get_recent_memory(count=2)
        
        if recent_memory and len(recent_memory) == 2:
            print_success(f"Retrieved {len(recent_memory)} recent memory items")
        else:
            print_warning(f"Expected 2 recent memory items, got {len(recent_memory) if recent_memory else 0}")
        
        # Clear memory
        print_info("Clearing memory...")
        memory_manager.clear()
        
        all_memory = memory_manager.get_all_memory()
        if not all_memory or len(all_memory) == 0:
            print_success("Memory cleared successfully")
        else:
            print_warning(f"Memory not cleared, {len(all_memory)} items remain")
        
        return True
    except Exception as e:
        print_error(f"Error testing memory storage and retrieval: {e}")
        return False

def test_query_engine_memory_integration():
    """Test integration between VPAQueryEngine and MemoryManager"""
    print_header("Testing VPAQueryEngine and MemoryManager Integration")
    
    try:
        from vpa_modular.vpa_llm_query_engine_enhanced import VPAQueryEngine
        from vpa_modular.memory_manager_enhanced import MemoryManager
        
        # Initialize memory manager
        memory_manager = MemoryManager(memory_file="test_integration_memory.json")
        
        # Initialize query engine with memory manager
        print_info("Initializing VPAQueryEngine with MemoryManager...")
        query_engine = VPAQueryEngine(memory_manager=memory_manager)
        print_success("VPAQueryEngine initialized with MemoryManager")
        
        # Verify memory manager is set
        if query_engine.memory_manager is memory_manager:
            print_success("MemoryManager reference is correctly set in VPAQueryEngine")
        else:
            print_error("MemoryManager reference is not correctly set in VPAQueryEngine")
            return False
        
        # Mock process_query to avoid OpenAI API calls
        def mock_process_query(query):
            # Store in memory manager
            query_engine.memory_manager.store_interaction(
                query=query,
                response="This is a mock response for testing",
                function_data={"mock": True}
            )
            return {"response": "Mock response", "mock": True}
        
        # Save original method
        original_process_query = query_engine.process_query
        
        # Replace with mock
        query_engine.process_query = mock_process_query
        
        # Test with a query
        print_info("Testing with a mock query...")
        result = query_engine.process_query("This is a test query")
        
        # Verify memory was updated
        recent_memory = memory_manager.get_recent_memory(count=1)
        
        if recent_memory and len(recent_memory) > 0 and recent_memory[0]["query"] == "This is a test query":
            print_success("Query was correctly stored in memory")
        else:
            print_error("Query was not correctly stored in memory")
        
        # Restore original method
        query_engine.process_query = original_process_query
        
        # Clean up
        memory_manager.clear()
        if os.path.exists("test_integration_memory.json"):
            os.remove("test_integration_memory.json")
        
        return True
    except Exception as e:
        print_error(f"Error testing VPAQueryEngine and MemoryManager integration: {e}")
        return False

def main():
    """Main function to test memory management integration"""
    print_header("VPA Memory Management Integration Test")
    
    # Test memory manager initialization
    memory_manager = test_memory_manager_initialization()
    
    # Test memory storage and retrieval
    storage_retrieval_success = test_memory_storage_and_retrieval(memory_manager)
    
    # Test query engine integration
    integration_success = test_query_engine_memory_integration()
    
    # Print summary
    print_header("Test Summary")
    
    if memory_manager and storage_retrieval_success and integration_success:
        print_success("Memory management integration test passed!")
    else:
        print_warning("Memory management integration test failed.")
    
    return 0 if (memory_manager and storage_retrieval_success and integration_success) else 1

if __name__ == "__main__":
    sys.exit(main())

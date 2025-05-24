"""
Memory Manager Module for VPA Query Engine

This module provides memory management functionality for the VPA Query Engine,
allowing it to maintain conversation context and retrieve relevant information
from past interactions.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np

from .vpa_logger import get_logger

class MemoryManager:
    """
    Memory manager for VPA Query Engine that stores and retrieves conversation history
    and relevant context for natural language queries.
    """
    
    def __init__(self, memory_file: Optional[str] = None, max_memory_items: int = 100):
        """
        Initialize the Memory Manager
        
        Parameters:
        - memory_file: Optional path to memory storage file
        - max_memory_items: Maximum number of memory items to store
        """
        self.logger = get_logger(module_name="MemoryManager")
        
        # Set memory file path
        if memory_file is None:
            # Get the root directory (vpa_modular_package)
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            memory_dir = os.path.join(root_dir, ".vpa", "memory")
            os.makedirs(memory_dir, exist_ok=True)
            memory_file = os.path.join(memory_dir, "vpa_memory.json")
        
        self.memory_file = memory_file
        self.max_memory_items = max_memory_items
        
        # Initialize memory storage
        self.memory = self._load_memory()
    
    def _load_memory(self) -> List[Dict[str, Any]]:
        """
        Load memory from file
        
        Returns:
        - List of memory items
        """
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                self.logger.debug(f"Loaded {len(memory)} memory items from {self.memory_file}")
                return memory
            else:
                self.logger.debug(f"Memory file {self.memory_file} not found, initializing empty memory")
                return []
        except Exception as e:
            self.logger.error(f"Error loading memory from {self.memory_file}: {e}")
            return []
    
    def _save_memory(self) -> None:
        """Save memory to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
            self.logger.debug(f"Saved {len(self.memory)} memory items to {self.memory_file}")
        except Exception as e:
            self.logger.error(f"Error saving memory to {self.memory_file}: {e}")
    
    def store_interaction(self, query: str, response: str, function_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a user interaction in memory
        
        Parameters:
        - query: User's query
        - response: System's response
        - function_data: Optional data from function calls
        """
        # Create memory item
        memory_item = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "function_data": function_data
        }
        
        # Add to memory
        self.memory.append(memory_item)
        
        # Trim memory if needed
        if len(self.memory) > self.max_memory_items:
            self.memory = self.memory[-self.max_memory_items:]
        
        # Save to file
        self._save_memory()
        
        self.logger.debug(f"Stored interaction in memory: {query[:50]}...")
    
    def get_context(self, query: str, max_items: int = 3) -> str:
        """
        Get relevant context for a query
        
        Parameters:
        - query: User's query
        - max_items: Maximum number of context items to return
        
        Returns:
        - Context string
        """
        if not self.memory:
            return ""
        
        query_keywords = set(query.lower().split())
        
        scored_items = []
        for item in self.memory:
            if not isinstance(item, dict):
                self.logger.warning(f"Skipping invalid memory item: {item}")
                continue
            
            item_text = f"{item.get('query', '')} {item.get('response', '')}".lower()
            
            matching_keywords = sum(1 for keyword in query_keywords if keyword in item_text)
            
            try:
                timestamp = datetime.fromisoformat(item.get('timestamp', ''))
                time_diff = (datetime.now() - timestamp).total_seconds()
                recency_score = 1.0 / (1.0 + time_diff / 86400.0)
            except (ValueError, TypeError):
                recency_score = 0.0
            
            score = 0.7 * matching_keywords + 0.3 * recency_score
            scored_items.append((score, item))
        
        scored_items.sort(reverse=True, key=lambda x: x[0])
        top_items = [item for _, item in scored_items[:max_items]]
        
        context_parts = []
        for item in top_items:
            ticker_info = ""
            function_data = item.get('function_data', {})
            if isinstance(function_data, dict):
                if 'ticker' in function_data:
                    ticker = function_data['ticker']
                    ticker_info = f" about {ticker}"
                elif 'tickers' in function_data:
                    tickers = function_data.get('tickers', [])
                    if tickers:
                        ticker_info = f" about {', '.join(tickers)}"
            
            query = item.get('query', '')
            response = item.get('response', '')
            if query and response:
                context_parts.append(f"User asked{ticker_info}: '{query}' and you responded: '{response[:100]}...'")
        
        return "\n".join(context_parts)
    
    def get_ticker_history(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get history of interactions related to a specific ticker
        
        Parameters:
        - ticker: Stock symbol
        
        Returns:
        - List of memory items related to the ticker
        """
        ticker_history = []
        
        for item in self.memory:
            # Check query
            if ticker.upper() in item['query'].upper():
                ticker_history.append(item)
                continue
            
            # Check function data
            function_data = item.get('function_data', {})
            if function_data:
                if function_data.get('ticker') == ticker:
                    ticker_history.append(item)
                    continue
                
                if ticker in function_data.get('tickers', []):
                    ticker_history.append(item)
                    continue
        
        return ticker_history
    
    def clear(self) -> None:
        """Clear all memory"""
        self.memory = []
        self._save_memory()
        self.logger.info("Memory cleared")
    
    def get_all_memory(self) -> List[Dict[str, Any]]:
        """Get all memory items"""
        return self.memory
    
    def get_recent_memory(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get most recent memory items
        
        Parameters:
        - count: Number of items to return
        
        Returns:
        - List of recent memory items
        """
        return self.memory[-count:]

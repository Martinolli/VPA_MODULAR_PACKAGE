"""
VPA Query Engine Module

This module provides a natural language interface to the VPA system,
allowing users to query and interact with VPA analysis through conversational language.
"""

import os
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

import openai
from openai import OpenAI

from .vpa_facade import VPAFacade
from .vpa_logger import get_logger
from .config_manager import get_config_manager
from .memory_manager_enhanced import MemoryManager

class VPAQueryEngine:
    """
    Query engine for VPA system that processes natural language queries
    and returns VPA analysis results.
    """
    
    def __init__(self, facade: Optional[VPAFacade] = None, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize the VPA Query Engine
        
        Parameters:
        - facade: Optional VPAFacade instance (if None, a new one will be created)
        - memory_manager: Optional MemoryManager instance (if None, a new one will be created)
        """
        self.logger = get_logger(module_name="VPAQueryEngine")
        self.config_manager = get_config_manager()
        
        # Initialize VPA facade
        self.facade = facade if facade is not None else VPAFacade(logger=self.logger)
        
        # Initialize memory manager
        self.memory_manager = memory_manager if memory_manager is not None else MemoryManager()
        
        # Initialize OpenAI client
        self.client = self._initialize_openai_client()
        
        # Define available functions
        self.functions = self._define_functions()
        
        # Track conversation history
        self.conversation_history = []
    
    def _initialize_openai_client(self) -> Optional[OpenAI]:
        """
        Initialize the OpenAI client with error handling
        
        Returns:
        - OpenAI client instance or None if initialization fails
        """
        try:
            # API key is expected to be in the environment variable OPENAI_API_KEY
            # or in the config manager
            api_key = self.config_manager.get_api_key('openai')
            client = OpenAI(api_key=api_key)
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set.", exc_info=True)
            return None
    
    def _define_functions(self) -> List[Dict[str, Any]]:
        """
        Define available functions for the OpenAI function calling API
        
        Returns:
        - List of function definitions
        """
        return [
            {
                "name": "analyze_ticker",
                "description": "Analyze a stock ticker using Volume Price Analysis (VPA)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock symbol to analyze (e.g., AAPL, MSFT)"
                        },
                        "timeframes": {
                            "type": "array",
                            "description": "List of timeframes to analyze",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "interval": {
                                        "type": "string",
                                        "description": "Timeframe interval (e.g., 1d, 1h, 15m)"
                                    },
                                    "period": {
                                        "type": "string",
                                        "description": "Period to analyze (e.g., 1m, 3m, 1y)"
                                    }
                                },
                                "required": ["interval"]
                            }
                        }
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "batch_analyze",
                "description": "Analyze multiple stock tickers using Volume Price Analysis (VPA)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "description": "List of stock symbols to analyze",
                            "items": {
                                "type": "string"
                            }
                        },
                        "timeframes": {
                            "type": "array",
                            "description": "List of timeframes to analyze",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "interval": {
                                        "type": "string",
                                        "description": "Timeframe interval (e.g., 1d, 1h, 15m)"
                                    },
                                    "period": {
                                        "type": "string",
                                        "description": "Period to analyze (e.g., 1m, 3m, 1y)"
                                    }
                                },
                                "required": ["interval"]
                            }
                        }
                    },
                    "required": ["tickers"]
                }
            },
            {
                "name": "explain_vpa_concepts",
                "description": "Explain Volume Price Analysis (VPA) concepts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "The VPA concept to explain (e.g., volume spread analysis, effort vs result)"
                        }
                    },
                    "required": ["concept"]
                }
            },
            {
                "name": "get_market_summary",
                "description": "Get a summary of current market conditions based on VPA analysis of major indices",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "indices": {
                            "type": "array",
                            "description": "List of market indices to analyze",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
        ]
    
    def _get_function_map(self) -> Dict[str, Callable]:
        """
        Get mapping of function names to actual implementation functions
        
        Returns:
        - Dictionary mapping function names to callables
        """
        return {
            "analyze_ticker": self._execute_analyze_ticker,
            "batch_analyze": self._execute_batch_analyze,
            "explain_vpa_concepts": self._execute_explain_vpa_concepts,
            "get_market_summary": self._execute_get_market_summary
        }
    
    def _execute_analyze_ticker(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analyze_ticker function
        
        Parameters:
        - params: Function parameters
        
        Returns:
        - Analysis results
        """
        ticker = params.get("ticker")
        timeframes = params.get("timeframes")
        
        if not ticker:
            return {"error": "Ticker symbol is required"}
        
        # Use default timeframes if none provided
        if not timeframes:
            timeframes = [{"interval": "1d", "period": "1y"}]
        
        try:
            # Perform analysis
            results = self.facade.analyze_ticker(ticker=ticker, timeframes=timeframes)
            
            if not results:
                return {"error": f"Failed to analyze {ticker}"}
            
            # Generate explanation
            explanation = self.facade.explain_signal(results)
            
            # Format results for response
            formatted_results = {
                "ticker": ticker,
                "signal": results.get("signal", {}),
                "current_price": results.get("current_price"),
                "risk_assessment": results.get("risk_assessment", {}),
                "explanation": explanation
            }
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error executing analyze_ticker for {ticker}: {e}", exc_info=True)
            return {"error": f"Error analyzing {ticker}: {str(e)}"}
    
    def _execute_batch_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute batch_analyze function
        
        Parameters:
        - params: Function parameters
        
        Returns:
        - Batch analysis results
        """
        tickers = params.get("tickers", [])
        timeframes = params.get("timeframes")
        
        if not tickers:
            return {"error": "At least one ticker symbol is required"}
        
        try:
            # Perform batch analysis
            results = self.facade.batch_analyze(tickers=tickers, timeframes=timeframes)
            
            if not results:
                return {"error": "Failed to analyze any tickers"}
            
            # Format results for response
            formatted_results = {
                "tickers_analyzed": len(results),
                "results": {}
            }
            
            for ticker, ticker_results in results.items():
                signal = ticker_results.get("signal", {})
                formatted_results["results"][ticker] = {
                    "signal_type": signal.get("type"),
                    "signal_strength": signal.get("strength"),
                    "current_price": ticker_results.get("current_price")
                }
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error executing batch_analyze: {e}", exc_info=True)
            return {"error": f"Error performing batch analysis: {str(e)}"}
    
    def _execute_explain_vpa_concepts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute explain_vpa_concepts function
        
        Parameters:
        - params: Function parameters
        
        Returns:
        - Concept explanation
        """
        concept = params.get("concept")
        
        if not concept:
            return {"error": "Concept is required"}
        
        # This would ideally come from a knowledge base
        # For now, we'll return placeholder explanations
        concept_explanations = {
            "volume spread analysis": "Volume Spread Analysis (VSA) is a methodology that analyzes the relationship between price, spread (range), and volume to identify market manipulation and institutional activity. It was developed by Tom Williams and is based on the works of Richard Wyckoff.",
            "effort vs result": "Effort vs Result is a key principle in Volume Price Analysis where 'effort' is represented by volume and 'result' is represented by price movement. When high volume (effort) produces little price movement (result), it indicates potential reversal. Conversely, when low volume produces significant price movement, it suggests the move may be unsustainable.",
            "no demand": "No Demand is a VSA principle that occurs when prices attempt to rise but volume is low, indicating lack of buying interest. This is a bearish sign suggesting that despite the upward price movement, there is insufficient demand to sustain the move.",
            "no supply": "No Supply is a VSA principle that occurs when prices attempt to fall but volume is low, indicating lack of selling interest. This is a bullish sign suggesting that despite the downward price movement, there is insufficient supply to sustain the move.",
            "stopping volume": "Stopping Volume is a VSA principle that occurs when high volume appears at the end of a downtrend with price closing in the middle or upper part of the bar. This indicates potential buying from smart money and a possible trend reversal.",
            "climactic action": "Climactic Action in VSA refers to extremely high volume with wide price spread, often marking the end of a trend. It represents exhaustion of buying (in uptrends) or selling (in downtrends) and suggests a potential reversal."
        }
        
        # Search for the concept (case-insensitive)
        concept_lower = concept.lower()
        for key, explanation in concept_explanations.items():
            if concept_lower in key or key in concept_lower:
                return {
                    "concept": key,
                    "explanation": explanation
                }
        
        return {
            "concept": concept,
            "explanation": "This concept is not in our current knowledge base. Please try another VPA concept such as 'volume spread analysis', 'effort vs result', 'no demand', 'no supply', 'stopping volume', or 'climactic action'."
        }
    
    def _execute_get_market_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_market_summary function
        
        Parameters:
        - params: Function parameters
        
        Returns:
        - Market summary
        """
        indices = params.get("indices", ["SPY", "QQQ", "IWM", "DIA"])
        
        try:
            # Analyze each index
            results = self.facade.batch_analyze(tickers=indices, timeframes=[{"interval": "1d", "period": "1m"}])
            
            if not results:
                return {"error": "Failed to analyze market indices"}
            
            # Determine overall market sentiment
            bullish_count = 0
            bearish_count = 0
            
            for ticker, ticker_results in results.items():
                signal = ticker_results.get("signal", {})
                signal_type = signal.get("type", "NEUTRAL")
                
                if signal_type == "BUY":
                    bullish_count += 1
                elif signal_type == "SELL":
                    bearish_count += 1
            
            # Determine overall sentiment
            if bullish_count > bearish_count:
                overall_sentiment = "BULLISH"
            elif bearish_count > bullish_count:
                overall_sentiment = "BEARISH"
            else:
                overall_sentiment = "NEUTRAL"
            
            # Format results for response
            formatted_results = {
                "overall_sentiment": overall_sentiment,
                "indices_analyzed": len(results),
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": len(results) - bullish_count - bearish_count,
                "index_signals": {}
            }
            
            for ticker, ticker_results in results.items():
                signal = ticker_results.get("signal", {})
                formatted_results["index_signals"][ticker] = {
                    "signal_type": signal.get("type"),
                    "signal_strength": signal.get("strength"),
                    "current_price": ticker_results.get("current_price")
                }
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error executing get_market_summary: {e}", exc_info=True)
            return {"error": f"Error getting market summary: {str(e)}"}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query
        
        Parameters:
        - query: User's natural language query
        
        Returns:
        - Response dictionary
        """
        if not self.client:
            return {"error": "OpenAI client not initialized. Please check your API key."}
        
        try:
            # Add query to conversation history
            self.conversation_history.append({"role": "user", "content": query})
            
            # Get conversation context from memory manager
            context = self.memory_manager.get_context(query)
            
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": "You are a financial analysis assistant specializing in Volume Price Analysis (VPA). You help users analyze stocks and understand market conditions using VPA principles."}
            ]
            
            # Add context if available
            if context:
                messages.append({"role": "system", "content": f"Context from previous conversations: {context}"})
            
            # Add conversation history (limited to last 10 messages)
            messages.extend(self.conversation_history[-10:])
            
            # Call OpenAI API with function calling
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use appropriate model
                messages=messages,
                functions=self.functions,
                function_call="auto",
                temperature=0.1
            )
            
            # Get the response message
            response_message = response.choices[0].message
            
            # Check if function call was made
            if hasattr(response_message, 'function_call') and response_message.function_call:
                # Extract function call details
                function_name = response_message.function_call.name
                function_args = json.loads(response_message.function_call.arguments)
                
                # Get function map
                function_map = self._get_function_map()
                
                # Execute function if it exists
                if function_name in function_map:
                    function_response = function_map[function_name](function_args)
                    
                    # Add function response to conversation
                    self.conversation_history.append({
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(function_response)
                    })
                    
                    # Get final response from OpenAI
                    final_response = self.client.chat.completions.create(
                        model="gpt-4o",  # Use appropriate model
                        messages=messages + [
                            response_message,
                            {
                                "role": "function",
                                "name": function_name,
                                "content": json.dumps(function_response)
                            }
                        ],
                        temperature=0.1
                    )
                    
                    final_content = final_response.choices[0].message.content
                    
                    # Add assistant response to conversation history
                    self.conversation_history.append({"role": "assistant", "content": final_content})
                    
                    # Store in memory manager
                    self.memory_manager.store_interaction(query, final_content, function_response)
                    
                    return {
                        "response": final_content,
                        "function_called": function_name,
                        "function_response": function_response
                    }
                else:
                    error_message = f"Function {function_name} not implemented"
                    self.logger.error(error_message)
                    return {"error": error_message}
            else:
                # Direct response (no function call)
                content = response_message.content
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": content})
                
                # Store in memory manager
                self.memory_manager.store_interaction(query, content)
                
                return {"response": content}
        
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return {"error": error_message}
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []
        self.memory_manager.clear()
        self.logger.info("Conversation history and memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history

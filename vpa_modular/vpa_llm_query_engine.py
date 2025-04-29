"""
Module implementing the VPA LLM Query Engine using OpenAI's API.

This module provides an interface to interact with the VPA system using natural
language queries, leveraging OpenAI's function calling capabilities.
"""

import os
import json
import logging
from openai import OpenAI, OpenAIError

# Assuming VPAFacade is importable from the vpa_modular package
try:
    from vpa_modular.vpa_facade import VPAFacade
except ImportError:
    print("Error: VPAFacade not found. Make sure vpa_modular package is accessible.")
    # Define a dummy class if facade is not available, to allow basic structure
    class VPAFacade:
        def analyze_ticker(self, ticker, primary_timeframe="1d", other_timeframes=None):
            # Dummy implementation for structure
            logging.warning("Using dummy VPAFacade.analyze_ticker")
            return {"error": "Dummy VPAFacade used", "ticker": ticker, "timeframe": primary_timeframe}
        
        def get_concept_explanation(self, concept_name):
            # Dummy implementation
            logging.warning("Using dummy VPAFacade.get_concept_explanation")
            return {"error": "Dummy VPAFacade used", "concept": concept_name, "explanation": "Explanation not available (dummy)."}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Function Definitions for OpenAI --- 
# (Based on vpa_query_parsing_approach.md)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_vpa_analysis",
            "description": "Performs Volume Price Analysis for a given stock ticker and timeframe, returning key findings like signals, patterns, and trend information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., \"AAPL\")"},
                    "timeframe": {"type": "string", "description": "The primary timeframe (e.g., \"1d\")", "default": "1d"},
                    "include_secondary_timeframes": {"type": "boolean", "description": "Include context from secondary timeframes?", "default": False}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_vpa_concept",
            "description": "Provides an explanation of a specific Volume Price Analysis concept, pattern, or term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept_name": {"type": "string", "description": "The name of the VPA concept (e.g., \"Selling Climax\")"}
                },
                "required": ["concept_name"]
            }
        }
    }
]

# --- VPA Query Engine Class --- 
class VPAQueryEngine:
    """Handles natural language queries about VPA using OpenAI and VPAFacade."""

    def __init__(self, vpa_facade: VPAFacade, openai_model="gpt-4o"):
        """Initializes the engine with VPAFacade and OpenAI client."""
        self.vpa_facade = vpa_facade
        self.openai_model = openai_model
        try:
            # API key is expected to be in the environment variable OPENAI_API_KEY
            self.openai_client = OpenAI()
            # Test connection (optional, but good practice)
            self.openai_client.models.list() 
            logger.info(f"OpenAI client initialized successfully for model {self.openai_model}.")
        except OpenAIError as e:
            logger.error(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set.", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI client initialization: {e}", exc_info=True)
            raise

    def _execute_get_vpa_analysis(self, ticker: str, timeframe: str = "1d", include_secondary_timeframes: bool = False):
        """Executes the VPA analysis function using the facade."""
        logger.info(f"Executing VPA analysis for {ticker} ({timeframe})...")
        try:
            # TODO: Ideally, facade should support point-in-time analysis
            # For now, use the standard analyze_ticker
            secondary_tfs = self.vpa_facade.config.get('secondary_timeframes', []) if include_secondary_timeframes else []
            results = self.vpa_facade.analyze_ticker(
                ticker=ticker, 
                primary_timeframe=timeframe, 
                other_timeframes=secondary_tfs
            )
            # Convert results to JSON string for the tool message content
            return json.dumps(results)
        except Exception as e:
            logger.error(f"Error executing VPA analysis for {ticker}: {e}", exc_info=True)
            return json.dumps({"error": f"Failed to perform VPA analysis for {ticker}: {str(e)}"})

    def _execute_explain_vpa_concept(self, concept_name: str):
        """Executes the concept explanation function."""
        logger.info(f"Executing explanation for concept: {concept_name}...")
        try:
            # Option 1: Try getting explanation from facade/knowledge base
            # explanation = self.vpa_facade.get_concept_explanation(concept_name)
            # if explanation and not explanation.get("error"): 
            #     return json.dumps(explanation)
            
            # Option 2: Just pass back the concept name, let LLM explain based on its knowledge
            # This is simpler initially and leverages the LLM's primary strength.
            return json.dumps({"concept_name": concept_name, "status": "Ready for LLM explanation"})
        except Exception as e:
            logger.error(f"Error executing concept explanation for {concept_name}: {e}", exc_info=True)
            return json.dumps({"error": f"Failed to process explanation request for {concept_name}: {str(e)}"})

    def handle_query(self, user_query: str):
        """Handles a user's natural language query."""
        logger.info(f"Received query: {user_query}")
        
        system_message = {
            "role": "system",
            "content": "You are an expert in Volume Price Analysis (VPA) based on the Wyckoff method and Anna Coulling's teachings. Your goal is to analyze stock market data using VPA principles, explain VPA concepts, and answer user questions accurately. When analysis is requested, use the provided functions to get VPA data and then interpret the results clearly, referencing specific VPA patterns, signals (like Selling Climax, No Demand, Shakeout, Tests, etc.), volume/spread relationships, and trend context. Provide concise yet informative explanations."
        }
        
        messages = [system_message, {"role": "user", "content": user_query}]
        
        try:
            # --- First API Call --- 
            logger.debug("Making initial API call to OpenAI...")
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                tools=tools,
                tool_choice="auto" # Let the model decide whether to use a function
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # --- Check if Function Call is Needed --- 
            if tool_calls:
                logger.info(f"LLM requested function call(s): {tool_calls}")
                messages.append(response_message) # Add assistant's reply to messages
                
                # --- Execute Function(s) --- 
                # Note: OpenAI currently only supports one function call per turn in practice, 
                # but the API allows multiple. We'll handle the first one for simplicity.
                tool_call = tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                function_result_content = ""
                if function_name == "get_vpa_analysis":
                    function_result_content = self._execute_get_vpa_analysis(**function_args)
                elif function_name == "explain_vpa_concept":
                    function_result_content = self._execute_explain_vpa_concept(**function_args)
                else:
                    logger.warning(f"LLM requested unknown function: {function_name}")
                    function_result_content = json.dumps({"error": f"Unknown function requested: {function_name}"})
                
                # --- Add Function Result to Messages --- 
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_result_content,
                    }
                )
                
                # --- Second API Call (with function result) --- 
                logger.debug("Making second API call to OpenAI with function result...")
                second_response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages
                )
                final_response_content = second_response.choices[0].message.content
                logger.info("Received final response from LLM after function call.")
                return final_response_content
                
            else:
                # --- No Function Call Needed - Direct Answer --- 
                logger.info("LLM provided a direct answer.")
                direct_answer = response_message.content
                return direct_answer

        except OpenAIError as e:
            logger.error(f"OpenAI API error during query handling: {e}", exc_info=True)
            return f"Error: Could not communicate with the AI assistant. {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON arguments from LLM: {e}", exc_info=True)
            return f"Error: Received malformed arguments from the AI assistant."
        except Exception as e:
            logger.error(f"An unexpected error occurred during query handling: {e}", exc_info=True)
            return f"Error: An unexpected issue occurred while processing your request. {str(e)}"

# --- Example Usage (for testing purposes) ---
if __name__ == '__main__':
    print("Running VPA LLM Query Engine Example...")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable not set.")
        print("Please set the environment variable before running.")
        # Example: export OPENAI_API_KEY='your-api-key'
        exit(1)
        
    try:
        # Initialize with a real or dummy facade
        # For real use, ensure VPAFacade is correctly configured and importable
        vpa_facade_instance = VPAFacade() 
        
        query_engine = VPAQueryEngine(vpa_facade=vpa_facade_instance)
        
        # --- Test Queries ---
        queries = [
            "What is a selling climax in VPA?",
            "Analyze AAPL using VPA for the daily timeframe.",
            "Can you check the VPA signal for MSFT on the 1h chart?"
        ]
        
        for query in queries:
            print(f"\n--- Query: {query} ---")
            response = query_engine.handle_query(query)
            print(f"Response:\n{response}")
            print("------------------------")
            
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")



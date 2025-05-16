"""
Module implementing the VPA LLM Query Engine using OpenAI's API.query
This module provides an interface to interact with the VPA system using natural
language queries, leveraging OpenAI's function calling capabilities.
"""

import os
import json
import logging
from openai import OpenAI, OpenAIError
from vpa_modular.vpa_llm_interface import VPALLMInterface
from vpa_modular.rag.retriever import retrieve_top_chunks
from vpa_modular.memory_manager import MemoryManager

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
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_trading_parameters",
            "description": "Suggests trading parameters for a given stock and trading style.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker symbol"},
                    "trading_style": {"type": "string", "description": "The trading style (e.g., 'day_trading', 'swing_trading')"}
                },
                "required": ["ticker", "trading_style"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "search_vpa_documents",
        "description": "Retrieves similar VPA concepts or examples from the embedded Anna Coulling book.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language question or keyword related to VPA concepts."
                }
            },
            "required": ["query"]
        }
    }
    }
]

# --- VPA Query Engine Class --- 
class VPAQueryEngine:
    """Handles natural language queries about VPA using OpenAI and VPAFacade."""

    def __init__(self, vpa_facade: VPAFacade, openai_model="gpt-4o"):
        """Initializes the engine with VPAFacade and OpenAI client."""
        self.memory = MemoryManager()  # Initialize memory manager
        self.vpa_facade = vpa_facade # Initialize VPAFacade
        self.openai_model = openai_model # Set the OpenAI model
        self.logger = logging.getLogger(__name__) # Initialize logger
        self.llm_interface = VPALLMInterface()  # Initialize VPALLMInterface

        # Add system message
        self.memory.add_system_message("""You are an expert in Volume Price Analysis (VPA) based on the Wyckoff method and Anna Coulling's teachings.
        Your goal is to analyze stock market data using VPA principles, explain VPA concepts, and answer user questions accurately.""")
        
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

    def _execute_search_vpa_documents(self, query: str) -> str:
        """Executes a search for VPA concepts in the embedded Anna Coulling book."""
         # Step 1: Retrieve top matching chunks
        top_k=10 # Define the Top K parameter for the number of chunks to retrieve
        top_chunks = retrieve_top_chunks(query, top_k)

        if not top_chunks:
            return json.dumps({"answer": "No relevant information found in the VPA document."})

        # Step 2: Build context from metadata-enhanced chunks
        context_text = "\n\n".join(
            f"Source: {chunk['source']}\n"
            f"Page: {chunk.get('metadata', {}).get('page', '?')} - "
            f"{chunk.get('metadata', {}).get('section', '?')}\n"
            f"{chunk['text']}"
            for chunk in top_chunks
        )

        # Step 3: Create a system message for OpenAI
        system_message = {
            "role": "system",
            "content": (
                "You are an expert assistant specializing in Volume Price Analysis (VPA) based on Anna Coulling's book, "
                "the Wyckoff Method, and related trading concepts. Provide comprehensive, accurate, and well-structured answers. "
                "Use examples where appropriate and always relate your answer back to VPA principles and the Wyckoff Method when relevant."
            )
        }

        # Step 4: Create a user prompt
        user_prompt = (
        f"Question: {query}\n\n"
        "Based on the following extracted passages from 'Volume Price Analysis' by Anna Coulling, "
        "please provide a comprehensive answer. Include the following elements:\n"
        "1. A clear and concise explanation of the concept or topic.\n"
        "2. How it relates to broader VPA principles.\n"
        "3. Practical application or example in trading scenarios.\n"
        "4. Any related VPA concepts that might be relevant.\n\n"
        f"Extracted Passages:\n{context_text}\n\n"
        "Comprehensive Answer:"
    )

        try:
            messages = self.memory.get_history()  # Get the conversation history
            messages.append({"role": "system", "content": system_message["content"]})
            messages.append({"role": "user", "content": user_prompt})

            response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            max_tokens=7000,
            temperature=0.5,
            presence_penalty=0.3,
            frequency_penalty=0.3,
            )
            # Step 5: Extract the answer from the response

            answer_text = response.choices[0].message.content.strip()
            self.memory.save_message("user", user_prompt)
            self.memory.save_message("assistant", answer_text)
        
        # Step 6: Package structured result
            result = {
            "answer": answer_text,
            "source_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk.get("source"),
                    "page": chunk.get("metadata", {}).get("page"),
                    "section": chunk.get("metadata", {}).get("section"),
                    "text": chunk["text"][:500] + "..."  # trim to avoid overloading
                }
                for chunk in top_chunks
            ]
            }
            return json.dumps(result)
        
        except Exception as e:
            self.logger.error(f"Error generating answer from VPA book chunks: {e}")
            return f"Error generating response based on embedded content: {str(e)}"

    def _execute_get_vpa_analysis(self, ticker, timeframe="1d", include_secondary_timeframes=False):
        """Executes VPA analysis using the VPAFacade and VPALLMInterface."""
        self.logger.info(f"Executing VPA analysis for {ticker} ({timeframe})...")
        try:
            # --- Revised Timeframe Handling ---
            # Define the primary timeframe based on LLM request
            # Assuming a default period like '1y'. Adjust if your config/facade expects something different.
            primary_tf_dict = {"interval": timeframe, "period": "1y"} 

            analysis_timeframes = [primary_tf_dict]

            if include_secondary_timeframes:
                try:
                    # Get default timeframes from config (as used in your VPAFacade)
                    default_timeframes = self.vpa_facade.config.get_timeframes()
                    # Add secondary timeframes from the default config, avoiding duplicates
                    for tf_dict in default_timeframes:
                        # Ensure the dictionary has the 'interval' key before comparing
                        if tf_dict.get("interval") != timeframe:
                            analysis_timeframes.append(tf_dict)
                except AttributeError:
                    self.logger.warning("VPAConfig does not have 'get_timeframes' method. Using only primary timeframe.")
                except Exception as e:
                    self.logger.warning(f"Could not get secondary timeframes from config: {e}. Using only primary timeframe.")

            self.logger.info(f"Analyzing {ticker} with timeframes: {analysis_timeframes}")

            # --- Call analyze_ticker with the correct signature ---
            # Pass the list of timeframe dictionaries to the 'timeframes' argument
            try:
                # Use VPALLMInterface to get the analysis
                llm_analysis = self.llm_interface.get_ticker_analysis(ticker)
                facade_results = self.vpa_facade.analyze_ticker(
                ticker=ticker,
                timeframes=analysis_timeframes 
                )

                results ={**llm_analysis, **facade_results}

                # --- Process and summarize results for LLM --- 
                # This summary structure assumes the keys returned by your analyze_ticker method.
                # Adjust if the actual keys in 'results' are different.
                summary = {
                    "ticker": results.get("ticker"),
                    "signal": results.get("signal"),
                    "risk_assessment": results.get("risk_assessment"),
                    "current_price": results.get("current_price"),
                    # Attempt to get a summary for the primary timeframe if available
                    "primary_analysis_summary": results.get("timeframe_analyses", {}).get(timeframe, {}).get("summary", "Analysis summary not available."), 
                    "confirmations": results.get("confirmations")
                }
                
                # Convert the summary dictionary to a JSON string for the LLM
                # Using default=str handles potential non-serializable types like datetime
                return json.dumps(summary, default=str)
            except Exception as e:
                self.logger.error(f"Error executing VPA analysis for {ticker}: {e}", exc_info=True)
                return json.dumps({"error": f"Failed to perform VPA analysis for {ticker}: {str(e)}"})
            
        except Exception as e:
            self.logger.error(f"Error executing VPA analysis for {ticker}: {e}", exc_info=True)
            # Return a clear error message as a JSON string for the LLM
            return json.dumps({"error": f"Failed to perform VPA analysis for {ticker}: {str(e)}"})

    def _execute_explain_vpa_concept(self, concept_name: str):
        """Executes the concept explanation function."""
        self.logger.info(f"Executing explanation for concept: {concept_name}...")
        try:
            explanation = self.llm_interface.explain_vpa_concept(concept_name)
            if not explanation:
                return json.dumps({"error": f"No explanation found for concept: {concept_name}"})
            return json.dumps({"explanation": explanation})
        except Exception as e:
            self.logger.error(f"Error executing concept explanation for {concept_name}: {e}", exc_info=True)
            return json.dumps({"error": f"Failed to process explanation request for {concept_name}: {str(e)}"})

    def _execute_suggest_trading_parameters(self, ticker, trading_style):
        try:
            suggestion = self.llm_interface.suggest_parameters(ticker, trading_style)
            return json.dumps(suggestion)
        except Exception as e:
            self.logger.error(f"Error suggesting parameters: {e}")
            return json.dumps({"error": f"Failed to suggest parameters for {ticker}: {str(e)}"})
    
    def _execute_function(self, function_name, **kwargs):
        if function_name == "get_vpa_analysis":
            return self._execute_get_vpa_analysis(**kwargs)
        elif function_name == "explain_vpa_concept":
            return self._execute_explain_vpa_concept(**kwargs)
        elif function_name == "suggest_trading_parameters":
            return self._execute_suggest_trading_parameters(**kwargs)
        elif function_name == "search_vpa_documents":
            return self._execute_search_vpa_documents(**kwargs)
        else:
            return json.dumps({"error": f"Unknown function requested: {function_name}"})
    
    def _validate_messages(self, messages):
        """
        Validates and sanitizes all messages to ensure they have valid content.
        This is a comprehensive fix to prevent "Invalid value for 'content': expected a string, got null" errors.
        """
        validated_messages = []
        for i, msg in enumerate(messages):
            # Create a new message dict to avoid modifying the original
            valid_msg = msg.copy() if isinstance(msg, dict) else {}
            
            # Ensure 'role' exists and is valid
            if 'role' not in valid_msg or not valid_msg['role']:
                self.logger.warning(f"Message at index {i} has missing or invalid 'role'. Setting to 'system'.")
                valid_msg['role'] = 'system'
            
            # Ensure 'content' exists and is a string
            if 'content' not in valid_msg or valid_msg['content'] is None:
                self.logger.warning(f"Message at index {i} with role '{valid_msg['role']}' has missing or null 'content'. Setting to empty string.")
                valid_msg['content'] = ""
            elif not isinstance(valid_msg['content'], str):
                self.logger.warning(f"Message at index {i} with role '{valid_msg['role']}' has non-string 'content'. Converting to string.")
                valid_msg['content'] = str(valid_msg['content'])
            
            # For function messages, ensure 'name' exists
            if valid_msg['role'] == 'function' and ('name' not in valid_msg or not valid_msg['name']):
                self.logger.warning(f"Function message at index {i} has missing 'name'. Setting to 'unknown_function'.")
                valid_msg['name'] = 'unknown_function'
            
            validated_messages.append(valid_msg)
            
        return validated_messages
    
    def handle_query(self, user_query: str):
        """Handles a user's natural language query."""
        self.logger.info(f"Received query: {user_query}")
        
        # Add the user query to the memory
        self.memory.save_message("user", user_query)
        
        try:
            while True:
                # Get history and validate all messages
                raw_messages = self.memory.get_history()
                messages = self._validate_messages(raw_messages)
                
                # Add current user query if not in history
                if not any(msg.get('role') == 'user' and msg.get('content') == user_query for msg in messages):
                    messages.append({"role": "user", "content": user_query})
                
                self.logger.debug(f"Sending validated messages to OpenAI: {messages}")
                
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                
                response_message = response.choices[0].message
                
                # Check if content exists before saving
                if response_message.content:
                    self.memory.save_message("assistant", response_message.content)
                else:
                    self.logger.warning("Received empty content from OpenAI")
                    # Save empty string instead of None
                    self.memory.save_message("assistant", "")
                
                if not response_message.tool_calls:
                    return response_message.content or "No response content received."
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    self.logger.debug(f"Executing function: {function_name} with args: {function_args}")
                    function_response = self._execute_function(function_name, **function_args)
                    
                    # Ensure function_response is a valid string
                    content_str = str(function_response) if function_response is not None else ""
                    
                    # Save the function response to memory
                    self.memory.save_message("function", json.dumps({
                        "name": function_name,
                        "response": content_str  # Use validated string
                    }))
                    
                    # Add function message with validated content
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": content_str
                    })
                    
        except Exception as e:
            self.logger.error(f"Error in handle_query: {e}", exc_info=True)
            return f"An error occurred while processing your query: {str(e)}"

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
            "Can you check the VPA signal for MSFT on the 1h chart?",
            "Explain the concept of accumulation in VPA.",
            "Suggest parameters for day trading TSLA."
        ]
        
        for query in queries:
            print(f"\n--- Query: {query} ---")
            response = query_engine.handle_query(query)
            print(f"Response:\n{response}")
            print("------------------------")
            
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")

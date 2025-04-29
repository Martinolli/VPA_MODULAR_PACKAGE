"""
Command-Line Interface (CLI) example for interacting with VPA LLM Query Engine.

Allows users to enter queries directly in the terminal.
"""

import os
import logging
from dotenv import load_dotenv

# Assuming the VPA modules are in the python path or installed
# Adjust the import path if necessary based on your project structure
try:
    from vpa_modular.vpa_facade import VPAFacade
    from vpa_modular.vpa_llm_query_engine import VPAQueryEngine
except ImportError as e:
    print(f"Import Error: {e}. Make sure vpa_modular package is accessible.")
    # Define dummy classes if modules are not found
    class VPAFacade:
        pass
    class VPAQueryEngine:
        def __init__(self, *args, **kwargs):
            print("Warning: Using dummy VPAQueryEngine.")
        def handle_query(self, query):
            return f"Dummy response for query: {query}"

# --- Logging Setup ---
# Reduce logging level for cleaner CLI output, but keep engine logs informative
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# You might want to set the engine\s logger level specifically if needed
# logging.getLogger("vpa_modular.vpa_llm_query_engine").setLevel(logging.INFO)

# --- Load Environment Variables ---
load_dotenv() # Load .env file if it exists (for OPENAI_API_KEY)

# --- Main CLI Loop ---
def main():
    print("--- VPA LLM Query CLI ---")
    print("Enter your VPA query below. Type \'exit\' or \'quit\' to end.")

    # --- Initialize VPA and LLM Engine ---
    vpa_facade_instance = None
    query_engine_instance = None
    initialization_error = None

    try:
        # Check for OpenAI API Key before initializing
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it (e.g., in a .env file or export). Example: export OPENAI_API_KEY=\'your-key\'")
        
        print("Initializing VPA and LLM engine...")
        # If VPAFacade needs specific config, provide it here
        vpa_facade_instance = VPAFacade()
        query_engine_instance = VPAQueryEngine(vpa_facade=vpa_facade_instance)
        print("Initialization complete.")

    except Exception as e:
        print(f"\nError during initialization: {e}")
        initialization_error = str(e)
        # Allow CLI to run even if init fails, to show the error

    # --- Query Loop ---
    while True:
        try:
            user_input = input("\nQuery> ")
            query = user_input.strip()

            if query.lower() in ["exit", "quit"]:
                print("Exiting CLI.")
                break
            
            if not query:
                continue

            if initialization_error:
                print(f"Cannot process query due to initialization error: {initialization_error}")
                continue
                
            if not query_engine_instance:
                 print("Error: Query engine is not available.")
                 continue

            print("Processing...")
            response = query_engine_instance.handle_query(query)
            print("\nResponse:")
            print(response)

        except EOFError:
            print("\nExiting CLI.")
            break
        except KeyboardInterrupt:
            print("\nExiting CLI.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            # Optionally add more detailed logging here

if __name__ == "__main__":
    main()


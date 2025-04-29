"""
Flask web application example demonstrating integration with VPA LLM Query Engine.

Serves a simple web interface and handles queries via an API endpoint.
"""

import os
import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Assuming the VPA modules are in the python path or installed
# Adjust the import path if necessary based on your project structure
try:
    from vpa_modular.vpa_facade import VPAFacade
    from vpa_modular.vpa_llm_query_engine import VPAQueryEngine
except ImportError as e:
    print(f"Import Error: {e}. Make sure vpa_modular package is accessible.")
    # Define dummy classes if modules are not found, to allow Flask app to start
    class VPAFacade:
        pass
    class VPAQueryEngine:
        def __init__(self, *args, **kwargs):
            print("Warning: Using dummy VPAQueryEngine.")
        def handle_query(self, query):
            return f"Dummy response for query: {query}"

# --- Flask App Setup ---
app = Flask(__name__)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv() # Load .env file if it exists (for OPENAI_API_KEY)

# --- Initialize VPA and LLM Engine ---
vpa_facade_instance = None
query_engine_instance = None
initialization_error = None

try:
    # Check for OpenAI API Key before initializing
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it (e.g., in a .env file).")
    
    logger.info("Initializing VPAFacade...")
    # If VPAFacade needs specific config, provide it here
    vpa_facade_instance = VPAFacade()
    logger.info("VPAFacade initialized.")
    
    logger.info("Initializing VPAQueryEngine...")
    query_engine_instance = VPAQueryEngine(vpa_facade=vpa_facade_instance)
    logger.info("VPAQueryEngine initialized.")

except Exception as e:
    logger.error(f"Failed to initialize VPA/LLM components: {e}", exc_info=True)
    initialization_error = str(e)

# --- Flask Routes ---

@app.route("/")
def index():
    """Serves the main HTML interface."""
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query_request():
    """Handles POST requests to the /query endpoint."""
    if initialization_error:
        logger.error("Cannot handle query due to initialization error.")
        return jsonify({"error": f"Server initialization failed: {initialization_error}"}), 500
        
    if not query_engine_instance:
         logger.error("Query engine not initialized.")
         return jsonify({"error": "Query engine is not available."}), 500

    try:
        data = request.get_json()
        if not data or "query" not in data:
            logger.warning("Received invalid query request: No query field.")
            return jsonify({"error": "Missing \'query\' field in request body"}), 400
        
        user_query = data["query"]
        logger.info(f"Received query via API: {user_query}")
        
        # Handle the query using the engine
        response_text = query_engine_instance.handle_query(user_query)
        
        logger.info("Sending response back to client.")
        return jsonify({"response": response_text})

    except Exception as e:
        logger.error(f"Error handling /query request: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- Run Flask App ---
if __name__ == "__main__":
    print("Starting Flask server for VPA LLM Query Interface...")
    print("Ensure OPENAI_API_KEY is set in your environment or a .env file.")
    # Runs on http://127.0.0.1:5000 by default
    # Use host=\'0.0.0.0\' to make it accessible externally if needed (e.g., via deploy_expose_port)
    app.run(debug=True, host="0.0.0.0", port=5000) 


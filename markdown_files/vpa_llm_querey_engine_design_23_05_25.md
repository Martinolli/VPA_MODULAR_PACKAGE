# VPA LLM Query Engine Design

## Overview

The `vpa_llm_query_engine.py` module will serve as the central user interface for the VPA Modular Package, orchestrating interactions between all components and providing a natural language interface for users. This design document outlines how the query engine will process user inputs, coordinate with other modules, and deliver results.

## Core Architecture

```bash
                  +----------------+
                  |                |
                  |     User       |
                  |                |
                  +-------+--------+
                          |
                          | (Natural Language Queries)
                          v
+----------------+    +---+----------+    +----------------+
|                |    |              |    |                |
| memory_manager +<-->+ vpa_llm_     +<-->+ vpa_llm_       |
|                |    | query_engine |    | interface      |
+----------------+    |              |    |                |
                      +---+----------+    +-------+--------+
                          |                       |
                          | (Function Calls)      | (Analysis Requests)
                          v                       v
+----------------+    +---+----------+    +-------+--------+
|                |    |              |    |                |
| RAG Components +<-->+ OpenAI       +<-->+ vpa_facade     |
|                |    | Function     |    |                |
+----------------+    | Calling      |    +-------+--------+
                      +---+----------+            |
                          |                       |
                          | (Results)             | (Visualization Requests)
                          v                       v
+----------------+    +---+----------+    +-------+--------+
|                |    |              |    |                |
| Training Data  +<-->+ Response     +<-->+ vpa_visualizer |
| Generator      |    | Formatter    |    |                |
+----------------+    +--------------+    +----------------+
```

## Query Processing Flow

1. **Query Reception**:
   - User submits a natural language query
   - Query is logged and added to conversation history via `memory_manager`

2. **Query Understanding**:
   - Query is sent to OpenAI with available function definitions
   - OpenAI determines appropriate function to call based on query intent
   - Function parameters are extracted from the query

3. **Function Execution**:
   - Appropriate function is executed based on OpenAI's determination:
     - `get_vpa_analysis`: Performs VPA analysis on a ticker
     - `explain_vpa_concept`: Provides explanation of VPA concepts
     - `suggest_trading_parameters`: Suggests parameters for trading styles
     - `search_vpa_documents`: Retrieves information from VPA literature

4. **Result Processing**:
   - Results from function execution are formatted for LLM consumption
   - LLM generates a natural language response based on results
   - Response is logged and added to conversation history

5. **Response Delivery**:
   - Formatted response is returned to the user
   - Any visualizations or attachments are included

## Enhanced Function Definitions

```python
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
                    "include_secondary_timeframes": {"type": "boolean", "description": "Include context from secondary timeframes?", "default": False},
                    "generate_visualizations": {"type": "boolean", "description": "Generate charts and visualizations?", "default": False},
                    "save_to_training_data": {"type": "boolean", "description": "Save analysis as training example?", "default": False}
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
                    "concept_name": {"type": "string", "description": "The name of the VPA concept (e.g., \"Selling Climax\")"},
                    "include_examples": {"type": "boolean", "description": "Include practical examples?", "default": True},
                    "include_visualizations": {"type": "boolean", "description": "Include concept visualizations?", "default": False}
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
                    "trading_style": {"type": "string", "description": "The trading style (e.g., 'day_trading', 'swing_trading')"},
                    "risk_tolerance": {"type": "string", "description": "Risk tolerance level (low, medium, high)", "default": "medium"}
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
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_vpa_training_data",
            "description": "Generates training data for a ticker over a specified period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker symbol"},
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                    "primary_timeframe": {"type": "string", "description": "Primary timeframe", "default": "1d"}
                },
                "required": ["ticker", "start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_vpa_visualization",
            "description": "Creates visualizations for VPA analysis results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker symbol"},
                    "timeframe": {"type": "string", "description": "Timeframe to visualize", "default": "1d"},
                    "chart_type": {"type": "string", "description": "Type of chart to create", "enum": ["price_volume", "pattern_analysis", "support_resistance", "dashboard", "signal_summary"]},
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format (optional)"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format (optional)"}
                },
                "required": ["ticker", "chart_type"]
            }
        }
    }
]
```

## Function Implementation Details

### 1. get_vpa_analysis

```python
def _execute_get_vpa_analysis(self, ticker, timeframe="1d", include_secondary_timeframes=False, 
                             generate_visualizations=False, save_to_training_data=False):
    """Executes VPA analysis using the VPAFacade and VPALLMInterface."""
    self.logger.info(f"Executing VPA analysis for {ticker} ({timeframe})...")
    
    try:
        # Define timeframes based on parameters
        primary_tf_dict = {"interval": timeframe, "period": "1y"}
        analysis_timeframes = [primary_tf_dict]
        
        if include_secondary_timeframes:
            try:
                default_timeframes = self.vpa_facade.config.get_timeframes()
                for tf_dict in default_timeframes:
                    if tf_dict.get("interval") != timeframe:
                        analysis_timeframes.append(tf_dict)
            except Exception as e:
                self.logger.warning(f"Could not get secondary timeframes: {e}")
        
        # Perform analysis
        analysis_results = self.vpa_facade.analyze_ticker(ticker=ticker, timeframes=analysis_timeframes)
        
        # Generate visualizations if requested
        visualization_paths = []
        if generate_visualizations and hasattr(self, 'visualizer'):
            try:
                # Create result extractor for visualization
                extractor = VPAResultExtractor({ticker: analysis_results})
                
                # Generate visualizations
                output_dir = f"visualizations/{ticker}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Price and volume chart
                price_path = f"{output_dir}/{ticker}_{timeframe}_price_volume.png"
                price_data = extractor.get_price_data(ticker, timeframe)
                self.visualizer.plot_price_volume_chart(price_data, ticker, timeframe, price_path)
                visualization_paths.append(price_path)
                
                # Pattern analysis
                pattern_path = f"{output_dir}/{ticker}_{timeframe}_patterns.png"
                pattern_analysis = extractor.get_pattern_analysis(ticker, timeframe)
                self.visualizer.plot_pattern_analysis(price_data, pattern_analysis, ticker, timeframe, output_dir)
                visualization_paths.append(pattern_path)
                
                # Support/resistance
                sr_path = f"{output_dir}/{ticker}_{timeframe}_support_resistance.png"
                sr_data = extractor.get_support_resistance(ticker, timeframe)
                self.visualizer.plot_support_resistance(price_data, sr_data, ticker, timeframe, sr_path)
                visualization_paths.append(sr_path)
            except Exception as e:
                self.logger.error(f"Error generating visualizations: {e}")
                visualization_paths = []
        
        # Save to training data if requested
        if save_to_training_data and hasattr(self, 'training_data_generator'):
            try:
                # Get current date for end_date
                end_date = datetime.now().strftime("%Y-%m-%d")
                # Calculate start_date as 1 year ago
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                
                # Generate training data
                self.training_data_generator.generate_training_data(
                    ticker, 
                    start_date, 
                    end_date, 
                    primary_timeframe=timeframe
                )
                self.logger.info(f"Generated training data for {ticker}")
            except Exception as e:
                self.logger.error(f"Error generating training data: {e}")
        
        # Format results for LLM
        llm_analysis = self.llm_interface.get_ticker_analysis(ticker)
        
        # Combine all results
        result = {
            **llm_analysis,
            "visualization_paths": visualization_paths,
            "training_data_saved": save_to_training_data
        }
        
        return json.dumps(result, default=str)
    
    except Exception as e:
        self.logger.error(f"Error executing VPA analysis for {ticker}: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to perform VPA analysis for {ticker}: {str(e)}"})
```

### 2. search_vpa_documents (Enhanced RAG)

```python
def _execute_search_vpa_documents(self, query: str, top_k: int = 5) -> str:
    """Executes a search for VPA concepts in the embedded Anna Coulling book with enhanced RAG."""
    
    # Step 1: Retrieve top matching chunks with metadata
    top_chunks = retrieve_top_chunks(query, top_k)
    
    if not top_chunks:
        return json.dumps({"answer": "No relevant information found in the VPA document."})
    
    # Step 2: Build context from metadata-enhanced chunks
    context_text = "\n\n".join(
        f"Source: {chunk['source']}\n"
        f"Page: {chunk.get('metadata', {}).get('page', '?')} - "
        f"Section: {chunk.get('metadata', {}).get('section', '?')}\n"
        f"{chunk['text']}"
        for chunk in top_chunks
    )
    
    # Step 3: Create a system message for OpenAI
    system_message = {
        "role": "system",
        "content": (
            "You are an expert assistant specializing in Volume Price Analysis (VPA) based on Anna Coulling's book, "
            "the Wyckoff Method, and related trading concepts. Provide comprehensive, accurate, and well-structured answers. "
            "Use examples where appropriate and always relate your answer back to VPA principles and the Wyckoff Method when relevant. "
            "Cite specific pages and sections when referencing the source material."
        )
    }
    
    # Step 4: Create a user prompt with improved instructions
    user_prompt = (
        f"Question: {query}\n\n"
        "Based on the following extracted passages from 'Volume Price Analysis' by Anna Coulling, "
        "please provide a comprehensive answer. Include the following elements:\n"
        "1. A clear and concise explanation of the concept or topic.\n"
        "2. How it relates to broader VPA principles and the Wyckoff Method.\n"
        "3. Practical application or example in trading scenarios.\n"
        "4. Any related VPA concepts that might be relevant.\n"
        "5. Specific citations to the source material (page numbers and sections).\n\n"
        f"Extracted Passages:\n{context_text}\n\n"
        "Comprehensive Answer:"
    )
    
    try:
        # Get conversation history and add system and user messages
        messages = self.memory.get_history()
        messages.append({"role": "system", "content": system_message["content"]})
        messages.append({"role": "user", "content": user_prompt})
        
        # Generate response
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            max_tokens=7000,
            temperature=0.5,
            presence_penalty=0.3,
            frequency_penalty=0.3,
        )
        
        # Extract answer
        answer_text = response.choices[0].message.content.strip()
        
        # Save to conversation history
        self.memory.save_message("user", query)  # Save original query, not the prompt
        self.memory.save_message("assistant", answer_text)
        
        # Package result with source information
        result = {
            "answer": answer_text,
            "source_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk.get("source"),
                    "page": chunk.get("metadata", {}).get("page"),
                    "section": chunk.get("metadata", {}).get("section"),
                    "text": chunk["text"][:300] + "..."  # Truncate for readability
                }
                for chunk in top_chunks
            ]
        }
        
        return json.dumps(result)
    
    except Exception as e:
        self.logger.error(f"Error generating answer from VPA book chunks: {e}")
        return json.dumps({"error": f"Error generating response: {str(e)}"})
```

## Class Structure Enhancements

```python
class VPAQueryEngine:
    """Handles natural language queries about VPA using OpenAI and VPAFacade."""

    def __init__(self, vpa_facade=None, openai_model="gpt-4o", log_level="INFO", log_file=None):
        """Initializes the engine with VPAFacade and OpenAI client."""
        
        # Initialize logger
        if log_file is None:
            log_file = "logs/vpa_query_engine.log"
        self.logger = VPALogger(log_level, log_file)
        
        # Initialize memory manager
        self.memory = MemoryManager()
        
        # Add system message
        self.memory.add_system_message("""You are an expert in Volume Price Analysis (VPA) based on the Wyckoff method and Anna Coulling's teachings.
        Your goal is to analyze stock market data using VPA principles, explain VPA concepts, and answer user questions accurately.""")
        
        # Initialize VPA components
        if vpa_facade is None:
            try:
                self.vpa_facade = VPAFacade()
            except Exception as e:
                self.logger.error(f"Failed to initialize VPAFacade: {e}")
                self.vpa_facade = None
        else:
            self.vpa_facade = vpa_facade
        
        # Initialize LLM interface
        self.llm_interface = VPALLMInterface()
        
        # Initialize OpenAI client
        try:
            self.openai_client = OpenAI()
            self.openai_model = openai_model
            self.openai_client.models.list()  # Test connection
            self.logger.info(f"OpenAI client initialized successfully for model {self.openai_model}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Initialize optional components
        try:
            from vpa_modular.vpa_visualizer import plot_price_volume_chart, plot_pattern_analysis, plot_support_resistance
            self.visualizer = type('Visualizer', (), {
                'plot_price_volume_chart': staticmethod(plot_price_volume_chart),
                'plot_pattern_analysis': staticmethod(plot_pattern_analysis),
                'plot_support_resistance': staticmethod(plot_support_resistance)
            })
            self.logger.info("Visualizer component initialized.")
        except ImportError:
            self.logger.warning("Visualizer component not available.")
            self.visualizer = None
        
        try:
            from vpa_modular.vpa_training_data_generator import VPATrainingDataGenerator
            self.training_data_generator = VPATrainingDataGenerator(self.vpa_facade)
            self.logger.info("Training data generator initialized.")
        except ImportError:
            self.logger.warning("Training data generator not available.")
            self.training_data_generator = None
    
    def process_query(self, query):
        """
        Process a natural language query about VPA.
        
        Parameters:
        - query: Natural language query string
        
        Returns:
        - Response to the query
        """
        self.logger.info(f"Processing query: {query}")
        
        try:
            # Save user query to memory
            self.memory.save_message("user", query)
            
            # Get conversation history
            messages = self.memory.get_history()
            
            # Call OpenAI with function definitions
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=7000,
                temperature=0.7
            )
            
            # Extract response
            response_message = response.choices[0].message
            
            # Check if function call was requested
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                # Execute function calls
                function_responses = []
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    self.logger.info(f"Executing function: {function_name} with args: {function_args}")
                    
                    # Execute the function
                    function_response = self._execute_function(function_name, **function_args)
                    function_responses.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })
                
                # Add function responses to messages
                messages.append(response_message)
                messages.extend(function_responses)
                
                # Get final response from OpenAI
                second_response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=7000,
                    temperature=0.7
                )
                
                # Extract final response
                final_response = second_response.choices[0].message.content
                
                # Save assistant response to memory
                self.memory.save_message("assistant", final_response)
                
                return final_response
            else:
                # No function call, just return the response
                response_content = response_message.content
                
                # Save assistant response to memory
                self.memory.save_message("assistant", response_content)
                
                return response_content
        
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            return f"I encountered an error while processing your query: {str(e)}"
```

## Integration with External Systems

### Web Interface Integration

The VPAQueryEngine can be exposed via a web interface using Flask or FastAPI:

```python
from flask import Flask, request, jsonify
from vpa_modular.vpa_llm_query_engine import VPAQueryEngine

app = Flask(__name__)
query_engine = VPAQueryEngine()

@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    response = query_engine.process_query(query)
    return jsonify({"response": response})

@app.route('/api/visualizations/<path:filename>')
def serve_visualization(filename):
    return send_from_directory('visualizations', filename)

if __name__ == '__main__':
    app.run(debug=True)
```

### Command Line Interface

A CLI can also be provided for direct interaction:

```python
import argparse
from vpa_modular.vpa_llm_query_engine import VPAQueryEngine

def main():
    parser = argparse.ArgumentParser(description='VPA Query Engine CLI')
    parser.add_argument('query', nargs='?', help='Query to process')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    query_engine = VPAQueryEngine()
    
    if args.interactive:
        print("VPA Query Engine Interactive Mode (type 'exit' to quit)")
        while True:
            query = input("> ")
            if query.lower() == 'exit':
                break
            response = query_engine.process_query(query)
            print(response)
    elif args.query:
        response = query_engine.process_query(args.query)
        print(response)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
```

## Future Enhancements

1. **Streaming Responses**:
   - Implement streaming for long responses
   - Show intermediate results during analysis

2. **Multi-Modal Support**:
   - Accept image inputs (e.g., charts for analysis)
   - Return rich outputs with embedded visualizations

3. **User Preferences**:
   - Store and apply user preferences for analysis
   - Customize responses based on user expertise level

4. **Batch Processing**:
   - Support batch analysis of multiple tickers
   - Generate comparative reports

5. **Real-Time Updates**:
   - Integrate with real-time data sources
   - Provide alerts based on VPA signals

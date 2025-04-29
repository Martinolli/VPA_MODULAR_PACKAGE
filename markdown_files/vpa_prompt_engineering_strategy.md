# VPA LLM Query Engine: Prompt Engineering Strategy (OpenAI)

## 1. Introduction

This document outlines the prompt engineering strategies for the `VPA LLM Query Engine` when interacting with the OpenAI API, particularly using the Chat Completions endpoint with Function Calling. Effective prompts are crucial for guiding the LLM to correctly understand user intent, utilize the provided VPA functions, and generate accurate, relevant, and well-formatted responses.

## 2. Core Prompt Components

The interaction with the OpenAI Chat Completions API involves a sequence of messages. Key components include:

*   **System Message:** Sets the overall context and persona for the LLM throughout the conversation.
*   **User Message:** Contains the user's raw natural language query.
*   **Assistant Message (with Tool Calls):** The LLM's response indicating that it needs to call one of the defined VPA functions.
*   **Tool Message:** A message sent by the engine *back* to the LLM, containing the result of the executed function call.
*   **Assistant Message (Final Response):** The LLM's final natural language answer to the user, generated after receiving the tool message result.

## 3. Prompting Strategies

**3.1. System Message:**
*   **Goal:** Establish the LLM's role as a VPA expert and define its primary task.
*   **Strategy:** Use a clear and concise system message.
*   **Example:**
    ```
    You are an expert in Volume Price Analysis (VPA) based on the Wyckoff method and Anna Coulling's teachings. Your goal is to analyze stock market data using VPA principles, explain VPA concepts, and answer user questions accurately. When analysis is requested, use the provided functions to get VPA data and then interpret the results clearly, referencing specific VPA patterns, signals (like Selling Climax, No Demand, Shakeout, Tests, etc.), volume/spread relationships, and trend context. Provide concise yet informative explanations.
    ```

**3.2. Initial User Query:**
*   **Goal:** Pass the user's request directly to the LLM.
*   **Strategy:** Send the raw user query as the content of the user message.
*   **Example:**
    ```json
    { "role": "user", "content": "Analyze MSFT daily chart with VPA." }
    ```

**3.3. Function Definitions (Passed with Initial Query):**
*   **Goal:** Clearly define the available tools (functions) for the LLM, including descriptions and parameters.
*   **Strategy:** Use the `tools` parameter in the OpenAI API call, providing a JSON schema for each function (`get_vpa_analysis`, `explain_vpa_concept`) as defined in `vpa_query_parsing_approach.md`.
*   **Example Snippet (for `tools` parameter):**
    ```json
    [
      {
        "type": "function",
        "function": {
          "name": "get_vpa_analysis",
          "description": "Performs Volume Price Analysis for a given stock ticker and timeframe, returning key findings like signals, patterns, and trend information.",
          "parameters": {
            "type": "object",
            "properties": {
              "ticker": { "type": "string", "description": "The stock ticker symbol (e.g., \"AAPL\")" },
              "timeframe": { "type": "string", "description": "The primary timeframe (e.g., \"1d\")", "default": "1d" },
              "include_secondary_timeframes": { "type": "boolean", "description": "Include context from secondary timeframes?", "default": false }
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
              "concept_name": { "type": "string", "description": "The name of the VPA concept (e.g., \"Selling Climax\")" }
            },
            "required": ["concept_name"]
          }
        }
      }
    ]
    ```

**3.4. Tool Message (Function Result):**
*   **Goal:** Provide the output of the executed function back to the LLM in a structured way.
*   **Strategy:** Send a message with `role: "tool"`. The `content` should contain the data returned by the function (e.g., the VPA analysis results as a JSON string or structured text). The `tool_call_id` must match the ID provided by the LLM in its request.
*   **Example (Result of `get_vpa_analysis`):**
    ```json
    {
      "role": "tool",
      "tool_call_id": "call_abc123...", // ID from the assistant's tool_calls
      "name": "get_vpa_analysis",
      "content": "{\"ticker\": \"MSFT\", \"timeframe\": \"1d\", \"analysis_timestamp\": \"...\", \"signal\": {\"type\": \"BUY\", \"strength\": \"MODERATE\"}, \"trend\": {\"status\": \"BULLISH\", \"strength\": \"STRONG\"}, \"patterns\": [\"Test of Support (Successful)\"], \"key_metrics\": {\"volume_class\": \"LOW\", \"candle_class\": \"NARROW\"}, \"support\": [295.50], \"resistance\": [310.00], \"summary\": \"Low volume narrow spread bar successfully tested support, indicating potential continuation of bullish trend.\"}" // Content is the JSON string result from VPAFacade
    }
    ```

**3.5. Prompting for Final Response (After Function Call):**
*   **Goal:** Instruct the LLM to synthesize the function result into a user-friendly natural language response.
*   **Strategy:** The conversation history sent in the second API call (including the system message, user query, assistant's tool call request, and the tool message with the result) implicitly prompts the LLM to use the provided data. The system message already guides the desired interpretation style.
*   **Refinement:** If needed, a specific instruction could be added to the *end* of the tool message content, e.g., `"... Use this data to provide a VPA analysis for the user."`

## 4. Handling Ambiguity and Edge Cases

*   **Unclear Queries:** If the LLM cannot determine the intent or required parameters, it might ask the user for clarification instead of calling a function. This is acceptable default behavior.
*   **Queries Outside Scope:** If the user asks something unrelated to VPA or the defined functions, the LLM should ideally state that it cannot fulfill the request, guided by the system message defining its role.
*   **Function Execution Errors:** If the `VPAFacade` fails, the tool message content should clearly indicate the error (e.g., `"content": "{\"error\": \"Failed to retrieve data for ticker XYZ\"}"`). The LLM should then relay this error appropriately to the user.

## 5. Iteration and Refinement

Prompt engineering is an iterative process. The effectiveness of these strategies will need to be evaluated during testing (Step 007). Adjustments to the system message, function descriptions, and the structure of data passed in the tool message may be necessary based on the quality of the LLM's responses.

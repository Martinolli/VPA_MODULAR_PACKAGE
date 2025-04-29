# VPA LLM Query Engine: User Query Parsing Approach

## 1. Introduction

This document defines the approach for parsing and understanding user natural language queries within the `VPA LLM Query Engine`. The goal is to reliably extract the user's intent and any necessary parameters (like ticker symbols, timeframes, or concept names) to fulfill the request using the VPA system and the OpenAI LLM.

## 2. Chosen Approach: OpenAI Function Calling

Instead of implementing separate parsing logic (like keyword matching, regex, or NLU libraries), we will leverage **OpenAI's Function Calling** capability. This approach integrates the query understanding directly into the interaction with the LLM.

**Rationale:**
*   **Leverages LLM Strength:** Utilizes the LLM's inherent natural language understanding capabilities to interpret user intent and extract entities.
*   **Robustness:** More resilient to variations in user phrasing compared to simple keyword or rule-based methods.
*   **Efficiency:** Combines query understanding and action determination within a single API interaction flow (potentially requiring a follow-up call after function execution).
*   **Integration:** Aligns well with our chosen provider (OpenAI) and uses a feature designed for this type of interaction.

## 3. Defined Functions for LLM

The `VPAQueryEngine` will expose the following functions to the OpenAI API. The LLM will be instructed to call these functions when appropriate based on the user's query.

**Function 1: `get_vpa_analysis`**
*   **Description:** Performs Volume Price Analysis for a given stock ticker and timeframe, returning key findings like signals, patterns, and trend information.
*   **Parameters:**
    *   `ticker`: (string, required) The stock ticker symbol (e.g., "AAPL", "MSFT").
    *   `timeframe`: (string, optional, default: "1d") The primary timeframe for analysis (e.g., "1d", "1h", "4h").
    *   `include_secondary_timeframes`: (boolean, optional, default: false) Whether to include analysis from pre-configured secondary timeframes for context.
*   **Engine Action:** Calls the relevant methods in `VPAFacade` (e.g., `analyze_ticker` or a future point-in-time equivalent) to get the analysis results.

**Function 2: `explain_vpa_concept`**
*   **Description:** Provides an explanation of a specific Volume Price Analysis concept, pattern, or term.
*   **Parameters:**
    *   `concept_name`: (string, required) The name of the VPA concept to explain (e.g., "Selling Climax", "No Demand Bar", "Accumulation Phase").
*   **Engine Action:** Can retrieve a pre-defined explanation from a knowledge base within the VPA system or simply pass the concept name back to the LLM in a subsequent prompt, asking it to provide the explanation based on its training.

*(Additional functions, like comparing tickers or finding stocks matching criteria, could be added later.)*

## 4. Workflow with Function Calling

1.  **User Query:** User provides a query (e.g., "What's the VPA signal for NVDA on the daily chart?", "Tell me about the Wyckoff accumulation schematic.").
2.  **Engine Call to OpenAI:** The `VPAQueryEngine` sends the user query to the OpenAI Chat Completions API, along with the definitions of the available functions (`get_vpa_analysis`, `explain_vpa_concept`).
3.  **LLM Response (Function Call):** If the LLM determines a function should be called, the API response will contain a `tool_calls` object specifying the function name and arguments (e.g., `{"name": "get_vpa_analysis", "arguments": "{\"ticker\": \"NVDA\", \"timeframe\": \"1d\"}"}`).
4.  **Engine Executes Function:** The `VPAQueryEngine` parses the function name and arguments from the API response.
    *   It calls the corresponding internal method (which interacts with `VPAFacade` or retrieves concept info).
    *   It obtains the results (e.g., VPA analysis data for NVDA, or confirmation that the concept name is valid).
5.  **Engine Call to OpenAI (with Function Result):** The engine makes a *second* call to the OpenAI API. This call includes:
    *   The original user query.
    *   The initial LLM response requesting the function call.
    *   A new message containing the *result* of the function execution (e.g., the VPA analysis data).
6.  **LLM Response (Final Answer):** The LLM receives the function result and generates a final, natural language response for the user, incorporating the data provided (e.g., "The VPA analysis for NVDA (1d) shows a potential No Supply signal...").
7.  **Return to User:** The engine returns this final text response to the user.

**Note:** If the initial user query doesn't require external data (e.g., "What is VPA?"), the LLM might respond directly without requesting a function call.

## 5. Implementation Details

*   The `openai` Python library provides direct support for handling `tool_calls` and sending function results back to the API.
*   The `VPAQueryEngine` will need internal methods corresponding to the defined functions, responsible for interacting with the `VPAFacade` and formatting the results appropriately for the LLM.
*   Robust error handling is needed for parsing function arguments and executing the internal methods.

This function calling approach provides a structured and powerful way to integrate the LLM's NLU capabilities with the VPA system's analytical functions.

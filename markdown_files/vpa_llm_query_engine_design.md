# VPA LLM Query Engine Design (OpenAI Integration)

## 1. Introduction

This document outlines the design for the `vpa_llm_query_engine.py` module. The purpose of this module is to provide an interactive interface for Volume Price Analysis (VPA) by leveraging the capabilities of an external Large Language Model (LLM) accessed via the OpenAI API. Users will be able to ask natural language questions about VPA concepts or request analysis for specific stock tickers, and the engine will use the existing VPA system and OpenAI to generate informative responses.

## 2. Core Components

*   **`VPAQueryEngine` Class:** The main class encapsulating the query processing logic.
*   **`VPAFacade` Integration:** Utilizes the existing `vpa_modular.vpa_facade.VPAFacade` to access VPA data fetching, processing, analysis, and signal generation capabilities.
*   **OpenAI API Client:** Uses the official `openai` Python library to interact with the OpenAI API (specifically the Chat Completions endpoint).
*   **User Query Input:** Accepts natural language queries as strings.
*   **LLM Response Output:** Returns the generated analysis or explanation from the LLM as a string.

## 3. Workflow

The core workflow for handling a user query will be:

1.  **Receive Query:** The `VPAQueryEngine` receives a natural language query string from the user (e.g., "Analyze AAPL using VPA for the daily timeframe", "Explain what a selling climax is").
2.  **Parse Query (Initial Simple Approach):** Perform basic parsing to identify the primary intent (e.g., `analyze_ticker`, `explain_concept`) and key entities (e.g., ticker symbol `AAPL`, timeframe `1d`, concept `selling climax`). *A more sophisticated parsing approach will be defined in Step 003.*
3.  **Fetch VPA Data (if applicable):**
    *   If the intent is `analyze_ticker`, use `VPAFacade` to fetch historical data, run the VPA analysis (processor, analyzer, signals) for the specified ticker and timeframe(s).
    *   If the intent is `explain_concept`, potentially retrieve pre-defined explanations or relevant context from the VPA system (or rely solely on the LLM's knowledge, guided by the prompt).
4.  **Construct Prompt:** Create a detailed and structured prompt for the OpenAI API. This prompt will include:
    *   A clear instruction defining the LLM's role (e.g., "You are a VPA expert analyzing stock data.").
    *   The user's original query for context.
    *   The relevant VPA data retrieved from the `VPAFacade` (e.g., signals, patterns, trend info, key metrics, support/resistance levels), formatted clearly (potentially as JSON or structured text within the prompt).
    *   Specific instructions on the desired output format (e.g., "Provide a concise analysis including the signal, key patterns, and reasoning."). *Prompt engineering strategies will be refined in Step 004.*
5.  **Call OpenAI API:**
    *   Initialize the OpenAI client (using an API key sourced securely, e.g., from an environment variable).
    *   Make a call to the Chat Completions endpoint (e.g., `client.chat.completions.create`) using the constructed prompt and specifying the desired model (e.g., "gpt-4o").
6.  **Handle API Response/Errors:**
    *   Check for errors returned by the OpenAI API.
    *   Extract the generated text content from the successful response.
    *   Implement basic error handling for API connection issues, authentication failures, etc.
7.  **Return Response:** Return the LLM's generated text response to the caller.

## 4. Configuration

*   **OpenAI API Key:** The engine will expect the OpenAI API key to be available as an environment variable named `OPENAI_API_KEY`. The code will *not* store or manage the key directly.
*   **OpenAI Model:** Configurable parameter, defaulting to a recommended model like "gpt-4o" or "gpt-4".
*   **API Parameters:** Allow configuration of parameters like `temperature` and `max_tokens` for the OpenAI API call to control creativity and response length.

## 5. Error Handling

The engine should gracefully handle potential errors, including:
*   Failures in fetching or analyzing data via `VPAFacade`.
*   Errors during query parsing.
*   OpenAI API errors (authentication, rate limits, connection issues, invalid requests).
*   Timeouts during API calls.

Appropriate logging and potentially user-friendly error messages should be implemented.

## 6. Dependencies

*   `openai` (Python library for OpenAI API access)
*   Existing VPA modules (`vpa_modular.vpa_facade`, etc.)
*   Standard Python libraries (`os` for environment variables, `logging`).

## 7. Future Enhancements

*   **Advanced Query Parsing:** Implement more robust Natural Language Understanding (NLU) to handle complex queries, ambiguity, and follow-up questions.
*   **Conversation History:** Maintain context across multiple user turns for more natural conversations.
*   **Streaming Responses:** Implement streaming for long-running analyses to provide faster initial feedback to the user.
*   **Function Calling (OpenAI):** Explore using OpenAI's function calling feature to allow the LLM to more directly request specific VPA data or actions from the `VPAFacade`.
*   **Multi-Provider Support:** Abstract the LLM interaction layer to potentially support other providers besides OpenAI in the future.


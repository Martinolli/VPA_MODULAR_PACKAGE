# VPA LLM Training Data Generator Design

This document outlines the design for a module (`vpa_training_data_generator.py`) dedicated to generating training data suitable for fine-tuning a Large Language Model (LLM) to analyze Volume Price Analysis (VPA) results.

## 1. Goal

The primary goal is to create input-output pairs where the input represents the state of VPA indicators and market data at a specific point in time, and the output is the corresponding VPA signal and a comprehensive textual explanation of the reasoning behind it.

## 2. Module Design

* **File:** `vpa_modular/vpa_training_data_generator.py`
* **Class:** `VPATrainingDataGenerator`
* **Dependencies:** `VPAFacade`, `VPADataFetcher` (or integration layer for local data), `pandas`, `json`, `datetime`.

## 3. Core Functionality

The generator will perform the following steps:

1.**Initialization:** Takes configuration (e.g., VPA parameters, output file path).
2.  **Data Loading:** Accepts ticker, start date, end date, and timeframes. Uses `VPADataFetcher` to retrieve historical data for the specified period and timeframes.
3.  **Iteration:** Iterates through the historical data, typically day-by-day or based on the primary analysis timeframe (e.g., '1d'). A suitable lookback period (e.g., 50-100 candles) is required before generation can start.
4.  **Point-in-Time Analysis:** For each timestamp `t` in the iteration:
    *Selects historical data up to `t`.
    *   Uses `VPAFacade.analyze_ticker_at_point` (a potential new method in the facade, or directly uses underlying modules) to perform a complete VPA analysis based *only* on data available up to `t`.
    *   This analysis includes calculating metrics, identifying trends, recognizing patterns, and generating a signal.
5.  **Feature Extraction (Input):** Extracts key VPA metrics and context from the analysis results to form the LLM input.
6.  **Target Generation (Output):** Extracts the final signal and generates a comprehensive textual explanation based on the analysis results.
7.  **Formatting:** Formats the input features and output analysis into a JSON object.
8.  **Output:** Appends the JSON object as a new line to the specified output JSON Lines (JSONL) file.

## 4. Data Structure (JSONL Format)

Each line in the output file will be a JSON object with the following structure:

```json
{
  "input": {
    "timestamp": "YYYY-MM-DDTHH:MM:SS",
    "ticker": "string",
    "primary_timeframe": "string (e.g., 1d)",
    "current_candle": {
      "open": float,
      "high": float,
      "low": float,
      "close": float,
      "volume": int
    },
    "recent_candles": [
      { "timestamp": "...", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ... },
      ...
    ], // List of last N candles (e.g., N=10)
    "vpa_metrics": {
      "relative_volume": float, // Volume / Avg Volume
      "volume_class": "string (e.g., VERY_HIGH, HIGH, AVERAGE, LOW, VERY_LOW)",
      "candle_spread_ratio": float, // Spread / Avg Spread
      "candle_class": "string (e.g., WIDE, NARROW, AVERAGE)",
      "upper_wick_ratio": float, // Upper Wick / Body
      "lower_wick_ratio": float // Lower Wick / Body
    },
    "trend_analysis": {
      "status": "string (e.g., BULLISH, BEARISH, SIDEWAYS)",
      "strength": "string (e.g., STRONG, WEAK)",
      "validation_signal": "string (e.g., VALIDATION, ANOMALY, NONE)"
    },
    "pattern_analysis": {
      "accumulation": { "detected": boolean, "details": "string" },
      "distribution": { "detected": boolean, "details": "string" },
      "buying_climax": { "detected": boolean, "details": "string" },
      "selling_climax": { "detected": boolean, "details": "string" },
      "testing_support": { "detected": boolean, "details": "string" },
      "testing_resistance": { "detected": boolean, "details": "string" }
    },
    "support_resistance": {
      "nearby_support": [float, ...],
      "nearby_resistance": [float, ...]
    },
    "multi_timeframe_context": {
      "secondary_tf_1": { "trend_status": "...", "signal_type": "..." },
      "secondary_tf_2": { "trend_status": "...", "signal_type": "..." }
    } // Optional summary from other timeframes
  },
  "output": {
    "signal": {
      "type": "string (e.g., BUY, SELL, HOLD, NO_ACTION)",
      "strength": "string (e.g., STRONG, MODERATE, WEAK)"
    },
    "explanation": "string (Comprehensive textual explanation - see section 5)"
  }
}
```

## 5. Explanation Generation

The `explanation` field in the output is crucial. It should be a detailed, human-readable text summarizing the VPA analysis and justifying the signal. It needs to dynamically incorporate data from the `input` section.

* **Structure:** Start with the overall signal, then elaborate on the contributing factors.
* **Content:**
  * Mention the ticker, timestamp, and primary timeframe.
  * State the signal type and strength.
  * Describe the current trend context (status, strength, validation).
  * Reference the current candle's characteristics (spread, volume class, wicks).
  * Mention any detected VPA patterns (accumulation, distribution, climax, tests) and their significance.
  * Refer to specific volume conditions (e.g., "high volume confirmation", "low volume test").
  * Incorporate nearby support/resistance levels.
  * Optionally include context from secondary timeframes.
  * Conclude with a summary reinforcing the signal.
* **Templating:** A templating engine or f-strings can be used to construct the explanation dynamically based on the analysis results.
  * *Example Snippet (BUY signal):* "...A BUY signal ([strength]) is suggested for [ticker] on [timestamp] ([primary_timeframe]). The market is currently in a [trend_status] trend. A key factor is the recent test of support at [support_level] on low volume ([relative_volume]), indicating potential absorption. Additionally, the current candle is a [candle_class] spread candle closing near its high with [volume_class] volume, confirming buying interest..."

## 6. Implementation Notes

* Need to handle potential errors during data fetching or analysis gracefully.
* Consider adding configuration options for the lookback period, number of recent candles (N), and output file path.
* The `VPAFacade` might need a new method like `analyze_ticker_at_point` that takes historical data up to a point `t` and returns the full structured analysis used for feature extraction.
* Generating high-quality explanations will require careful logic and potentially iterative refinement.

## 7. Next Steps

1. Confirm this design with the user.
2. Proceed to implement the `VPATrainingDataGenerator` class and associated functions (Step 004).

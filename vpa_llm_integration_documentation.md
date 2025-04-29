# VPA LLM Integration: Training Data Generation Documentation

## 1. Introduction

This document provides a comprehensive overview of the approach and implementation for generating training data suitable for fine-tuning Large Language Models (LLMs) to perform Volume Price Analysis (VPA).

The primary goal is to create a dataset of input-output pairs where:

* **Input:** Represents the VPA context (indicators, patterns, trend, market data) at a specific point in time.

* **Output:** Consists of the corresponding VPA signal (Buy/Sell/Hold) and a detailed, human-readable explanation justifying the signal based on VPA principles and the input context.

This training data can then be used to fine-tune LLMs to understand VPA concepts and generate insightful market analysis.

## 2. Overall Approach

The approach involves a dedicated Python module, `VPATrainingDataGenerator`, which leverages the existing VPA modular components (`VPAFacade`, `VPADataFetcher`, etc.) to:

1. Load historical market data for a specified ticker and period.
2. Iterate through the historical data (e.g., daily).
3. Perform a point-in-time VPA analysis at each step using data available up to that moment.
4. Extract relevant VPA metrics, patterns, and context to form the structured **input** features.
5. Generate the corresponding VPA signal and a comprehensive textual **output** explanation.
6. Save these input-output pairs in JSON Lines (JSONL) format, a standard for LLM training.

## 3. Training Data Generator Module (`vpa_training_data_generator.py`)

* **Location:** `vpa_modular/vpa_training_data_generator.py`
* **Class:** `VPATrainingDataGenerator`
* **Core Functionality:**
  * Initializes with a `VPAFacade` instance and an output directory.
  * Loads historical data using a data provider (e.g., `YFinanceProvider`).
  * Iterates through the primary timeframe data, ensuring a minimum lookback period.
  * **Crucially, requires a point-in-time analysis capability within `VPAFacade` (e.g., `analyze_ticker_at_point(ticker, data_up_to_t)`) to avoid lookahead bias. The current implementation uses a placeholder/mock for this step.**
  * Extracts input features using `_extract_input_features`.
  * Generates the textual explanation using `_generate_explanation`.
  * Writes the structured input/output pair to a JSONL file.

*(Refer to the source code `/home/ubuntu/VPA_MODULAR_PACKAGE/vpa_modular/vpa_training_data_generator.py` for full implementation details)*

## 4. Training Data Format (JSONL)

Each line in the output JSONL file represents one training example and follows this structure:

```json
{
  "input": {
    "timestamp": "YYYY-MM-DDTHH:MM:SS",
    "ticker": "string",
    "primary_timeframe": "string (e.g., 1d)",
    "current_candle": { /* OHLCV data */ },
    "recent_candles": [ /* List of last N candles */ ],
    "vpa_metrics": { /* volume_class, candle_class, ratios, etc. */ },
    "trend_analysis": { /* status, strength, validation */ },
    "pattern_analysis": { /* accumulation, distribution, climax, testing flags & details */ },
    "support_resistance": { /* nearby_support, nearby_resistance levels */ },
    "multi_timeframe_context": { /* Optional summary from other timeframes */ }
  },
  "output": {
    "signal": {
      "type": "string (e.g., BUY, SELL, HOLD)",
      "strength": "string (e.g., STRONG, MODERATE, WEAK)"
    },
    "explanation": "string (Comprehensive textual explanation incorporating input data)"
  }
}
```

*(Refer to the design document `/home/ubuntu/vpa_llm_training_data_design.md` for detailed field descriptions)*

## 5. Explanation Generation

The `output.explanation` field is designed to be a comprehensive, data-driven narrative. It dynamically incorporates:

* Signal type and strength.
* Trend context.
* Current candle characteristics (volume, spread, wicks).
* Detected VPA patterns.
* Support and resistance levels.
* Reasoning derived from VPA rules.

This rich textual output provides the LLM with clear examples of how to articulate VPA analysis.

## 6. Usage Example: Generating Data

The script `example_generate_training_data.py` demonstrates how to use the `VPATrainingDataGenerator`:

```python
# Example snippet from example_generate_training_data.py

from vpa_modular.vpa_facade import VPAFacade
from vpa_modular.vpa_training_data_generator import VPATrainingDataGenerator
import logging

# --- Configuration ---
TICKER = "MSFT"
START_DATE = "2023-01-01"
END_DATE = "2023-06-30"
PRIMARY_TIMEFRAME = "1d"
SECONDARY_TIMEFRAMES = ["1h"]
OUTPUT_DIRECTORY = "./llm_training_data_output"
MIN_LOOKBACK = 50

# --- Setup & Initialization ---
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\')
logger = logging.getLogger(\'GenerateTrainingDataExample\')

vpa_facade = VPAFacade() # Assumes default config
generator = VPATrainingDataGenerator(vpa_facade, output_dir=OUTPUT_DIRECTORY)

# --- Run Generation ---
logger.info(f"Starting data generation for {TICKER}...")
generator.generate_training_data(
    ticker=TICKER,
    start_date=START_DATE,
    end_date=END_DATE,
    primary_timeframe=PRIMARY_TIMEFRAME,
    other_timeframes=SECONDARY_TIMEFRAMES,
    min_lookback=MIN_LOOKBACK
)
logger.info(f"Data generation process completed for {TICKER}.")
```

**Note:** Before running, ensure the minor syntax errors in the logging configuration strings within `vpa_training_data_generator.py` and `example_generate_training_data.py` are corrected (remove extra backslashes).

*(Refer to `/home/ubuntu/VPA_MODULAR_PACKAGE/example_generate_training_data.py` for the full script)*

## 7. Usage Example: Preparing Data for LLM Fine-Tuning

The generated JSONL file can be loaded and processed for fine-tuning using frameworks like Hugging Face `transformers`.

```python
# Conceptual Example from llm_training_examples.md

import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_vpa_training_data(jsonl_file_path):
    # ... (loading logic) ...

def format_instruction_prompt(example):
    # ... (formats input features and output explanation into a prompt) ...
    input_text = f"Analyze VPA for {example['input']['ticker']}... [details] ..."
    output_text = example['output']['explanation']
    prompt = f"### Instruction:\nProvide VPA explanation.\n### Input:\n{input_text}\n### Response:\n{output_text}"
    return {"text": prompt}

# Load data
loaded_data = load_vpa_training_data("path/to/your_data.jsonl")

# Create HF Dataset
hf_dataset = Dataset.from_list(loaded_data)

# Format and Tokenize
MODEL_NAME = "meta-llama/Llama-2-7b-hf" # Example
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ... (handle padding token) ...

formatted_dataset = hf_dataset.map(format_instruction_prompt)
tokenized_dataset = formatted_dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512),
    batched=True
)

# 'tokenized_dataset' is ready for the Trainer API
```

*(Refer to `/home/ubuntu/VPA_MODULAR_PACKAGE/llm_training_examples.md` for more detailed examples and inference prompts)*

## 8. Important Considerations

* **Point-in-Time Analysis:** The accuracy of the training data heavily relies on the `VPAFacade` performing true point-in-time analysis to avoid lookahead bias. This capability needs to be implemented or verified in the facade.
* **Data Quality:** The quality of the historical market data used as input directly impacts the quality of the generated training data.
* **Explanation Quality:** The logic in `_generate_explanation` is critical. It may require refinement to produce consistently high-quality, accurate, and natural-sounding explanations.
* **Feature Engineering:** The choice of input features in the JSONL structure can be further optimized based on LLM performance during fine-tuning.

This documentation provides a solid foundation for generating VPA-specific training data for LLMs.

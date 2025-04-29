# Examples: Using Generated VPA Training Data for LLMs

This document provides conceptual examples of how the generated JSONL training data (`*_vpa_training_data.jsonl`) can be used with Large Language Models (LLMs), particularly for fine-tuning.

**Note:** These are illustrative examples. Actual implementation will depend on the specific LLM framework (e.g., Hugging Face Transformers, PyTorch, TensorFlow), the chosen model architecture, and the fine-tuning strategy.

## 1. Loading and Parsing the JSONL Data

The first step is always to load the generated data. Since it's in JSON Lines format, each line is a valid JSON object.

```python
import json
import pandas as pd

def load_vpa_training_data(jsonl_file_path):
    """Loads VPA training data from a JSONL file."""
    data = []
    try:
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}\nLine: {line.strip()}")
        print(f"Loaded {len(data)} examples from {jsonl_file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {jsonl_file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return []

# Example usage:
training_data_file = "./llm_training_data_output/AAPL_vpa_training_data.jsonl" # Path to your generated file
loaded_data = load_vpa_training_data(training_data_file)

# Optional: Convert to Pandas DataFrame for easier manipulation
if loaded_data:
    # Flatten the nested structure if needed, or use json_normalize
    # Simple example: creating columns for input and output dicts
    df = pd.DataFrame(loaded_data)
    print("\nSample DataFrame head:")
    print(df.head())

    # Example: Accessing input features of the first example
    if not df.empty:
        first_input = df.iloc[0]['input']
        print("\nInput features of the first example:")
        # print(json.dumps(first_input, indent=2))

        first_output = df.iloc[0]['output']
        print("\nOutput of the first example:")
        # print(json.dumps(first_output, indent=2))
```

## 2. Preparing Data for Hugging Face `transformers`

Fine-tuning often involves formatting the input and output into a single string prompt or using specific input/output fields, then tokenizing this data.

```python
# Conceptual Example using Hugging Face datasets and transformers

# Assuming 'loaded_data' is the list of dicts from the previous step

# --- Option A: Formatting as Instruction Tuning Prompt --- 
def format_instruction_prompt(example):
    input_features = example['input']
    output_analysis = example['output']
    
    # Create a detailed textual representation of the input features
    # This needs careful crafting to be informative for the LLM
    input_text = f"Analyze the VPA context for {input_features['ticker']} on {input_features['timestamp']} ({input_features['primary_timeframe']}).\n"
    input_text += f"Trend: {input_features['trend_analysis']['status']} ({input_features['trend_analysis']['strength']}).\n"
    input_text += f"Current Candle Volume: {input_features['vpa_metrics']['volume_class']}, Spread: {input_features['vpa_metrics']['candle_class']}.\n"
    # ... add more details from input_features ...
    
    # The desired output is the explanation
    output_text = output_analysis['explanation']
    
    # Combine into a standard instruction format (e.g., Alpaca format)
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Provide a Volume Price Analysis (VPA) explanation based on the given market context.

### Input:
{input_text}

### Response:
{output_text}"""
    return {"text": prompt}

# --- Option B: Using Separate Input/Output Fields --- 
def format_input_output_fields(example):
    # Similar to Option A, but keep input and output separate
    input_features = example['input']
    output_analysis = example['output']
    
    input_text = f"VPA Context: {input_features['ticker']} {input_features['timestamp']} ... [details] ..."
    output_text = output_analysis['explanation'] # Or just the signal: f"{output_analysis['signal']['type']} ({output_analysis['signal']['strength']})"
    
    return {"input_text": input_text, "output_text": output_text}

# --- Apply Formatting and Tokenization --- 
# Requires installation: pip install datasets transformers torch
from datasets import Dataset
from transformers import AutoTokenizer

# Choose a tokenizer suitable for your target LLM
MODEL_NAME = "meta-llama/Llama-2-7b-hf" # Example
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Add padding token if missing (common for Llama)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

# Create Hugging Face Dataset
hf_dataset = Dataset.from_list(loaded_data)

# Apply formatting (choose Option A or B)
formatted_dataset = hf_dataset.map(format_instruction_prompt) # Using Option A
# formatted_dataset = hf_dataset.map(format_input_output_fields) # Using Option B

# Define tokenization function
def tokenize_function(examples):
    # Adjust field name based on formatting option ('text' or 'input_text'/'output_text')
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512) # Adjust max_length

# Tokenize the dataset
tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# Remove original columns if needed
tokenized_dataset = tokenized_dataset.remove_columns(hf_dataset.column_names)

print("\nSample of tokenized dataset:")
print(tokenized_dataset[0])

# This 'tokenized_dataset' can now be used with the Hugging Face Trainer API
# for fine-tuning.
```

## 3. Example Prompt for Inference (After Fine-Tuning)

Once an LLM is fine-tuned on this data, you could provide it with new VPA context (formatted similarly to the training input) and ask for an analysis.

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Provide a Volume Price Analysis (VPA) explanation based on the given market context.

### Input:
Analyze the VPA context for NVDA on 2024-04-29T16:00:00 (1d).
Trend: BULLISH (STRONG).
Current Candle Volume: HIGH, Spread: WIDE.
Detected Patterns: Test of Resistance (Failed on low volume).
Support/Resistance: Nearby Resistance at 950.00.

### Response:
```

**Expected LLM Output (based on fine-tuning):**

```text
VPA Analysis for NVDA on 2024-04-29T16:00:00 (1d):

**Signal:** SELL (MODERATE)

**Trend Context:** The primary trend is currently BULLISH (STRONG). 

**Current Candle Analysis:**
- Spread: WIDE (Ratio: 1.50 vs avg)
- Volume: HIGH (Ratio: 1.60 vs avg)
- Significant upper wick present (Ratio: 0.80).

**Detected Patterns:**
- Test Of Resistance: Failed test of 950.00 on low volume, indicating potential weakness.

**Support/Resistance:**
- Nearby Resistance Levels: 950.00

**Signal Reasoning:** Although the trend is bullish, the failure to break resistance at 950.00 on low volume, combined with a significant upper wick on the wide-spread, high-volume candle, suggests supply is overcoming demand at this level. This anomaly indicates a potential reversal or pullback.

**Conclusion:** Based on the BULLISH trend, the HIGH volume on the WIDE spread candle, and the presence of Test Of Resistance, a MODERATE SELL signal is generated.
```

These examples illustrate how the structured JSONL data can be transformed and utilized in common LLM workflows. The key is the careful design of the input feature representation and the desired output format during the generation phase. 


# VPA Modular Package Overview

## Module Summaries

### 1. memory_manager.py

**Purpose**: Manages conversation history and system messages for LLM interactions.
**Key Features**:

- Persistent storage of message history in JSON format
- Maintains system messages separately from user-assistant exchanges
- Provides methods to save, retrieve, and clear conversation history
- Supports adding system messages for context setting

### 2. vpa_analyzer.py

**Purpose**: Core analysis engine for Volume Price Analysis patterns and signals.
**Key Components**:

- `CandleAnalyzer`: Analyzes individual candles and their volume for VPA signals
- `TrendAnalyzer`: Identifies trend characteristics and direction
- `PatternRecognizer`: Detects VPA patterns like accumulation, distribution, and climaxes
- `PointInTimeAnalyzer`: Performs analysis at specific historical points (for training data)
- `MultiTimeframeAnalyzer`: Coordinates analysis across multiple timeframes

### 3. vpa_config.py

**Purpose**: Configuration management for the VPA algorithm.
**Key Features**:

- Loads configuration from file or uses defaults
- Provides access to various parameter groups (volume thresholds, candle thresholds, etc.)
- Supports parameter updates via dot notation
- Contains default configuration values for all VPA components

### 4. vpa_data.py

**Purpose**: Data acquisition and management from various sources.
**Key Components**:

- `DataProvider`: Base class for data providers
- `PolygonIOProvider`: Implementation for Polygon.io data source
- `CSVProvider`: Implementation for local CSV files
- `MultiTimeframeProvider`: Fetches data for multiple timeframes

### 5. vpa_facade.py

**Purpose**: Simplified API for VPA analysis, coordinating all components.
**Key Features**:

- Initializes and coordinates all VPA components
- Provides high-level methods for ticker analysis
- Handles multi-timeframe analysis and signal generation
- Supports point-in-time analysis for training data generation
- Offers methods for signal explanation and batch analysis

### 6. vpa_llm_interface.py

**Purpose**: Specific integration for LLMs with the VPA algorithm.
**Key Features**:

- Processes natural language queries about VPA
- Provides concept explanations for VPA terminology
- Formats analysis results for LLM consumption
- Suggests parameters based on trading goals
- Generates code examples for VPA tasks

### 7. vpa_llm_query_engine.py

**Purpose**: Central interface for natural language interactions with VPA.
**Key Features**:

- Uses OpenAI's function calling capabilities
- Provides tools for VPA analysis, concept explanation, and parameter suggestions
- Integrates with RAG for retrieving VPA knowledge
- Manages conversation memory and context
- Handles error cases and validation

### 8. vpa_logger.py

**Purpose**: Logging framework for VPA operations.
**Key Features**:

- Configurable logging levels and output destinations
- Specialized logging methods for VPA-specific events
- Performance tracking for operations
- Supports both console and file logging

### 9. vpa_processor.py

**Purpose**: Data processing and feature calculation for VPA analysis.
**Key Features**:

- Preprocesses price and volume data
- Calculates candle properties and volume metrics
- Classifies candles and volume based on thresholds
- Determines price and volume direction

### 10. vpa_result_extractor.py

**Purpose**: Extracts and organizes analysis results for consumption.
**Key Features**:

- Extracts structured data from analysis results
- Provides methods to access specific parts of the analysis
- Supports extraction of testing signals
- Organizes results by ticker and timeframe

### 11. vpa_signals.py

**Purpose**: Signal generation and risk assessment based on VPA analysis.
**Key Components**:

- `SignalGenerator`: Generates trading signals from analysis results
- `RiskAssessor`: Calculates risk metrics for potential trades
- Provides methods to identify different signal strengths
- Gathers supporting evidence for signals

### 12. vpa_training_data_generator.py

**Purpose**: Generates training data for fine-tuning LLMs on VPA analysis.
**Key Features**:

- Loads historical data for specified tickers and timeframes
- Performs point-in-time analysis at each historical point
- Extracts input features from analysis results
- Generates comprehensive explanations for each data point
- Saves training data in JSONL format

### 13. vpa_visualizer.py

**Purpose**: Visualization tools for VPA analysis results.
**Key Features**:

- Creates price and volume charts
- Visualizes pattern analysis and support/resistance levels
- Generates summary reports and dashboards
- Provides specialized visualizations for signals and multi-timeframe trends

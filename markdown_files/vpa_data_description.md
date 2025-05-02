# vpa_data

The provided code defines the VPA Data Module, which is responsible for acquiring and managing market data for the Volume Price Analysis (VPA) algorithm. It includes multiple classes for fetching data from different sources, such as Yahoo Finance (yfinance), CSV files, and multi-timeframe data. Here's a detailed explanation of the code:

## Key Classes and Their Responsibilities

### 1. DataProvider (Base Class)

Purpose: Acts as an abstract base class for data providers. It defines the interface that all data provider implementations must follow.
Methods:
get_price_data(ticker, timeframe, period): Abstract method to fetch price data (OHLC).
get_volume_data(ticker, timeframe, period): Abstract method to fetch volume data.
get_data(ticker, timeframe, period): Combines price and volume data by calling the above two methods.
Usage: This class is not meant to be instantiated directly. Instead, it serves as a blueprint for other data provider classes

### 2. YFinanceProvider

Purpose: Implements the DataProvider interface using the yfinance library to fetch market data from Yahoo Finance.
Methods:
get_data(ticker, interval='1d', period='1y', start_date=None, end_date=None):
Fetches both price and volume data for a given stock ticker.
Supports specifying the interval (e.g., '1d', '1h') and period (e.g., '1y', '6mo').
Optionally allows specifying a custom date range (start_date and end_date).
Handles potential errors (e.g., missing data or invalid columns).
Returns:
price_data: A DataFrame containing OHLC (Open, High, Low, Close) data.
volume_data: A Series containing volume data.
get_price_data(ticker, interval='1d', period='1y', start_date=None, end_date=None):
Fetches only the price data (OHLC) for the given ticker.
get_volume_data(ticker, interval='1d', period='1y', start_date=None, end_date=None):
Fetches only the volume data for the given ticker.
Key Features:
Handles both standard column names and those with ticker suffixes (e.g., close_AAPL).
Ensures the data has a proper datetime index and converts it to UTC.
Handles missing or malformed data gracefully

### 3. CSVProvider

Purpose: Implements the DataProvider interface to fetch data from CSV files.
Methods:
get_data(ticker, interval='1d', period=None):
Reads price and volume data from a CSV file.
Ensures the file contains required columns (open, high, low, close, volume) and a datetime column.
Converts the datetime column to a proper index.
Returns:
price_data: A DataFrame containing OHLC data.
volume_data: A Series containing volume data.
get_price_data(ticker, interval='1d', period=None):
Fetches only the price data from the CSV file.
get_volume_data(ticker, interval='1d', period=None):
Fetches only the volume data from the CSV file.
Key Features:
Dynamically constructs the file path based on the ticker and interval.
Standardizes column names to lowercase for consistency.
Handles missing files or columns with appropriate error messages

### 4. MultiTimeframeProvider

Purpose: Fetches market data for multiple timeframes using a specified data provider (e.g., YFinanceProvider or CSVProvider).
Methods:
get_multi_timeframe_data(ticker, timeframes):
Fetches price and volume data for a list of timeframes.
Each timeframe is specified as a dictionary with interval and period keys.
Returns a dictionary where each key corresponds to a timeframe (e.g., '1d', '1h'), and the value contains the price and volume data for that timeframe.
Key Features:
Iterates over the provided timeframes and fetches data for each.
Handles errors for individual timeframes without stopping the entire process.
Ensures at least one timeframe's data is successfully fetched.

## How the Code Works

### Fetching Data with YFinanceProvider

Example:

```python
from vpa_modular.vpa_data import YFinanceProvider

provider = YFinanceProvider()
price_data, volume_data = provider.get_data("AAPL", interval="1d", period="1y")
print(price_data.head())
print(volume_data.head())
```

### Fetching Data from CSV Files

Example:

```python
from vpa_modular.vpa_data import CSVProvider

provider = CSVProvider(data_path="data/")
price_data, volume_data = provider.get_data("AAPL", interval="1d")
print(price_data.head())
print(volume_data.head())
```

### Fetching Multi-Timeframe Data

Example:

```python
from vpa_modular.vpa_data import MultiTimeframeProvider, YFinanceProvider

provider = YFinanceProvider()
multi_provider = MultiTimeframeProvider(provider)

timeframes = [
    {"interval": "1d", "period": "1y"},
    {"interval": "1h", "period": "30d"}
]
data = multi_provider.get_multi_timeframe_data("AAPL", timeframes)
for tf, tf_data in data.items():
    print(f"Timeframe: {tf}")
    print(tf_data["price_data"].head())
    print(tf_data["volume_data"].head())
```

## Error Handling

YFinanceProvider:
Raises a ValueError if data is missing or malformed.
Handles exceptions during data download and provides meaningful error messages.
CSVProvider:
Raises a FileNotFoundError if the CSV file is missing.
Ensures required columns are present and raises a ValueError if they are not.
MultiTimeframeProvider:
Logs errors for individual timeframes but continues fetching data for others

## Use Cases

Real-Time Data Analysis:
Use YFinanceProvider to fetch live market data for analysis.
Backtesting:
Use CSVProvider to load historical data from local files for backtesting strategies.
Multi-Timeframe Analysis:
Use MultiTimeframeProvider to analyze data across multiple timeframes (e.g., daily and hourly)

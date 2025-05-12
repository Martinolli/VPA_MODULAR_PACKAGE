# VPA System: Polygon.io Data Provider Migration Guide

This document outlines the changes made to the VPA (Volume Price Analysis) modular system to migrate the primary data source from yfinance to Polygon.io. It also provides guidance on configuration and best practices for using the new data provider.

## 1. Rationale for Migration

The primary motivation for this migration was to address issues related to the previous data provider (yfinance) and to leverage a more robust and potentially feature-rich financial data API. Polygon.io offers a comprehensive suite of market data APIs, including historical and real-time data for stocks, options, forex, and crypto, which can support current and future VPA system enhancements.

## 2. Key Changes Implemented

The core changes were focused on the `vpa_modular.vpa_data` and `vpa_modular.vpa_facade` modules.

### 2.1. `vpa_data.py` Modifications

* **New `PolygonIOProvider` Class:**
  * A new class, `PolygonIOProvider`, was implemented as a subclass of `DataProvider`.
  * This class uses the official `polygon-api-client` library to fetch data from Polygon.io.
  * It requires a Polygon.io API key, which is retrieved from the `POLYGON_API_KEY` environment variable.
  * The `get_data` method handles fetching aggregate bars (OHLCV) for specified tickers, intervals (daily, hourly, minute), and date ranges (either via a period string like "1y", "1mo", "7d" or specific start/end dates).
  * It includes logic to parse interval strings (e.g., "1d", "1h", "5m") into the format required by the Polygon.io API (multiplier and timespan).
  * It calculates the `from_` date based on the `period` string if start/end dates are not provided.
  * Data returned from Polygon.io is transformed into pandas DataFrames with columns named `open`, `high`, `low`, `close`, and `volume`, and a `datetime` index, ensuring compatibility with the rest of the VPA processing pipeline.
  * Error handling for API calls and empty data responses is included.
* **`YFinanceProvider` Deprecation (Conceptual):**
  * While the `YFinanceProvider` class might still exist in the file for reference or potential future use, its active usage has been replaced by `PolygonIOProvider` in the `VPAFacade`.
* **`MultiTimeframeProvider` Update:**
  * This class was already designed to work with any `DataProvider` instance. No structural changes were needed, but it now receives an instance of `PolygonIOProvider` when initialized by the `VPAFacade`.

### 2.2. `vpa_facade.py` Modifications

* **Data Provider Instantiation:**
  * In the `VPAFacade.__init__` method, the `self.data_provider` is now initialized as an instance of `PolygonIOProvider`:

    ```python
    self.data_provider = PolygonIOProvider()
    ```

  * Consequently, `self.multi_tf_provider` also uses the `PolygonIOProvider` indirectly.

## 3. Configuration and Setup

### 3.1. Polygon.io API Key

To use the new data provider, a valid Polygon.io API key is **required**.

1. **Obtain an API Key:** Sign up at [Polygon.io](https://polygon.io/) to get your API key.
2. **Set Environment Variable:** The `PolygonIOProvider` expects the API key to be available as an environment variable named `POLYGON_API_KEY`.
   * **Using a `.env` file (Recommended for local development):**
     * Create a file named `.env` in the root directory of your `VPA_MODULAR_PACKAGE_LATEST` project.
     * Add the following line to the `.env` file, replacing `YOUR_POLYGON_API_KEY` with your actual key:

       ```env
       POLYGON_API_KEY=YOUR_POLYGON_API_KEY
       ```

     * Ensure your Python environment has the `python-dotenv` package installed (`pip3 install python-dotenv`). The test script and potentially your application scripts should call `load_dotenv()` to load this file.
   * **Setting System Environment Variable:** Alternatively, you can set `POLYGON_API_KEY` as a system-wide environment variable. The method for this varies by operating system.

### 3.2. Dependencies

Ensure the following Python packages are installed:

* `polygon-api-client`
* `pandas`
* `numpy`
* `python-dotenv` (if using `.env` files)

You can typically install these using pip:

```bash
pip3 install polygon-api-client pandas numpy python-dotenv
```

## 4. Usage

No changes are required in how you interact with the `VPAFacade` for analysis. The facade abstracts the data provider, so methods like `analyze_ticker`, `get_signals`, etc., will now use Polygon.io data automatically.

## 5. Validation and Testing

A test script, `test_polygon_integration.py`, has been created in the project root to help validate the Polygon.io integration.

* **Purpose:** This script directly calls the `PolygonIOProvider.get_data` method (via the facade instance) to fetch data for various tickers and timeframes (daily, hourly, minute).
* **Running the Test:**
  1. Ensure your `POLYGON_API_KEY` is correctly set up (see section 3.1).
  2. Navigate to the `VPA_MODULAR_PACKAGE_LATEST` directory in your terminal.
  3. Run the script: `python3 test_polygon_integration.py`
* **Expected Outcome:** The script will print the status of data fetching for each test scenario, including the shape and head/tail of the fetched DataFrames. If data is fetched successfully, it indicates the Polygon.io integration is working.

## 6. Data Format and Compatibility

The `PolygonIOProvider` is designed to return data in the same pandas DataFrame format (OHLCV with a datetime index) as previously expected by the VPA system. This ensures backward compatibility with the existing data processing, analysis, and signal generation modules.

* **Timestamps:** Polygon.io provides timestamps in milliseconds. These are converted to UTC `datetime` objects and set as the DataFrame index.
* **Adjusted Prices:** The provider requests adjusted prices from Polygon.io by default (`adjusted=True`).

## 7. Updated Best Practices

* **API Key Management:** Always keep your `POLYGON_API_KEY` secure. Do not commit it directly into your codebase or share it publicly. Use environment variables or `.env` files.
* **API Rate Limits:** Be mindful of Polygon.io API rate limits, especially if you are on a free or lower-tier plan. The current implementation fetches data in bulk for the requested period but avoid making excessively frequent or large data requests in tight loops.
* **Interval and Period Specificity:** When requesting data, ensure the `interval` and `period` (or `start_date`/`end_date`) parameters are valid and appropriate for your analysis needs. The `PolygonIOProvider` has specific parsing logic for these.
* **Error Handling:** The provider includes basic error handling. When integrating into larger applications, ensure you have robust error handling around calls to the `VPAFacade` to manage potential data fetching issues (e.g., network errors, invalid tickers, API errors).
* **Data Validation:** While the provider aims for compatibility, always perform sanity checks on the fetched data, especially when working with new tickers or less common timeframes, to ensure it meets your analysis requirements.

This migration to Polygon.io provides a more stable and potentially richer data foundation for the VPA system. Remember to test thoroughly in your environment with your API key.

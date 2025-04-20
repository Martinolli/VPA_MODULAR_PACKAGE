"""
VPA Data Validator Module

This module provides tools for validating data consistency across timeframes
and ensuring data is suitable for backtesting.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VPADataValidator')

def datetime_json_serializer(obj):
    """
    Helper function for json.dumps to serialize datetime objects.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class VPADataValidator:
    """Validate data consistency across timeframes for VPA analysis"""
    
    def __init__(self, base_dir="fetched_data"):
        """
        Initialize the data validator
        
        Parameters:
        - base_dir: Base directory for stored data
        """
        self.base_dir = base_dir
        self.validation_dir = os.path.join(base_dir, "validation")
        os.makedirs(self.validation_dir, exist_ok=True)
    
    def validate_ticker(self, ticker, start_date=None, end_date=None, timeframes=None):
        """
        Validate data for a ticker across timeframes
        
        Parameters:
        - ticker: Stock symbol
        - start_date: Start date for validation (string or datetime)
        - end_date: End date for validation (string or datetime)
        - timeframes: List of timeframe strings to validate
        
        Returns:
        - Dictionary with validation results
        """
        if timeframes is None:
            timeframes = ["1d", "1h", "15m"]
        
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        validation_results = {
            "ticker": ticker,
            "validation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "timeframes": {},
            "cross_timeframe_validation": {}
        }
        
        # Load data for each timeframe
        timeframe_data = {}
        for timeframe in timeframes:
            data = self._load_data(ticker, timeframe)
            if data is not None:
                timeframe_data[timeframe] = data
                
                # Validate individual timeframe
                validation_results["timeframes"][timeframe] = self._validate_timeframe(
                    data, timeframe, start_date, end_date
                )
        
        # Cross-timeframe validation
        if len(timeframe_data) > 1:
            validation_results["cross_timeframe_validation"] = self._validate_cross_timeframe(
                ticker, timeframe_data
            )
        
        # Generate validation report
        self._generate_validation_report(ticker, validation_results)
        
        return validation_results
    
    def _load_data(self, ticker, timeframe):
        """
        Load data for a ticker at a specific timeframe
        
        Parameters:
        - ticker: Stock symbol
        - timeframe: Timeframe string ('1d', '1h', '15m')
        
        Returns:
        - DataFrame with loaded data or None if data doesn't exist
        """
        file_path = os.path.join(self.base_dir, timeframe, f"{ticker}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"No data file found for {ticker} at {timeframe} timeframe")
            return None
        
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(data)} rows of {timeframe} data for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Error loading data for {ticker} at {timeframe} timeframe: {str(e)}")
            return None
    
    def _validate_timeframe(self, data, timeframe, start_date=None, end_date=None):
        """
        Validate data for a specific timeframe
        
        Parameters:
        - data: DataFrame with data to validate
        - timeframe: Timeframe string ('1d', '1h', '15m')
        - start_date: Start date for validation
        - end_date: End date for validation
        
        Returns:
        - Dictionary with validation results
        """
        # Check if data is empty
        if data.empty:
            return {
                "status": "empty",
                "message": "Data file is empty"
            }
        
        # Check date range
        data_start = data.index.min()
        data_end = data.index.max()
        
        timeframe_result = {
            "status": "valid",
            "rows": len(data),
            "data_start": data_start.strftime('%Y-%m-%d'),
            "data_end": data_end.strftime('%Y-%m-%d'),
            "issues": []
        }
        
        # Check if data covers the requested date range
        if start_date is not None and data_start > start_date:
            timeframe_result["issues"].append({
                "type": "date_range",
                "message": f"Data starts at {data_start.strftime('%Y-%m-%d')} but requested start date is {start_date.strftime('%Y-%m-%d')}"
            })
        
        if end_date is not None and data_end < end_date:
            timeframe_result["issues"].append({
                "type": "date_range",
                "message": f"Data ends at {data_end.strftime('%Y-%m-%d')} but requested end date is {end_date.strftime('%Y-%m-%d')}"
            })
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            timeframe_result["issues"].append({
                "type": "missing_columns",
                "message": f"Missing required columns: {', '.join(missing_columns)}"
            })
        
        # Check for gaps in the data
        if timeframe == "1d":
            # For daily data, check for missing trading days
            business_days = pd.date_range(start=data_start, end=data_end, freq='B')
            missing_days = business_days.difference(data.index)
            if len(missing_days) > 0:
                timeframe_result["issues"].append({
                    "type": "data_gaps",
                    "message": f"Missing {len(missing_days)} trading days",
                    "missing_days": [day.strftime('%Y-%m-%d') for day in missing_days[:10]]  # Show first 10 missing days
                })
        
        # Check for outliers in price data
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                # Calculate z-scores
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = data[z_scores > 3]
                if not outliers.empty:
                    timeframe_result["issues"].append({
                        "type": "outliers",
                        "message": f"Found {len(outliers)} outliers in {col} column",
                        "outlier_count": len(outliers)
                    })
        
        # Check for zero or negative values in volume
        if 'volume' in data.columns:
            zero_volume = data[data['volume'] <= 0]
            if not zero_volume.empty:
                timeframe_result["issues"].append({
                    "type": "invalid_volume",
                    "message": f"Found {len(zero_volume)} rows with zero or negative volume",
                    "zero_volume_count": len(zero_volume)
                })
        
        # Update status based on issues
        if timeframe_result["issues"]:
            timeframe_result["status"] = "issues_found"
        
        return timeframe_result
    
    def _validate_cross_timeframe(self, ticker, timeframe_data):
        """
        Validate consistency across timeframes
        
        Parameters:
        - ticker: Stock symbol
        - timeframe_data: Dictionary with DataFrames for each timeframe
        
        Returns:
        - Dictionary with cross-timeframe validation results
        """
        cross_validation = {
            "status": "valid",
            "issues": []
        }
        
        # Check if daily data aligns with hourly data
        if "1d" in timeframe_data and "1h" in timeframe_data:
            daily_data = timeframe_data["1d"]
            hourly_data = timeframe_data["1h"]
            
            # Get common dates
            daily_dates = daily_data.index.normalize().unique()
            hourly_dates = hourly_data.index.normalize().unique()
            common_dates = daily_dates.intersection(hourly_dates)
            
            if len(common_dates) == 0:
                cross_validation["issues"].append({
                    "type": "no_common_dates",
                    "message": "No common dates between daily and hourly data"
                })
            else:
                # Compare daily OHLC with hourly OHLC for common dates
                for date in common_dates:
                    daily_row = daily_data[daily_data.index.normalize() == date]
                    hourly_rows = hourly_data[hourly_data.index.normalize() == date]
                    
                    if not daily_row.empty and not hourly_rows.empty:
                        # Check high price
                        daily_high = daily_row['high'].iloc[0]
                        hourly_high = hourly_rows['high'].max()
                        if abs(daily_high - hourly_high) / daily_high > 0.01:  # 1% tolerance
                            cross_validation["issues"].append({
                                "type": "high_price_mismatch",
                                "message": f"High price mismatch on {date.strftime('%Y-%m-%d')}: daily={daily_high}, hourly={hourly_high}"
                            })
                        
                        # Check low price
                        daily_low = daily_row['low'].iloc[0]
                        hourly_low = hourly_rows['low'].min()
                        if abs(daily_low - hourly_low) / daily_low > 0.01:  # 1% tolerance
                            cross_validation["issues"].append({
                                "type": "low_price_mismatch",
                                "message": f"Low price mismatch on {date.strftime('%Y-%m-%d')}: daily={daily_low}, hourly={hourly_low}"
                            })
                        
                        # Check volume
                        if 'volume' in daily_row.columns and 'volume' in hourly_rows.columns:
                            daily_volume = daily_row['volume'].iloc[0]
                            hourly_volume_sum = hourly_rows['volume'].sum()
                            if abs(daily_volume - hourly_volume_sum) / daily_volume > 0.05:  # 5% tolerance
                                cross_validation["issues"].append({
                                    "type": "volume_mismatch",
                                    "message": f"Volume mismatch on {date.strftime('%Y-%m-%d')}: daily={daily_volume}, hourly sum={hourly_volume_sum}"
                                })
        
        # Check if hourly data aligns with 15-minute data
        if "1h" in timeframe_data and "15m" in timeframe_data:
            hourly_data = timeframe_data["1h"]
            minute_data = timeframe_data["15m"]
            
            # Get common dates
            hourly_dates = hourly_data.index.normalize().unique()
            minute_dates = minute_data.index.normalize().unique()
            common_dates = hourly_dates.intersection(minute_dates)
            
            if len(common_dates) == 0:
                cross_validation["issues"].append({
                    "type": "no_common_dates",
                    "message": "No common dates between hourly and 15-minute data"
                })
        
        # Update status based on issues
        if cross_validation["issues"]:
            cross_validation["status"] = "issues_found"
        
        return cross_validation
    
    def _generate_validation_report(self, ticker, validation_results):
        """
        Generate validation report with visualizations
        
        Parameters:
        - ticker: Stock symbol
        - validation_results: Dictionary with validation results
        """
        # Save validation results to JSON
        report_dir = os.path.join(self.validation_dir, ticker)
        os.makedirs(report_dir, exist_ok=True)
        
        json_path = os.path.join(report_dir, "validation_results.json")
        with open(json_path, 'w') as f:
            json.dump(validation_results, f, indent=4, default=datetime_json_serializer)
        
        # Generate HTML report
        html_report = self._generate_html_report(ticker, validation_results)
        html_path = os.path.join(report_dir, "validation_report.html")
        with open(html_path, 'w') as f:
            f.write(html_report)
        
        # Generate visualizations if data is available
        for timeframe in validation_results["timeframes"]:
            if validation_results["timeframes"][timeframe]["status"] != "empty":
                data = self._load_data(ticker, timeframe)
                if data is not None:
                    self._generate_visualizations(ticker, timeframe, data, report_dir)
        
        logger.info(f"Generated validation report for {ticker} at {report_dir}")
    
    def _generate_html_report(self, ticker, validation_results):
        """
        Generate HTML validation report
        
        Parameters:
        - ticker: Stock symbol
        - validation_results: Dictionary with validation results
        
        Returns:
        - HTML report as string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Validation Report for {ticker}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .valid {{ color: green; }}
                .issues {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .issue-list {{ margin-left: 20px; }}
                .issue-item {{ margin-bottom: 5px; }}
            </style>
        </head>
        <body>
            <h1>Data Validation Report for {ticker}</h1>
            <p>Validation Time: {validation_results["validation_time"]}</p>
            
            <h2>Timeframe Validation</h2>
        """
        
        # Add timeframe validation results
        for timeframe, result in validation_results["timeframes"].items():
            status_class = "valid" if result["status"] == "valid" else "issues" if result["status"] == "issues_found" else "error"
            
            html += f"""
            <h3>{timeframe} Timeframe <span class="{status_class}">({result["status"]})</span></h3>
            """
            
            if result["status"] != "empty" and result["status"] != "missing":
                html += f"""
                <table>
                    <tr>
                        <th>Data Start</th>
                        <th>Data End</th>
                        <th>Rows</th>
                    </tr>
                    <tr>
                        <td>{result.get("data_start", "N/A")}</td>
                        <td>{result.get("data_end", "N/A")}</td>
                        <td>{result.get("rows", "N/A")}</td>
                    </tr>
                </table>
                """
                
                if "issues" in result and result["issues"]:
                    html += "<h4>Issues:</h4><div class='issue-list'>"
                    for issue in result["issues"]:
                        html += f"<div class='issue-item'><strong>{issue['type']}:</strong> {issue['message']}</div>"
                    html += "</div>"
            else:
                html += f"<p>{result.get('message', 'No data available')}</p>"
        
        # Add cross-timeframe validation results
        if "cross_timeframe_validation" in validation_results:
            cross_result = validation_results["cross_timeframe_validation"]
            status_class = "valid" if cross_result["status"] == "valid" else "issues"
            
            html += f"""
            <h2>Cross-Timeframe Validation <span class="{status_class}">({cross_result["status"]})</span></h2>
            """
            
            if "issues" in cross_result and cross_result["issues"]:
                html += "<h4>Issues:</h4><div class='issue-list'>"
                for issue in cross_result["issues"]:
                    html += f"<div class='issue-item'><strong>{issue['type']}:</strong> {issue['message']}</div>"
                html += "</div>"
            else:
                html += "<p>No cross-timeframe issues found.</p>"
        
        # Add visualization references
        html += """
            <h2>Visualizations</h2>
            <p>The following visualizations are available in the report directory:</p>
            <ul>
                <li>Price charts for each timeframe</li>
                <li>Volume charts for each timeframe</li>
                <li>Data completeness visualizations</li>
            </ul>
        """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_visualizations(self, ticker, timeframe, data, report_dir):
        """
        Generate visualizations for data validation
        
        Parameters:
        - ticker: Stock symbol
        - timeframe: Timeframe string
        - data: DataFrame with data
        - report_dir: Directory to save visualizations
        """
        # Create directory for visualizations
        viz_dir = os.path.join(report_dir, "visualizations", timeframe)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate price chart
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['close'], label='Close Price')
        plt.title(f"{ticker} Close Price - {timeframe} Timeframe")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "price_chart.png"))
        plt.close()
        
        # Generate volume chart
        plt.figure(figsize=(12, 6))
        plt.bar(data.index, data['volume'], alpha=0.7)
        plt.title(f"{ticker} Volume - {timeframe} Timeframe")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "volume_chart.png"))
        plt.close()
        
        # Generate data completeness visualization (calendar heatmap for daily data)
        if timeframe == "1d" and len(data) > 20:
            try:
                # Create a date range covering the entire period
                start_date = data.index.min().normalize()
                end_date = data.index.max().normalize()
                all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Create a DataFrame with all dates and mark which ones have data
                date_df = pd.DataFrame(index=all_dates)
                date_df['has_data'] = date_df.index.isin(data.index.normalize())
                date_df['year'] = date_df.index.year
                date_df['month'] = date_df.index.month
                date_df['day'] = date_df.index.day
                date_df['weekday'] = date_df.index.weekday
                
                # Plot calendar heatmap
                plt.figure(figsize=(15, 8))
                
                # Group by year and month
                years = date_df.index.year.unique()
                months = range(1, 13)
                
                # Create a grid of subplots
                fig, axes = plt.subplots(len(years), 12, figsize=(20, len(years) * 2))
                fig.suptitle(f"Data Completeness for {ticker} - Daily Timeframe", fontsize=16)
                
                # Flatten axes array if there's only one year
                if len(years) == 1:
                    axes = axes.reshape(1, -1)
                
                # Plot each year-month combination
                for i, year in enumerate(years):
                    for j, month in enumerate(months):
                        ax = axes[i, j]
                        
                        # Get data for this month
                        month_data = date_df[(date_df['year'] == year) & (date_df['month'] == month)]
                        
                        if not month_data.empty:
                            # Create a 7x6 grid for the days (rows=weekdays, cols=weeks)
                            data_grid = np.zeros((7, 6))
                            data_grid.fill(np.nan)  # Fill with NaN for days not in month
                            
                            # Fill in the data
                            for _, row in month_data.iterrows():
                                week = (row['day'] - 1) // 7
                                weekday = row['weekday']
                                if week < 6:  # Ensure we don't go out of bounds
                                    data_grid[weekday, week] = 1 if row['has_data'] else 0
                            
                            # Plot heatmap
                            sns.heatmap(data_grid, ax=ax, cmap=['red', 'green'], cbar=False, 
                                       xticklabels=False, yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                            ax.set_title(f"{year}-{month:02d}")
                        else:
                            ax.axis('off')
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(viz_dir, "data_completeness.png"))
                plt.close()
            except Exception as e:
                logger.warning(f"Error generating calendar heatmap: {str(e)}")
    
    def validate_multiple_tickers(self, tickers, start_date=None, end_date=None, timeframes=None):
        """
        Validate data for multiple tickers
        
        Parameters:
        - tickers: List of stock symbols
        - start_date: Start date for validation
        - end_date: End date for validation
        - timeframes: List of timeframe strings to validate
        
        Returns:
        - Dictionary with validation results for each ticker
        """
        results = {}
        
        for ticker in tickers:
            logger.info(f"Validating data for {ticker}")
            ticker_results = self.validate_ticker(ticker, start_date, end_date, timeframes)
            results[ticker] = ticker_results
        
        return results
    
    def check_backtesting_readiness(self, ticker, start_date, end_date, timeframes=None):
        """
        Check if data is ready for backtesting
        
        Parameters:
        - ticker: Stock symbol
        - start_date: Start date for backtesting
        - end_date: End date for backtesting
        - timeframes: List of timeframe strings required for backtesting
        
        Returns:
        - Dictionary with readiness status and issues
        """
        if timeframes is None:
            timeframes = ["1d", "1h", "15m"]
        
        # Validate data
        validation_results = self.validate_ticker(ticker, start_date, end_date, timeframes)
        
        # Check readiness
        readiness = {
            "ticker": ticker,
            "start_date": start_date.strftime('%Y-%m-%d') if isinstance(start_date, (datetime, pd.Timestamp)) else start_date,
            "end_date": end_date.strftime('%Y-%m-%d') if isinstance(end_date, (datetime, pd.Timestamp)) else end_date,
            "is_ready": True,
            "issues": []
        }
        
        # Check each timeframe
        for timeframe in timeframes:
            if timeframe not in validation_results["timeframes"]:
                readiness["is_ready"] = False
                readiness["issues"].append({
                    "timeframe": timeframe,
                    "message": "No data available"
                })
                continue
            
            tf_result = validation_results["timeframes"][timeframe]
            
            if tf_result["status"] == "empty" or tf_result["status"] == "missing":
                readiness["is_ready"] = False
                readiness["issues"].append({
                    "timeframe": timeframe,
                    "message": tf_result.get("message", "No data available")
                })
                continue
            
            # Check for date range issues
            date_range_issues = [issue for issue in tf_result.get("issues", []) if issue["type"] == "date_range"]
            if date_range_issues:
                readiness["is_ready"] = False
                for issue in date_range_issues:
                    readiness["issues"].append({
                        "timeframe": timeframe,
                        "message": issue["message"]
                    })
        
        # Check cross-timeframe issues
        if "cross_timeframe_validation" in validation_results:
            cross_issues = validation_results["cross_timeframe_validation"].get("issues", [])
            if cross_issues:
                for issue in cross_issues:
                    readiness["issues"].append({
                        "timeframe": "cross-timeframe",
                        "message": issue["message"]
                    })
        
        return readiness

# Example usage
if __name__ == "__main__":
    # Create data validator
    validator = VPADataValidator()
    
    # Validate data for a ticker
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    validation_results = validator.validate_ticker(ticker, start_date, end_date)
    print(f"Validation results for {ticker}:")
    print(json.dumps(validation_results, indent=4))
    
    # Check backtesting readiness
    readiness = validator.check_backtesting_readiness(ticker, start_date, end_date)
    print(f"Backtesting readiness for {ticker}:")
    print(json.dumps(readiness, indent=4))

"""
VPA Data Fetcher Fix for Non-Standard CSV Format

This module provides a fix for handling non-standard CSV formats in the VPA data fetcher.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger('VPADataFetcher')

def fix_csv_format(file_path):
    """
    Fix a non-standard CSV file format by converting it to a standard format
    that can be properly read by pandas.
    
    Parameters:
    - file_path: Path to the CSV file to fix
    
    Returns:
    - Boolean indicating success or failure
    """
    try:
        # First, check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Read the file to check its format
        try:
            # Try to read the file with pandas first
            df = pd.read_csv(file_path, nrows=5)
            
            # Check if this is already a properly formatted file with all required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if all(col in [c.lower() for c in df.columns] for col in required_columns):
                logger.info(f"File {file_path} already has all required columns in standard format")
                return True
            
            # Check if this is a file with capitalized column names
            capitalized_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in capitalized_columns):
                logger.info(f"File {file_path} has capitalized column names, will rename them")
                # Read the full file
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Rename columns to lowercase
                df.columns = [col.lower() for col in df.columns]
                # Save the fixed file
                df.to_csv(file_path)
                logger.info(f"Successfully renamed columns to lowercase for {file_path}")
                return True
            
            # Check for the problematic format with metadata rows
            with open(file_path, 'r') as f:
                first_lines = [next(f) for _ in range(5) if f.readable()]
            
            is_problematic = False
            if len(first_lines) >= 3:
                if ('Price' in first_lines[0] or 'Ticker' in first_lines[1] or 'Date' in first_lines[2]):
                    is_problematic = True
            
            if not is_problematic:
                # If we get here, the file format is non-standard but not the specific problematic format
                logger.warning(f"File {file_path} has a non-standard format but not the specific problematic format")
                # Try to fix it by ensuring column names are lowercase
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Map column names to expected lowercase names
                column_mapping = {
                    'Price': 'close',  # Assuming Price is equivalent to close
                    'Adj Close': 'adj_close',
                    'Close': 'close',
                    'High': 'high',
                    'Low': 'low',
                    'Open': 'open',
                    'Volume': 'volume'
                }
                
                # Rename columns using the mapping
                df = df.rename(columns=lambda x: column_mapping.get(x, x.lower()))
                
                # Save the fixed file
                df.to_csv(file_path)
                logger.info(f"Applied general column name fix for {file_path}")
                return True
            
            logger.info(f"Detected non-standard CSV format with metadata rows in {file_path}, applying fix")
            
            # Read the file with pandas, skipping the metadata rows
            df = pd.read_csv(file_path, skiprows=3)
            
            # Ensure the date column is properly formatted
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Map column names to expected lowercase names
            column_mapping = {
                'Price': 'close',  # Assuming Price is equivalent to close
                'Adj Close': 'adj_close',
                'Close': 'close',
                'High': 'high',
                'Low': 'low',
                'Open': 'open',
                'Volume': 'volume'
            }
            
            # Rename columns using the mapping
            df = df.rename(columns=lambda x: column_mapping.get(x, x.lower()))
            
            # Ensure all required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Required columns missing after fix: {', '.join(missing_columns)}")
                
                # If we're missing 'close' but have 'adj_close', use that
                if 'close' in missing_columns and 'adj_close' in df.columns:
                    df['close'] = df['adj_close']
                    missing_columns.remove('close')
                
                # If we still have missing columns, try to derive them from what we have
                if missing_columns:
                    logger.warning(f"Attempting to derive missing columns: {', '.join(missing_columns)}")
                    
                    # If we have any price column, use it for missing price columns
                    price_columns = ['open', 'high', 'low', 'close']
                    available_price_cols = [col for col in price_columns if col in df.columns]
                    
                    if available_price_cols:
                        # Use the first available price column for all missing price columns
                        reference_col = available_price_cols[0]
                        for col in missing_columns:
                            if col in price_columns:
                                df[col] = df[reference_col]
                                logger.info(f"Derived {col} from {reference_col}")
                    
                    # If volume is missing, create a dummy column with 0s
                    if 'volume' in missing_columns:
                        df['volume'] = 0
                        logger.info("Created dummy volume column with 0s")
            
            # Save the fixed file
            df.to_csv(file_path)
            
            logger.info(f"Successfully fixed CSV format for {file_path}")
            return True
            
        except pd.errors.EmptyDataError:
            logger.error(f"File {file_path} is empty")
            return False
        except pd.errors.ParserError:
            # If pandas can't parse it, try a more manual approach
            logger.warning(f"Pandas couldn't parse {file_path}, trying manual approach")
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Check if this looks like our problematic format
            if len(lines) >= 3 and ('Price' in lines[0] or 'Ticker' in lines[1] or 'Date' in lines[2]):
                # Remove the first three lines
                with open(file_path, 'w') as f:
                    f.writelines(lines[3:])
                
                # Now try to read it with pandas
                try:
                    df = pd.read_csv(file_path)
                    
                    # Ensure the date column is properly formatted
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    
                    # Rename columns to lowercase
                    df.columns = [col.lower() for col in df.columns]
                    
                    # Save the fixed file
                    df.to_csv(file_path)
                    
                    logger.info(f"Successfully fixed CSV format for {file_path} using manual approach")
                    return True
                except Exception as e:
                    logger.error(f"Error fixing CSV format after manual approach: {str(e)}")
                    return False
            else:
                logger.error(f"File {file_path} has an unknown format that couldn't be parsed")
                return False
        
    except Exception as e:
        logger.error(f"Error fixing CSV format for {file_path}: {str(e)}")
        return False

def fix_all_csv_files(base_dir="fetched_data"):
    """
    Fix all CSV files in the data directory
    
    Parameters:
    - base_dir: Base directory for stored data
    
    Returns:
    - Dictionary with results for each fixed file
    """
    results = {}
    
    # Process each timeframe directory
    for timeframe in ["1d", "1h", "15m"]:
        timeframe_dir = os.path.join(base_dir, timeframe)
        if not os.path.exists(timeframe_dir):
            continue
        
        # Process each CSV file in the timeframe directory
        for filename in os.listdir(timeframe_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(timeframe_dir, filename)
                ticker = filename.replace('.csv', '')
                
                logger.info(f"Processing {ticker} at {timeframe} timeframe")
                success = fix_csv_format(file_path)
                
                results[f"{ticker}_{timeframe}"] = {
                    "status": "fixed" if success else "error",
                    "file_path": file_path
                }
    
    return results

def enhance_data_fetcher_load_method(original_load_method):
    """
    Create an enhanced version of the data fetcher's load_data method
    that automatically fixes CSV format issues.
    
    Parameters:
    - original_load_method: The original load_data method from VPADataFetcher
    
    Returns:
    - Enhanced load_data method
    """
    def enhanced_load_data(self, ticker, timeframe):
        """
        Enhanced load_data method that fixes CSV format issues
        
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
            # Try to fix the CSV format if needed
            fix_csv_format(file_path)
            
            # Now load the data with pandas
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Verify that required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns after loading {file_path}: {', '.join(missing_columns)}")
                
                # Try to fix column names again
                column_mapping = {
                    'Price': 'close',
                    'Adj Close': 'adj_close',
                    'Close': 'close',
                    'High': 'high',
                    'Low': 'low',
                    'Open': 'open',
                    'Volume': 'volume'
                }
                
                # Also try capitalized versions
                for key in list(column_mapping.keys()):
                    column_mapping[key.lower()] = column_mapping[key]
                
                # Rename columns using the mapping
                data = data.rename(columns=lambda x: column_mapping.get(x, x.lower()))
                
                # Check again for missing columns
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    logger.warning(f"Still missing required columns: {', '.join(missing_columns)}")
                    
                    # If we're missing 'close' but have 'adj_close', use that
                    if 'close' in missing_columns and 'adj_close' in data.columns:
                        data['close'] = data['adj_close']
                        missing_columns.remove('close')
                    
                    # If we still have missing columns, try to derive them from what we have
                    if missing_columns:
                        logger.warning(f"Attempting to derive missing columns: {', '.join(missing_columns)}")
                        
                        # If we have any price column, use it for missing price columns
                        price_columns = ['open', 'high', 'low', 'close']
                        available_price_cols = [col for col in price_columns if col in data.columns]
                        
                        if available_price_cols:
                            # Use the first available price column for all missing price columns
                            reference_col = available_price_cols[0]
                            for col in missing_columns:
                                if col in price_columns:
                                    data[col] = data[reference_col]
                                    logger.info(f"Derived {col} from {reference_col}")
                        
                        # If volume is missing, create a dummy column with 0s
                        if 'volume' in missing_columns and 'volume' not in data.columns:
                            data['volume'] = 0
                            logger.info("Created dummy volume column with 0s")
                
                # Save the fixed data back to the file
                data.to_csv(file_path)
            
            logger.info(f"Loaded {len(data)} rows of {timeframe} data for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Error loading data for {ticker} at {timeframe} timeframe: {str(e)}")
            return None
    
    return enhanced_load_data

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Fix all CSV files in the data directory
    results = fix_all_csv_files()
    print(f"Fixed {len([r for r in results.values() if r['status'] == 'fixed'])} files")
    print(f"Errors in {len([r for r in results.values() if r['status'] == 'error'])} files")

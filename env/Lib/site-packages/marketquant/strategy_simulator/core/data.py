from functools import lru_cache
import pandas as pd
from marketquant.strategy_simulator.core.cli.cli_output import CLIOutput

class DataEngine:
    def __init__(self, data_source):
        self.data_source = data_source

    @lru_cache(maxsize=None)  # Dev Note: This is used to improve execution time by caching results.
    def fetch_data(self):
        CLIOutput.print_welcome_message()
        print("\033[92mFetching data...\033[0m")
        try:
            raw_data = self.data_source.get_data()
            if raw_data is None or len(raw_data) == 0:
                print("\033[92mWarning: No data was returned by the data source. Check the time range or data provider.\033[0m")
                return pd.DataFrame()

            print(f"\033[92mSuccessfully fetched {len(raw_data)} records of raw data.\033[0m")
            return self._standardize_data(raw_data)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def _standardize_data(self, data):
        # Dev Note: this will standardize data to a common format.
        print("\033[92mStandardizing the raw data to a common format...\033[0m")

        df = pd.DataFrame(data)
        if df.empty:
            print("\033[92mWarning: DataFrame is empty after fetching data. Please verify the input parameters.\033[0m")
            return df

        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])

        print(f"\033[92mData has been standardized. Available data from \033[95m{df['Date'].min()}\033[0m to \033[95m{df['Date'].max()}\033[0m.\033[0m")
        return df

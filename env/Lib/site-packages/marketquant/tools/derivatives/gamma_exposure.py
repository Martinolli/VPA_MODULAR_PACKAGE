from datetime import datetime
import pandas as pd
from marketquant.tools.utils.bar_plotter import BarPlotter

class GammaExposure:
    def __init__(self, client):
        """
        Initializes the GammaExposure class with a Schwab API client.
        :param client: Initialized Schwab API client.
        """
        self.client = client

    @classmethod
    def run(cls, client, symbol, plot_strikes=50, barchart=False, netexposure=False, positive_color='blue', negative_color='red'):
        """
        Creates an instance and calculates gamma exposure in one step.
        :param client: Initialized Schwab API client.
        :param symbol: Stock ticker symbol (e.g., 'AAPL').
        :param plot_strikes: Number of unique strikes to include in the chart. All expiration dates are included for those strikes.
        :param barchart: Whether to plot the gamma exposure chart.
        :param netexposure: Whether to calculate net exposure (True) or separate puts and calls (False).
        :param positive_color: Color for positive exposure bars (default: 'blue').
        :param negative_color: Color for negative exposure bars (default: 'red').
        :return: Pandas DataFrame with gamma exposure data.
        """
        instance = cls(client)
        df = instance.get_gamma_exposure(symbol, plot_strikes, netexposure)

        # Plot the gamma exposure if barchart is True
        if barchart:
            plotter = BarPlotter(df, x_col="strikePrice", y_col="gammaExposure")
            if plotter.confirm_valid_data():
                plotter.plot_barchart(
                    positive_color=positive_color,
                    negative_color=negative_color,
                    title=f"Gamma Exposure for {symbol}"
                )
                print(df)

        return df

    def get_option_chains(self, symbol):
        """
        Fetches option chains for the given symbol.
        :param symbol: Stock ticker symbol (e.g., 'AAPL').
        :return: All option chains for the symbol.
        """
        option_chains_response = self.client.option_chains(
            symbol=symbol,
            contractType="ALL",
            includeUnderlyingQuote=True,
            strategy="SINGLE"
        ).json()

        # Fetching call and put option chains
        calls = option_chains_response.get('callExpDateMap', {})
        puts = option_chains_response.get('putExpDateMap', {})

        if not calls and not puts:
            raise ValueError(f"No valid option chains found for {symbol}")

        options = self.flatten_option_chain(calls, is_call=True) + self.flatten_option_chain(puts, is_call=False)

        return options, option_chains_response['underlyingPrice']

    def flatten_option_chain(self, option_chain, is_call):
        """
        Flattens the option chain dictionary into a list of option contracts.
        :param option_chain: The option chain dictionary from Schwab.
        :param is_call: Boolean to indicate whether it's a call option.
        :return: Flattened list of option contracts with correct gamma sign.
        """
        flattened = []
        for exp_date, strikes in option_chain.items():
            for strike_price, contracts in strikes.items():
                for contract in contracts:
                    contract['expirationDate'] = exp_date
                    contract['strikePrice'] = float(strike_price)
                    contract['gamma'] = contract.get('gamma', 0)  # Handle cases where gamma might be missing
                    contract['openInterest'] = contract.get('openInterest', 0)  # Ensure open interest is present
                    contract['gamma'] = contract['gamma'] if is_call else -contract['gamma']  # Adjust gamma sign for puts
                    flattened.append(contract)
        return flattened

    def calculate_gamma_exposure(self, option_chain, netexposure, spot_price, plot_strikes):
        """
        Calculates gamma exposure for each option in the chain.
        :param option_chain: List of option contracts.
        :param netexposure: Whether to calculate net gamma exposure per strike.
        :param spot_price: Current spot price of the underlying stock.
        :param plot_strikes: Number of strikes to include.
        :return: List of gamma exposures for each contract or net per strike.
        """
        gamma_exposures = {}
        unique_strikes = set()
        for option in option_chain:
            strike = option['strikePrice']
            gamma = option['gamma']
            open_interest = option['openInterest']
            contract_size = 100

            # Ensure we only process up to the required number of unique strikes
            if len(unique_strikes) < plot_strikes or strike in unique_strikes:
                unique_strikes.add(strike)

                # Calculate gamma exposure using the correct formula
                exposure = spot_price * gamma * open_interest * contract_size * spot_price * 0.01

                if netexposure:
                    # Sum the exposure at each strike
                    if strike in gamma_exposures:
                        gamma_exposures[strike] += exposure
                    else:
                        gamma_exposures[strike] = exposure
                else:
                    gamma_exposures.setdefault(strike, [])
                    gamma_exposures[strike].append({
                        'symbol': option['symbol'],
                        'strikePrice': strike,
                        'gammaExposure': exposure,
                        'expirationDate': option['expirationDate']
                    })

        if netexposure:
            return [{'strikePrice': strike, 'gammaExposure': exposure} for strike, exposure in gamma_exposures.items()]
        else:
            return [exp for exposures in gamma_exposures.values() for exp in exposures]

    def get_gamma_exposure(self, symbol, plot_strikes=50, netexposure=False):
        """
        Gets the gamma exposure for the specified symbol.
        :param symbol: Stock ticker symbol (e.g., 'AAPL').
        :param plot_strikes: Number of strikes closest to the money to include in the plot.
        :param netexposure: Whether to calculate net gamma exposure per strike.
        :return: Pandas DataFrame with gamma exposure data.
        """
        option_chain, spot_price = self.get_option_chains(symbol)

        gamma_exposure = self.calculate_gamma_exposure(option_chain, netexposure, spot_price, plot_strikes)

        df = pd.DataFrame(gamma_exposure)
        return df.sort_values(by='gammaExposure', ascending=False)

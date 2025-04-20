import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import pickle
from pathlib import Path
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import warnings

# Import VPA modules
from .vpa_config import VPAConfig
from .vpa_data import YFinanceProvider, MultiTimeframeProvider
from .vpa_processor import DataProcessor
from .vpa_analyzer import CandleAnalyzer, TrendAnalyzer, PatternRecognizer, SupportResistanceAnalyzer
from .vpa_signals import SignalGenerator
from .vpa_facade import VPAFacade
from .vpa_logger import VPALogger
from .vpa_utils import plot_candlestick, create_vpa_report

# Set up logger
logger = VPALogger()

class BacktestDataManager:
    """
    Manages historical data for backtesting, ensuring point-in-time analysis
    without look-ahead bias.
    """
    
    def __init__(self, data_provider=None, start_date=None, end_date=None, timeframes=None):
        """
        Initialize the BacktestDataManager.
        
        Args:
            data_provider: Data provider instance (default: YFinanceProvider)
            start_date: Start date for backtesting (str or datetime)
            end_date: End date for backtesting (str or datetime)
            timeframes: List of timeframes to use (default: from VPAConfig)
        """
        self.config = VPAConfig()
        self.data_provider = data_provider or YFinanceProvider()
        self.start_date = pd.to_datetime(start_date) if start_date else pd.to_datetime('2020-01-01')
        self.end_date = pd.to_datetime(end_date) if end_date else pd.to_datetime('2023-12-31')
        self.timeframes = timeframes or self.config.get_timeframes()
        self.data_cache = {}
        
        logger.info(f"Initialized BacktestDataManager with date range: {self.start_date} to {self.end_date}")
    
    def get_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for a ticker across all timeframes.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary of DataFrames with timeframes as keys
        """
        if ticker in self.data_cache:
            return self.data_cache[ticker]
        
        logger.info(f"Fetching historical data for {ticker} from {self.start_date} to {self.end_date}")
        
        data = {}
        for timeframe in self.timeframes:
            try:
                # Extract interval and period from timeframe dictionary
                interval = timeframe['interval']
                period = timeframe.get('period', '1y')  # Default to 1y if period not specified
                
                # Call data provider with correct parameters
                price_data, volume_data = self.data_provider.get_data(
                    ticker, 
                    interval=interval, 
                    period=period,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                # Combine price and volume data
                df = price_data.copy()
                df['volume'] = volume_data
                
                if df is not None and not df.empty:
                    data[interval] = df
                else:
                    logger.warning(f"No data available for {ticker} on {interval} timeframe")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker} on {timeframe} timeframe: {str(e)}")
        
        if not data:
            logger.error(f"Failed to fetch any data for {ticker}")
            return {}
        
        self.data_cache[ticker] = data
        return data

    def get_data_window(self, ticker: str, current_date: Union[str, pd.Timestamp], 
                    lookback_days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Get a window of historical data up to a specific date.
        This ensures point-in-time analysis without look-ahead bias.
        
        Args:
            ticker: Ticker symbol
            current_date: The current date in the backtest
            lookback_days: Number of days to look back for analysis
            
        Returns:
            Dictionary of DataFrames with timeframes as keys, limited to data
            available up to current_date
        """
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        start_date = current_date - pd.Timedelta(days=lookback_days)
        
        # Get full data if not in cache
        if ticker not in self.data_cache:
            try:
                self.get_data(ticker)
            except Exception as e:
                logger.error(f"Error retrieving data for {ticker}: {str(e)}")
                return {}
        
        if ticker not in self.data_cache:
            logger.error(f"No data available for {ticker}")
            return {}
    
        # Create a window for each timeframe
        windowed_data = {}
        for timeframe, df in self.data_cache[ticker].items():
            try:
                # Filter data up to current_date
                window = df[df.index <= current_date].copy()
                if not window.empty:
                    windowed_data[timeframe] = window
            except Exception as e:
                logger.error(f"Error creating window for {ticker} on {timeframe}: {str(e)}")
        
        return windowed_data
    
    def get_trade_dates(self, ticker: str, timeframe: str = '1d') -> List[pd.Timestamp]:
        """
        Get a list of all trading dates for a ticker.
        
        Args:
            ticker: Ticker symbol
            timeframe: Timeframe to use for date extraction
            
        Returns:
            List of trading dates as pandas Timestamps
        """
        if ticker not in self.data_cache:
            self.get_data(ticker)
        
        if ticker not in self.data_cache or timeframe not in self.data_cache[ticker]:
            logger.error(f"No data available for {ticker} on {timeframe} timeframe")
            return []
        
        return self.data_cache[ticker][timeframe].index.tolist()
    
    def split_data(self, ticker: str, train_ratio: float = 0.7, 
                  validation_ratio: float = 0.15) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            ticker: Ticker symbol
            train_ratio: Ratio of data to use for training
            validation_ratio: Ratio of data to use for validation
            
        Returns:
            Dictionary with 'train', 'validation', and 'test' keys, each containing
            a dictionary of DataFrames with timeframes as keys
        """
        if ticker not in self.data_cache:
            self.get_data(ticker)
        
        if ticker not in self.data_cache:
            logger.error(f"No data available for {ticker}")
            return {'train': {}, 'validation': {}, 'test': {}}
        
        result = {'train': {}, 'validation': {}, 'test': {}}
        
        for timeframe, df in self.data_cache[ticker].items():
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + validation_ratio))
            
            result['train'][timeframe] = df.iloc[:train_end].copy()
            result['validation'][timeframe] = df.iloc[train_end:val_end].copy()
            result['test'][timeframe] = df.iloc[val_end:].copy()
        
        return result


class TradeSimulator:
    """
    Simulates trade execution based on VPA signals, accounting for
    slippage, commissions, and position sizing.
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.001, 
                 slippage_percent: float = 0.001,
                 risk_per_trade: float = 0.02,
                 max_positions: int = 5):
        """
        Initialize the TradeSimulator.
        
        Args:
            initial_capital: Starting capital for the simulation
            commission_rate: Commission rate as a percentage of trade value
            slippage_percent: Slippage as a percentage of price
            risk_per_trade: Risk per trade as a percentage of capital
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_percent = slippage_percent
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        
        self.positions = {}  # ticker -> {entry_price, shares, entry_date, stop_loss}
        self.closed_trades = []
        self.equity_curve = []
        self.trade_history = []
        
        logger.info(f"Initialized TradeSimulator with {initial_capital} capital")
    
    def calculate_position_size(self, price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk per trade.
        
        Args:
            price: Current price
            stop_loss: Stop loss price
            
        Returns:
            Number of shares to buy
        """
        if price <= stop_loss:
            logger.warning(f"Invalid stop loss: price={price}, stop_loss={stop_loss}")
            return 0
        
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share == 0:
            logger.warning("Risk per share is zero, using 1% of price as risk")
            risk_per_share = price * 0.01
        
        shares = int(risk_amount / risk_per_share)
        
        # Check if we have enough capital
        cost = shares * price * (1 + self.commission_rate)
        if cost > self.capital:
            shares = int(self.capital / (price * (1 + self.commission_rate)))
        
        return max(1, shares)  # Ensure at least 1 share
    
    def enter_position(self, ticker: str, date: pd.Timestamp, price: float, 
                      signal_type: str, stop_loss: float, take_profit: float = None) -> bool:
        """
        Enter a new position.
        
        Args:
            ticker: Ticker symbol
            date: Entry date
            price: Entry price
            signal_type: Type of signal (BUY or SELL)
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            
        Returns:
            True if position was entered, False otherwise
        """
        # Check if we already have this position
        if ticker in self.positions:
            logger.info(f"Already have position in {ticker}, skipping")
            return False
        
        # Check if we have reached max positions
        if len(self.positions) >= self.max_positions:
            logger.info(f"Maximum positions ({self.max_positions}) reached, skipping {ticker}")
            return False
        
        # Apply slippage
        if signal_type == "BUY":
            adjusted_price = price * (1 + self.slippage_percent)
        else:  # SELL (short)
            adjusted_price = price * (1 - self.slippage_percent)
        
        # Calculate position size
        shares = self.calculate_position_size(adjusted_price, stop_loss)
        
        if shares == 0:
            logger.warning(f"Could not calculate valid position size for {ticker}")
            return False
        
        # Calculate cost
        cost = shares * adjusted_price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        if total_cost > self.capital:
            logger.warning(f"Not enough capital to enter position in {ticker}")
            return False
        
        # Update capital
        self.capital -= total_cost
        
        # Record position
        self.positions[ticker] = {
            'entry_price': adjusted_price,
            'shares': shares,
            'entry_date': date,
            'signal_type': signal_type,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'cost': total_cost
        }
        
        # Record trade
        self.trade_history.append({
            'ticker': ticker,
            'action': 'ENTER',
            'date': date,
            'price': adjusted_price,
            'shares': shares,
            'value': cost,
            'commission': commission,
            'signal_type': signal_type,
            'capital': self.capital
        })
        
        logger.info(f"Entered {signal_type} position in {ticker}: {shares} shares at {adjusted_price}")
        return True
    
    def exit_position(self, ticker: str, date: pd.Timestamp, price: float, reason: str) -> bool:
        """
        Exit an existing position.
        
        Args:
            ticker: Ticker symbol
            date: Exit date
            price: Exit price
            reason: Reason for exit (STOP_LOSS, TAKE_PROFIT, SIGNAL_CHANGE, etc.)
            
        Returns:
            True if position was exited, False otherwise
        """
        if ticker not in self.positions:
            logger.warning(f"No position in {ticker} to exit")
            return False
        
        position = self.positions[ticker]
        
        # Apply slippage
        if position['signal_type'] == "BUY":
            adjusted_price = price * (1 - self.slippage_percent)
        else:  # SELL (short)
            adjusted_price = price * (1 + self.slippage_percent)
        
        # Calculate proceeds
        shares = position['shares']
        proceeds = shares * adjusted_price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission
        
        # Calculate profit/loss
        if position['signal_type'] == "BUY":
            pl = net_proceeds - position['cost']
        else:  # SELL (short)
            pl = position['cost'] - net_proceeds
        
        # Update capital
        self.capital += net_proceeds
        
        # Record closed trade
        trade_duration = (date - position['entry_date']).days
        self.closed_trades.append({
            'ticker': ticker,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': adjusted_price,
            'shares': shares,
            'pl': pl,
            'pl_percent': (pl / position['cost']) * 100,
            'signal_type': position['signal_type'],
            'exit_reason': reason,
            'duration': trade_duration
        })
        
        # Record trade history
        self.trade_history.append({
            'ticker': ticker,
            'action': 'EXIT',
            'date': date,
            'price': adjusted_price,
            'shares': shares,
            'value': proceeds,
            'commission': commission,
            'reason': reason,
            'pl': pl,
            'capital': self.capital
        })
        
        # Remove position
        del self.positions[ticker]
        
        logger.info(f"Exited position in {ticker}: {shares} shares at {adjusted_price}, P/L: {pl:.2f}")
        return True
    
    def update_positions(self, date: pd.Timestamp, ticker_prices: Dict[str, float]) -> None:
        """
        Update positions with current prices and check for stop loss/take profit.
        
        Args:
            date: Current date
            ticker_prices: Dictionary of current prices for each ticker
        """
        for ticker, position in list(self.positions.items()):
            if ticker not in ticker_prices:
                logger.warning(f"No price data for {ticker}, skipping update")
                continue
            
            current_price = ticker_prices[ticker]
            
            # Check stop loss
            if position['signal_type'] == "BUY" and current_price <= position['stop_loss']:
                logger.info(f"Stop loss triggered for {ticker}")
                self.exit_position(ticker, date, position['stop_loss'], "STOP_LOSS")
            elif position['signal_type'] == "SELL" and current_price >= position['stop_loss']:
                logger.info(f"Stop loss triggered for {ticker}")
                self.exit_position(ticker, date, position['stop_loss'], "STOP_LOSS")
            
            # Check take profit
            elif position['take_profit'] is not None:
                if position['signal_type'] == "BUY" and current_price >= position['take_profit']:
                    logger.info(f"Take profit triggered for {ticker}")
                    self.exit_position(ticker, date, position['take_profit'], "TAKE_PROFIT")
                elif position['signal_type'] == "SELL" and current_price <= position['take_profit']:
                    logger.info(f"Take profit triggered for {ticker}")
                    self.exit_position(ticker, date, position['take_profit'], "TAKE_PROFIT")
    
    def update_equity_curve(self, date: pd.Timestamp, ticker_prices: Dict[str, float]) -> None:
        """
        Update the equity curve with current portfolio value.
        
        Args:
            date: Current date
            ticker_prices: Dictionary of current prices for each ticker
        """
        # Calculate current portfolio value
        portfolio_value = self.capital
        
        for ticker, position in self.positions.items():
            if ticker in ticker_prices:
                current_price = ticker_prices[ticker]
                position_value = position['shares'] * current_price
                portfolio_value += position_value
        
        # Record equity curve
        self.equity_curve.append({
            'date': date,
            'equity': portfolio_value,
            'cash': self.capital,
            'positions': len(self.positions)
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the backtest.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.equity_curve:
            logger.warning("No equity curve data to calculate performance metrics")
            return {}
        
        # Create equity curve dataframe
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        equity_df['return'] = equity_df['equity'].pct_change().fillna(0)
        
        # Calculate metrics
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        
        total_return = (final_equity / initial_equity) - 1
        
        # Calculate annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate volatility
        daily_returns = equity_df['return']
        annualized_volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Calculate drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = 1 - (equity_df['equity'] / equity_df['cummax'])
        max_drawdown = equity_df['drawdown'].max()
        
        # Trade metrics
        if self.closed_trades:
            trades_df = pd.DataFrame(self.closed_trades)
            winning_trades = trades_df[trades_df['pl'] > 0]
            losing_trades = trades_df[trades_df['pl'] < 0]
            
            num_trades = len(trades_df)
            num_winning = len(winning_trades)
            num_losing = len(losing_trades)
            
            win_rate = num_winning / num_trades if num_trades > 0 else 0
            
            avg_win = winning_trades['pl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pl'].mean() if not losing_trades.empty else 0
            
            profit_factor = abs(winning_trades['pl'].sum() / losing_trades['pl'].sum()) if not losing_trades.empty and losing_trades['pl'].sum() != 0 else float('inf')
            
            avg_trade_duration = trades_df['duration'].mean()
        else:
            num_trades = 0
            num_winning = 0
            num_losing = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        return {
            'initial_capital': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': num_trades,
            'num_winning': num_winning,
            'num_losing': num_losing,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration
        }


class VPABacktester:
    """
    Main backtesting class that integrates data management, VPA analysis,
    and trade simulation to evaluate the VPA strategy.
    """
    
    def __init__(self, 
                 start_date: Union[str, pd.Timestamp] = None,
                 end_date: Union[str, pd.Timestamp] = None,
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_percent: float = 0.001,
                 risk_per_trade: float = 0.02,
                 max_positions: int = 5,
                 timeframes: List[str] = None,
                 vpa_config: VPAConfig = None,
                 benchmark_ticker: str = "SPY",
                 position_sizing_method: str = "fixed_risk"):
        """
        Initialize the VPABacktester.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Starting capital for the simulation
            commission_rate: Commission rate as a percentage of trade value
            slippage_percent: Slippage as a percentage of price
            risk_per_trade: Risk per trade as a percentage of capital
            max_positions: Maximum number of concurrent positions
            timeframes: List of timeframes to use
            vpa_config: VPA configuration instance
            benchmark_ticker: Ticker symbol for benchmark comparison (default: SPY)
            position_sizing_method: Method for position sizing ('fixed_risk', 'kelly', 'volatility')
        """
        self.config = vpa_config or VPAConfig()
        self.timeframes = timeframes or self.config.get_timeframes()
        self.benchmark_ticker = benchmark_ticker
        self.position_sizing_method = position_sizing_method
        
        # Initialize components
        self.data_manager = BacktestDataManager(
            start_date=start_date,
            end_date=end_date,
            timeframes=self.timeframes
        )
        
        self.trade_simulator = TradeSimulator(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_percent=slippage_percent,
            risk_per_trade=risk_per_trade,
            max_positions=max_positions
        )
        
        # Initialize VPA components
        self.data_processor = DataProcessor(self.config)
        self.candle_analyzer = CandleAnalyzer(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config)
        self.pattern_recognizer = PatternRecognizer(self.config)
        self.support_resistance_analyzer = SupportResistanceAnalyzer(self.config)
        self.signal_generator = SignalGenerator(self.config)
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized VPABacktester with date range: {start_date} to {end_date}")
    
    def analyze_at_date(self, ticker: str, current_date: pd.Timestamp, 
                       lookback_days: int = 365) -> Dict[str, Any]:
        """
        Perform VPA analysis at a specific date.
        
        Args:
            ticker: Ticker symbol
            current_date: Current date in the backtest
            lookback_days: Number of days to look back for analysis
            
        Returns:
            Dictionary with analysis results
        """
        # Get data window up to current_date
        data_window = self.data_manager.get_data_window(ticker, current_date, lookback_days)
        
        if not data_window:
            logger.warning(f"No data available for {ticker} at {current_date}")
            return None
        
        # Process data for each timeframe
        processed_data = {}
        for timeframe, df in data_window.items():
            processed_data[timeframe] = self.data_processor.preprocess_data(df)
        
        # Analyze each timeframe
        timeframe_results = {}
        for timeframe, data in processed_data.items():
            # Skip if not enough data
            if len(data['price']) < 20:  # Minimum data required
                logger.warning(f"Not enough data for {ticker} on {timeframe} timeframe")
                continue
            
            # Get the latest data point for analysis
            idx = len(data['price']) - 1
            
            # Analyze candle
            candle_result = self.candle_analyzer.analyze_candle(data, idx)
            
            # Analyze trend
            trend_result = self.trend_analyzer.analyze_trend(data, idx)
            
            # Recognize patterns
            pattern_result = self.pattern_recognizer.recognize_patterns(data, idx)
            
            # Analyze support/resistance
            sr_result = self.support_resistance_analyzer.analyze_levels(data, idx)
            
            # Combine results
            timeframe_results[timeframe] = {
                'candle': candle_result,
                'trend': trend_result,
                'patterns': pattern_result,
                'support_resistance': sr_result,
                'price_data': data['price'].iloc[-1].to_dict(),
                'volume_data': data['volume'].iloc[-1]
            }
        
        # Generate signal based on all timeframes
        signal_result = self.signal_generator.generate_signal(timeframe_results)
        
        # Add risk assessment
        if signal_result['type'] != "NO_ACTION":
            risk_result = self.signal_generator.assess_risk(
                ticker, 
                timeframe_results, 
                signal_result
            )
        else:
            risk_result = None
        
        return {
            'ticker': ticker,
            'date': current_date,
            'timeframes': timeframe_results,
            'signal': signal_result,
            'risk': risk_result
        }
    
    def run_backtest(self, tickers: Union[str, List[str]], 
                    frequency: str = '1d',
                    rebalance_frequency: str = '1w',
                    lookback_days: int = 365,
                    include_benchmark: bool = True) -> Dict[str, Any]:
        """
        Run a backtest for one or more tickers.
        
        Args:
            tickers: Ticker symbol or list of ticker symbols
            frequency: Frequency to check for signals ('1d', '1w', etc.)
            rebalance_frequency: Frequency to rebalance portfolio
            lookback_days: Number of days to look back for analysis
            include_benchmark: Whether to include benchmark comparison
            
        Returns:
            Dictionary with backtest results
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        logger.info(f"Starting backtest for {tickers}")
        
        # Get all dates for the backtest
        all_dates = []
        for ticker in tickers:
            dates = self.data_manager.get_trade_dates(ticker)
            all_dates.extend(dates)
        
        all_dates = sorted(list(set(all_dates)))
        
        if not all_dates:
            logger.error("No dates available for backtest")
            return {}
            
        # Get benchmark data if requested
        benchmark_data = None
        if include_benchmark and self.benchmark_ticker:
            logger.info(f"Fetching benchmark data for {self.benchmark_ticker}")
            try:
                benchmark_data = self.data_manager.get_data(self.benchmark_ticker)
                if not benchmark_data or '1d' not in benchmark_data:
                    logger.warning(f"Could not retrieve benchmark data for {self.benchmark_ticker}")
                    benchmark_data = None
            except Exception as e:
                logger.error(f"Error fetching benchmark data: {str(e)}")
                benchmark_data = None
        
        # Convert frequency to pandas frequency string
        freq_map = {'1d': 'B', '1w': 'W-FRI', '1m': 'MS'}
        pd_freq = freq_map.get(frequency, 'B')  # Default to business day
        
        # Get dates to check for signals based on frequency
        if pd_freq == 'B':  # Business day
            signal_dates = all_dates
        else:
            # Resample to get dates at the specified frequency
            date_series = pd.Series(index=all_dates, data=1)
            resampled = date_series.resample(pd_freq).first()
            signal_dates = resampled.index.tolist()
        
        # Filter signal dates to be within the data range
        signal_dates = [d for d in signal_dates if d >= all_dates[0] and d <= all_dates[-1]]
        
        logger.info(f"Running backtest on {len(signal_dates)} signal dates")
        
        # Run the backtest
        for current_date in signal_dates:
            logger.info(f"Processing date: {current_date}")
            
            # Get current prices for all tickers
            ticker_prices = {}
            for ticker in tickers:
                data = self.data_manager.get_data_window(ticker, current_date, 1)
                if data and '1d' in data and not data['1d'].empty:
                    ticker_prices[ticker] = data['1d']['close'].iloc[-1]
            
            # Update positions with current prices
            self.trade_simulator.update_positions(current_date, ticker_prices)
            
            # Update equity curve
            self.trade_simulator.update_equity_curve(current_date, ticker_prices)
            
            # Check for new signals
            for ticker in tickers:
                # Skip if we already have a position in this ticker
                if ticker in self.trade_simulator.positions:
                    continue
                
                # Analyze ticker at current date
                analysis = self.analyze_at_date(ticker, current_date, lookback_days)
                
                if analysis is None:
                    continue
                
                signal = analysis['signal']
                risk = analysis['risk']
                
                # Process signal
                if signal['type'] == "BUY" and signal['strength'] >= 0.7:  # Only take strong signals
                    if risk and 'stop_loss' in risk and risk['stop_loss'] > 0:
                        current_price = ticker_prices.get(ticker)
                        if current_price:
                            self.trade_simulator.enter_position(
                                ticker=ticker,
                                date=current_date,
                                price=current_price,
                                signal_type="BUY",
                                stop_loss=risk['stop_loss'],
                                take_profit=risk.get('take_profit')
                            )
                elif signal['type'] == "SELL" and signal['strength'] >= 0.7:  # Only take strong signals
                    if risk and 'stop_loss' in risk and risk['stop_loss'] > 0:
                        current_price = ticker_prices.get(ticker)
                        if current_price:
                            self.trade_simulator.enter_position(
                                ticker=ticker,
                                date=current_date,
                                price=current_price,
                                signal_type="SELL",
                                stop_loss=risk['stop_loss'],
                                take_profit=risk.get('take_profit')
                            )
        
        # Close all remaining positions at the end of the backtest
        final_date = all_dates[-1]
        for ticker in list(self.trade_simulator.positions.keys()):
            if ticker in ticker_prices:
                self.trade_simulator.exit_position(
                    ticker=ticker,
                    date=final_date,
                    price=ticker_prices[ticker],
                    reason="END_OF_BACKTEST"
                )
        
        # Calculate performance metrics
        performance = self.trade_simulator.get_performance_metrics()
        
        # Calculate benchmark performance if available
        benchmark_performance = None
        if benchmark_data is not None and '1d' in benchmark_data:
            benchmark_df = benchmark_data['1d']
            # Filter benchmark data to match backtest period
            benchmark_df = benchmark_df[(benchmark_df.index >= all_dates[0]) & 
                                       (benchmark_df.index <= all_dates[-1])]
            
            if not benchmark_df.empty:
                # Calculate benchmark returns
                benchmark_returns = benchmark_df['close'].pct_change().dropna()
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                
                # Calculate benchmark metrics
                benchmark_total_return = benchmark_cumulative.iloc[-1] - 1 if len(benchmark_cumulative) > 0 else 0
                benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1 if len(benchmark_returns) > 0 else 0
                benchmark_volatility = benchmark_returns.std() * np.sqrt(252) if len(benchmark_returns) > 0 else 0
                benchmark_sharpe = benchmark_annualized_return / benchmark_volatility if benchmark_volatility > 0 else 0
                
                # Calculate drawdown
                benchmark_drawdown = 1 - benchmark_df['close'] / benchmark_df['close'].cummax()
                benchmark_max_drawdown = benchmark_drawdown.max()
                
                benchmark_performance = {
                    'total_return': benchmark_total_return,
                    'total_return_pct': benchmark_total_return * 100,
                    'annualized_return': benchmark_annualized_return,
                    'annualized_return_pct': benchmark_annualized_return * 100,
                    'annualized_volatility': benchmark_volatility,
                    'sharpe_ratio': benchmark_sharpe,
                    'max_drawdown': benchmark_max_drawdown,
                    'max_drawdown_pct': benchmark_max_drawdown * 100
                }
                
                # Calculate alpha and beta
                if len(self.trade_simulator.equity_curve) > 0:
                    # Create equity curve dataframe
                    equity_df = pd.DataFrame(self.trade_simulator.equity_curve)
                    equity_df.set_index('date', inplace=True)
                    
                    # Calculate strategy returns
                    equity_df['return'] = equity_df['equity'].pct_change().fillna(0)
                    
                    # Align benchmark returns with strategy returns
                    aligned_returns = pd.DataFrame({
                        'strategy': equity_df['return'],
                        'benchmark': benchmark_returns
                    }).dropna()
                    
                    if len(aligned_returns) > 0:
                        # Calculate beta (using covariance and variance)
                        beta = aligned_returns.cov().loc['strategy', 'benchmark'] / aligned_returns['benchmark'].var() if aligned_returns['benchmark'].var() > 0 else 0
                        
                        # Calculate alpha (annualized)
                        rf_rate = 0.02  # Assume 2% risk-free rate
                        alpha = performance['annualized_return'] - (rf_rate + beta * (benchmark_annualized_return - rf_rate))
                        
                        # Add to benchmark performance
                        benchmark_performance['alpha'] = alpha
                        benchmark_performance['alpha_pct'] = alpha * 100
                        benchmark_performance['beta'] = beta
        
        # Store results
        self.results = {
            'tickers': tickers,
            'start_date': all_dates[0],
            'end_date': all_dates[-1],
            'performance': performance,
            'benchmark_performance': benchmark_performance,
            'benchmark_ticker': self.benchmark_ticker if benchmark_performance else None,
            'equity_curve': self.trade_simulator.equity_curve,
            'trades': self.trade_simulator.closed_trades,
            'trade_history': self.trade_simulator.trade_history
        }
        
        logger.info(f"Backtest completed with {performance['num_trades']} trades")
        logger.info(f"Final equity: {performance['final_equity']:.2f}, Return: {performance['total_return_pct']:.2f}%")
        
        return self.results
    
    def plot_equity_curve(self, figsize=(12, 6), save_path=None, include_benchmark=True):
        """
        Plot the equity curve from the backtest.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            include_benchmark: Whether to include benchmark comparison
            
        Returns:
            Matplotlib figure
        """
        if not self.results or 'equity_curve' not in self.results:
            logger.warning("No backtest results to plot")
            return None
        
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df.set_index('date', inplace=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        ax.plot(equity_df.index, equity_df['equity'], label='VPA Strategy', linewidth=2, color='blue')
        
        # Add benchmark if available
        if include_benchmark and 'benchmark_performance' in self.results and self.results['benchmark_performance']:
            benchmark_ticker = self.results.get('benchmark_ticker', 'Benchmark')
            try:
                # Get benchmark data
                benchmark_data = self.data_manager.get_data(benchmark_ticker)
                if benchmark_data and '1d' in benchmark_data:
                    benchmark_df = benchmark_data['1d']
                    # Filter to match backtest period
                    benchmark_df = benchmark_df[(benchmark_df.index >= equity_df.index[0]) & 
                                              (benchmark_df.index <= equity_df.index[-1])]
                    
                    if not benchmark_df.empty:
                        # Normalize benchmark to match initial capital
                        initial_price = benchmark_df['close'].iloc[0]
                        benchmark_equity = benchmark_df['close'] / initial_price * self.trade_simulator.initial_capital
                        
                        # Plot benchmark
                        ax.plot(benchmark_df.index, benchmark_equity, label=f'{benchmark_ticker}', 
                               linewidth=2, color='green', alpha=0.7, linestyle='--')
            except Exception as e:
                logger.warning(f"Error plotting benchmark: {str(e)}")
        
        # Add initial capital line
        ax.axhline(y=self.trade_simulator.initial_capital, color='r', linestyle='--', 
                  label=f'Initial Capital ({self.trade_simulator.initial_capital:,.0f})')
        
        # Format the plot
        ax.set_title('Equity Curve', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        
        return fig
    
    def plot_drawdown(self, figsize=(12, 6), save_path=None, include_benchmark=True):
        """
        Plot the drawdown from the backtest with optional benchmark comparison.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            include_benchmark: Whether to include benchmark comparison
            
        Returns:
            Matplotlib figure
        """
        if not self.results or 'equity_curve' not in self.results:
            logger.warning("No backtest results to plot")
            return None
        
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df.set_index('date', inplace=True)
        
        # Calculate drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['cummax']) - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdown
        ax.fill_between(equity_df.index, 0, equity_df['drawdown'] * 100, color='r', alpha=0.3, label='VPA Strategy')
        ax.plot(equity_df.index, equity_df['drawdown'] * 100, color='r', linewidth=1)
        
        # Add benchmark if available
        if include_benchmark and 'benchmark_performance' in self.results and self.results['benchmark_performance']:
            benchmark_ticker = self.results.get('benchmark_ticker', 'Benchmark')
            try:
                # Get benchmark data
                benchmark_data = self.data_manager.get_data(benchmark_ticker)
                if benchmark_data and '1d' in benchmark_data:
                    benchmark_df = benchmark_data['1d']
                    # Filter to match backtest period
                    benchmark_df = benchmark_df[(benchmark_df.index >= equity_df.index[0]) & 
                                              (benchmark_df.index <= equity_df.index[-1])]
                    
                    if not benchmark_df.empty:
                        # Calculate benchmark drawdown
                        benchmark_df['cummax'] = benchmark_df['close'].cummax()
                        benchmark_df['drawdown'] = (benchmark_df['close'] / benchmark_df['cummax']) - 1
                        
                        # Plot benchmark drawdown
                        ax.plot(benchmark_df.index, benchmark_df['drawdown'] * 100, 
                               color='g', linewidth=1.5, linestyle='--', 
                               label=f'{benchmark_ticker} Drawdown')
            except Exception as e:
                logger.warning(f"Error plotting benchmark drawdown: {str(e)}")
        
        # Format the plot
        ax.set_title('Drawdown', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(bottom=min(equity_df['drawdown'].min() * 100 * 1.1, -5), top=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Drawdown plot saved to {save_path}")
        
        return fig
    
    def plot_trade_analysis(self, figsize=(15, 10), save_path=None):
        """
        Plot trade analysis charts.
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if not self.results or 'trades' not in self.results or not self.results['trades']:
            logger.warning("No trade data to plot")
            return None
        
        trades_df = pd.DataFrame(self.results['trades'])
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Trade P/L distribution
        axs[0, 0].hist(trades_df['pl'], bins=20, color='skyblue', edgecolor='black')
        axs[0, 0].axvline(x=0, color='r', linestyle='--')
        axs[0, 0].set_title('Trade P/L Distribution')
        axs[0, 0].set_xlabel('Profit/Loss ($)')
        axs[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Trade P/L by ticker
        if len(trades_df['ticker'].unique()) <= 10:  # Only if not too many tickers
            ticker_pl = trades_df.groupby('ticker')['pl'].sum().sort_values()
            ticker_pl.plot(kind='barh', ax=axs[0, 1], color='lightgreen')
            axs[0, 1].set_title('Total P/L by Ticker')
            axs[0, 1].set_xlabel('Profit/Loss ($)')
            axs[0, 1].grid(axis='x', alpha=0.3)
        else:
            # Too many tickers, show top 10 and bottom 10
            ticker_pl = trades_df.groupby('ticker')['pl'].sum().sort_values()
            top_bottom = pd.concat([ticker_pl.head(5), ticker_pl.tail(5)])
            top_bottom.plot(kind='barh', ax=axs[0, 1], color='lightgreen')
            axs[0, 1].set_title('Top/Bottom 5 Tickers by P/L')
            axs[0, 1].set_xlabel('Profit/Loss ($)')
            axs[0, 1].grid(axis='x', alpha=0.3)
        
        # Plot 3: Win/Loss by exit reason
        exit_counts = trades_df.groupby(['exit_reason', trades_df['pl'] > 0]).size().unstack()
        if exit_counts.shape[1] == 2:  # Ensure we have both True and False columns
            exit_counts.columns = ['Loss', 'Win']
            exit_counts.plot(kind='bar', ax=axs[1, 0], stacked=True, color=['salmon', 'lightgreen'])
            axs[1, 0].set_title('Win/Loss by Exit Reason')
            axs[1, 0].set_xlabel('Exit Reason')
            axs[1, 0].set_ylabel('Count')
            axs[1, 0].legend(loc='upper right')
        else:
            axs[1, 0].text(0.5, 0.5, 'Insufficient data for exit reason analysis', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[1, 0].transAxes)
            axs[1, 0].set_title('Win/Loss by Exit Reason')
        
        # Plot 4: Trade duration vs P/L
        axs[1, 1].scatter(trades_df['duration'], trades_df['pl'], 
                         c=trades_df['pl'].apply(lambda x: 'g' if x > 0 else 'r'),
                         alpha=0.6)
        axs[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[1, 1].set_title('Trade Duration vs P/L')
        axs[1, 1].set_xlabel('Duration (days)')
        axs[1, 1].set_ylabel('Profit/Loss ($)')
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Trade analysis plot saved to {save_path}")
        
        return fig
    
    def create_backtest_report(self, output_dir: str) -> Dict[str, str]:
        """
        Create a comprehensive backtest report.
        
        Args:
            output_dir: Directory to save the report files
            
        Returns:
            Dictionary with paths to report files
        """
        if not self.results:
            logger.warning("No backtest results to report")
            return {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        report_files = {}
        
        # Save performance metrics
        performance_path = os.path.join(output_dir, "performance_metrics.json")
        with open(performance_path, 'w') as f:
            json.dump(self.results['performance'], f, indent=4)
        report_files['performance'] = performance_path
        
        # Save trades
        trades_path = os.path.join(output_dir, "trades.csv")
        trades_df = pd.DataFrame(self.results['trades'])
        trades_df.to_csv(trades_path, index=False)
        report_files['trades'] = trades_path
        
        # Save equity curve
        equity_path = os.path.join(output_dir, "equity_curve.csv")
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df.to_csv(equity_path, index=False)
        report_files['equity'] = equity_path
        
        # Create plots
        equity_plot_path = os.path.join(output_dir, "equity_curve.png")
        self.plot_equity_curve(save_path=equity_plot_path)
        report_files['equity_plot'] = equity_plot_path
        
        drawdown_plot_path = os.path.join(output_dir, "drawdown.png")
        self.plot_drawdown(save_path=drawdown_plot_path)
        report_files['drawdown_plot'] = drawdown_plot_path
        
        if self.results['trades']:
            trade_plot_path = os.path.join(output_dir, "trade_analysis.png")
            self.plot_trade_analysis(save_path=trade_plot_path)
            report_files['trade_plot'] = trade_plot_path
        
        # Create HTML report
        html_path = os.path.join(output_dir, "backtest_report.html")
        self._create_html_report(html_path)
        report_files['html'] = html_path
        
        logger.info(f"Backtest report created in {output_dir}")
        
        return report_files
    
    def _create_html_report(self, html_path: str) -> None:
        """
        Create an HTML report of the backtest results.
        
        Args:
            html_path: Path to save the HTML report
        """
        if not self.results:
            logger.warning("No backtest results for HTML report")
            return
        
        # Get performance metrics
        perf = self.results['performance']
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VPA Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
                .metric-card {{ background-color: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .metric-name {{ font-size: 14px; color: #666; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart {{ margin: 30px 0; text-align: center; }}
                .chart img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>VPA Backtest Report</h1>
                
                <div>
                    <p><strong>Tickers:</strong> {', '.join(self.results['tickers'])}</p>
                    <p><strong>Period:</strong> {self.results['start_date'].strftime('%Y-%m-%d')} to {self.results['end_date'].strftime('%Y-%m-%d')}</p>
                </div>
                
                <h2>Performance Summary</h2>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-name">Total Return</div>
                        <div class="metric-value {'positive' if perf['total_return'] >= 0 else 'negative'}">{perf['total_return_pct']:.2f}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Annualized Return</div>
                        <div class="metric-value {'positive' if perf['annualized_return'] >= 0 else 'negative'}">{perf['annualized_return_pct']:.2f}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Sharpe Ratio</div>
                        <div class="metric-value {'positive' if perf['sharpe_ratio'] >= 1 else 'negative'}">{perf['sharpe_ratio']:.2f}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Max Drawdown</div>
                        <div class="metric-value negative">{perf['max_drawdown_pct']:.2f}%</div>
                    </div>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-name">Win Rate</div>
                        <div class="metric-value {'positive' if perf['win_rate'] >= 0.5 else 'negative'}">{perf['win_rate_pct']:.2f}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Profit Factor</div>
                        <div class="metric-value {'positive' if perf['profit_factor'] >= 1 else 'negative'}">{perf['profit_factor']:.2f}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Total Trades</div>
                        <div class="metric-value">{perf['num_trades']}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Avg Trade Duration</div>
                        <div class="metric-value">{perf['avg_trade_duration']:.1f} days</div>
                    </div>
                </div>
                
                <h2>Equity Curve</h2>
                <div class="chart">
                    <img src="equity_curve.png" alt="Equity Curve">
                </div>
                
                <h2>Drawdown</h2>
                <div class="chart">
                    <img src="drawdown.png" alt="Drawdown">
                </div>
        """
        
        # Add trade analysis chart if available
        if self.results['trades']:
            html_content += """
                <h2>Trade Analysis</h2>
                <div class="chart">
                    <img src="trade_analysis.png" alt="Trade Analysis">
                </div>
            """
        
        # Add trade table
        if self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_df = trades_df.sort_values('exit_date', ascending=False)
            
            # Format the trades table
            trades_html = "<h2>Recent Trades</h2><table>"
            trades_html += "<tr><th>Ticker</th><th>Signal</th><th>Entry Date</th><th>Exit Date</th><th>Entry Price</th><th>Exit Price</th><th>P/L</th><th>P/L %</th><th>Exit Reason</th></tr>"
            
            # Show the most recent 20 trades
            for _, trade in trades_df.head(20).iterrows():
                pl_class = "positive" if trade['pl'] > 0 else "negative"
                trades_html += f"""
                <tr>
                    <td>{trade['ticker']}</td>
                    <td>{trade['signal_type']}</td>
                    <td>{trade['entry_date'].strftime('%Y-%m-%d')}</td>
                    <td>{trade['exit_date'].strftime('%Y-%m-%d')}</td>
                    <td>${trade['entry_price']:.2f}</td>
                    <td>${trade['exit_price']:.2f}</td>
                    <td class="{pl_class}">${trade['pl']:.2f}</td>
                    <td class="{pl_class}">{trade['pl_percent']:.2f}%</td>
                    <td>{trade['exit_reason']}</td>
                </tr>
                """
            
            trades_html += "</table>"
            html_content += trades_html
        
        # Add benchmark comparison if available
        if 'benchmark_performance' in self.results and self.results['benchmark_performance']:
            benchmark = self.results['benchmark_performance']
            benchmark_ticker = self.results.get('benchmark_ticker', 'Benchmark')
            
            html_content += f"""
                <h2>Benchmark Comparison ({benchmark_ticker})</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-name">Benchmark Return</div>
                        <div class="metric-value {'positive' if benchmark['total_return'] >= 0 else 'negative'}">{benchmark['total_return_pct']:.2f}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Alpha</div>
                        <div class="metric-value {'positive' if benchmark.get('alpha', 0) >= 0 else 'negative'}">{benchmark.get('alpha_pct', 0):.2f}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Beta</div>
                        <div class="metric-value">{benchmark.get('beta', 0):.2f}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-name">Benchmark Max Drawdown</div>
                        <div class="metric-value negative">{benchmark['max_drawdown_pct']:.2f}%</div>
                    </div>
                </div>
            """
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_path}")
    
    def run_walk_forward_analysis(self, ticker: str, 
                                 window_size: int = 365, 
                                 step_size: int = 90,
                                 lookback_days: int = 180,
                                 include_benchmark: bool = True) -> Dict[str, Any]:
        """
        Run walk-forward analysis to test strategy robustness.
        
        Args:
            ticker: Ticker symbol
            window_size: Size of each walk-forward window in days
            step_size: Number of days to step forward for each window
            lookback_days: Number of days to look back for analysis
            include_benchmark: Whether to include benchmark comparison
            
        Returns:
            Dictionary with walk-forward analysis results
        """
        logger.info(f"Starting walk-forward analysis for {ticker}")
        
        # Get full data
        full_data = self.data_manager.get_data(ticker)
        if not full_data or '1d' not in full_data:
            logger.error(f"No data available for {ticker}")
            return {}
        
        # Get all dates
        all_dates = full_data['1d'].index.tolist()
        if len(all_dates) < window_size:
            logger.error(f"Not enough data for walk-forward analysis: {len(all_dates)} days available, {window_size} required")
            return {}
        
        # Create walk-forward windows
        windows = []
        for i in range(0, len(all_dates) - window_size, step_size):
            start_idx = i
            end_idx = i + window_size
            if end_idx > len(all_dates):
                end_idx = len(all_dates)
            
            window_start = all_dates[start_idx]
            window_end = all_dates[end_idx - 1]
            
            windows.append({
                'start_date': window_start,
                'end_date': window_end,
                'window_number': len(windows) + 1
            })
        
        if not windows:
            logger.error("No valid windows for walk-forward analysis")
            return {}
        
        logger.info(f"Created {len(windows)} walk-forward windows")
        
        # Run backtest for each window
        window_results = []
        for window in windows:
            logger.info(f"Running backtest for window {window['window_number']}: {window['start_date']} to {window['end_date']}")
            
            # Create a new backtester for this window
            window_backtester = VPABacktester(
                start_date=window['start_date'],
                end_date=window['end_date'],
                initial_capital=self.trade_simulator.initial_capital,
                commission_rate=self.trade_simulator.commission_rate,
                slippage_percent=self.trade_simulator.slippage_percent,
                risk_per_trade=self.trade_simulator.risk_per_trade,
                max_positions=self.trade_simulator.max_positions,
                timeframes=self.timeframes,
                vpa_config=self.config,
                benchmark_ticker=self.benchmark_ticker if include_benchmark else None
            )
            
            # Run backtest
            result = window_backtester.run_backtest(ticker, lookback_days=lookback_days)
            
            if result:
                window_results.append({
                    'window': window,
                    'performance': result['performance'],
                    'benchmark_performance': result.get('benchmark_performance'),
                    'trades': result['trades'],
                    'equity_curve': result['equity_curve']
                })
        
        if not window_results:
            logger.error("No valid results from any window")
            return {}
        
        # Analyze window results
        performance_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 
                              'max_drawdown', 'win_rate', 'profit_factor']
        
        metrics_by_window = {}
        for metric in performance_metrics:
            metrics_by_window[metric] = [w['performance'].get(metric, 0) for w in window_results]
        
        # Calculate aggregate statistics
        aggregate_stats = {}
        for metric, values in metrics_by_window.items():
            aggregate_stats[f'{metric}_mean'] = np.mean(values)
            aggregate_stats[f'{metric}_median'] = np.median(values)
            aggregate_stats[f'{metric}_std'] = np.std(values)
            aggregate_stats[f'{metric}_min'] = np.min(values)
            aggregate_stats[f'{metric}_max'] = np.max(values)
        
        # Calculate consistency metrics
        total_trades = sum(len(w['trades']) for w in window_results)
        profitable_windows = sum(1 for w in window_results if w['performance']['total_return'] > 0)
        window_consistency = profitable_windows / len(window_results) if window_results else 0
        
        # Store results
        walk_forward_results = {
            'ticker': ticker,
            'windows': windows,
            'window_results': window_results,
            'metrics_by_window': metrics_by_window,
            'aggregate_stats': aggregate_stats,
            'total_trades': total_trades,
            'profitable_windows': profitable_windows,
            'window_consistency': window_consistency,
            'window_consistency_pct': window_consistency * 100
        }
        
        logger.info(f"Walk-forward analysis completed with {len(window_results)} windows")
        logger.info(f"Window consistency: {window_consistency * 100:.2f}%")
        
        return walk_forward_results
    
    def plot_walk_forward_results(self, walk_forward_results: Dict[str, Any], 
                                 figsize=(15, 10), save_path=None):
        """
        Plot walk-forward analysis results.
        
        Args:
            walk_forward_results: Results from run_walk_forward_analysis
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if not walk_forward_results or 'window_results' not in walk_forward_results:
            logger.warning("No walk-forward results to plot")
            return None
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Returns by window
        window_numbers = [w['window']['window_number'] for w in walk_forward_results['window_results']]
        returns = [w['performance']['total_return_pct'] for w in walk_forward_results['window_results']]
        
        bars = axs[0, 0].bar(window_numbers, returns, color=['g' if r >= 0 else 'r' for r in returns])
        axs[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[0, 0].set_title('Returns by Window')
        axs[0, 0].set_xlabel('Window Number')
        axs[0, 0].set_ylabel('Return (%)')
        axs[0, 0].grid(axis='y', alpha=0.3)
        
        # Add mean return line
        mean_return = np.mean(returns)
        axs[0, 0].axhline(y=mean_return, color='blue', linestyle='--', 
                         label=f'Mean: {mean_return:.2f}%')
        axs[0, 0].legend()
        
        # Plot 2: Sharpe ratio by window
        sharpe_ratios = [w['performance']['sharpe_ratio'] for w in walk_forward_results['window_results']]
        
        bars = axs[0, 1].bar(window_numbers, sharpe_ratios, color=['g' if s >= 1 else 'orange' if s >= 0 else 'r' for s in sharpe_ratios])
        axs[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[0, 1].axhline(y=1, color='green', linestyle='--', alpha=0.3)
        axs[0, 1].set_title('Sharpe Ratio by Window')
        axs[0, 1].set_xlabel('Window Number')
        axs[0, 1].set_ylabel('Sharpe Ratio')
        axs[0, 1].grid(axis='y', alpha=0.3)
        
        # Add mean Sharpe line
        mean_sharpe = np.mean(sharpe_ratios)
        axs[0, 1].axhline(y=mean_sharpe, color='blue', linestyle='--', 
                         label=f'Mean: {mean_sharpe:.2f}')
        axs[0, 1].legend()
        
        # Plot 3: Win rate by window
        win_rates = [w['performance']['win_rate_pct'] for w in walk_forward_results['window_results']]
        
        bars = axs[1, 0].bar(window_numbers, win_rates, color=['g' if w >= 50 else 'r' for w in win_rates])
        axs[1, 0].axhline(y=50, color='black', linestyle='--', alpha=0.3)
        axs[1, 0].set_title('Win Rate by Window')
        axs[1, 0].set_xlabel('Window Number')
        axs[1, 0].set_ylabel('Win Rate (%)')
        axs[1, 0].grid(axis='y', alpha=0.3)
        
        # Add mean win rate line
        mean_win_rate = np.mean(win_rates)
        axs[1, 0].axhline(y=mean_win_rate, color='blue', linestyle='--', 
                         label=f'Mean: {mean_win_rate:.2f}%')
        axs[1, 0].legend()
        
        # Plot 4: Trades per window
        trades_per_window = [len(w['trades']) for w in walk_forward_results['window_results']]
        
        bars = axs[1, 1].bar(window_numbers, trades_per_window, color='skyblue')
        axs[1, 1].set_title('Trades per Window')
        axs[1, 1].set_xlabel('Window Number')
        axs[1, 1].set_ylabel('Number of Trades')
        axs[1, 1].grid(axis='y', alpha=0.3)
        
        # Add mean trades line
        mean_trades = np.mean(trades_per_window)
        axs[1, 1].axhline(y=mean_trades, color='blue', linestyle='--', 
                         label=f'Mean: {mean_trades:.1f}')
        axs[1, 1].legend()
        
        # Add overall title
        plt.suptitle(f"Walk-Forward Analysis Results for {walk_forward_results['ticker']}", 
                    fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Walk-forward results plot saved to {save_path}")
        
        return fig
    
    def run_monte_carlo_simulation(self, num_simulations: int = 1000, 
                                  confidence_level: float = 0.95,
                                  save_path: str = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to assess the range of possible outcomes.
        
        Args:
            num_simulations: Number of Monte Carlo simulations to run
            confidence_level: Confidence level for the simulation results
            save_path: Path to save the simulation plot
            
        Returns:
            Dictionary with Monte Carlo simulation results
        """
        if not self.results or 'trades' not in self.results or not self.results['trades']:
            logger.warning("No trade data for Monte Carlo simulation")
            return {}
        
        logger.info(f"Running {num_simulations} Monte Carlo simulations")
        
        # Get trade data
        trades_df = pd.DataFrame(self.results['trades'])
        
        # Calculate trade returns as percentage of capital at entry
        trades_df['return_pct'] = trades_df['pl_percent'] / 100  # Convert to decimal
        
        # Run simulations
        initial_capital = self.trade_simulator.initial_capital
        num_trades = len(trades_df)
        
        # Create array to store simulation results
        simulation_results = np.zeros((num_simulations, num_trades + 1))
        simulation_results[:, 0] = initial_capital  # Set initial capital
        
        # Run simulations
        for i in range(num_simulations):
            # Resample trade returns with replacement
            sampled_returns = np.random.choice(trades_df['return_pct'].values, size=num_trades, replace=True)
            
            # Calculate equity curve
            capital = initial_capital
            for j in range(num_trades):
                # Apply return to capital
                trade_return = sampled_returns[j]
                trade_size = capital * self.trade_simulator.risk_per_trade * 10  # Approximate position size
                capital += trade_size * trade_return
                simulation_results[i, j+1] = capital
        
        # Calculate statistics
        final_capitals = simulation_results[:, -1]
        
        # Sort final capitals for percentile calculations
        sorted_capitals = np.sort(final_capitals)
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        lower_bound = np.percentile(final_capitals, lower_percentile * 100)
        median = np.median(final_capitals)
        upper_bound = np.percentile(final_capitals, upper_percentile * 100)
        
        # Calculate return statistics
        returns = (final_capitals / initial_capital) - 1
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        
        # Calculate probability of profit
        prob_profit = np.mean(returns > 0)
        
        # Calculate drawdowns for each simulation
        max_drawdowns = np.zeros(num_simulations)
        for i in range(num_simulations):
            equity_curve = simulation_results[i, :]
            peak = np.maximum.accumulate(equity_curve)
            drawdown = 1 - equity_curve / peak
            max_drawdowns[i] = np.max(drawdown)
        
        # Calculate drawdown statistics
        mean_max_drawdown = np.mean(max_drawdowns)
        median_max_drawdown = np.median(max_drawdowns)
        worst_drawdown = np.max(max_drawdowns)
        
        # Plot results
        if save_path:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot equity curves
            for i in range(min(100, num_simulations)):  # Plot up to 100 simulations for clarity
                ax1.plot(simulation_results[i, :], color='blue', alpha=0.1)
            
            # Plot confidence interval
            x = np.arange(num_trades + 1)
            lower_curve = np.percentile(simulation_results, lower_percentile * 100, axis=0)
            upper_curve = np.percentile(simulation_results, upper_percentile * 100, axis=0)
            median_curve = np.median(simulation_results, axis=0)
            
            ax1.plot(median_curve, color='blue', linewidth=2, label='Median')
            ax1.fill_between(x, lower_curve, upper_curve, color='blue', alpha=0.2, 
                            label=f'{confidence_level*100:.0f}% Confidence Interval')
            ax1.axhline(y=initial_capital, color='r', linestyle='--', 
                       label=f'Initial Capital ({initial_capital:,.0f})')
            
            ax1.set_title('Monte Carlo Simulation of Equity Curves', fontsize=14)
            ax1.set_xlabel('Trade Number')
            ax1.set_ylabel('Capital ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            
            # Plot histogram of final capitals
            ax2.hist(final_capitals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.axvline(x=initial_capital, color='r', linestyle='--', 
                       label=f'Initial Capital ({initial_capital:,.0f})')
            ax2.axvline(x=median, color='blue', linewidth=2, 
                       label=f'Median Final Capital ({median:,.0f})')
            ax2.axvline(x=lower_bound, color='green', linestyle=':', 
                       label=f'Lower Bound ({lower_bound:,.0f})')
            ax2.axvline(x=upper_bound, color='green', linestyle=':', 
                       label=f'Upper Bound ({upper_bound:,.0f})')
            
            ax2.set_title('Distribution of Final Capital', fontsize=14)
            ax2.set_xlabel('Final Capital ($)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis as currency
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            
            plt.tight_layout()
            plt.savefig(save_path)
            logger.info(f"Monte Carlo simulation plot saved to {save_path}")
        
        # Store results
        monte_carlo_results = {
            'num_simulations': num_simulations,
            'confidence_level': confidence_level,
            'initial_capital': initial_capital,
            'median_final_capital': median,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'mean_return': mean_return,
            'mean_return_pct': mean_return * 100,
            'median_return': median_return,
            'median_return_pct': median_return * 100,
            'std_return': std_return,
            'std_return_pct': std_return * 100,
            'prob_profit': prob_profit,
            'prob_profit_pct': prob_profit * 100,
            'mean_max_drawdown': mean_max_drawdown,
            'mean_max_drawdown_pct': mean_max_drawdown * 100,
            'median_max_drawdown': median_max_drawdown,
            'median_max_drawdown_pct': median_max_drawdown * 100,
            'worst_drawdown': worst_drawdown,
            'worst_drawdown_pct': worst_drawdown * 100
        }
        
        logger.info(f"Monte Carlo simulation completed")
        logger.info(f"Median final capital: ${median:,.2f}, Return: {median_return*100:.2f}%")
        logger.info(f"Probability of profit: {prob_profit*100:.2f}%")
        
        return monte_carlo_results
    
    def optimize_parameters(self, ticker: str, param_grid: Dict[str, List[Any]],
                           metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            ticker: Ticker symbol
            param_grid: Dictionary of parameters to optimize with lists of values to try
            metric: Performance metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting parameter optimization for {ticker}")
        
        # Validate metric
        valid_metrics = ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor']
        if metric not in valid_metrics:
            logger.warning(f"Invalid metric: {metric}. Using sharpe_ratio instead.")
            metric = 'sharpe_ratio'
        
        # Generate parameter combinations
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Track best parameters and results
        best_value = float('-inf')
        best_params = None
        best_result = None
        all_results = []
        
        # Test each parameter combination
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_keys, combination))
            
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Create a new backtester with these parameters
            backtester = VPABacktester(
                start_date=self.data_manager.start_date,
                end_date=self.data_manager.end_date,
                **{k: v for k, v in params.items() if k in [
                    'initial_capital', 'commission_rate', 'slippage_percent',
                    'risk_per_trade', 'max_positions'
                ]}
            )
            
            # Update VPA config parameters if needed
            if any(k.startswith('vpa_') for k in params):
                vpa_params = {k[4:]: v for k, v in params.items() if k.startswith('vpa_')}
                backtester.config.update_parameters(vpa_params)
            
            # Run backtest
            result = backtester.run_backtest(ticker)
            
            if not result or 'performance' not in result:
                logger.warning(f"No valid results for combination {i+1}")
                continue
            
            # Get the metric value
            value = result['performance'].get(metric, float('-inf'))
            
            # Track result
            all_results.append({
                'params': params,
                'value': value,
                'performance': result['performance']
            })
            
            # Update best if better
            if value > best_value:
                best_value = value
                best_params = params
                best_result = result
                logger.info(f"New best {metric}: {best_value} with params: {best_params}")
        
        # Sort all results by the metric
        all_results.sort(key=lambda x: x['value'], reverse=True)
        
        # Return best parameters and results
        return {
            'best_params': best_params,
            'best_value': best_value,
            'best_result': best_result,
            'all_results': all_results
        }
    
    def save_backtest(self, filepath: str) -> None:
        """
        Save backtest results to a file.
        
        Args:
            filepath: Path to save the backtest results
        """
        if not self.results:
            logger.warning("No backtest results to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save results
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Backtest results saved to {filepath}")
    
    @classmethod
    def load_backtest(cls, filepath: str) -> 'VPABacktester':
        """
        Load backtest results from a file.
        
        Args:
            filepath: Path to load the backtest results from
            
        Returns:
            VPABacktester instance with loaded results
        """
        # Create a new instance
        backtester = cls()
        
        # Load results
        with open(filepath, 'rb') as f:
            backtester.results = pickle.load(f)
        
        logger.info(f"Backtest results loaded from {filepath}")
        
        return backtester


# Example usage
if __name__ == "__main__":
    # Create a backtester
    backtester = VPABacktester(
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_percent=0.001,
        risk_per_trade=0.02,
        max_positions=5
    )
    
    # Run a backtest
    results = backtester.run_backtest(['AAPL', 'MSFT', 'GOOGL'])
    
    # Create a report
    report_files = backtester.create_backtest_report("backtest_reports")
    
    print(f"Backtest completed. Report available at: {report_files.get('html')}")

"""
VPA LLM Interface Module

This module provides specific integration for LLMs with the VPA algorithm.
"""

import json
from .vpa_facade import VPAFacade

class VPALLMInterface:
    """Interface for LLM integration with VPA"""
    
    def __init__(self, config_file=None):
        """
        Initialize the VPA LLM interface
        
        Parameters:
        - config_file: Optional path to configuration file
        """
        self.vpa = VPAFacade(config_file)
        self.concept_explanations = self._load_concept_explanations()
    
    def _load_concept_explanations(self):
        """
        Load explanations of VPA concepts
        
        Returns:
        - Dictionary with concept explanations
        """
        # This would typically load from a file, but for simplicity we'll define inline
        return {
            "vpa_overview": """
                Volume Price Analysis (VPA) is a trading methodology that analyzes the relationship 
                between price action and volume to reveal market sentiment and identify trading opportunities.
                
                VPA is based on the work of iconic traders like Charles Dow, Jesse Livermore, and Richard Wyckoff,
                and focuses on how volume confirms or contradicts price movements. The core principle is that
                volume precedes price, meaning significant volume changes often signal upcoming price movements.
                
                Key concepts in VPA include:
                1. Volume confirms price - When price and volume move in the same direction, it validates the move
                2. Volume contradicts price - When price and volume move in opposite directions, it signals potential reversal
                3. Effort vs. Result - Comparing the effort (volume) with the result (price movement)
            """,
            
            "accumulation": """
                Accumulation in VPA refers to a pattern where large operators or institutions are buying
                an asset while trying to keep the price relatively stable to avoid driving it up before
                they complete their buying.
                
                Key characteristics of accumulation:
                1. Sideways price movement with narrowing range
                2. High volume on down days but price doesn't fall much (absorption)
                3. Low volume on up days
                4. Tests of support that hold with decreasing volume
                
                Accumulation typically occurs after a downtrend and precedes an uptrend.
            """,
            
            "distribution": """
                Distribution in VPA refers to a pattern where large operators or institutions are selling
                an asset while trying to keep the price relatively stable to avoid driving it down before
                they complete their selling.
                
                Key characteristics of distribution:
                1. Sideways price movement with narrowing range
                2. High volume on up days but price doesn't rise much (supply)
                3. Low volume on down days
                4. Tests of resistance that fail with decreasing volume
                
                Distribution typically occurs after an uptrend and precedes a downtrend.
            """,
            
            "buying_climax": """
                A buying climax in VPA is a pattern that marks the end of an uptrend, characterized by:
                
                1. Extremely high volume (often the highest in the trend)
                2. Wide range up candle
                3. Price at or near the high of the trend
                4. Often followed by a reversal or significant pullback
                
                A buying climax represents the last surge of buying before smart money begins to distribute.
                It often shows exhaustion of buying pressure and is a bearish signal.
            """,
            
            "selling_climax": """
                A selling climax in VPA is a pattern that marks the end of a downtrend, characterized by:
                
                1. Extremely high volume (often the highest in the trend)
                2. Wide range down candle
                3. Price at or near the low of the trend
                4. Often followed by a reversal or significant bounce
                
                A selling climax represents the last surge of selling before smart money begins to accumulate.
                It often shows exhaustion of selling pressure and is a bullish signal.
            """,
            
            "testing": """
                Testing in VPA refers to price probes of support or resistance levels with specific
                volume characteristics that reveal the strength of these levels.
                
                Key characteristics of testing:
                1. Support test: Price moves below a previous low but with lower volume
                2. Resistance test: Price moves above a previous high but with lower volume
                
                The outcome of these tests provides valuable information:
                - If support holds on low volume, it's strong support
                - If resistance breaks on high volume, it's a valid breakout
                - If resistance fails on low volume, it's a false breakout
            """,
            
            "effort_vs_result": """
                Effort vs. Result is a core concept in VPA that compares the volume (effort)
                with the price movement (result).
                
                Key principles:
                1. High effort (volume) with small result (price movement) indicates potential reversal
                2. Low effort with large result indicates weakness in the move
                3. Equal effort and result indicates a healthy trend
                
                Examples:
                - High volume up day with small price gain: supply is meeting demand (bearish)
                - Low volume down day with large price drop: no support present (bearish)
                - High volume down day with small price drop: demand is meeting supply (bullish)
            """
        }
    
    def process_query(self, query):
        """
        Process a natural language query about VPA
        
        Parameters:
        - query: Natural language query string
        
        Returns:
        - Response to the query
        """
        query = query.lower()
        
        # Check if query is asking for concept explanation
        for concept, explanation in self.concept_explanations.items():
            concept_terms = concept.replace('_', ' ').split()
            if all(term in query for term in concept_terms):
                return self.explain_vpa_concept(concept)
        
        # Check if query is asking for ticker analysis
        ticker_keywords = ["analyze", "analysis", "signal", "trade", "buy", "sell"]
        if any(keyword in query for keyword in ticker_keywords):
            # Extract potential ticker symbols (uppercase words 1-5 characters)
            import re
            potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
            
            if potential_tickers:
                return self.get_ticker_analysis(potential_tickers[0])
        
        # Specific check for "Analyze [TICKER] using VPA" format
        analyze_pattern = r"analyze\s+(\w+)\s+using\s+vpa"
        match = re.search(analyze_pattern, query)
        if match:
            ticker = match.group(1).upper()
            return self.get_ticker_analysis(ticker)
        
        # Default response
        return """
            I can help you with Volume Price Analysis (VPA). You can:
            1. Ask about VPA concepts (e.g., "What is accumulation in VPA?")
            2. Request analysis of a specific ticker (e.g., "Analyze AAPL using VPA")
            3. Get trading signals for a stock (e.g., "What's the VPA signal for MSFT?")
        """
    
    def get_ticker_analysis(self, ticker):
        """
        Get a complete analysis for a ticker in a format suitable for LLM
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        
        Returns:
        - Dictionary with structured analysis data
        """
        try:
            # Get full analysis from facade
            analysis = self.vpa.analyze_ticker(ticker)
            
            # Format for LLM consumption
            llm_friendly_analysis = {
                "ticker": ticker,
                "current_price": float(analysis["current_price"]),
                "signal": {
                    "type": analysis["signal"]["type"],
                    "strength": analysis["signal"]["strength"],
                    "details": analysis["signal"]["details"]
                },
                "risk": {
                    "stop_loss": float(analysis["risk_assessment"]["stop_loss"]),
                    "take_profit": float(analysis["risk_assessment"]["take_profit"]),
                    "risk_reward": float(analysis["risk_assessment"]["risk_reward_ratio"])
                },
                "patterns": {}
            }
            
            # Extract pattern information
            primary_tf = list(analysis["timeframe_analyses"].keys())[0]
            patterns = analysis["timeframe_analyses"][primary_tf]["pattern_analysis"]
            
            for pattern_name, pattern_data in patterns.items():
                if pattern_name != "testing":  # Special handling for testing
                    llm_friendly_analysis["patterns"][pattern_name] = {
                        "detected": pattern_data["detected"],
                        "details": pattern_data["details"] if "details" in pattern_data else ""
                    }
            
            # Add natural language explanation
            llm_friendly_analysis["explanation"] = self.vpa.explain_signal(ticker)
            
            return llm_friendly_analysis
            
        except Exception as e:
            return {
                "ticker": ticker,
                "error": str(e),
                "explanation": f"Unable to analyze {ticker}: {str(e)}"
            }
    
    def explain_vpa_concept(self, concept):
        """
        Explain a VPA concept in natural language
        
        Parameters:
        - concept: Name of the VPA concept
        
        Returns:
        - String with explanation
        """
        if concept in self.concept_explanations:
            return self.concept_explanations[concept].strip()
        else:
            return f"Concept '{concept}' not found in VPA knowledge base."
    
    def suggest_parameters(self, ticker, goal):
        """
        Suggest VPA parameters based on a specific goal
        
        Parameters:
        - ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        - goal: Trading goal (e.g., 'day_trading', 'swing_trading', 'long_term')
        
        Returns:
        - Dictionary with suggested parameters
        """
        # Default parameters
        default_params = {
            "volume": {
                "very_high_threshold": 2.0,
                "high_threshold": 1.3,
                "low_threshold": 0.6,
                "very_low_threshold": 0.3,
                "lookback_period": 50
            },
            "candle": {
                "wide_threshold": 1.3,
                "narrow_threshold": 0.6,
                "wick_threshold": 1.5,
                "lookback_period": 20
            }
        }
        
        # Adjust based on goal
        if goal == "day_trading":
            suggested_params = default_params.copy()
            suggested_params["volume"]["lookback_period"] = 20
            suggested_params["candle"]["lookback_period"] = 10
            suggested_params["volume"]["high_threshold"] = 1.2  # More sensitive
            suggested_params["timeframes"] = [
                {"interval": "5m", "period": "1d"},
                {"interval": "15m", "period": "3d"},
                {"interval": "1h", "period": "5d"}
            ]
            
        elif goal == "swing_trading":
            suggested_params = default_params.copy()
            suggested_params["volume"]["lookback_period"] = 50
            suggested_params["candle"]["lookback_period"] = 20
            suggested_params["timeframes"] = [
                {"interval": "1h", "period": "10d"},
                {"interval": "4h", "period": "30d"},
                {"interval": "1d", "period": "90d"}
            ]
            
        elif goal == "long_term":
            suggested_params = default_params.copy()
            suggested_params["volume"]["lookback_period"] = 100
            suggested_params["candle"]["lookback_period"] = 50
            suggested_params["volume"]["high_threshold"] = 1.5  # Less sensitive
            suggested_params["timeframes"] = [
                {"interval": "1d", "period": "1y"},
                {"interval": "1wk", "period": "3y"},
                {"interval": "1mo", "period": "5y"}
            ]
            
        else:
            suggested_params = default_params.copy()
            suggested_params["timeframes"] = [
                {"interval": "1d", "period": "1y"},
                {"interval": "1h", "period": "60d"},
                {"interval": "15m", "period": "5d"}
            ]
        
        # Add explanation
        explanation = f"These parameters are optimized for {goal} with {ticker}. "
        
        if goal == "day_trading":
            explanation += "They use shorter lookback periods and more sensitive thresholds to capture intraday movements."
        elif goal == "swing_trading":
            explanation += "They use medium lookback periods and balanced thresholds to identify multi-day swings."
        elif goal == "long_term":
            explanation += "They use longer lookback periods and less sensitive thresholds to filter out short-term noise."
        
        return {
            "parameters": suggested_params,
            "explanation": explanation
        }
    
    def generate_code_example(self, task, ticker="AAPL"):
        """
        Generate code example for a specific VPA task
        
        Parameters:
        - task: Task description (e.g., 'analyze_ticker', 'backtest', 'scan_market')
        - ticker: Example ticker to use
        
        Returns:
        - String with Python code example
        """
        if task == "analyze_ticker":
            return f"""
# Example: Analyze a single ticker with VPA
from vpa_facade import VPAFacade

# Initialize the VPA facade
vpa = VPAFacade()

# Analyze the ticker
results = vpa.analyze_ticker("{ticker}")

# Print the signal
print(f"Signal for {ticker}: {{results['signal']['type']}} ({{results['signal']['strength']}})")
print(f"Details: {{results['signal']['details']}}")

# Print risk assessment
print(f"Stop Loss: ${{results['risk_assessment']['stop_loss']:.2f}}")
print(f"Take Profit: ${{results['risk_assessment']['take_profit']:.2f}}")
print(f"Risk-Reward Ratio: {{results['risk_assessment']['risk_reward_ratio']:.2f}}")
"""
            
        elif task == "backtest":
            return f"""
# Example: Backtest VPA strategy
from vpa_facade import VPAFacade
import pandas as pd
import yfinance as yf

# Initialize the VPA facade
vpa = VPAFacade()

# Define backtest parameters
ticker = "{ticker}"
start_date = "2022-01-01"
end_date = "2023-01-01"
initial_capital = 10000

# Get historical data
data = yf.download(ticker, start=start_date, end=end_date)

# Initialize results tracking
equity = [initial_capital]
position = 0
entry_price = 0

# Loop through each day (excluding first 50 days for lookback)
for i in range(50, len(data)):
    # Get data up to current day
    current_data = data.iloc[:i+1]
    
    # Get current price
    current_price = current_data['Close'].iloc[-1]
    
    # Analyze with VPA (would need to modify facade to accept DataFrame)
    # This is simplified for example purposes
    signal = vpa.get_signals(ticker)
    
    # Execute trades based on signals
    if signal['signal']['type'] == 'BUY' and position == 0:
        # Calculate position size
        position = equity[-1] / current_price
        entry_price = current_price
        print(f"BUY at {{current_price:.2f}}")
    
    elif signal['signal']['type'] == 'SELL' and position > 0:
        # Close position
        equity.append(position * current_price)
        position = 0
        print(f"SELL at {{current_price:.2f}}")
    
    # Update equity if in position
    if position > 0:
        equity.append(position * current_price)
    else:
        equity.append(equity[-1])

# Calculate performance metrics
total_return = (equity[-1] - initial_capital) / initial_capital * 100
print(f"Total Return: {{total_return:.2f}}%")
"""
            
        elif task == "scan_market":
            return """
# Example: Scan market for VPA signals
from vpa_facade import VPAFacade

# Initialize the VPA facade
vpa = VPAFacade()

# Define list of tickers to scan
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD"]

# Scan for strong buy signals
buy_signals = vpa.scan_for_signals(tickers, signal_type="BUY", signal_strength="STRONG")

# Print results
print("Strong Buy Signals:")
for ticker, result in buy_signals.items():
    print(f"{ticker}: {result['signal']['details']}")
    print(f"  Stop Loss: ${result['risk_assessment']['stop_loss']:.2f}")
    print(f"  Take Profit: ${result['risk_assessment']['take_profit']:.2f}")
    print(f"  Risk-Reward: {result['risk_assessment']['risk_reward_ratio']:.2f}")
    print()

# Scan for strong sell signals
sell_signals = vpa.scan_for_signals(tickers, signal_type="SELL", signal_strength="STRONG")

# Print results
print("Strong Sell Signals:")
for ticker, result in sell_signals.items():
    print(f"{ticker}: {result['signal']['details']}")
    print()
"""
        
        else:
            return """
# Basic VPA usage example
from vpa_facade import VPAFacade

# Initialize the VPA facade
vpa = VPAFacade()

# Analyze a ticker
results = vpa.analyze_ticker("AAPL")

# Print the results
print(f"Signal: {results['signal']['type']} ({results['signal']['strength']})")
print(f"Details: {results['signal']['details']}")
"""

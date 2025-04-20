"""
VPA Signals Module

This module provides signal generation and risk assessment for the VPA algorithm.
"""

import pandas as pd
import numpy as np
from .vpa_config import VPAConfig
from .vpa_processor import DataProcessor

class SignalGenerator:
    """Generate trading signals based on VPA analysis"""
    
    def __init__(self, config=None):
        self.config = config or VPAConfig()
        self.signal_params = self.config.get_signal_parameters()
    
    def generate_signals(self, timeframe_analyses, confirmations):
        """
        Generate trading signals based on VPA analysis across timeframes
        
        Parameters:
        - timeframe_analyses: Dictionary with analysis results for each timeframe
        - confirmations: Dictionary with confirmation analysis across timeframes
        
        Returns:
        - Dictionary with signal information
        """
        # Check for strong signals
        strong_buy = self.is_strong_buy_signal(timeframe_analyses, confirmations)
        strong_sell = self.is_strong_sell_signal(timeframe_analyses, confirmations)
        
        # Check for moderate signals
        moderate_buy = self.is_moderate_buy_signal(timeframe_analyses, confirmations)
        moderate_sell = self.is_moderate_sell_signal(timeframe_analyses, confirmations)
        
        # Generate signal
        if strong_buy:
            signal = {
                "type": "BUY",
                "strength": "STRONG",
                "details": "Strong buy signal confirmed across multiple timeframes"
            }
        elif strong_sell:
            signal = {
                "type": "SELL",
                "strength": "STRONG",
                "details": "Strong sell signal confirmed across multiple timeframes"
            }
        elif moderate_buy:
            signal = {
                "type": "BUY",
                "strength": "MODERATE",
                "details": "Moderate buy signal with some timeframe confirmation"
            }
        elif moderate_sell:
            signal = {
                "type": "SELL",
                "strength": "MODERATE",
                "details": "Moderate sell signal with some timeframe confirmation"
            }
        else:
            signal = {
                "type": "NO_ACTION",
                "strength": "NEUTRAL",
                "details": "No clear signal detected"
            }
        
        # Add supporting evidence
        signal["evidence"] = self.gather_signal_evidence(timeframe_analyses, confirmations, signal["type"])
        
        return signal
    
    def is_strong_buy_signal(self, timeframe_analyses, confirmations):
        """
        Check if a strong buy signal is present
        
        Parameters:
        - timeframe_analyses: Dictionary with analysis results for each timeframe
        - confirmations: Dictionary with confirmation analysis across timeframes
        
        Returns:
        - Boolean indicating if a strong buy signal is present
        """
        # Get parameters from config
        bullish_confirmation_threshold = self.signal_params.get("bullish_confirmation_threshold", 1)
        
        # Check for bullish confirmations across timeframes
        if len(confirmations["bullish"]) <= bullish_confirmation_threshold:
            return False
        
        # Check for bullish patterns
        bullish_patterns = 0
        for tf in timeframe_analyses:
            patterns = timeframe_analyses[tf]["pattern_analysis"]
            
            # Check for accumulation
            if patterns["accumulation"]["detected"]:
                bullish_patterns += 1
            
            # Check for selling climax
            if patterns["selling_climax"]["detected"]:
                bullish_patterns += 1
            
            # Check for support tests
            if patterns["testing"]["detected"]:
                for test in patterns["testing"]["tests"]:
                    if test["type"] == "SUPPORT_TEST":
                        bullish_patterns += 1
        
        # Check for bullish candle signals in the primary timeframe
        primary_tf = list(timeframe_analyses.keys())[0]  # Assuming first timeframe is primary
        candle_analysis = timeframe_analyses[primary_tf]["candle_analysis"]
        trend_analysis = timeframe_analyses[primary_tf]["trend_analysis"]
        
        bullish_candle = candle_analysis["signal_strength"] == "BULLISH"
        bullish_trend = trend_analysis["signal_strength"] == "BULLISH"
        
        # Determine if strong buy signal is present
        return (len(confirmations["bullish"]) >= bullish_confirmation_threshold and 
                bullish_patterns >= 1 and 
                (bullish_candle or bullish_trend))
    
    def is_strong_sell_signal(self, timeframe_analyses, confirmations):
        """
        Check if a strong sell signal is present
        
        Parameters:
        - timeframe_analyses: Dictionary with analysis results for each timeframe
        - confirmations: Dictionary with confirmation analysis across timeframes
        
        Returns:
        - Boolean indicating if a strong sell signal is present
        """
        # Get parameters from config
        bearish_confirmation_threshold = self.signal_params.get("bearish_confirmation_threshold", 1)
        
        # Check for bearish confirmations across timeframes
        if len(confirmations["bearish"]) <= bearish_confirmation_threshold:
            return False
        
        # Check for bearish patterns
        bearish_patterns = 0
        for tf in timeframe_analyses:
            patterns = timeframe_analyses[tf]["pattern_analysis"]
            
            # Check for distribution
            if patterns["distribution"]["detected"]:
                bearish_patterns += 1
            
            # Check for buying climax
            if patterns["buying_climax"]["detected"]:
                bearish_patterns += 1
            
            # Check for resistance tests
            if patterns["testing"]["detected"]:
                for test in patterns["testing"]["tests"]:
                    if test["type"] == "RESISTANCE_TEST":
                        bearish_patterns += 1
        
        # Check for bearish candle signals in the primary timeframe
        primary_tf = list(timeframe_analyses.keys())[0]  # Assuming first timeframe is primary
        candle_analysis = timeframe_analyses[primary_tf]["candle_analysis"]
        trend_analysis = timeframe_analyses[primary_tf]["trend_analysis"]
        
        bearish_candle = candle_analysis["signal_strength"] == "BEARISH"
        bearish_trend = trend_analysis["signal_strength"] == "BEARISH"
        
        # Determine if strong sell signal is present
        return (len(confirmations["bearish"]) >= bearish_confirmation_threshold and 
                bearish_patterns >= 1 and 
                (bearish_candle or bearish_trend))
    
    def is_moderate_buy_signal(self, timeframe_analyses, confirmations):
        """
        Check if a moderate buy signal is present
        
        Parameters:
        - timeframe_analyses: Dictionary with analysis results for each timeframe
        - confirmations: Dictionary with confirmation analysis across timeframes
        
        Returns:
        - Boolean indicating if a moderate buy signal is present
        """
        # Check for bullish signals in at least one timeframe
        bullish_candles = 0
        bullish_trends = 0
        
        for tf in timeframe_analyses:
            candle_analysis = timeframe_analyses[tf]["candle_analysis"]
            trend_analysis = timeframe_analyses[tf]["trend_analysis"]
            
            if candle_analysis["signal_strength"] == "BULLISH":
                bullish_candles += 1
            
            if trend_analysis["signal_strength"] == "BULLISH":
                bullish_trends += 1
        
        # Get parameters from config
        bullish_candles_threshold = self.signal_params.get("bullish_candles_threshold", 2)
        
        # Check for bullish patterns
        bullish_patterns = 0
        for tf in timeframe_analyses:
            patterns = timeframe_analyses[tf]["pattern_analysis"]
            
            # Check for accumulation
            if patterns["accumulation"]["detected"]:
                bullish_patterns += 1
            
            # Check for selling climax
            if patterns["selling_climax"]["detected"]:
                bullish_patterns += 1
        
        # Determine if moderate buy signal is present
        return (bullish_candles >= bullish_candles_threshold or 
                bullish_trends >= 1 or 
                bullish_patterns >= 1)
    
    def is_moderate_sell_signal(self, timeframe_analyses, confirmations):
        """
        Check if a moderate sell signal is present
        
        Parameters:
        - timeframe_analyses: Dictionary with analysis results for each timeframe
        - confirmations: Dictionary with confirmation analysis across timeframes
        
        Returns:
        - Boolean indicating if a moderate sell signal is present
        """
        # Check for bearish signals in at least one timeframe
        bearish_candles = 0
        bearish_trends = 0
        
        for tf in timeframe_analyses:
            candle_analysis = timeframe_analyses[tf]["candle_analysis"]
            trend_analysis = timeframe_analyses[tf]["trend_analysis"]
            
            if candle_analysis["signal_strength"] == "BEARISH":
                bearish_candles += 1
            
            if trend_analysis["signal_strength"] == "BEARISH":
                bearish_trends += 1
        
        # Get parameters from config
        bearish_candles_threshold = self.signal_params.get("bearish_candles_threshold", 2)
        
        # Check for bearish patterns
        bearish_patterns = 0
        for tf in timeframe_analyses:
            patterns = timeframe_analyses[tf]["pattern_analysis"]
            
            # Check for distribution
            if patterns["distribution"]["detected"]:
                bearish_patterns += 1
            
            # Check for buying climax
            if patterns["buying_climax"]["detected"]:
                bearish_patterns += 1
        
        # Determine if moderate sell signal is present
        return (bearish_candles >= bearish_candles_threshold or 
                bearish_trends >= 1 or 
                bearish_patterns >= 1)
    
    def gather_signal_evidence(self, timeframe_analyses, confirmations, signal_type):
        """
        Gather evidence supporting the signal
        
        Parameters:
        - timeframe_analyses: Dictionary with analysis results for each timeframe
        - confirmations: Dictionary with confirmation analysis across timeframes
        - signal_type: Type of signal (BUY, SELL, NO_ACTION)
        
        Returns:
        - Dictionary with supporting evidence
        """
        evidence = {
            "candle_signals": [],
            "trend_signals": [],
            "pattern_signals": [],
            "timeframe_confirmations": []
        }
        
        # Gather evidence based on signal type
        if signal_type == "BUY":
            # Add bullish candle signals
            for tf in timeframe_analyses:
                candle_analysis = timeframe_analyses[tf]["candle_analysis"]
                if candle_analysis["signal_strength"] == "BULLISH":
                    evidence["candle_signals"].append({
                        "timeframe": tf,
                        "details": candle_analysis["details"]
                    })
            
            # Add bullish trend signals
            for tf in timeframe_analyses:
                trend_analysis = timeframe_analyses[tf]["trend_analysis"]
                if trend_analysis["signal_strength"] == "BULLISH":
                    evidence["trend_signals"].append({
                        "timeframe": tf,
                        "details": trend_analysis["details"]
                    })
            
            # Add bullish pattern signals
            for tf in timeframe_analyses:
                patterns = timeframe_analyses[tf]["pattern_analysis"]
                
                if patterns["accumulation"]["detected"]:
                    evidence["pattern_signals"].append({
                        "timeframe": tf,
                        "pattern": "Accumulation",
                        "details": patterns["accumulation"]["details"]
                    })
                
                if patterns["selling_climax"]["detected"]:
                    evidence["pattern_signals"].append({
                        "timeframe": tf,
                        "pattern": "Selling Climax",
                        "details": patterns["selling_climax"]["details"]
                    })
            
            # Add timeframe confirmations
            evidence["timeframe_confirmations"] = confirmations["bullish"]
        
        elif signal_type == "SELL":
            # Add bearish candle signals
            for tf in timeframe_analyses:
                candle_analysis = timeframe_analyses[tf]["candle_analysis"]
                if candle_analysis["signal_strength"] == "BEARISH":
                    evidence["candle_signals"].append({
                        "timeframe": tf,
                        "details": candle_analysis["details"]
                    })
            
            # Add bearish trend signals
            for tf in timeframe_analyses:
                trend_analysis = timeframe_analyses[tf]["trend_analysis"]
                if trend_analysis["signal_strength"] == "BEARISH":
                    evidence["trend_signals"].append({
                        "timeframe": tf,
                        "details": trend_analysis["details"]
                    })
            
            # Add bearish pattern signals
            for tf in timeframe_analyses:
                patterns = timeframe_analyses[tf]["pattern_analysis"]
                
                if patterns["distribution"]["detected"]:
                    evidence["pattern_signals"].append({
                        "timeframe": tf,
                        "pattern": "Distribution",
                        "details": patterns["distribution"]["details"]
                    })
                
                if patterns["buying_climax"]["detected"]:
                    evidence["pattern_signals"].append({
                        "timeframe": tf,
                        "pattern": "Buying Climax",
                        "details": patterns["buying_climax"]["details"]
                    })
            
            # Add timeframe confirmations
            evidence["timeframe_confirmations"] = confirmations["bearish"]
        
        return evidence


class RiskAssessor:
    """Assess risk for potential trades"""
    
    def __init__(self, config=None):
        self.config = config or VPAConfig()
        self.risk_params = self.config.get_risk_parameters()
    
    def assess_trade_risk(self, signal, current_price, support_resistance):
        """
        Assess risk for a potential trade
        
        Parameters:
        - signal: Dictionary with signal information
        - current_price: Current price of the asset
        - support_resistance: Dictionary with support and resistance levels
        
        Returns:
        - Dictionary with risk assessment
        """
        # Calculate stop loss and take profit levels
        stop_loss = self.calculate_stop_loss(signal, current_price, support_resistance)
        take_profit = self.calculate_take_profit(signal, current_price, support_resistance)
        
        # Calculate risk-reward ratio
        if signal["type"] == "BUY":
            risk = current_price - stop_loss
            reward = take_profit - current_price
        else:  # SELL or NO_ACTION
            risk = stop_loss - current_price
            reward = current_price - take_profit
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Determine position size based on risk
        position_size = self.calculate_position_size(current_price, stop_loss)
        
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward_ratio,
            "position_size": position_size,
            "risk_per_share": abs(current_price - stop_loss)
        }
    
    def calculate_stop_loss(self, signal, current_price, support_resistance):
        """
        Calculate appropriate stop loss level
        
        Parameters:
        - signal: Dictionary with signal information
        - current_price: Current price of the asset
        - support_resistance: Dictionary with support and resistance levels
        
        Returns:
        - Stop loss price level
        """
        # Get parameters from config
        default_stop_loss_percent = self.risk_params.get("default_stop_loss_percent", 0.02)
        support_resistance_buffer = self.risk_params.get("support_resistance_buffer", 0.005)
        
        if signal["type"] == "BUY":
            # For buy signals, place stop loss below recent support
            support_levels = support_resistance["support"]
            
            if support_levels:
                # Find closest support level below current price
                valid_supports = [level for level in support_levels if level["price"] < current_price]
                
                if valid_supports:
                    # Sort by price (descending) to get closest support below current price
                    valid_supports.sort(key=lambda x: x["price"], reverse=True)
                    stop_level = valid_supports[0]["price"]
                    
                    # Add buffer below support
                    stop_loss = stop_level * (1 - support_resistance_buffer)
                else:
                    # No valid support found, use default percentage
                    stop_loss = current_price * (1 - default_stop_loss_percent)
            else:
                # No support levels found, use default percentage
                stop_loss = current_price * (1 - default_stop_loss_percent)
        
        elif signal["type"] == "SELL":
            # For sell signals, place stop loss above recent resistance
            resistance_levels = support_resistance["resistance"]
            
            if resistance_levels:
                # Find closest resistance level above current price
                valid_resistances = [level for level in resistance_levels if level["price"] > current_price]
                
                if valid_resistances:
                    # Sort by price (ascending) to get closest resistance above current price
                    valid_resistances.sort(key=lambda x: x["price"])
                    stop_level = valid_resistances[0]["price"]
                    
                    # Add buffer above resistance
                    stop_loss = stop_level * (1 + support_resistance_buffer)
                else:
                    # No valid resistance found, use default percentage
                    stop_loss = current_price * (1 + default_stop_loss_percent)
            else:
                # No resistance levels found, use default percentage
                stop_loss = current_price * (1 + default_stop_loss_percent)
        
        else:  # NO_ACTION
            # For no action signals, use default percentage
            stop_loss = current_price * (1 - default_stop_loss_percent)
        
        return stop_loss
    
    def calculate_take_profit(self, signal, current_price, support_resistance):
        """
        Calculate appropriate take profit level
        
        Parameters:
        - signal: Dictionary with signal information
        - current_price: Current price of the asset
        - support_resistance: Dictionary with support and resistance levels
        
        Returns:
        - Take profit price level
        """
        # Get parameters from config
        default_take_profit_percent = self.risk_params.get("default_take_profit_percent", 0.05)
        support_resistance_buffer = self.risk_params.get("support_resistance_buffer", 0.005)
        
        if signal["type"] == "BUY":
            # For buy signals, place take profit at or above recent resistance
            resistance_levels = support_resistance["resistance"]
            
            if resistance_levels:
                # Find closest resistance level above current price
                valid_resistances = [level for level in resistance_levels if level["price"] > current_price]
                
                if valid_resistances:
                    # Sort by price (ascending) to get closest resistance above current price
                    valid_resistances.sort(key=lambda x: x["price"])
                    target_level = valid_resistances[0]["price"]
                    
                    # Add buffer above resistance
                    take_profit = target_level * (1 + support_resistance_buffer)
                else:
                    # No valid resistance found, use default percentage
                    take_profit = current_price * (1 + default_take_profit_percent)
            else:
                # No resistance levels found, use default percentage
                take_profit = current_price * (1 + default_take_profit_percent)
        
        elif signal["type"] == "SELL":
            # For sell signals, place take profit at or below recent support
            support_levels = support_resistance["support"]
            
            if support_levels:
                # Find closest support level below current price
                valid_supports = [level for level in support_levels if level["price"] < current_price]
                
                if valid_supports:
                    # Sort by price (descending) to get closest support below current price
                    valid_supports.sort(key=lambda x: x["price"], reverse=True)
                    target_level = valid_supports[0]["price"]
                    
                    # Add buffer below support
                    take_profit = target_level * (1 - support_resistance_buffer)
                else:
                    # No valid support found, use default percentage
                    take_profit = current_price * (1 - default_take_profit_percent)
            else:
                # No support levels found, use default percentage
                take_profit = current_price * (1 - default_take_profit_percent)
        
        else:  # NO_ACTION
            # For no action signals, use default percentage
            take_profit = current_price * (1 + default_take_profit_percent)
        
        return take_profit
    
    def calculate_position_size(self, current_price, stop_loss, risk_per_trade=0.01, account_size=10000):
        """
        Calculate appropriate position size based on risk
        
        Parameters:
        - current_price: Current price of the asset
        - stop_loss: Stop loss price level
        - risk_per_trade: Percentage of account to risk per trade (default: 1%)
        - account_size: Total account size (default: $10,000)
        
        Returns:
        - Position size (number of shares)
        """
        # Calculate dollar risk per trade
        dollar_risk = account_size * risk_per_trade
        
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_loss)
        
        # Calculate position size
        if risk_per_share > 0:
            position_size = dollar_risk / risk_per_share
        else:
            position_size = 0
        
        return position_size

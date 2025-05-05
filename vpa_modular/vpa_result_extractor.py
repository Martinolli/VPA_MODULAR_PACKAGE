# vpa_result_extractor.py

import pandas as pd
from typing import Dict, Any, List

class VPAResultExtractor:
    def __init__(self, results: Dict[str, Any]):
        self.raw_results = results
        self.extracted_data = self._extract_data()

    def _extract_data(self) -> Dict[str, Any]:
        extracted = {}
        for ticker, ticker_data in self.raw_results.items():
            extracted[ticker] = {
                'current_price': ticker_data.get('current_price'),
                'signal': ticker_data.get('signal', {}),
                'risk_assessment': ticker_data.get('risk_assessment', {}),
                'timeframes': {}
            }
            
            for timeframe, tf_data in ticker_data.get('timeframe_analyses', {}).items():
                extracted[ticker]['timeframes'][timeframe] = {
                    'price_data': pd.DataFrame(tf_data.get('processed_data', {}).get('price', {})),
                    'volume_data': pd.DataFrame(tf_data.get('processed_data', {}).get('volume', {})),
                    'candle_analysis': tf_data.get('candle_analysis', {}),
                    'trend_analysis': tf_data.get('trend_analysis', {}),
                    'pattern_analysis': tf_data.get('pattern_analysis', {}),
                    'support_resistance': tf_data.get('support_resistance', {})
                }
        
        return extracted

    def get_tickers(self) -> list:
        return list(self.extracted_data.keys())

    def get_ticker_data(self, ticker: str) -> Dict[str, Any]:
        return self.extracted_data.get(ticker, {})

    def get_timeframes(self, ticker: str) -> list:
        return list(self.extracted_data.get(ticker, {}).get('timeframes', {}).keys())

    def get_price_data(self, ticker: str, timeframe: str) -> pd.DataFrame:
        return self.extracted_data.get(ticker, {}).get('timeframes', {}).get(timeframe, {}).get('price_data', pd.DataFrame())

    def get_volume_data(self, ticker: str, timeframe: str) -> pd.DataFrame:
        return self.extracted_data.get(ticker, {}).get('timeframes', {}).get(timeframe, {}).get('volume_data', pd.DataFrame())

    def get_candle_analysis(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        return self.extracted_data.get(ticker, {}).get('timeframes', {}).get(timeframe, {}).get('candle_analysis', {})

    def get_trend_analysis(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        return self.extracted_data.get(ticker, {}).get('timeframes', {}).get(timeframe, {}).get('trend_analysis', {})

    def get_pattern_analysis(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        return self.extracted_data.get(ticker, {}).get('timeframes', {}).get(timeframe, {}).get('pattern_analysis', {})

    def get_support_resistance(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        return self.extracted_data.get(ticker, {}).get('timeframes', {}).get(timeframe, {}).get('support_resistance', {})

    def get_signal(self, ticker: str) -> Dict[str, Any]:
        return self.extracted_data.get(ticker, {}).get('signal', {})

    def get_signal_evidence(self, ticker: str) -> Dict[str, List]:
        return self.get_signal(ticker).get('evidence', {})
    
    def get_risk_assessment(self, ticker: str) -> Dict[str, Any]:
        return self.extracted_data.get(ticker, {}).get('risk_assessment', {})
    
    def get_current_price(self, ticker: str) -> float:
        return self.extracted_data.get(ticker, {}).get('current_price', 0.0)
    
def extract_testing_signals(results: dict) -> dict:
    """
    Extract 'Testing' pattern information for each ticker and timeframe.

    Returns a dictionary like:
    {
        "AAPL": {
            "1d": [ { "type": ..., "price": ..., "index": ... }, ... ],
            "1h": [...],
            ...
        },
        ...
    }
    """
    testing_data = {}

    for ticker, ticker_data in results.items():
        testing_data[ticker] = {}

        timeframes = ticker_data.get("timeframes", {})
        for tf, tf_data in timeframes.items():
            pattern_analysis = tf_data.get("Pattern Analysis", {})
            testing_section = pattern_analysis.get("Testing", {})

            if testing_section.get("Detected") and isinstance(testing_section.get("Tests"), list):
                testing_data[ticker][tf] = testing_section["Tests"]

    return testing_data

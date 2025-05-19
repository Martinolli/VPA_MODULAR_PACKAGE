# **Module: vpa_signals.py**

This module provides **signal generation** and **risk assessment** logic for the VPA (Volume Price Analysis) algorithm. It contains two main classes:

- `SignalGenerator`: Generates trading signals (BUY, SELL, etc.) based on multi-timeframe VPA analysis.
- `RiskAssessor`: Calculates stop loss, take profit, risk-reward ratio, and position sizing for trades.

---

## **Class: SignalGenerator**

### **Attributes**

- **config**: Holds the configuration object (usually a `VPAConfig` instance).
- **signal_params**: Dictionary of signal-related parameters, loaded from config.

### **Key Methods**

#### **generate_signals(timeframe_analyses, confirmations)**

- **Purpose**: Main entry point. Generates a trading signal (BUY, SELL, NO_ACTION) and attaches supporting evidence.
- **Inputs**:
  - `timeframe_analyses`: Dict of analysis results for each timeframe.
  - `confirmations`: Dict of confirmation results across timeframes.
- **Returns**: Dict with signal type, strength, details, and evidence.

#### **is_strong_buy_signal(timeframe_analyses, confirmations)**

- **Purpose**: Checks if a strong buy signal is present.
- **Logic**: Requires enough bullish confirmations, bullish patterns (accumulation, selling climax, support test), and bullish candle/trend in the primary timeframe.

#### **is_strong_sell_signal(timeframe_analyses, confirmations)**

- **Purpose**: Checks if a strong sell signal is present.
- **Logic**: Requires enough bearish confirmations, bearish patterns (distribution, buying climax, resistance test), and bearish candle/trend in the primary timeframe.

#### **is_moderate_buy_signal(timeframe_analyses, confirmations)**

- **Purpose**: Checks for a moderate buy signal.
- **Logic**: Looks for bullish candles/trends or bullish patterns in at least one timeframe.

#### **is_moderate_sell_signal(timeframe_analyses, confirmations)**

- **Purpose**: Checks for a moderate sell signal.
- **Logic**: Looks for bearish candles/trends or bearish patterns in at least one timeframe.

#### **gather_signal_evidence(timeframe_analyses, confirmations, signal_type)**

- **Purpose**: Collects supporting evidence for the generated signal.
- **Returns**: Dict with lists of candle signals, trend signals, pattern signals, and timeframe confirmations.

---

## **Class: RiskAssessor**

### *Attributes**

- **config**: Holds the configuration object (usually a `VPAConfig` instance).

- **risk_params**: Dictionary of risk-related parameters, loaded from config.

### *Key Methods**

#### **assess_trade_risk(signal, current_price, support_resistance)**

- **Purpose**: Main entry point. Calculates stop loss, take profit, risk-reward ratio, position size, and risk per share.
- **Returns**: Dict with all risk assessment metrics.

#### **calculate_stop_loss(signal, current_price, support_resistance)**

- **Purpose**: Determines the stop loss price based on signal type and support/resistance levels.
- **Logic**: For BUY, uses closest support below price; for SELL, uses closest resistance above price; otherwise, uses a default percentage.

#### **calculate_take_profit(signal, current_price, support_resistance)**

- **Purpose**: Determines the take profit price based on signal type and support/resistance levels.
- **Logic**: For BUY, uses closest resistance above price; for SELL, uses closest support below price; otherwise, uses a default percentage.

#### **calculate_position_size(current_price, stop_loss, risk_per_trade=0.01, account_size=10000)**

- **Purpose**: Calculates position size (number of shares) based on account size and risk per trade.
- **Logic**: Uses the difference between current price and stop loss to determine risk per share.

---

## **Expected Behavior**

- **SignalGenerator** analyzes multi-timeframe VPA results and produces a trading signal with supporting evidence.
- **RiskAssessor** uses the signal, current price, and support/resistance levels to compute risk management metrics for the trade.
- Both classes are configurable via a `VPAConfig` object, allowing for flexible thresholds and risk parameters.

---

## **Typical Usage Example**

```python
signal_gen = SignalGenerator(config)
signal = signal_gen.generate_signals(timeframe_analyses, confirmations)

risk_assessor = RiskAssessor(config)
risk = risk_assessor.assess_trade_risk(signal, current_price, support_resistance)
```

---

## **Summary Table**

| Class            | Purpose                        | Main Methods                                      |
|------------------|--------------------------------|---------------------------------------------------|
| SignalGenerator  | Generate trading signals       | generate_signals, is_strong_buy_signal, ...       |
| RiskAssessor     | Assess trade risk/position     | assess_trade_risk, calculate_stop_loss, ...       |

---

This module is central to turning VPA analysis results into actionable trading decisions and risk management plans

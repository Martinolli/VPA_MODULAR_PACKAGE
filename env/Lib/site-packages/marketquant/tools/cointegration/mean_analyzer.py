import yfinance as yf
import plotly.graph_objects as go
import numpy as np

class MeanAnalyzer:
    def __init__(self, stock1, stock2, start=None, end=None, interval='1d', show_signals=True):
        self.stock1 = stock1
        self.stock2 = stock2
        self.interval = interval  # Adding interval parameter for time aggregation
        self.show_signals = show_signals

        # If an interval is provided, don't use the start and end date
        if interval == '1d':
            self.start = start
            self.end = end
        else:
            self.start = None
            self.end = None

        # Download stock data with optional time interval aggregation
        self.data1 = yf.download(stock1, start=self.start, end=self.end, interval=interval)
        self.data2 = yf.download(stock2, start=self.start, end=self.end, interval=interval)

        # Check if we have any data; raise an error if not
        if self.data1.empty or self.data2.empty:
            raise ValueError(f"No data found for {stock1} or {stock2} with interval {interval}")

        # Prepare data
        self.price1 = self.data1['Close']
        self.price2 = self.data2['Close']
        self.price1_norm = self.price1 / self.price1.iloc[0]
        self.price2_norm = self.price2 / self.price2.iloc[0]
        self.normalized_spread = self.price1_norm - self.price2_norm

        # Calculate log returns
        self.log_return1 = np.log(self.price1 / self.price1.shift(1)).dropna()
        self.log_return2 = np.log(self.price2 / self.price2.shift(1)).dropna()
        self.log_return1, self.log_return2 = self.log_return1.align(self.log_return2, join='inner')
        self.spread = self.log_return1 - self.log_return2

        # Mean and Std Dev of log return spread
        self.spread_mean = self.spread.mean()
        self.spread_std = self.spread.std()
        self.threshold_upper = self.spread_mean + 4 * self.spread_std
        self.threshold_lower = self.spread_mean - 4 * self.spread_std

        # Generate signals
        self.long_signals_stock1, self.long_signals_stock2 = self._generate_signals()

    def _generate_signals(self):
        long_signals_stock1 = []
        long_signals_stock2 = []
        for i, value in enumerate(self.spread):
            if value > self.threshold_upper:
                long_signals_stock2.append((self.spread.index[i], self.price2.iloc[i], 'up'))
                long_signals_stock1.append((self.spread.index[i], self.price1.iloc[i], 'down'))
            elif value < self.threshold_lower:
                long_signals_stock1.append((self.spread.index[i], self.price1.iloc[i], 'up'))
                long_signals_stock2.append((self.spread.index[i], self.price2.iloc[i], 'down'))
        return long_signals_stock1, long_signals_stock2

    def plot(self):
        fig = go.Figure()

        # Top pane 1: POOL Linear stock prices with grey thinner lines
        fig.add_trace(go.Scatter(x=self.price1.index, y=self.price1, mode='lines', name=f'{self.stock1} Price (Linear)',
                                 line=dict(color='#403625', width=1), yaxis='y1'))

        # Top pane 2: SWK Linear stock prices with grey thinner lines
        fig.add_trace(go.Scatter(x=self.price2.index, y=self.price2, mode='lines', name=f'{self.stock2} Price (Linear)',
                                 line=dict(color='#824800', width=1), yaxis='y2'))

        # Second pane: Normalized stock prices with grey thinner lines
        fig.add_trace(go.Scatter(x=self.price1.index, y=self.price1_norm, mode='lines', name=f'{self.stock1} Price (Normalized)',
                                 line=dict(color='#403625', width=1), yaxis='y3'))
        fig.add_trace(go.Scatter(x=self.price2.index, y=self.price2_norm, mode='lines', name=f'{self.stock2} Price (Normalized)',
                                 line=dict(color='#824800', width=1), yaxis='y3'))

        fig.add_trace(go.Scatter(x=self.normalized_spread.index, y=self.normalized_spread, mode='lines',
                                 name='Normalized Price Spread', line=dict(color='#210736', width=1), yaxis='y4'))
        fig.add_trace(go.Scatter(x=self.normalized_spread.index, y=[self.normalized_spread.mean()] * len(self.normalized_spread),
                                 mode='lines', name='Mean of Normalized Spread', line=dict(color='lightgrey', width=1), yaxis='y4'))

        fig.add_trace(go.Scatter(x=self.spread.index, y=self.spread, mode='lines', name=f'{self.stock1} - {self.stock2} Log Return Spread',
                                 line=dict(color='#210736', width=2), yaxis='y5'))
        fig.add_trace(go.Scatter(x=self.spread.index, y=[self.spread_mean] * len(self.spread), mode='lines',
                                 name='Log Return Spread Mean', line=dict(color='#360707', dash='dot'), yaxis='y5'))
        fig.add_trace(go.Scatter(x=self.spread.index, y=[self.threshold_upper] * len(self.spread), mode='lines',
                                 name='Upper Bound (3rd Std Dev)', line=dict(color='darkgrey', dash='dash'), yaxis='y5'))
        fig.add_trace(go.Scatter(x=self.spread.index, y=[self.threshold_lower] * len(self.spread), mode='lines',
                                 name='Lower Bound (3rd Std Dev)', line=dict(color='darkgrey', dash='dash'), yaxis='y5'))

        if self.show_signals:
            for signal in self.long_signals_stock1:
                arrow_color = 'limegreen' if signal[2] == 'up' else 'red'
                fig.add_annotation(
                    x=signal[0],
                    yref="y1", y=0,
                    text="↑" if signal[2] == 'up' else "↓",
                    showarrow=False,
                    font=dict(color=arrow_color, size=14),
                    yshift=-15 if signal[2] == 'down' else 15
                )

            for signal in self.long_signals_stock2:
                arrow_color = 'limegreen' if signal[2] == 'up' else 'red'
                fig.add_annotation(
                    x=signal[0],
                    yref="y2", y=0,
                    text="↑" if signal[2] == 'up' else "↓",
                    showarrow=False,
                    font=dict(color=arrow_color, size=14),
                    yshift=-15 if signal[2] == 'down' else 15
                )

        fig.update_layout(
            height=1000,
            title=f'{self.stock1} vs {self.stock2} - Mean Analyzer',

            plot_bgcolor='rgba(117, 117, 117, 1)',
            paper_bgcolor='rgba(92, 92, 92, 1)',

            font=dict(color='white'),

            xaxis=dict(showgrid=True, gridcolor='rgba(150, 150, 150, 0.3)', title='', domain=[0, 1]),

            yaxis1=dict(title=f'{self.stock1} Price', domain=[0.75, 1], showgrid=True,
                        gridcolor='rgba(150, 150, 150, 0.3)',
                        titlefont=dict(size=10),
                        tickfont=dict(size=8)),
            yaxis2=dict(title=f'{self.stock2} Price', domain=[0.50, 0.74], showgrid=True,
                        gridcolor='rgba(150, 150, 150, 0.3)',
                        titlefont=dict(size=10),
                        tickfont=dict(size=8)),
            yaxis3=dict(title='Normalized Price', domain=[0.35, 0.49], showgrid=True,
                        gridcolor='rgba(150, 150, 150, 0.3)',
                        titlefont=dict(size=10),
                        tickfont=dict(size=8)),
            yaxis4=dict(title='Normalized Price Spread', domain=[0.20, 0.34], showgrid=True,
                        gridcolor='rgba(150, 150, 150, 0.3)',
                        titlefont=dict(size=10),
                        tickfont=dict(size=8)),
            yaxis5=dict(title='Log Return Spread', domain=[0, 0.19], showgrid=True,
                        gridcolor='rgba(150, 150, 150, 0.3)',
                        titlefont=dict(size=10),
                        tickfont=dict(size=8)),

            hovermode='x unified',
            showlegend=True,
            legend=dict(bgcolor='rgba(92, 92, 92, 0.8)'),
        )

        fig.update_xaxes(rangeslider_visible=False)

        fig.show()


chart = MeanAnalyzer('POOL', 'SWK', start='2022-01-01', end='2024-09-13', interval='1d', show_signals=False)

chart.plot()





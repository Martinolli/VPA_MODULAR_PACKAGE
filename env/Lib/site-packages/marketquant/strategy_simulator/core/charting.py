import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import mplcursors
import warnings


import warnings
warnings.filterwarnings("ignore")


class TradeChart:
    def __init__(self, data, trades, pnl):
        self.data = data
        self.trades = trades
        self.pnl = pnl

    def plot_chart(self):
        plt.style.use('dark_background')

        dates = self.data['Date']
        prices = self.data['Close']

        fig, ax = plt.subplots(figsize=(12, 6))

        if self.pnl >= 0:
            line_color = 'grey'
            gradient_color = [(0, 1, 0, 0.1), (0, 1, 0, 0.6)]
        else:
            line_color = 'grey'
            gradient_color = [(1, 0, 0, 0.1), (1, 0, 0, 0.6)]

        cmap = LinearSegmentedColormap.from_list('pnl_gradient', gradient_color)

        ax.plot(dates, prices, label='Closing Price', color=line_color, lw=2)

        y_min = np.min(prices) * 0.95
        z = np.empty((100, len(dates)))
        z[:] = np.linspace(0, 1, 100)[:, None]

        ax.imshow(z, aspect='auto', cmap=cmap,
                  extent=[mdates.date2num(dates.min()), mdates.date2num(dates.max()), y_min, np.max(prices)], alpha=0.3)

        fig.autofmt_xdate()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        buy_dates = []
        buy_prices = []
        sell_dates = []
        sell_prices = []
        short_dates = []
        short_prices = []
        cover_dates = []
        cover_prices = []

        for trade in self.trades:
            action, price, date = self._parse_trade(trade)
            if action == 'Buy':
                buy_dates.append(date)
                buy_prices.append(price)
            elif action == 'Sell':
                sell_dates.append(date)
                sell_prices.append(price)
            elif action == 'Short':
                short_dates.append(date)
                short_prices.append(price)
            elif action == 'Cover':
                cover_dates.append(date)
                cover_prices.append(price)

        ax.scatter(buy_dates, buy_prices, color='lime', marker='v', label='Buy', s=50, zorder=5)
        ax.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell', s=50, zorder=5)

        ax.scatter(short_dates, short_prices, color='red', marker='^', label='Short', s=50, zorder=5)
        ax.scatter(cover_dates, cover_prices, color='lime', marker='^', label='Cover', s=50, zorder=5)

        ax.set_title(f"Chart with Trades", fontsize=16, color='white')

        ax.set_xlabel("Date", fontsize=12, color='white')
        ax.set_ylabel("Price", fontsize=12, color='white')

        ax.set_ylim([np.min(prices) * 0.95, np.max(prices) * 1.05])

        cursor = mplcursors.cursor(hover=True)

        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.get_bbox_patch().set_facecolor("white")
            sel.annotation.get_bbox_patch().set_alpha(0.5)
            sel.annotation.set_text(f"Price: {sel.target[1]:.2f}")

        plt.tight_layout()

        print("Building chart...")

        try:
            print(f"Chart Shown")
            plt.show(block=True)
        except:
            print(f"There was an error showing the chart {e}")

    def _parse_trade(self, trade):
        parts = trade.split()
        action = parts[0]
        price = float(parts[3])
        date_str = " ".join(parts[5:])
        date = pd.to_datetime(date_str)
        return action, price, date

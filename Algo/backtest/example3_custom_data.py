import multiprocessing as mp

from backtesting import Backtest, Strategy
from backtesting.lib import crossover, resample_apply, barssince
import talib
import pandas as pd

df = pd.read_csv('data\AAPL.csv', index_col=0, parse_dates=True)


def custom_optimizer(series):
    return abs(series['Return [%]'] / series['Max. Drawdown [%]'])


class RSIStrategy(Strategy):
    upper_band = 70
    lower_band = 30
    rsi_window = 14

    def init(self):
        self.daily_rsi = self.I(talib.RSI, self.data.Close, self.rsi_window)
        self.weekly_rsi = resample_apply("W-FRI", talib.RSI, self.data.Close, self.rsi_window)

    def next(self):
        price = self.data.Close[-1]  # yesterday's price

        if not self.position:
            if crossover(self.daily_rsi, self.upper_band) and self.daily_rsi[-1] > self.weekly_rsi[-1]:
                # rsi is a number between 1 and 100, it measures the smoothed average of price increases vs smoothed
                # average of price increase and price decreases. It is large if price has been increasing much more
                # than decreasing over the rsi_window, and small if it has been decreasing more and for longer than
                # increasing. If RSI is large, the stock is considered overbought and vice-versa, hence we sell.
                # Here I also compare weekly RSI vs daily RSI: daily is averaged over a short timeframe, weekly is over
                # a larger timeframe. This helps to isolate short term movements from long term trends.
                self.buy(tp=1.30 * price, sl=0.9 * price, size=0.33)
                # take profit = +15%, stop loss = -5% relative to the most recent closing price.
                # self.position.close()

        elif (crossover(self.daily_rsi, self.lower_band) and self.daily_rsi[-1] < self.weekly_rsi[-1]
              or barssince(self.daily_rsi > self.lower_band) == 3):
            # if daily_rsi < lower_band and daily_rsi < weekly_rsi, we see an out-of-trend oversale -> buy.
            # alternatively, if daily_rsi has last been over lower_band 3 periods ago, then we also buy.
            self.position.close()
            # self.buy(tp=1.10 * price, sl=0.95 * price, size=0.33)
            # take profit = +15%, stop loss = -5% relative to the most recent closing price.


if __name__ == '__main__':
    mp.set_start_method('spawn')  # 'fork' does not work

    bt = Backtest(df, RSIStrategy, cash=1e5, commission=0.002, exclusive_orders=True)

    """stats, heatmap = bt.optimize(
        method='grid',
        upper_band=range(10, 85, 5),
        lower_band=range(10, 85, 5),
        rsi_window=range(2, 14, 2),
        maximize=custom_optimizer,  # alternatively, choose 'Sharpe Ratio' or others
        constraint=lambda param: param.upper_band > param.lower_band,
        return_heatmap=True
    )"""
    stats = bt.run()
    print(stats)
    print(stats._strategy)
    bt.plot(filename='plots/aapl_plot.html')
    # print(heatmap)



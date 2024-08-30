import multiprocessing as mp
from backtesting import Backtest, Strategy
from backtesting.test import GOOG
from backtesting.lib import crossover, plot_heatmaps
import talib
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def custom_optimizer(series):
    return abs(series['Return [%]'] / series['Max. Drawdown [%]'])


class RSIStrategy(Strategy):
    upper_band = 70
    lower_band = 30
    rsi_window = 14

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_window)

    def next(self):
        if crossover(self.rsi, self.upper_band):  # momentum strategy
            self.position.close()
        if crossover(self.rsi, self.lower_band):
            self.buy()


if __name__ == '__main__':
    mp.set_start_method('spawn')  # 'fork' does not work

    bt = Backtest(GOOG, RSIStrategy, cash=1e5, commission=0.002, exclusive_orders=True)

    stats, heatmap = bt.optimize(
        method='grid',
        upper_band=range(10, 85, 5),
        lower_band=range(10, 85, 5),
        rsi_window=range(2, 14, 2),
        maximize=custom_optimizer,  # alternatively, choose 'Sharpe Ratio' or others
        constraint=lambda param: param.upper_band > param.lower_band,
        return_heatmap=True
    )
    # stats = bt.run()
    print(stats)
    print(stats._strategy)
    bt.plot(filename='plots/plot.html')
    print(heatmap)
    plot_heatmaps(heatmap, filename='plots/heatmap.html')

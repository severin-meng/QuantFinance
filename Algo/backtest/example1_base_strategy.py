from backtesting import Backtest, Strategy
from backtesting.test import GOOG
from backtesting.lib import crossover
import talib


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


bt = Backtest(GOOG, RSIStrategy, cash=1e5, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
bt.plot(filename='plots/plot.html')

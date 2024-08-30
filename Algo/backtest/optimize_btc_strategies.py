import multiprocessing as mp

from backtesting import Backtest, Strategy
from backtesting.lib import crossover, resample_apply, barssince, plot_heatmaps
import pandas as pd
import numpy as np
from strategies import MACDStrategy, CHDLStrategy, EMASlopeStrategy, BBStrategy, MACDPullbackStrategy

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def custom_optimizer(series):
    return abs(series['Return [%]'] / series['Max. Drawdown [%]'])


# TODO: combine with other indicators:
#  trendlines, support and resistance levels, oscillators like RSI or stochastic oscillator.


def run_macd(df, optimize=True):
    bt = Backtest(df, MACDStrategy, cash=1e8, commission=0.004, exclusive_orders=True)  # huge size to trade fractional btc
    if optimize:
        stats, heatmap = bt.optimize(
            max_tries=200,
            method='skopt',
            macd_short_range=range(2, 20, 1),
            macd_long_range=range(12, 52, 1),
            macd_signal_range=range(8, 40, 1),
            n_atr=range(6, 18, 1),
            volume_range=range(15, 30, 1),
            long_ema=range(50, 500, 10),
            # size=list(np.linspace(0.1, 0.99, 10)),
            maximize='SQN',  # alternatively, choose 'Sharpe Ratio' or others
            constraint=lambda param: param.macd_long_range > param.macd_short_range,
            return_heatmap=True
        )
        print(stats)
        print(stats._strategy)
        plot_heatmaps(heatmap, filename='plots/macd_heatmap.html')
    else:
        stats = bt.run()
        print(stats)
    # print(stats._trades)
    bt.plot(filename='plots/macd_plot.html')
    print("DONE")


def run_chdl(df, optimize=True):
    bt = Backtest(df, CHDLStrategy, cash=1e8, commission=0.004, exclusive_orders=True)  # huge size to trade fractional btc
    if optimize:
        stats, heatmap = bt.optimize(
            method='grid',
            atr_length=range(2, 40, 4),
            multiplier=range(1, 5, 1),
            high_length=range(20, 60, 5),
            low_length=range(20, 60, 5),
            maximize='SQN',  # alternatively, choose 'Sharpe Ratio' or others
            # constraint=lambda param: param.macd_long_range > param.macd_short_range,
            return_heatmap=True
        )
        print(stats)
        print(stats._strategy)
        print(stats._trades)
        plot_heatmaps(heatmap, filename='plots/chdl_heatmap.html')
    else:
        stats = bt.run()
        print(stats)
        print(stats._trades)
    bt.plot(filename='plots/chdl_plot.html')
    print("DONE")


def run_slope_ema(df, optimize=True):
    bt = Backtest(df, EMASlopeStrategy, cash=1e8, commission=0.004, exclusive_orders=True)  # huge size to trade fractional btc
    if optimize:
        stats, heatmap = bt.optimize(
            max_tries=200,
            method='skopt',
            n_atr=range(6, 18, 1),
            long_length=range(50, 250, 10),
            short_length=range(2, 50, 2),
            maximize='SQN',  # alternatively, choose 'Sharpe Ratio' or others
            constraint=lambda param: (
                    param.long_length > param.short_length and
                    param.slope_long_length > param.slope_short_length and
                    param.convexity_long_length > param.convexity_short_length),
            return_heatmap=True
        )
        print(stats)
        print(stats._strategy)
        plot_heatmaps(heatmap, filename='plots/SlopeEMA_heatmap.html')
    else:
        stats = bt.run()
        print(stats)
    # print(stats._trades)
    bt.plot(filename='plots/SlopeEMA_plot.html')
    print("DONE")


def run_bbands(df, optimize=True):
    bt = Backtest(df, BBStrategy, cash=1e8, commission=0.004, exclusive_orders=True)  # huge size to trade fractional btc
    if optimize:
        stats, heatmap = bt.optimize(
            max_tries=2000,
            method='skopt',
            stddev_long_entry_cutoff=range(300, 1200, 50),
            stddev_short_entry_cutoff=range(300, 1200, 50),
            stddev_exit_cutoff=range(300, 1200, 50),
            stop_loss_short=list(np.linspace(1.005, 1.1, 4)),
            stop_loss_long=list(np.linspace(0.995, 0.9, 4)),
            stddev_regime_switch=range(1000, 2100, 100),
            maximize='Return [%]',  # alternatively, choose 'Sharpe Ratio' or others
            return_heatmap=True
        )
        print(stats)
        print(stats._strategy)
        plot_heatmaps(heatmap, filename='plots/BB_heatmap.html')
    else:
        stats = bt.run()
        print(stats)
    # print(stats._trades)
    bt.plot(filename='plots/BB_plot.html')
    print("DONE")


def run_macd_pullback(df, optimize=True):
    bt = Backtest(df, MACDPullbackStrategy, cash=1e8, commission=0.004, exclusive_orders=True)  # huge size to trade fractional btc
    if optimize:
        stats, heatmap = bt.optimize(
            max_tries=200,
            method='skopt',
            # volume_range=range(15, 30, 1),
            momentum_threshold=range(100, 1000, 50),
            momentum_bars=range(2, 8, 1),
            pullback_bars=range(2, 8, 1),
            # size=list(np.linspace(0.1, 0.99, 10)),
            maximize='SQN',  # alternatively, choose 'Sharpe Ratio' or others
            return_heatmap=True
        )
        print(stats)
        print(stats._strategy)
        plot_heatmaps(heatmap, filename='plots/macdPB_heatmap.html')
    else:
        stats = bt.run()
        print(stats)
    # print(stats._trades)
    bt.plot(filename='plots/macdPB_plot.html')
    print("DONE")


if __name__ == '__main__':
    # df = pd.read_csv(r'D:\Quant Finance\PyRepo\Algo\notebooks\data\btc_usd_test_1m_15min.csv', index_col=0, parse_dates=True)
    df_full = pd.read_csv(r'D:\Quant Finance\PyRepo\Algo\notebooks\data\btc_usd_test_2m_5min.csv',
                          index_col=0, parse_dates=True)
    df1, df2 = np.array_split(df_full, 2)

    mp.set_start_method('spawn')  # 'fork' does not work
    # run_chdl(optimize=False)
    # run_macd(df1, optimize=False)
    # run_slope_ema(df1, optimize=False)
    run_bbands(df1, optimize=False)
    # run_macd_pullback(df1, optimize=False)

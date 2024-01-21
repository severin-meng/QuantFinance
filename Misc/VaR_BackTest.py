import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from tabulate import tabulate

import warnings
warnings.filterwarnings('ignore')


class VaRBackTest:
    def __init__(self, price_data, confidence, horizon, sample_ddof=1):
        self.prices = price_data
        self.simple_returns = self.prices.pct_change().dropna()['SP500']
        self.log_returns = [np.log(ret + 1) for ret in self.simple_returns]
        self.inv_cumulative_factor = (stats.norm.ppf(1 - confidence))
        self.sample_ddof = sample_ddof
        self.horizon = horizon

    def rolling_std_deviation(self, window_size):
        std_deviation_forecast = []
        for i in range(len(self.log_returns) - window_size):
            std_deviation_forecast.append(np.std(self.log_returns[i:i + window_size], ddof=self.sample_ddof))
        return std_deviation_forecast

    def exponentially_weighted_moving_average(self, decay):
        truncated_log_returns = self.log_returns
        initial_stddev = np.std(self.log_returns, ddof=self.sample_ddof)
        variance_forecast = [initial_stddev ** 2]
        for i, ret in enumerate(truncated_log_returns):
            init_variance = variance_forecast[i]
            variance_forecast.append(decay * init_variance + (1 - decay) * ret ** 2)
        volatility_forecast = [np.sqrt(variance) for variance in variance_forecast]
        # discard the initial std deviation as it is not a local quantity -> not used for backtesting
        return volatility_forecast[1:]

    def backtest_var(self, volatility_estimate, cutoff, plot_title):
        var_series = [self.inv_cumulative_factor * std_dev * np.sqrt(self.horizon) for std_dev in volatility_estimate]

        # 10-day log returns
        cumulative_log_returns = [sum(self.log_returns[cutoff + i:cutoff + self.horizon + i]) for i in
                                  range(len(self.log_returns) - cutoff - self.horizon + 1)]
        var_series = var_series[:len(cumulative_log_returns)]  # cut off VaR series at end

        breaches = 0
        consecutive_count = 0
        is_consecutive = False
        breach_data = {"return": [], "time": []}
        for day, var_data in enumerate(list(zip(cumulative_log_returns, var_series))):
            log_return, var = var_data
            if log_return < var < 0:
                breaches += 1
                breach_data["return"].append(log_return)
                breach_data["time"].append(day)
                if is_consecutive:
                    consecutive_count += 1
                is_consecutive = True
            else:
                is_consecutive = False
        test_cases = len(cumulative_log_returns)
        table_data = [["Sample Size", test_cases],
                      ["Total Breaches", breaches],
                      ["% Breaches", f"{round(breaches / test_cases * 100, 4)}%"],
                      ["Total Consecutive Breaches", consecutive_count],
                      ["% Consecutive Breaches", f"{round(consecutive_count / test_cases * 100, 4)}%"]]
        print(tabulate(table_data))

        plt.plot(range(len(cumulative_log_returns)), cumulative_log_returns, label=f'{horizon}-day Log Returns', linewidth=1.0)
        plt.plot(range(len(var_series)), var_series, label='%.0f' % (100 * confidence) + f'% /{horizon}-day VaR', linewidth=1.0)
        plt.scatter(breach_data["time"], breach_data["return"], label='VaR Breach', c='k')
        plt.legend(loc='best')
        plt.xlabel("Days")
        plt.title(plot_title)
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv('data/Data_SP500.csv', index_col=0, parse_dates=True)
    confidence = 0.99
    horizon = 10
    value_at_risk_backtest = VaRBackTest(df, confidence, horizon, sample_ddof=1)

    rolling_stepsize = 21
    rolling_volatility = value_at_risk_backtest.rolling_std_deviation(window_size=rolling_stepsize)
    # rolling backtest
    value_at_risk_backtest.backtest_var(
        rolling_volatility, cutoff=rolling_stepsize+1,
        plot_title='Fig. 4: S&P500 VaR Backtesting using Equally Weighted Rolling Volatility method')

    ewma_forecast = value_at_risk_backtest.exponentially_weighted_moving_average(decay=0.72)
    # ewma backtest
    value_at_risk_backtest.backtest_var(
        ewma_forecast, cutoff=2,
        plot_title='Fig. 5: S&P500 VaR Backtesting using Exponentially Weighted Volatility Estimate')

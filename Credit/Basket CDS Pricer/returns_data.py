import os

import numpy as np
import pandas as pd
import yfinance as yf


__all__ = ['get_returns', 'TICKERS']


TICKERS = ['JPM', 'C', 'GS', 'BAC', 'MS']


def print_full(x):
    """
    :param x: Dataframe to be printed in full
    :return:  None, prints full Dataframe to console without truncation
    """
    pd.set_option('display.max_rows', x.shape[0])
    pd.set_option('display.max_columns', x.shape[1])  # or 1000
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.precision', 2)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.expand_frame_repr')
    pd.reset_option('display.precision')


def download_prices(tickers, start, end, save):
    """
    :param tickers: list of Yahoo Finance tickers to be downloaded
    :param start: start date, YYYY-MM-DD
    :param end: end date, YYYY-MM-DD
    :param save: set to True to save downloaded prices
    :return: closing prices of tickers from start to end date
    """
    df = yf.download(tickers, start=start, end=end)

    # reset datetime index to date only index
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index('Date', inplace=True)

    if isinstance(tickers, list) and len(tickers) > 1:
        keep_cols = [('Close', ticker) for ticker in tickers]
        drop_cols = [c for c in df.columns if c not in keep_cols]
        close_data = df.drop(drop_cols, axis=1)
        close_data.columns = [col[1] for col in close_data.columns.values]
    else:
        keep_cols = ['Close']
        drop_cols = [c for c in df.columns if c not in keep_cols]
        close_data = df.drop(drop_cols, axis=1)
        close_data.columns = [tickers[0]]
    close_data = close_data[list(tickers)]  # adjust order to match order of tickers list

    current_path = os.getcwd()
    target_path = os.sep.join([current_path, 'data'])
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    if save:
        close_data.to_csv(os.sep.join([target_path, 'equity_prices.csv']))
    return close_data


def get_prices(target='data/equity_prices.csv'):
    """
    :param target: data folder to read prices from (csv)
    :return: dataframe read from target csv. Use this instead of downloading repeatedly.
    """
    try:
        current_path = os.getcwd()
        target_path = os.sep.join([current_path, target])
        df = pd.read_csv(target_path, index_col=0, dayfirst=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(exc)
    except Exception:
        raise
    return df


def inspect_data(frame):
    """
    :param frame: dataframe to inspect
    :return: print Null, NaN values in dataframe as well as shape
    """
    print(f"Null data: \n{frame.isnull().sum()} \n")
    print(f"NaN data: \n{frame.isna().sum()} \n")
    print(f"Shape: {frame.shape}")


def get_returns(tickers=TICKERS, start='2018-05-12', end='2023-05-12', inspect=True, download=False, save=True):
    """
    :param tickers: list of tickers to get returns timeseries from
    :param start: timeseries start date
    :param end: timeseries end date
    :param inspect: set to True to inspect timeseries for Null and NaN entries
    :param download: set to True to download data from yfinance, set to False to read from file
    :param save: set to True to save downloaded data from yfinance to csv
    :return: timeseries of daily log returns for given tickers
    """
    if download:
        df = download_prices(tickers, start, end, save)
    else:
        df = get_prices()

    returns = np.log(df).diff()
    returns.drop(index=returns.index[0], axis=0, inplace=True)

    returns.dropna(inplace=True)
    if inspect:
        print("Checking returns data for invalid values..")
        inspect_data(returns)
        print("\nFirst five rows:")
        print_full(returns.head())
        print("\nLast five rows:")
        print_full(returns.tail())
    return returns


def test_run():
    """
    :return: test run, load returns and check if log returns are gaussian distributed by uniformity plot after
             applying inverse normal cdf to z-scores of log returns, can also plot prices
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    from scipy.stats import norm, zscore

    import seaborn as sns

    sns.set(rc={'figure.figsize': (7, 7)})

    def plot_gaussian_uniformity(log_returns):
        tickers = list(log_returns.columns)
        z_scores = zscore(log_returns, ddof=1)
        fig, ax = plt.subplots(1, len(tickers), figsize=(12, 2.5), constrained_layout=True)
        for index, ticker in enumerate(tickers):
            dat = z_scores[ticker]
            pseudo_uniform = norm.cdf(dat)
            ax[index].hist(pseudo_uniform, bins=100, density=True)
            ax[index].set_title(ticker)
        fig.suptitle('Uniformity plots of historical log returns using normality assumption.', fontsize=16)
        plt.show()

    def plot_prices(prices, offset=False):
        tickers = list(prices.columns)
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in prices.index]
        years = mdates.YearLocator()
        months = mdates.MonthLocator()
        yearsFmt = mdates.DateFormatter('%Y')
        fig, ax = plt.subplots(figsize=(12, 7))
        if offset:
            offset = np.max(np.array(prices))
        else:
            offset = 0
        print(offset)
        for index, ticker in enumerate(tickers):
            ax.plot(dates, np.array(prices[ticker]) + 2*offset*index, label=f'{ticker}')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

        plt.grid(True)
        plt.xlabel('Time (years)')
        plt.legend(loc='best')
        plt.title('Time Series of Log Returns')
        # fig.autofmt_xdate()

        plt.show()

    data = get_returns(TICKERS, download=False, save=False)
    # print(data.head())
    # plot_gaussian_uniformity(data)
    # prices = get_prices()
    # plot_prices(prices)
    returns = zscore(get_returns(), ddof=1)
    plot_prices(returns, offset=False)


if __name__ == '__main__':
    test_run()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from load_spx_data import get_prices


def linear_fit_global():
    data = get_prices()
    data.index = pd.to_datetime(data.index)
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    ax1.plot(data.index, data['^VIX'], 'b-')
    ax2.plot(data.index, data['^SPX'], 'k-')

    ax1.set_title("VIX and SPX500 time series")
    ax1.set_ylabel('VIX')
    ax2.set_ylabel('SPX')

    ax1.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(data['^SPX'], data['^VIX'], marker='.', c=list(range(len(data.index))), s=2)

    ax.set_title('VIX vs SPX')
    ax.set_ylabel('VIX')
    ax.set_xlabel('SPX')

    plt.show()

    pass


def linear_fit_piecewise():
    pass


def quadratic_fit():
    pass


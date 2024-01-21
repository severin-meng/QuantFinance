import os

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

sns.set(rc={'figure.figsize': (14, 7)})

import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    train_test_split,
)


__all__ = ['get_feature_data',
           'get_class_balance']


def print_full(x):
    pd.set_option('display.max_rows', x.shape[0])
    pd.set_option('display.max_columns', x.shape[1])  # or 1000
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.precision', 2)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.expand_frame_repr')
    pd.reset_option('display.precision')


def download_data(ticker='^SSMI', start='2018-05-12', end='2023-05-12'):
    df = yf.download(ticker, start=start, end=end)

    # reset datetime index to date only index
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index('Date', inplace=True)

    current_path = os.getcwd()
    target_path = os.sep.join([current_path, 'data'])
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    df.to_csv(os.sep.join([target_path, 'SMI.csv']))
    return df


def get_data(target='data/SMI.csv'):
    try:
        current_path = os.getcwd()
        target_path = os.sep.join([current_path, target])
        df = pd.read_csv(target_path, index_col=0, dayfirst=True)
    except FileNotFoundError:
        df = download_data()
    except Exception:
        raise
    return df


def add_return(df):
    df['Return'] = np.log(df['Close'].shift(-1) / df['Close'])
    return df


def inspect_data(frame):
    print(f"Null data: \n{frame.isnull().sum()} \n")
    print(f"NaN data: \n{frame.isna().sum()} \n")
    print(f"Shape: {frame.shape}")


def create_features(frame, threshold=0):
    df = frame.copy()
    df = df.drop(['Adj Close', 'Volume'], axis=1)

    df['O-C'] = df['Open'] - df['Close']
    df['H-L'] = df['High'] - df['Low']
    df['Return'] = np.log(df['Close'].shift(-1) / df['Close'])

    # lagged returns
    return_range = [1, 2, 3, 4, 5]  # sign of RET_0 is what we try to predict -> must not use it as a feature!
    for k in return_range:
        df[f'RET_{k}'] = df['Return'].shift(k)

    df['SIGN'] = np.sign(df['RET_1'])

    # momentum
    momentum_range = [1, 2, 5, 10, 20, 30, 50]
    for k in momentum_range:
        df[f'MOM_{k}'] = df['Close'] - df['Close'].shift(k)

    # moving average
    ma_range = [5, 20, 50]
    for k in ma_range:
        df[f'MA_{k}'] = df['Close'].rolling(window=k).mean()

    ewma_range = [5, 20, 50]
    for k in ewma_range:
        df[f'EWMA_{k}'] = df['Close'].ewm(span=k, adjust=False).mean()

    # compute label
    df['Label'] = np.where(df['Return'] > threshold, 1, 0)  # positive returns are 1, else 0

    # drop NaN values
    df.dropna(inplace=True)
    return df


def step1(inspect=True, download=False):
    # get and split data into train and test set
    if download:
        df = download_data()
    else:
        df = get_data()

    if inspect:
        print("Inspecting data")
        inspect_data(df)

    # after ensuring the data is clean i split into training and test data to prevent data leakage.
    train, test = train_test_split(df, test_size=0.2, random_state=0, shuffle=False)

    train = add_return(train)
    train.dropna(inplace=True)

    all_data = add_return(df)
    all_data.dropna(inplace=True)
    if inspect:
        print("\nFirst five rows:")
        print_full(train.head())
        print("\nLast five rows:")
        print_full(train.tail())
    return train, test, all_data


def step2(train, show_plot=False):
    # eda, find threshold
    ret = pd.DataFrame(train['Return'], columns=['Return'])
    norm_cutoff = 0.3262
    # scale threshold back!
    std = train['Return'].std()
    mean = train['Return'].mean()
    threshold = norm_cutoff * std + mean
    if show_plot:
        normalized_ret = pd.DataFrame(StandardScaler().fit_transform(ret), columns=['Norm. Returns'])

        fig, ax = plt.subplots()
        ax.hist(normalized_ret['Norm. Returns'], label="Normalized Returns", bins=150, density=True)
        print("\nPrinting statistics of normalized returns.")
        print("mean: \t\t", normalized_ret['Norm. Returns'].mean())
        print("std dev: \t", normalized_ret['Norm. Returns'].std())
        print("skew: \t\t", normalized_ret['Norm. Returns'].skew())
        print("kurtosis: \t", normalized_ret['Norm. Returns'].kurtosis())
        # narrow peak, fat tailed, slightly skewed towards positive returns

        rng = np.linspace(-10, 7.5, 100)
        ax.plot(rng, norm.pdf(rng), label="Standard Normal")

        # plot threshold
        ax.vlines(norm_cutoff, 0, 0.6, colors=['k'], linestyles='dashed', label=f'{norm_cutoff}')
        ax.legend(loc='best')
        plt.show()

        fig, ax = plt.subplots()
        ax.hist(train['Return'], label="Returns", bins=150, density=False)
        ax.vlines(threshold, 0, 100, colors=['k'], linestyles='dashed', label=f'{threshold}')
        ax.legend(loc='best')
        plt.show()

    return threshold


def step3(full, threshold, inspect=True):
    full_data = create_features(full, threshold=threshold)
    train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=0, shuffle=False)
    # put a gap between train and test set, to mitigate data leakage
    test_data.drop(index=test_data.index[0], axis=0, inplace=True)

    if inspect:
        print("\nInspecting train data")
        inspect_data(train_data)
        print("\nInspecting test data")
        inspect_data(test_data)
        print("\nInspecting full data")
        inspect_data(full_data)

        print(f"\nFirst five train rows:")
        print_full(train_data.head(5))
        print(f"\nLast five train rows:")
        print_full(train_data.tail(5))

        print(f"\nFirst five test rows:")
        print_full(test_data.head(5))
        print(f"\nLast five test rows:")
        print_full(test_data.tail(5))

        print("\nDescribe full data set:")
        print_full(train_data.describe())

    X_train = train_data.drop(['Label', 'Return'], axis=1)
    y_train = train_data['Label']

    X_test = test_data.drop(['Label', 'Return'], axis=1)
    y_test = test_data['Label']
    return X_train, y_train, X_test, y_test, train_data


def get_class_balance(y, bias=1, inspect=False):
    if inspect:
        print("\nChecking for class imbalance")
        print(np.sum(y), y.shape, np.sum(y) / len(y))
    balanced_weights = {
        1: len(y) / (2 * np.sum(y)),
        0: len(y) / (2 * (len(y) - np.sum(y)))
    }
    weights = balanced_weights.copy()
    weights[1] *= bias
    return weights


def get_feature_data(show_return_distribution=False, inspect=False):
    train, test, all_data = step1(inspect=inspect)
    threshold = step2(train, show_plot=show_return_distribution)
    X_train, y_train, X_test, y_test, train_data = step3(all_data, threshold, inspect=inspect)
    return train_data, X_train, y_train, X_test, y_test


if __name__ == '__main__':
    rain_data, X_train, y_train, X_test, y_test = get_feature_data(show_return_distribution=True, inspect=False)
    print(get_class_balance(y_train, inspect=False, bias=1.02))

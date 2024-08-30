import pandas as pd
import pandas_ta as ta
import numpy as np
from functools import reduce


def momentum_state(momentum, threshold):
    return np.where(momentum > threshold, 1, np.where(momentum < -threshold, -1, 0))


def vwma(data: pd.DataFrame, timeperiod: int = 100):
    tot_vol = data.Volume.rolling(timeperiod).sum()
    return (data.Close * data.Volume).rolling(timeperiod).mean() / tot_vol


def chandelier_exit(data: pd.DataFrame, atr_length: int = 14, multiplier: int = 2, high_length: int = 22,
                    low_length: int = 22):
    # return direction indicator of Chandelier Exit
    # direction = 1.0 means: closing > high-ATR and closing > low+ATR   -> close short position
    # direction = -1.0 means: closing < low+ATR drift and closing < high-ATR  -> close long position
    # direction = 0.0: does not exist, as 0 and nan are forward filled.
    chdl_data = data.ta.chandelier_exit(high_length=high_length, low_length=low_length, atr_length=atr_length,
                                        multiplier=multiplier)
    _props = f"_{high_length}_{low_length}_{atr_length}_{multiplier}"
    return chdl_data[f"CHDLREXTd{_props}"]


def slope(values, accuracy=1):
    assert accuracy in (1, 2, 3)
    if accuracy == 1:
        return values - np.roll(values, 1)
    elif accuracy == 2:
        return 3/2 * values - 2*np.roll(values, 1) + 1/2 * np.roll(values, 2)
    elif accuracy == 3:
        return 11/6 * values - 3 * np.roll(values, 1) + 3/2 * np.roll(values, 2) - 1/3 * np.roll(values, 3)


def convexity(values: np.ndarray, accuracy=1):
    assert accuracy in (1, 2, 3)
    if accuracy == 1:
        return values - 2*np.roll(values, 1) + np.roll(values, 2)
    elif accuracy == 2:
        return 2 * values - 5*np.roll(values, 1) + 4 * np.roll(values, 2) - np.roll(values, 3)
    elif accuracy == 3:
        return (35/12 * values - 26/3 * np.roll(values, 1) + 19/2 * np.roll(values, 2)
                - 14/3 * np.roll(values, 3) + 11/12 * np.roll(values, 4))


def slope_diff(slope1, slope2):
    return slope1 - slope2


def conv_diff(conv1, conv2):
    return conv1 - conv2
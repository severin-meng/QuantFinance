import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (7, 7)})


# USD yield curve params from FED sheet
BETA0 = 4.41805645
BETA1 = -1.273505705
BETA2 = 5.567297043
BETA3 = -1.647982213
TAU1 = 0.359196851
TAU2 = 4.996274168

B_BASE = (BETA0, BETA1, BETA2, BETA3, TAU1, TAU2)

__all__ = [
    'discount_factor',
    'load_spot_rate_params'
]


def svenssons_inst_fwd(m: float, b: tuple) -> float:
    """
    :param m: settlement time
    :param b: svensson parameters
    :return: instantaneous forward rate
    """
    # instantaneous forward rate
    beta0, beta1, beta2, beta3, tau1, tau2 = b
    return (beta0 +
            beta1*np.exp(-m/tau1) +
            beta2*m/tau1 * np.exp(-m/tau1) +
            beta3 * m/tau2 * np.exp(-m/tau2))


def svensson_spot_rate(m: float, b: tuple) -> float:
    """
    :param m: time from today to spot date
    :param b: svensson parameters
    :return: spot rate
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = b

    def frac(tau):
        return (1 - np.exp(-m/tau))/(m/tau)
    return (beta0 +
            beta1 * frac(tau1) +
            beta2 * (frac(tau1) - np.exp(-m/tau1)) +
            beta3 * (frac(tau2) - np.exp(-m/tau2)))


def discount_factor(m: float, b: tuple = B_BASE, spot_rate_shift=0) -> float:
    """
    :param m: time from today to spot date
    :param b: svensson parameters
    :param spot_rate_shift: optional parallel shift to spot rate
    :return: discount factor
    """
    if m == 0:
        return 1
    disc_rate = svensson_spot_rate(m, b) + spot_rate_shift
    return np.exp(-disc_rate/100 * m)


def forward_rate(m0: float, m: float, b: tuple) -> float:
    """
    :param m0: forward start time
    :param m: forward end time
    :param b: svensson parameters
    :return: forward rate
    """
    assert m > m0
    total = discount_factor(m, b)
    initial = discount_factor(m0, b)
    cont_fwd_rate = np.log(initial/total)/(m - m0)
    return cont_fwd_rate


def load_spot_rate_params():
    """
    :return: svensson yield curve parameters loaded from the csv
    """
    path = os.sep.join([os.getcwd(), rf'\data\yield_curve_params.csv'])
    col_order = ['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']
    parameters = pd.read_csv(path, index_col=0)
    return parameters[col_order].iloc[0]


def plot_spot_rate(params, save=False):
    """
    :param params: svennson parameters
    :param save: boolean, figure is save if true
    :return: None, plot spot rate over time
    """
    time_range = np.linspace(0, 10, 1000)
    discount_rates = [svensson_spot_rate(r, params) for r in time_range]
    plt.figure(figsize=(9, 6))
    plt.plot(time_range, discount_rates)
    plt.title('Spot Rate Curve')
    plt.xlabel('Tenor (y)')
    plt.ylabel(r'Spot Rate (%)')
    plt.grid(True)
    if save:
        plt.savefig(f'plots/SpotRates.png', dpi='figure')
    plt.show()


def plot_discount_factors(params, save=False):
    """
    :param params: svennson parameters
    :param save: boolean, figure is save if true
    :return: None, plot discount factor curve over time
    """
    time_range = np.linspace(0, 10, 1000)
    discount_rates = [discount_factor(r, params) for r in time_range]
    plt.figure(figsize=(9, 6))
    plt.plot(time_range, discount_rates)
    plt.title('Discount Factor Curve')
    plt.xlabel('Tenor (y)')
    plt.ylabel('Discount Factor')
    plt.grid(True)
    if save:
        plt.savefig(f'plots/DiscountFactors.png', dpi='figure')
    plt.show()


def run_test(save=False):
    """
    :param save:  boolean, figure is save if true
    :return: None, plot spot rate and discount factors over time
    """
    params = load_spot_rate_params()
    plot_spot_rate(params, save=save)
    plot_discount_factors(params, save=save)


if __name__ == '__main__':
    run_test(save=False)

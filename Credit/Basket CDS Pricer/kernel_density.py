import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.integrate import quad
from scipy.stats import norm, zscore, kstest
import os
from itertools import combinations

import seaborn as sns

sns.set(rc={'figure.figsize': (7, 7)})


__all__ = ['kernel_density_estimation',
           'plot_gaussian_uniformity',
           'plot_fitted_uniformity',
           'fit_kernel_density']


def plot_gaussian_uniformity(log_returns, save=False):
    """
    :param log_returns: time series of log returns
    :param save: set to true to save gaussian uniformity plot
    :return: histogram plot, checks uniformity under gaussian hypothesis by applying the gaussian cdf to the log returns
    """
    tickers = list(log_returns.columns)
    z_scores = zscore(log_returns, ddof=1)
    fig, ax = plt.subplots(1, len(tickers), figsize=(12, 2.5), constrained_layout=True)
    for index, ticker in enumerate(tickers):
        data = z_scores[ticker]
        pseudo_uniform = norm.cdf(data)
        ax[index].hist(pseudo_uniform, bins=20, density=True)
        ax[index].set_title(ticker)
    fig.suptitle('Historical Pseudo-Uniforms using Normal CDF', fontsize=16)
    if save:
        plt.savefig("plots/ReturnsGaussianUniformity.png", dpi='figure')
    plt.show()


def plot_fitted_uniformity(fitted_cdf, save=False):
    """
    :param fitted_cdf: empirical cdf of historical pseudo-uniforms (obtained from kernel-smoothed log returns pdf)
    :param save: set to true to save kernel-smoothed pseudo-uniform uniformity plot
    :return: histogram plot, checks uniformity of historical pseudo-uniforms by applying kernel-smoothed cdf to log
             returns
    """
    tickers = fitted_cdf.columns
    fig, ax = plt.subplots(1, len(tickers), figsize=(12, 2.5), constrained_layout=True)
    for index, ticker in enumerate(tickers):
        ax[index].hist(fitted_cdf[ticker], bins=20, density=True)
        ax[index].set_title(ticker)
    fig.suptitle('Historical Pseudo-Uniforms using Empirical CDF', fontsize=16)
    if save:
        plt.savefig("plots/KernelFittedUniformity.png", dpi='figure')
    plt.show()


def plot_gaussian_copula(log_returns, save=False):
    """
    :param log_returns: time series of log returns
    :param save: set to true to save pairwise co-dependece scatter plots after applying gaussian cdf
    :return: plot pairwise co-dependece scatter plots after applying gaussian cdf to log returns
    """
    z_scores = zscore(log_returns, ddof=1)
    pseudo_uniform = norm.cdf(z_scores).T
    tickers = log_returns.columns

    combos = list(combinations(np.arange(5), 2))
    nbr_plots = len(combos)
    fig, ax = plt.subplots(2, 5, figsize=(13, 5), constrained_layout=True)
    for i in range(nbr_plots):
        a, b = combos[i]
        ax[i % 2, int(i / 2)].scatter(pseudo_uniform[a], pseudo_uniform[b], marker='.', s=5)
        ax[i % 2, int(i / 2)].set_xlabel(tickers[a])
        ax[i % 2, int(i / 2)].set_ylabel(tickers[b])
    fig.suptitle("Co-Dependence of Historical Pseudo-Uniforms using Normal CDF")
    if save:
        plt.savefig("plots/ReturnsGaussianCopula.png", dpi='figure')
    plt.show()


def plot_fitted_copula(fitted_cdf, save=False):
    """
    :param fitted_cdf: empirical cdf of historical pseudo-uniforms (obtained from kernel-smoothed log returns pdf)
    :param save: set to true to save pairwise co-dependece scatter plots of historical pseudo-uniforms
    :return: plot pairwise co-dependece scatter plots of historical pseudo-uniforms
    """
    tickers = fitted_cdf.columns
    combos = list(combinations(np.arange(5), 2))
    nbr_plots = len(combos)
    fig, ax = plt.subplots(2, 5, figsize=(13, 5), constrained_layout=True)
    for i in range(nbr_plots):
        a, b = combos[i]
        ax[i % 2, int(i / 2)].scatter(fitted_cdf[tickers[a]], fitted_cdf[tickers[b]], marker='.', s=5)
        ax[i % 2, int(i / 2)].set_xlabel(tickers[a])
        ax[i % 2, int(i / 2)].set_ylabel(tickers[b])
    fig.suptitle("Co-Dependence of Historical Pseudo-Uniforms using Empirical CDF")
    if save:
        plt.savefig("plots/KernelFittedCopula.png", dpi='figure')
    plt.show()


def kernel_density_estimation(returns, kernel_type='gau', bw='silverman', bw_factor=1.0, interpolate=True,
                              perform_ks_test=True):
    """
    for a more detailed description of the parameters kernel_type, bw, bw_factor: see official statsmodels documentation
    :param returns: timeseries of historical log returns
    :param kernel_type: kernel type to fit to returns data. 'gau' (gaussian) is fast as it has an FFT implementation.
    :param bw: bandwidth setting, float, string or callable is possible.
    :param bw_factor: numerical factor applied to chosen bandwidth
    :param interpolate: set to true to use np.interp linear interpolation between empirical cdf estimated on its
                        support to compute empirical cdf of returns, set to false to use numerical quadrature instead
                        (quadrature is much slower than interpolation, improved accuracy has little impact).
    :param perform_ks_test: set to true to run kolmogorov-smirnov test to check uniformity of the empirical cdf
                            evaluated at log returns, result is printed to console
    :return: pandas dataframe of kernel-smoothed empirical cdf evaluated at historical log returns
    """
    tickers = returns.columns
    cdf_values = {}
    for ticker in tickers:
        return_series = returns[ticker]
        kde = sm.nonparametric.KDEUnivariate(return_series)
        kde.fit(kernel=kernel_type, bw=bw, adjust=bw_factor, fft=kernel_type == 'gau')  # estimate the densities
        # the kde is now fitted at its support, an array of numbers similar to the returns
        # kde.cdf gives an array of cdf values at the support
        # to get the cdf at a different place (from returns) one has to interpolate or integrate the pdf.
        if interpolate:
            cdf_values[ticker] = np.interp(return_series, kde.support, kde.cdf)
        else:
            cdf_values[ticker] = integrate(return_series, kde.evaluate)
        if perform_ks_test:
            gaussian_res = kstest(zscore(return_series), 'norm')
            print(f"Kolmogorov-Smirnov test for ticker \'{ticker}\' assuming normal distribution: "
                  f"statistic = {gaussian_res.statistic}, p value = {gaussian_res.pvalue}.")
            result = kstest(cdf_values[ticker], 'uniform')
            print(f"Kolmogorov-Smirnov uniformity test for ticker \'{ticker}\' after Kernel Density Estimation: "
                  f"statistic = {result.statistic}, p value = {result.pvalue}.")
    return pd.DataFrame(cdf_values)


def integrate(target_val, function):
    """
    :param target_val: upper integration bounds
    :param function: function to integrate from -infinity to target_val
    :return: integral from -infinity to target val of provided function
    """
    integrated_vals = np.zeros_like(target_val)
    for index, val in enumerate(target_val):
        integral = quad(function, -np.inf, val)
        integrated_vals[index] = integral
    return integrated_vals


def interpolate_cdf(xval, support, cdf):
    """
    :param xval: x values where one wishes to know the cdf function at
    :param support: grid where cdf is known
    :param cdf: cdf evaluated at support
    :return: cdf evaluated at xval using linear interpolation. Use np.interp instead
    """
    if xval < support[0]:
        return 0
    if xval < support[-1]:
        return 1
    index_below = np.argmax([x for x in support if x <= xval])
    index_above = index_below + 1
    xlow, xhigh = support[index_below], support[index_above]
    ylow, yhigh = cdf[index_below], cdf[index_above]
    slope = (yhigh-ylow)/(xhigh - ylow)
    return ylow + slope * (xval-xlow)


def fit_kernel_density(log_returns, kernel_type='gau', bw='silverman', bw_factor=1.0, interpolate=True,
                       perform_ks_test=True, plot=False, save=True, from_file=False):
    """
    :param log_returns: timeseries of historical log returns
    :param kernel_type: kernel type used for kernel density estimation. 'gau' recommended due to FFT implementation
    :param bw: bandwidth of kernel smoothing algorithm
    :param bw_factor: numerical factor applied to bandwidth
    :param interpolate: set to True to use linear interpolation of empirical cdf at support to approximate empirical
                        cdf at log returns
    :param perform_ks_test: set to True to apply ks test to check uniformity of empirical cdf evaluated at log returns
    :param plot: set to True to plot co-dependence of pseudo-uniform log-returns
    :param save: set to True to save pseudo-uniforms
    :param from_file: set to True to read pseudo-uniforms from file instead of performing kernel density estimation
    :return: empirical cdf obtained via kernel density estimation evaluated at historical log returns
    """
    if from_file:
        path = os.sep.join([os.getcwd(), rf'\data\uniform_returns.csv'])
        uniform_returns = pd.read_csv(path, index_col=0, parse_dates=True)
        return uniform_returns

    uniform_returns = kernel_density_estimation(log_returns, kernel_type=kernel_type, bw=bw, bw_factor=bw_factor,
                                                interpolate=interpolate, perform_ks_test=perform_ks_test)
    uniform_returns.index = log_returns.index
    if plot:
        plot_gaussian_uniformity(log_returns, save=save)
        plot_fitted_uniformity(uniform_returns, save=save)
        plot_gaussian_copula(log_returns, save=save)
        plot_fitted_copula(uniform_returns, save=save)
    if save:
        path = os.sep.join([os.getcwd(), rf'\data\uniform_returns.csv'])
        uniform_returns.to_csv(path)
    return uniform_returns


def test_run():
    """
    :return: test function to test performance and functionality of kernel density estimation
    """
    from returns_data import get_returns
    ln_returns = get_returns(inspect=False, save=False)
    fit_kernel_density(ln_returns, from_file=False, save=False, plot=True, perform_ks_test=False)


def plot_cdf(ticker='JPM'):
    from returns_data import get_returns
    ln_returns = get_returns(inspect=False, save=False).sort_values(ticker)
    path = os.sep.join([os.getcwd(), rf'\data\uniform_returns.csv'])
    uniform_returns = pd.read_csv(path, index_col=0, parse_dates=True)

    # plot_dat = sorted(list(zip(ln_returns[ticker], uniform_returns[ticker])), key=lambda x: x[0])
    sorted_returns = sorted(ln_returns[ticker].values)
    # sorted_uniforms = [i[1] for i in plot_dat]

    sorted_returns_df = pd.DataFrame({ticker: sorted_returns})

    """smooth_uniforms = kernel_density_estimation(sorted_returns_df, kernel_type='gau', bw='silverman',
                                                bw_factor=0.1,
                                                interpolate=True, perform_ks_test=False)
    normal_uniforms = kernel_density_estimation(sorted_returns_df, kernel_type='gau', bw='silverman',
                                                bw_factor=0.8,
                                                interpolate=True, perform_ks_test=False)
    rough_uniforms = kernel_density_estimation(sorted_returns_df, kernel_type='gau', bw='silverman',
                                               bw_factor=2.5,
                                               interpolate=True, perform_ks_test=False)"""

    mean = np.mean(sorted_returns)
    std = np.std(sorted_returns)
    gaussians = norm.cdf(sorted_returns, loc=mean, scale=std)

    # plt.figure(10, 7)
    sns.ecdfplot(sorted_returns, label='Empirical CDF', linewidth=2)
    """plt.plot(sorted_returns, smooth_uniforms, label='Small bandwidth KDE', linewidth=2)
    plt.plot(sorted_returns, normal_uniforms, label='Medium bandwidth KDE', linewidth=2)
    plt.plot(sorted_returns, rough_uniforms, label='Large bandwidth KDE', linewidth=2)"""
    plt.plot(sorted_returns, gaussians, label='Normal CDF')
    plt.xlim(-0.1, 0.1)
    plt.legend(loc='best')
    plt.title("Kernel Density Estimation")
    plt.show()


if __name__ == '__main__':
    # test_run()
    plot_cdf()

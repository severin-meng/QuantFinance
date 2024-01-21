from hazard_rate_bootstrapping import survival_probability
import numpy as np

__all__ = ['default_time_func']


def imply_default_time(uniform, hazard_rates, hazard_dates, cumulative_survival):
    """
    :param uniform: number between 0 and 1 (0 and 1 possible), denotes cdf at default time
    :param hazard_rates: piece-wise constant hazard rates per underlying
    :param hazard_dates: cutoff time of piece-wise constant hazard rates
    :param cumulative_survival: cumulative survival distribution evaluated at cutoff times
    :return: default time implied from input uniform "cdf at default time", i.e. analytical inverse cdf of uniform
    """
    assert 1 >= uniform >= 0
    if uniform == 0:
        return 0
    if uniform == 1:
        return hazard_dates[-1] + uniform
    if np.log(1-uniform) < cumulative_survival[-1]:
        # defaults after maturity -> return generic result, does not matter
        return hazard_dates[-1] + 1

    integer_maturity_index = np.max(np.where(cumulative_survival > np.log(1-uniform)))
    integer_maturity = hazard_dates[integer_maturity_index]
    integer_survival = cumulative_survival[integer_maturity_index]
    residual = np.log(1-uniform) - integer_survival
    if residual == 0:
        return integer_maturity
    fraction = -residual / hazard_rates[integer_maturity_index + 1]
    return integer_maturity + fraction


def get_default_times(uniforms, hazard_df):
    """
    :param uniforms: uniform cdf at default time
    :param hazard_df: dataframe containing piece-wise constant hazard rates and cutoff times
    :return: default times matching input cdf of default times using analytical inverse cdf
    """
    if hazard_df is None:
        return uniforms
    tickers = hazard_df.columns
    dates = np.array(hazard_df.index)
    hazard_timespans = np.zeros_like(dates)
    hazard_timespans[1:] = np.diff(dates)

    default_times_arr = np.zeros_like(uniforms)

    for index, ticker in enumerate(tickers):
        rates = np.array(hazard_df[ticker])
        uniform_per_ticker = np.array(uniforms[:][index])
        cumulative_survival = -np.cumsum(rates * hazard_timespans)
        default_times = np.array([imply_default_time(u, rates, dates, cumulative_survival) for u in uniform_per_ticker])
        default_times_arr[:][index] = default_times
    return default_times_arr


def default_time_func(hazard_df, interp=False):
    """
    :param hazard_df: dataframe containing piece-wise constant hazard rates and cutoff times
    :param interp: set to True to use interpolated inverse cdf to get default time, set to False to use analytical
                   inverse cdf (much slower, added accuracy not needed in JP Morgan model)
    :return: function that computes the inverse cdf of default times
    """
    if interp:
        grid_cdf = cdf_payment_grid(hazard_df)

        def funct(uniforms):
            return get_default_times_interpolated(uniforms, grid_cdf, hazard_df)
    else:
        def funct(uniforms):
            return get_default_times(uniforms, hazard_df)
    return funct


def cdf_payment_grid(hazard_df, interest_freq=0.25):
    """
    :param hazard_df: dataframe containing piece-wise constant hazard rates and cutoff times
    :param interest_freq: interest payment frequency
    :return: default time cdf evaluated at interest payment times per ticker
    """
    dates = np.array(hazard_df.index)
    payments = round(dates[-1]/interest_freq)
    delta_ts = [0.] + [interest_freq for i in range(payments)]
    interest_time_series = list(np.cumsum(delta_ts))
    interest_time_series += [interest_time_series[-1] + 1]
    # get uniform value for all interest time series payments
    grid_cdf = {}
    for ticker in hazard_df.columns:
        rates = np.array(hazard_df[ticker])
        uniform_payments = [1 - survival_probability(t, rates, dates) for t in interest_time_series[:-1]]
        uniform_payments.append(1)
        # x value: uniform: cdf of default time
        # y value: interest time series
        grid_cdf[ticker] = [uniform_payments, interest_time_series]
    return grid_cdf


def get_default_times_interpolated(uniforms, grid_cdf, hazard_df):
    """
    :param uniforms: cdf of sampled default time
    :param grid_cdf: grid of cdf of default time evaluated at interest payment times
    :param hazard_df: dataframe containing piece-wise constant hazard rates and cutoff times
    :return: linearly interpolated default times from sampled cdf of default times
    """
    default_times_arr = np.zeros_like(uniforms)
    for index, ticker in enumerate(hazard_df.columns):
        u, t = grid_cdf[ticker]
        uniform_per_ticker = np.array(uniforms[:][index])
        default_times_arr[:][index] = np.interp(uniform_per_ticker, u, t)
    return default_times_arr

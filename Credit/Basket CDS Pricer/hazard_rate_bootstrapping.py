import numpy as np
import pandas as pd
from discount_factor import discount_factor as discount_func
from scipy.optimize import root_scalar
from math import ceil
import matplotlib.pyplot as plt

import os

import seaborn as sns

sns.set(rc={'figure.figsize': (7, 7)})

__all__ = [
    'survival_probability',
    'fit_hazard_rates',
    'load_cds_prices',
    'risky_bond_price']


def find_root(func, xmin=0, xmax=0.04, retries=10):
    """
    :param func: function to find root of
    :param xmin: lower bound of interval to search root in
    :param xmax: upper bound of interval to search root in
    :param retries: number of retries of increas9jg upper bound of search interval by a factor of two if func(xmin) and
                    func(xmax) have the same sign
    :return: root of func, uses bisection algorithm
    """
    try:
        res = root_scalar(func, method='bisect', bracket=[xmin, xmax], xtol=1e-15)
    except Exception as e:
        if retries > 0:
            res = find_root(func, xmin, xmax=2*xmax, retries=retries-1)
        else:
            raise e

    if not res.converged:
        raise Exception("Failed to imply hazard rate from cds spreads.")
    return res


def survival_probability(t, hazard_rates, hazard_dates):
    """
    :param t: survival time
    :param hazard_rates: piece-wise constant hazard rates
    :param hazard_dates: cutoff times of hazard rates
    :return: probability of surviving until time t, i.e. cdf of survival at t
    """
    if t == 0:
        return 1
    delta_t = np.diff(hazard_dates)
    rates = hazard_rates[1:]
    dates = list(hazard_dates[1:])
    if t <= dates[0]:
        return np.exp(-t * rates[0])
    elif t > dates[-1]:
        # assume constant extrapolation
        sums = np.dot(delta_t, rates)
        sums += (t-dates[-1]) * rates[-1]
        return np.exp(-sums)
    # somewhere in between
    max_date = max([d for d in dates if d < t])
    index = np.where(dates == max_date)[0][0]
    sums = np.dot(delta_t[:(index + 1)], rates[:(index + 1)])
    sums += (t-max_date) * rates[index+1]
    return np.exp(-sums)


def hazard_rate_bootstrapper(maturities, spreads, interest_freq=0.25, accrued_time=0.0, recovery=0.4,
                             halfway_default=True, discount_factor=discount_func, clean_price=False):
    """
    :param maturities: maturities of single-underlying cds in years
    :param spreads: market spreads of cds
    :param interest_freq: interest payment frequency in years
    :param accrued_time: accrued time of cds
    :param recovery: assumed recovery rate for calculating hazard rates
    :param halfway_default: set to True to use JP Morgan model for halfway accrued interest payments in case of default
    :param discount_factor: callable that returns discount factor given time
    :param clean_price: set to True to assume a clean price instead of a dirty price
    :return: dataframe containing hazard rates and cutoff dates implied from market spreads
    """

    mat = np.zeros(len(maturities) + 1)
    mat[1:] = maturities
    maturities = mat

    spr = np.zeros(len(spreads) + 1)
    spr[1:] = spreads
    spreads = spr

    assert accrued_time < maturities[1]
    # make sure maturities, spreads have a leading zero
    hazard_rates = np.zeros(len(spreads))
    hazard_dates = [0] + [m - accrued_time for m in maturities[1:]]
    hazard_dates = np.array(hazard_dates)

    # loss rate
    loss = 1.0 - recovery

    # first hazard rate
    spread1 = spreads[1]
    maturity1 = maturities[1]
    if accrued_time > 1.0/252:
        payments1 = ceil((maturity1 - accrued_time) / interest_freq)
    else:
        payments1 = round(maturity1/interest_freq)

    first_payment1 = interest_freq - accrued_time % interest_freq
    delta_ts1 = [interest_freq for i in range(payments1)]  # year fractions
    if first_payment1 > 1/252:
        delta_ts1[0] = first_payment1
    interest_time_series1 = list(np.cumsum(delta_ts1))
    discount_factors1 = [discount_factor(t) for t in interest_time_series1]

    if not clean_price:
        # if we want the dirty price, then one has to pay full interest for every period,
        # especially for the potentially incomplete first one.
        delta_ts1[0] = interest_freq

    def first_cds_valuation(hazard, intra_period=halfway_default):
        sums = 0
        for disc, t, delta_t in zip(discount_factors1, interest_time_series1, delta_ts1):
            t_prev = max(0, t-delta_t)
            if intra_period:
                bracket = loss - spread1 * delta_t/2 - np.exp(-hazard * (t-t_prev))*(loss + spread1 * delta_t/2)
            else:
                bracket = loss - np.exp(-hazard * (t-t_prev)) * (loss + spread1 * delta_t)
            sums += disc * np.exp(-hazard * t_prev) * bracket
        return sums

    hazard_rates[1] = find_root(first_cds_valuation).root

    default_legs = list(np.zeros_like(maturities))
    base_premium_legs = list(np.zeros_like(maturities))

    survival_probs1 = [1] + [survival_probability(t, hazard_rates, hazard_dates) for t in interest_time_series1]

    mydct = {'Maturity': [maturity1 for i in range(payments1 + 1)],
             'Spread': [spread1 for i in range(payments1 + 1)],
             'Hazard': [hazard_rates[1] for i in range(payments1 + 1)],
             'Interest Paydate': [0.0] + interest_time_series1,
             'Dt': [0.0] + delta_ts1,
             'Df': [1.0] + discount_factors1,
             'Survival': survival_probs1,
             'DeltaSurvival': [0] + list(-np.diff(survival_probs1)),
             'AverageSurvival': [0] + list(np.array(survival_probs1[1:]) - 0.5 * np.diff(survival_probs1))
             }

    df = pd.DataFrame(mydct)

    default_legs[1] = loss * np.dot(df['Df'], df['DeltaSurvival'])
    if halfway_default:
        base_premium_legs[1] = - np.sum(df['Dt'] * df['Df'] * df['AverageSurvival'])
    else:
        base_premium_legs[1] = - np.sum(df['Dt'] * df['Df'] * df['Survival'])

    # build full time series of interest payments
    interest_time_series = [0.] + interest_time_series1
    alpha_time_series = [0.] + delta_ts1
    tenor_diffs = np.diff(hazard_dates[1:])
    nbr_payments = [1, payments1]
    discount_factors = [1.0] + discount_factors1
    for tenor_diff, start_date in zip(tenor_diffs, hazard_dates[1:-1]):
        payments = round(tenor_diff/interest_freq)
        delta_ts = [interest_freq for i in range(payments)]
        interest_ts = np.ones_like(delta_ts) * start_date + np.cumsum(delta_ts)
        alpha_time_series += delta_ts
        interest_time_series += list(interest_ts)
        nbr_payments.append(payments)
        discount_factors += [discount_factor(t) for t in interest_ts]

    mydct = {'Maturity': [maturities[idx] for idx, nbr_payment in enumerate(nbr_payments)
                          for i in range(nbr_payment)],
             'Spread': [spreads[idx] for idx, nbr_payment in enumerate(nbr_payments)
                        for i in range(nbr_payment)],
             'Hazard': [hazard_rates[idx] for idx, nbr_payment in enumerate(nbr_payments)
                        for i in range(nbr_payment)],
             'Interest Paydate': interest_time_series,
             'Dt': alpha_time_series,
             'Df': discount_factors,
             'Survival': survival_probs1 + list(np.ones(len(alpha_time_series) - len(survival_probs1))),
             'DeltaSurvival': [0] + list(-np.diff(survival_probs1)) + list(
                 np.zeros(len(alpha_time_series) - len(survival_probs1))),
             'AverageSurvival': [0] + list(np.array(survival_probs1[1:]) - 0.5 * np.diff(survival_probs1)) + list(
                 np.ones(len(alpha_time_series) - len(survival_probs1))),
             }

    df = pd.DataFrame(mydct)

    # subsequent hazard rates
    for index, (spread, maturity) in enumerate(zip(spreads[2:], maturities[2:])):
        payments = nbr_payments[index+2]
        df_start = int(np.sum(nbr_payments[:index+2]))
        df_end = int(df_start + payments)
        data_range = range(df_start, df_end)

        delta_ts = df.loc[data_range, 'Dt']
        interest_paydates = df.loc[data_range, 'Interest Paydate']
        discount_factors = df.loc[data_range, 'Df']

        # calculate previous tenor's contribution
        prev_contribution = np.sum(default_legs[:index+2]) + spread * np.sum(base_premium_legs[:index+2])

        # current contribution
        def any_maturity_cds_valuation(latest_hazard, intra_period=halfway_default):
            sums = prev_contribution
            for disc, t, delta_t in zip(discount_factors, interest_paydates, delta_ts):
                if intra_period:
                    first = disc * df.loc[df_start-1, 'Survival'] * np.exp(
                        -latest_hazard * (t - delta_t - df.loc[df_start-1, 'Interest Paydate']))
                    a_k_i = (loss - spread * delta_t / 2) / (loss + spread * delta_t/2)
                    second = (loss + spread * delta_t / 2) * (a_k_i - np.exp(-latest_hazard * delta_t))
                else:
                    first = disc * df.loc[df_start - 1, 'Survival'] * np.exp(
                        -latest_hazard * (t - delta_t - df.loc[df_start - 1, 'Interest Paydate']))
                    a_k_i = loss / (loss + spread * delta_t)
                    second = (loss + spread * delta_t) * (a_k_i - np.exp(-latest_hazard * delta_t))
                sums += first * second
            return sums

        hazard_rates[index+2] = find_root(any_maturity_cds_valuation).root
        df.loc[data_range, 'Hazard'] = hazard_rates[index+2]
        df.loc[data_range, 'Survival'] = [
            survival_probability(t, hazard_rates, hazard_dates) for t in interest_paydates]
        df.loc[data_range, 'DeltaSurvival'] = list(
            -np.diff(df.loc[range(df_start-1, df_end), 'Survival']))
        df.loc[data_range, 'AverageSurvival'] = list(np.array(
            df.loc[data_range, 'Survival']) - 0.5*np.diff(df.loc[range(df_start - 1, df_end), 'Survival']))

        default_legs[index+2] = loss * np.dot(df.loc[data_range, 'Df'], df.loc[data_range, 'DeltaSurvival'])
        if halfway_default:
            base_premium_legs[index+2] = -np.sum(
                df.loc[data_range, 'Dt'] * df.loc[data_range, 'Df'] * df.loc[data_range, 'AverageSurvival'])
        else:
            base_premium_legs[index + 2] = -np.sum(
                df.loc[data_range, 'Dt'] * df.loc[data_range, 'Df'] * df.loc[data_range, 'Survival'])

    hazard_dict = {'HazardDates': hazard_dates,
                   'Hazard': hazard_rates}
    hazard_df = pd.DataFrame(hazard_dict)
    return hazard_df


def load_cds_prices():
    """
    :return: read cds market spreads from file
    """
    path = os.sep.join([os.getcwd(), rf'\data\cds_quotes.csv'])
    quotes = pd.read_csv(path, index_col=0)
    return quotes


def plot_quotes(quotes, save=False):
    """
    :param quotes: cds market spreads to plot
    :param save: set to True to save plot of market spreads
    :return: show plot of cds market spreads
    """
    tickers = quotes.columns
    colors = plt.cm.get_cmap('hsv', len(tickers)+2)
    plt.figure(figsize=(12, 7))
    for index, ticker in enumerate(tickers):
        plt.plot(quotes.index, quotes[ticker], label=ticker, c=colors(index))
    plt.legend(loc='best')
    plt.title('CDS Quotes')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Market Spread (bps)')
    if save:
        plt.savefig('plots/MarketQuotes.png', dpi='figure')
    plt.show()


def plot_hazard_rates(hazard_df, save=False):
    """
    :param hazard_df: dataframe containing hazard rates and cutoff dates
    :param save: set to True to save plot of piece-wise constant hazard rates
    :return: show plot of piece-wise constant hazard rates
    """
    tickers = hazard_df.columns
    dates = np.array(hazard_df.index)
    colors = plt.cm.get_cmap('hsv', len(tickers)+2)
    plt.figure(figsize=(12, 7))
    for index, ticker in enumerate(tickers):
        rates = np.array(hazard_df[ticker])
        plt.hlines(rates[1:], dates[:-1], dates[1:], label=ticker, colors=colors(index))
    plt.legend(loc='lower right')
    plt.xlabel('Maturity (years)')
    plt.title(f'Market-Implied Hazard Rates')
    plt.ylim(0)
    if save:
        plt.savefig('plots/HazardRates.png', dpi='figure')
    plt.show()


def plot_survival_probs(hazard_df, save=False):
    """
    :param hazard_df: dataframe containing hazard rates and cutoff dates
    :param save: set to True to save plot of exponential survival probability
    :return: show plot of survival probability parameterized by hazard rates
    """
    tickers = hazard_df.columns
    dates = np.array(hazard_df.index)
    colors = plt.cm.get_cmap('hsv', len(tickers)+2)
    time_space = np.linspace(dates[0], dates[-1], 100)
    plt.figure(figsize=(12, 7))
    for index, ticker in enumerate(tickers):
        rates = np.array(hazard_df[ticker])
        probs = [survival_probability(t, rates, dates) for t in time_space]
        plt.plot(time_space, probs, label=ticker, c=colors(index))
    plt.legend(loc='best')
    plt.xlabel('Time (years)')
    plt.title(f'Survival Probability')
    if save:
        plt.savefig('plots/SurvivalProb.png', dpi='figure')
    plt.show()


def fit_hazard_rates(cds_quotes, save=False, from_file=False,  interest_freq=0.25, accrued_time=0.0, recovery=0.4,
                     halfway_default=True, discount_factor=discount_func, clean_price=False, plot=True):
    """
    :param cds_quotes: cds market quotes to fit hazard rates to
    :param save: set to True to save implied hazard rates and plots
    :param from_file: set to True to read hazard rates from file instead of bootstrapping to cds market spreads
    :param interest_freq: frequency of interest payments
    :param accrued_time: accrued time of cds
    :param recovery: assumed recovery rate
    :param halfway_default: set to True to use JP Morgan halfway accrued interest payment in case of default
    :param discount_factor: callable returning discount factor at given time
    :param clean_price: set to true to interpret market spread as clean instead of dirty
    :param plot: set to true to plot market quotes, hazard rates and implied survival probabilities
    :return: dataframe containing bootstrapped hazard rates and cutoff dates implied from market spreads
    """
    if from_file:
        path = os.sep.join([os.getcwd(), rf'\data\hazard_rates.csv'])
        hazard_df = pd.read_csv(path, index_col=0)
        return hazard_df

    tickers = cds_quotes.columns
    if isinstance(recovery, float):
        recovery = {ticker: recovery for ticker in tickers}
    hazard_rates = {}
    hazard_dates = np.zeros(len(cds_quotes.index) + 1)
    hazard_dates[1:] = cds_quotes.index - accrued_time
    for ticker in tickers:
        maturities = cds_quotes.index
        spreads = cds_quotes[ticker] / 10000  # convert to bps

        hazard_data = hazard_rate_bootstrapper(maturities, spreads, interest_freq=interest_freq,
                                               accrued_time=accrued_time, recovery=recovery[ticker],
                                               halfway_default=halfway_default, discount_factor=discount_factor,
                                               clean_price=clean_price)
        hazard_rates[ticker] = np.array(hazard_data['Hazard'])
    hazard_df = pd.DataFrame(hazard_rates, index=hazard_dates)
    if plot:
        plot_quotes(cds_quotes, save=save)
        plot_hazard_rates(hazard_df, save=save)
        plot_survival_probs(hazard_df, save=save)
    if save:
        path = os.sep.join([os.getcwd(), rf'\data\hazard_rates.csv'])
        hazard_df.to_csv(path)
    return hazard_df


def single_cds_pricer(maturity, hazard_dates, hazard_rates, interest_freq=0.25, accrued_time=0.0, recovery=0.4,
                      halfway_default=True, discount_factor=discount_func, clean_price=False):
    """
    :param maturity: maturity of cds to price
    :param hazard_dates: cutoff times for piece-wise constant hazard rates
    :param hazard_rates: piece-wise constant hazard rates
    :param interest_freq: interest payment frequency
    :param accrued_time: accrued time of cds
    :param recovery: assumed recovery rate
    :param halfway_default: set to True to use JP Morgan halfway accrued interest payment in case of default
    :param discount_factor: callable returning discount factor at given time
    :param clean_price: set to true to interpret market spread as clean instead of dirty
    :return: fair spread of single cds assuming given hazard rate structure
    idea: price cds contracts using fitted hazard data as a backtest -> one should get the original market quotes back!
    """
    if accrued_time > 1.0/252:
        payments = ceil((maturity - accrued_time) / interest_freq)
    else:
        payments = round(maturity/interest_freq)

    first_payment = interest_freq - accrued_time % interest_freq
    delta_ts = [interest_freq for i in range(payments)]  # year fractions
    if first_payment > 1/252:
        delta_ts[0] = first_payment
    interest_time_series = list(np.cumsum(delta_ts))
    discount_factors = np.array([discount_factor(t) for t in interest_time_series])

    if not clean_price:
        # if we want the dirty price, then one has to pay full interest for every period,
        # especially for the potentially incomplete first one.
        delta_ts[0] = interest_freq

    survival_probs = np.array([1] + [survival_probability(t, hazard_rates, hazard_dates) for t in interest_time_series])
    diff_survival = np.diff(survival_probs)
    avg_survival = np.array(survival_probs[:-1] + 0.5 * diff_survival)

    default_leg = (1 - recovery) * np.dot(diff_survival, discount_factors)
    if halfway_default:
        premium_leg = -np.sum(delta_ts * discount_factors * avg_survival)
    else:
        premium_leg = -np.sum(delta_ts * discount_factors * survival_probs[1:])
    spread = default_leg / premium_leg
    return spread


def risky_bond_price(maturity, hazard_df, ticker, discount_factor=discount_func, recovery=0.4, risky=True, steps=1000):
    """
    :param maturity: maturity of zero coupon bond
    :param hazard_df: dataframe containing piece-wise constant hazard rates and respective cutoff times
    :param ticker: ticker of underlying, required to pick correct set of hazard rates
    :param discount_factor: callable returning discount factor at given time
    :param recovery: assumed recovery rate of zero coupon bond
    :param risky: set to True to enable risk of default, set to False to compute riskless ZCB (=discount factor)
    :param steps: number of discrete time steps where a default payment could occur.
    :return: theoretical price of a risky zero coupon bond
    """
    dates = np.array(hazard_df.index)
    rates = np.array(hazard_df[ticker])
    disc = discount_factor(maturity)
    if risky:
        survival_prob = survival_probability(maturity, rates, dates)
        default_times = np.linspace(0, maturity, steps)
        disc_factors = np.array([discount_factor(t) for t in default_times])
        surv_probs = np.array([survival_probability(t, rates, dates) for t in default_times])
        default_probs = np.diff(1 - surv_probs)  # probability of default per bucket
        return disc * survival_prob + recovery * np.sum(default_probs * disc_factors[1:])
    else:
        return disc


def backtest(cds_quotes=None, hazard_df=None):
    """
    :param cds_quotes: cds market prices, optional
    :param hazard_df: dataframe containing piece-wise constant hazard rates and respective cutoff times, optional
    :return: prints difference between computed cds spreads using bootstrapped hazard rates and input market quotes
    """
    # backtest: calculate cds prices from hazard rates, compare with original cds quotes.
    if cds_quotes is None:
        cds_quotes = load_cds_prices()
    tickers = cds_quotes.columns
    maturities = cds_quotes.index
    if hazard_df is None:
        hazard_df = fit_hazard_rates(cds_quotes, from_file=True)
    errors = {}
    for ticker in tickers:
        errs = []
        for mat in maturities:
            calc_spread = single_cds_pricer(mat, np.array(hazard_df.index), np.array(hazard_df[ticker])) * 10000
            quote = cds_quotes[ticker][mat]
            error = calc_spread - quote
            errs.append(error)
        errors[ticker] = errs
    err_df = pd.DataFrame(errors, index=maturities)
    print("Overall backtest errors:")
    print(err_df)


def test_run():
    """
    :return: test function to test functionality of hazard rate bootstrapper, shows difference between risky and
             riskless zero coupon bond.
    """
    prices = load_cds_prices()
    haz_df = fit_hazard_rates(prices, save=False, from_file=False)
    # backtest(prices, haz_df)
    mats = np.linspace(0.1, 10, 30)
    prices = []
    for mat in mats:
        riskless = risky_bond_price(mat, haz_df, ticker='JPM', risky=False)
        risky = risky_bond_price(mat, haz_df, ticker='JPM', risky=True)
        prices.append([riskless, risky])
    plt.plot(mats, [p[0] for p in prices], label='riskless')
    plt.plot(mats, [p[1] for p in prices], label='risky')
    plt.legend(loc='best')
    plt.title("Risky vs Riskless Zero Coupon Bond")
    plt.show()


if __name__ == "__main__":
    # test_run()
    prices = load_cds_prices()
    fit_hazard_rates(prices)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from discount_factor import discount_factor as base_discount_factor, load_spot_rate_params
from hazard_rate_bootstrapping import fit_hazard_rates, load_cds_prices, risky_bond_price
from uniform_to_default import default_time_func
from copula_sampling import gaussian_copula_sampling, students_t_copula_sampling
from returns_data import get_returns, TICKERS
from correlations import get_correlations
from kernel_density import fit_kernel_density
from copula_fitter import t_copula_fit_dof
from price_basket_cds import calculate_spread


class BasketCdsPricer:
    """
    Basket CDS pricing class. Implements pricing market spreads, sensitivities, convergence analysis.
    """
    def __init__(self, tickers=TICKERS, interest_frequency=0.25, accrued_time_basket=0.0, accrued_time_single_cds=0.0,
                 halfway_default=True, clean_price=False, base_recovery=0.4, corr_start='2018-05-12',
                 corr_end='2023-05-12', cds_maturity=5, plot=True, download=False, correlation_type='kendall',
                 fit=False):
        """
        :param tickers: list of corporate stock tickers
        :param interest_frequency: set to True to use accrued interest payment upon default of cds underlying
        :param accrued_time_basket: accrued time of basket cds
        :param accrued_time_single_cds: accrued time of single cds (used for bootstrapping)
        :param halfway_default: set to True to account for accrued interest payments upon default
        :param clean_price: set to True to interpret spreads as clean prices, set to False for dirty
        :param base_recovery: recovery rate used for bootstrapping and pricing of cds
        :param corr_start: start date of return time series for correlation estimation
        :param corr_end: end date of return time series for correlation estimation
        :param cds_maturity: basket cds maturity
        :param plot: set to True to show plots generated during data fitting (hazard rate bootstrapping,
                     correlation estimation etc)
        :param download: set to True to download equity prices for correlation estimation, otherwise
                         pre-downloaded data is used.
        :param correlation_type: correlation type used for student-t copula fitting and sampling. choose from 'kendall',
                                 'spearman', 'pearson'.
        :param fit: set to True to fit parameters instead of reading pre-fitted data from files. includes correlation
                    estimation, hazard rate bootstrapping, maximum likelihood estimation for student-t copula fitting
                    and kernel density estimation
        """
        self.tickers = tickers
        self.cds_market_data = load_cds_prices()
        self.spot_rate_params = load_spot_rate_params()
        self.freq = interest_frequency
        self.accrued_basket = accrued_time_basket
        self.accrued_single = accrued_time_single_cds
        self.halfway_default = bool(halfway_default)
        self.clean_price = bool(clean_price)
        self.recovery = base_recovery
        self.maturity = cds_maturity

        self.plot = bool(plot)
        self.download = bool(download)
        self.start = corr_start
        self.end = corr_end
        self.corr_type = correlation_type
        self.fit = bool(fit)

        self.default_func, self.t_correlation, self.g_correlation, self.t_dof = self._fit_data()

    def _fit_data(self, save=True):
        """
        :param save: set to True to save fitted parameters to the data folder
        :return: default time inverse cdf, correlation matrix for student-t copula, correlation matrix for gaussian
                 copula, degrees of freedom for student-t copula
        """
        if self.fit:
            print("Fitting Hazard Curves..")
            hazard_data = fit_hazard_rates(self.cds_market_data, save=save, interest_freq=self.freq,
                                           accrued_time=self.accrued_single, recovery=self.recovery,
                                           halfway_default=self.halfway_default, discount_factor=self._discount_factor,
                                           clean_price=self.clean_price, plot=self.plot)
            print("Fetching Log Returns..")
            log_returns = get_returns(tickers=self.tickers, start=self.start, end=self.end, inspect=self.plot,
                                      download=self.download)
            print("Fitting kernel density..")
            uniforms = fit_kernel_density(log_returns, kernel_type='gau', bw='silverman', interpolate=True,
                                          perform_ks_test=self.plot, plot=self.plot, save=save)
            print("Estimating Correlations..")
            t_correlation = get_correlations(None, uniform_returns=uniforms, corr_type=self.corr_type,
                                             density_estimation=True, save=save, plot=self.plot)
            g_correlation = get_correlations(None, uniform_returns=uniforms, corr_type='pearson',
                                             density_estimation=True, save=save, plot=self.plot)
            print("Estimating t-copula degrees of freedom..")
            t_dof = t_copula_fit_dof(uniforms, t_correlation, plot=self.plot, save=save)
        else:
            hazard_data = fit_hazard_rates(None, from_file=True, discount_factor=self._discount_factor)
            g_correlation = get_correlations(None, corr_type=self.corr_type, density_estimation=True, from_file=True)
            t_correlation = get_correlations(None, corr_type='pearson', density_estimation=True, from_file=True)
            t_dof = t_copula_fit_dof(None, from_file=True)
        default_func = default_time_func(hazard_data, interp=True)
        return default_func, t_correlation, g_correlation, t_dof

    def _refit_hazard(self, market_recovery=None, market_data_scale=None, market_cds_data=None, return_hazard=False,
                      discount_override=None):
        """
        this functions allows refitting the hazard rates and subsequent inverse default time cdf
        :param market_recovery: recovery rate of single underlying cds
        :param market_data_scale: scale the market cds quotes to see sensitivity wrt market quotes
        :param market_cds_data: replacement market cds quotes
        :param return_hazard: set to True to return the bootstrapped hazard rates
        :param discount_override: alternative discount factor function used for bootstrapping
        :return: inverse default time cdf and optionally the bootstrapped hazard rates
        """
        if discount_override is None:
            discount_override = self._discount_factor
        if market_recovery is None:
            market_recovery = self.recovery
        market_data = self.cds_market_data
        if market_data_scale is not None:
            market_data *= market_data_scale
        elif market_cds_data is not None:
            market_data = market_cds_data
        hazard_data = fit_hazard_rates(market_data, save=False, interest_freq=self.freq,
                                       accrued_time=self.accrued_single, recovery=market_recovery,
                                       halfway_default=self.halfway_default, discount_factor=discount_override,
                                       clean_price=self.clean_price, plot=False)
        default_func = default_time_func(hazard_data, interp=True)
        if return_hazard:
            return default_func, hazard_data
        return default_func

    def _discount_factor(self, time):
        """
        :param time: time of discount factor
        :return: discount factor for given time
        """
        return base_discount_factor(time, self.spot_rate_params)

    def get_price_risky_zcb(self, ticker, hazard_df, recovery_override=None, discount_override=None, steps=1000):
        """
        :param ticker: ticker of underlying for which you want to get the risky zero coupon bond.
        :param hazard_df: hazard rates and cutoff dates, to calculate default time probabilities
        :param recovery_override: optionally override the recovery rate of the ZCB
        :param discount_override: optionally provide a different discount factor function
        :param steps: number of time steps within the ZCB lifetime it can default and pay back the recovered amount at
        :return: theoretical price of a risky zero coupon bond
        """
        if recovery_override is None:
            recovery_override = self.recovery
        if discount_override is None:
            discount_override = self._discount_factor
        prc = risky_bond_price(self.maturity, hazard_df, ticker, discount_factor=discount_override,
                               recovery=recovery_override, steps=steps)
        return prc

    def get_price_basket_cds(self, copula='t', k=1, corr_override=None, basket_recovery_override=None, exponent=15,
                             method='sobol', plot_copula=False, factor=1e4, iterative=False, theor_price=False,
                             spread=1, discount_override=None):
        """
        entry point for basket cds pricing
        :param copula: copula model, 't' for student-t and 'gau' for gaussian
        :param k: k-th to default
        :param corr_override: optional alternative correlation matrix
        :param basket_recovery_override: optional alternative recovery rate for the basket cds
        :param exponent: base 2 exponent of nbr of simulations, such that 2^exponent = nbr simulations
        :param method: random or low-discrepancy number generation method, 'sobol', 'halton' (low discrepancy sequences)
                       and 'pseudo' (pseudo-random Mersenne-Twister) are available
        :param plot_copula: set to True to show plots from the copula sampling process
        :param factor: factor to multiply computed spread with. Typically, spreads are quoted in bps, which is why the
                       default is 10^4
        :param iterative: set to True to receive multiple spreads for an increasing number of simulations
        :param theor_price: set to True to receive the theoretical price instead of the spread
        :param spread: optionally provide a spread to compute a theoretical price from
        :param discount_override: optional override to the discount factor function
        :return: basket cds spread (or optionally, theoretical price) and optionally the number of simulation steps
                 per returned spread.
        """
        if corr_override is None:
            corr_override = self.t_correlation if copula == 't' else self.g_correlation
        if basket_recovery_override is None:
            basket_recovery_override = self.recovery
        if discount_override is None:
            discount_override = self._discount_factor
        if copula == 't':
            sampled_def_times = students_t_copula_sampling(self.default_func, corr_override, dof=self.t_dof,
                                                           exponent=exponent, method=method, plot=plot_copula)
        else:
            sampled_def_times = gaussian_copula_sampling(self.default_func, corr_override, exponent=exponent,
                                                         method=method, plot=plot_copula)
        if iterative:
            simulations, spreads = calculate_spread(
                k, sampled_def_times, maturity=self.maturity, accrued_time=self.accrued_basket, interest_freq=self.freq,
                recovery=basket_recovery_override, halfway_accrual=self.halfway_default, clean_price=self.clean_price,
                discount_factor=discount_override, iterative=True, theor_price=theor_price, basket_spread=spread)
            return simulations, factor * spreads
        else:
            spread = calculate_spread(
                k, sampled_def_times, maturity=self.maturity, accrued_time=self.accrued_basket, interest_freq=self.freq,
                recovery=basket_recovery_override, halfway_accrual=self.halfway_default, clean_price=self.clean_price,
                discount_factor=discount_override, iterative=False, theor_price=theor_price, basket_spread=spread)
            return factor * spread

    def correlation_sensitivity(self, save=False, copula='gau', mode='constant', corr1='C', corr2='GS',
                                base_exponent=19):
        """
        calculate and plot correlation sensitivity of basket spread
        :param save: set to True to save the plot
        :param copula: copula method, 'gau' for gaussian, 't' for student-t, 'both' for both methods in one plot
        :param mode: correlation shift method, 'constant' for constant correlation, 'single' to only adjust correlation
                     between corr1 and corr2 (factor), and 'full' for factor correlation adjustment on the full matrix
        :param corr1: ticker 1 for single correlation shift
        :param corr2: ticker 2 for single correlation shift
        :param base_exponent: nbr_simulations = 2^base_exponent
        :return: None, plots and optionally saves the correlation sensitivity of the basket cds spread
        """
        print("Starting Correlation sensitivity analysis..")
        dim = self.t_correlation.shape[0]

        if mode == 'constant':
            def _factor_correlation(factr, **kwargs):
                assert 0 <= factr < 1
                base = np.ones((dim, dim)) * factr
                np.fill_diagonal(base, 1)
                min_eig = min(np.linalg.eigh(base)[0])
                if min_eig <= 0:
                    print(f"Correlation shifted with factor {factr} is not positive semidefinite!")
                    return self.t_correlation if copula == 't' else self.g_correlation
                tickers = self.t_correlation.columns
                return pd.DataFrame(base, index=tickers, columns=tickers)

            factor_range = np.linspace(0.01, 0.99, 15)
            title_addon = 'constant levels'
            x_axis_name = 'Uniform correlation'
        elif mode == 'single' and corr1 in self.tickers and corr2 in self.tickers:
            def _factor_correlation(factr, cop=copula):
                correl = self.t_correlation.copy() if cop == 't' else self.g_correlation.copy()

                res = correl
                base_corr = correl[corr1][corr2]
                res[corr1][corr2] = base_corr + factor * (1 - base_corr)
                res[corr2][corr1] = res[corr1][corr2]

                min_eig = min(np.linalg.eigh(res)[0])
                if min_eig <= 0:
                    print(f"Correlation shifted with factor {factr} is not positive semidefinite!")
                    return self.t_correlation if copula == 't' else self.g_correlation
                return res
            factor_range = np.linspace(-1.2, 0.85, 20)
            title_addon = 'constant levels'
            x_axis_name = f'Relative Correlation Shift {corr1}/{corr2}'
        else:
            def _factor_correlation(factr, cop=copula):
                correl = self.t_correlation.copy() if copula == 't' else self.g_correlation.copy()
                assert -6 < factr < 1
                res = correl + factr * (np.ones_like(correl) - correl)
                min_eig = min(np.linalg.eigh(res)[0])
                if min_eig <= 0:
                    print(f"Correlation shifted with factor {factr} is not positive semidefinite!")
                    return self.t_correlation if cop == 't' else self.g_correlation
                return res

            factor_range = np.linspace(-5.5, 0.95, 20)
            title_addon = 'percentage change'
            x_axis_name = 'Correlation scaling factor'

        print("Calculating prices...")
        if copula == 'both':
            t_prices = {k: [] for k in range(1, 6)}
            g_prices = {k: [] for k in range(1, 6)}
            for k in range(1, 6):
                for factor in factor_range:
                    corr_t = _factor_correlation(factor, cop='t')
                    corr_g = _factor_correlation(factor, cop='gau')
                    t_prices[k].append(self.get_price_basket_cds(copula='t', k=k, corr_override=corr_t,
                                                                 exponent=base_exponent))
                    g_prices[k].append(self.get_price_basket_cds(copula='gau', k=k, corr_override=corr_g,
                                                                 exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            for k in range(1, 6):
                plt.plot(factor_range, t_prices[k], label=f'student-t k = {k}', ls='--')
                plt.plot(factor_range, g_prices[k], label=f'gaussian k = {k}')
        else:
            prices = {k: [] for k in range(1, 6)}
            for k in range(1, 6):
                prices[k] = []
                for factor in factor_range:
                    corr = _factor_correlation(factor)
                    prices[k].append(self.get_price_basket_cds(copula=copula, k=k, corr_override=corr,
                                                               exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            for k in range(1, 6):
                plt.plot(factor_range, prices[k], label=f'default k = {k}')

        plt.legend(loc='center left')
        plt.xlabel(x_axis_name)
        plt.ylabel('Fair Spread (bps)')
        plt.title(f"Correlation Sensitivity ({title_addon})")
        if save:
            plt.savefig('plots/CorrSensitivity.png', dpi='figure')
        plt.show()

        print("End of Correlation sensitivity analysis..")

    def basket_recovery_sensitivity(self, base_exponent=15, copula='gau', save=False):
        """
        basket recovery rate sensitivity plotting
        :param base_exponent: nbr_simulations = 2^base_exponent
        :param copula: copula method, 'gau' for gaussian, 't' for student-t, 'both' for both methods in one plot
        :param save: set to True to save the plot
        :return: None, plots and optionally saves the basket recovery rate sensitivity of the basket cds spread
        """
        print("Starting recovery rate sensitivity analysis..")
        recovery_range = np.linspace(0.1, 0.9, 15)

        print("Calculating prices..")
        if copula == 'both':
            g_prices = {k: [] for k in range(1, 6)}
            t_prices = {k: [] for k in range(1, 6)}
            for k in range(1, 6):
                for r in recovery_range:
                    g_prices[k].append(self.get_price_basket_cds(k=k, copula='gau', basket_recovery_override=r,
                                                                 exponent=base_exponent))
                    t_prices[k].append(self.get_price_basket_cds(k=k, copula='t', basket_recovery_override=r,
                                                                 exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            for k in range(1, 6):
                plt.plot(recovery_range, t_prices[k], label=f'k = {k}, student-t', ls='--')
                plt.plot(recovery_range, g_prices[k], label=f'k = {k}, gaussian')
        else:
            prices = {k: [] for k in range(1, 6)}
            for k in range(1, 6):
                for r in recovery_range:
                    prices[k].append(self.get_price_basket_cds(k=k, copula=copula, basket_recovery_override=r,
                                                               exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            for k in range(1, 6):
                plt.plot(recovery_range, prices[k], label=f'default k = {k}')
        plt.legend(loc='best')
        plt.xlabel('Recovery Rate R')
        plt.ylabel('Fair Spread (bps)')
        plt.title("Basket Recovery Rate Sensitivity")
        if save:
            plt.savefig('plots/BasketRecoverySensitivity.png', dpi='figure')
        plt.show()
        print("End of recovery rate sensitivity analysis.")

    def market_recovery_sensitivity(self, copula='gau', base_exponent=15, save=False, single='',
                                    fixed_basket_recovery=False):
        """
        single market cds recovery rate sensitivity plotting
        :param copula: copula method, 'gau' for gaussian, 't' for student-t, 'both' for both methods in one plot
        :param base_exponent: nbr_simulations = 2^base_exponent
        :param save: set to True to save the plot
        :param single: optionally provide a single ticker. Only for this ticker will the recovery rate be adjusted
        :param fixed_basket_recovery: set to True to keep basket recovery rate fixed when adjusting the market recovery
        :return: Plots and optionally saves the single market cds recovery rate sensitivity of the basket cds spread
        """
        print("Starting market recovery rate sensitivity..")
        market_recovery_range = np.linspace(0.1, 0.9, 15)
        print("Calculating prices..")
        if single and single in self.tickers:
            def market_recovery(rec):
                recovery = {ticker: self.recovery for ticker in self.tickers}
                recovery[single] = rec
                return recovery
            x_label = f' for {single}'
        else:
            def market_recovery(rec):
                return rec
            x_label = ''

        if copula == 'both':
            g_prices = {k: [] for k in range(1, 6)}
            t_prices = {k: [] for k in range(1, 6)}
            for recov in market_recovery_range:
                self.default_func = self._refit_hazard(market_recovery=market_recovery(recov))
                for k in range(1, 6):
                    if not fixed_basket_recovery:
                        g_prices[k].append(self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent,
                                                                     basket_recovery_override=recov))
                        t_prices[k].append(self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent,
                                                                     basket_recovery_override=recov))
                    else:
                        g_prices[k].append(self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent))
                        t_prices[k].append(self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            for k in range(1, 6):
                plt.plot(market_recovery_range, t_prices[k], label=f'k = {k}, student-t', ls='--')
                plt.plot(market_recovery_range, g_prices[k], label=f'k = {k}, gaussian')
        else:
            prices = {k: [] for k in range(1, 6)}
            for recov in market_recovery_range:
                self.default_func = self._refit_hazard(market_recovery=market_recovery(recov))
                for k in range(1, 6):
                    if not fixed_basket_recovery:
                        prices[k].append(self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent,
                                                                   basket_recovery_override=recov))
                    else:
                        prices[k].append(self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            for k in range(1, 6):
                plt.plot(market_recovery_range, prices[k], label=f'default k = {k}')
        plt.legend(loc='best')
        plt.xlabel(f'Recovery Rate R{x_label}')
        plt.ylabel('Fair Spread (bps)')
        market = 'Market ' if not fixed_basket_recovery else ''
        plt.title(f"{market}Recovery Rate Sensitivity")
        if save:
            plt.savefig('plots/MarketRecoverySensitivity.png', dpi='figure')
        plt.show()
        print("End of market recovery rate sensitivity analysis.")

        self.default_func = self._refit_hazard()

    def market_cds_sensitivity(self, copula='gau', base_exponent=15, save=False, single=''):
        """
        single market spread sensitivity plotting
        :param copula: copula method, 'gau' for gaussian, 't' for student-t, 'both' for both methods in one plot
        :param base_exponent: nbr_simulations = 2^base_exponent
        :param save: set to True to save the plot
        :param single: optionally provide a single ticker. Only adjust market spread of this single ticker
        :return: Plots and optionally saves the market cds spread sensitivity of the basket cds spread
        """
        print("Starting market spread sensitivity..")
        market_spread_factors = np.linspace(0.1, 5, 15)
        print("Calculating prices..")
        if single and single in self.tickers:
            def market_cds_scaling(factor):
                scaled_data = factor * self.cds_market_data[single]
                total_data = self.cds_market_data.copy()
                total_data[single] = scaled_data
                return total_data
            x_label = f' for {single}'
        else:
            def market_cds_scaling(factor):
                return factor * self.cds_market_data
            x_label = ''

        if copula == 'both':
            g_prices = {k: [] for k in range(1, 6)}
            t_prices = {k: [] for k in range(1, 6)}
            for spread_factor in market_spread_factors:
                for k in range(1, 6):
                    self.default_func = self._refit_hazard(market_cds_data=market_cds_scaling(spread_factor))
                    g_prices[k].append(self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent))
                    t_prices[k].append(self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            for k in range(1, 6):
                plt.plot(market_spread_factors, t_prices[k], label=f'k = {k}, student-t', ls='--')
                plt.plot(market_spread_factors, g_prices[k], label=f'k = {k}, gaussian')
        else:
            prices = {k: [] for k in range(1, 6)}
            for spread_factor in market_spread_factors:
                for k in range(1, 6):
                    self.default_func = self._refit_hazard(market_cds_data=market_cds_scaling(spread_factor))
                    prices[k].append(self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            for k in range(1, 6):
                plt.plot(market_spread_factors, prices[k], label=f'default k = {k}')
        plt.legend(loc='best')
        plt.xlabel(f'Market Spread Factor{x_label}')
        plt.ylabel('Fair Spread (bps)')
        plt.title("Market Spread Sensitivity")
        if save:
            plt.savefig('plots/MarketRSpreadSensitivity.png', dpi='figure')
        plt.show()
        print("End of market spread sensitivity analysis.")

        self.default_func = self._refit_hazard()

    def market_cds_delta_hedge(self, copula='gau', base_exponent=20, save=False, ticker='JPM', market_spread_ds=20,
                               steps=1000):
        """
        hedge ratio for hedging basket cds with risky zero-coupon bond
        :param copula: copula method, 'gau' for gaussian, 't' for student-t, 'both' for both methods in one plot
        :param base_exponent: nbr_simulations = 2^base_exponent
        :param save: set to True to save the plot
        :param ticker: ticker for which to hedge against
        :param market_spread_ds: shift size of market spread in bps
        :param steps: number of time steps within the ZCB lifetime it can default and pay back the recovered amount at
        :return: Plots and optionally saves the hedge ratio of the basket spread against a risky zero coupon spread
        """
        print("Starting market spread delta hedging..")
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not recognized.")

        base_cds_data = self.cds_market_data.copy()
        shifted_cds_data = base_cds_data
        shifted_cds_data[ticker] += market_spread_ds
        # shifted_cds_data += market_spread_ds

        print("Calculating prices..")
        _, hazard_df = self._refit_hazard(return_hazard=True)
        zcb_base = 1 / 5 * self.get_price_risky_zcb(ticker, hazard_df, steps=steps)
        if copula == 'both':

            g_basket_spreads_base = [self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent, factor=1)
                                     for k in range(1, 6)]
            t_basket_spreads_base = [self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent, factor=1)
                                     for k in range(1, 6)]
            basket_prices_base = np.zeros(5)  # base basket price is 0

            self.default_func, hazard_df = self._refit_hazard(market_cds_data=shifted_cds_data, return_hazard=True)
            zcb_shift = 1 / 5 * self.get_price_risky_zcb(ticker, hazard_df, steps=steps)

            g_basket_prices_shift = [self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent, factor=1,
                                                               theor_price=True, spread=g_basket_spreads_base[k - 1])
                                     for k in range(1, 6)]
            t_basket_prices_shift = [self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent, factor=1,
                                                               theor_price=True, spread=t_basket_spreads_base[k - 1])
                                     for k in range(1, 6)]
            self.default_func = self._refit_hazard()
            zcb_delta = (zcb_shift - zcb_base) / market_spread_ds
            g_basket_deltas = np.array([(g_basket_prices_shift[k] - basket_prices_base[k]) / market_spread_ds
                                        for k in range(0, 5)])
            t_basket_deltas = np.array([(t_basket_prices_shift[k] - basket_prices_base[k]) / market_spread_ds
                                        for k in range(0, 5)])
            g_hedge_ratios = g_basket_deltas / zcb_delta
            t_hedge_ratios = t_basket_deltas / zcb_delta
            plt.figure(figsize=(12, 7))
            dx = 0.2
            rects = plt.bar(np.arange(1, 6)-dx, g_hedge_ratios, label='gaussian copula', width=0.35)
            plt.bar_label(rects, padding=3, fmt=" %.2f ")
            rects = plt.bar(np.arange(1, 6)+dx, t_hedge_ratios, label='student-t copula', width=0.35)
            plt.bar_label(rects, padding=3, fmt=" %.2f ")
        else:
            basket_spreads_base = [self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent, factor=1)
                                   for k in range(1, 6)]
            basket_prices_base = np.zeros(5)  # base basket price is 0

            self.default_func, hazard_df = self._refit_hazard(market_cds_data=shifted_cds_data, return_hazard=True)
            zcb_shift = 1/5 * self.get_price_risky_zcb(ticker, hazard_df, steps=steps)
            basket_prices_shift = [self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent, factor=1,
                                                             theor_price=True, spread=basket_spreads_base[k-1])
                                   for k in range(1, 6)]
            self.default_func = self._refit_hazard()
            zcb_delta = (zcb_shift - zcb_base) / market_spread_ds
            basket_deltas = np.array([(basket_prices_shift[k] - basket_prices_base[k])/market_spread_ds
                                      for k in range(0, 5)])
            hedge_ratios = basket_deltas / zcb_delta
            plt.figure(figsize=(12, 7))
            rects = plt.bar(range(1, 6), hedge_ratios)
            plt.bar_label(rects, padding=3, fmt=" %.2f ")
        plt.title(f"Basket Credit Default Hedge Ratio for {ticker}")
        plt.xlabel('k-th to default')
        plt.ylabel('Hedge Ratio')
        plt.legend(loc='best')
        if save:
            plt.savefig('plots/HedgeRatio.png', dpi='figure')
        plt.show()
        print("End of market spread delta hedging..")

    def convergence_analysis_global(self, k=1, save=False, copula='gau', base_exponent=20):
        """
        plot convergence of basket spread
        Compares Sobol' low-discrepancy sequence vs pseudo-random Mersenne Twister.
        :param k: k-th to default, optionally 'all'.
        :param save: set to True to save the plot
        :param copula: copula method, 'gau' for gaussian, 't' for student-t. For k <> 'all', 'both' is also possible.
        :param base_exponent: maximum nbr_simulations = 2^base_exponent
        :return: Plots and optionally saves a global convergence plot of the basket spreads
        """
        print(f"Starting global convergence analysis..")
        prices = {}
        methods = ['sobol', 'pseudo']
        markers = {'sobol': 'o', 'pseudo': 'x'}
        copula_title = {'gau': 'Gaussian ', 't': 'Student-T '}
        print("Calculating prices...")
        if k == 'all':
            for k in range(1, 6):
                prices[k] = {}
                for method in methods:
                    steps, prices[k][method] = self.get_price_basket_cds(
                        copula=copula, k=k, exponent=base_exponent, method=method, iterative=True)
            plt.figure(figsize=(12, 7))
            plt.title(f"{copula_title[copula]}Spread Convergence")
            for k in range(1, 6):
                for method in methods:
                    plt.semilogx(steps, prices[k][method], label=f'k={k}, {method}', marker=markers[method])
        else:
            if copula == 'both':
                g_prices = {method: [] for method in methods}
                t_prices = {method: [] for method in methods}
                for method in methods:
                    steps, g_prices[method] = self.get_price_basket_cds(
                        copula='gau', k=k, exponent=base_exponent, method=method, iterative=True)
                    steps, t_prices[method] = self.get_price_basket_cds(
                        copula='t', k=k, exponent=base_exponent, method=method, iterative=True)
                plt.figure(figsize=(12, 7))
                plt.title(f"Spread Convergence, default k={k}")
                for method in methods:
                    plt.semilogx(steps, g_prices[method], label=f'gaussian, {method}', marker=markers[method])
                    plt.semilogx(steps, t_prices[method], label=f'student-t, {method}', marker=markers[method])
            else:
                for method in methods:
                    prices[method] = []
                    steps, prices[method] = self.get_price_basket_cds(
                        copula=copula, k=k, exponent=base_exponent, method=method, iterative=True)
                plt.figure(figsize=(12, 7))
                plt.title(f"{copula_title[copula]}Spread Convergence, default k={k}")
                for method in methods:
                    plt.semilogx(steps, prices[method], label=f'{method}', marker=markers[method])
        plt.legend(loc='upper left')
        plt.xlabel('Number of Simulations')
        plt.ylabel('Fair Spread (bps)')
        if save:
            plt.savefig('plots/SpreadConvergence.png', dpi='figure')
        plt.show()
        print(f"End of global convergence analysis.")

    def convergence_analysis_error(self, k=1, save=False, copula='gau', base_exponent=21):
        """
        plot convergence of the error in the basket spread. The true spread is computed using 8 times more simulation
        steps than the maximum that is plotted, computed separately.
        Compares Sobol' low-discrepancy sequence vs pseudo-random Mersenne Twister.
        :param k: k-th to default, optionally 'all'
        :param save: set to True to save the plot
        :param copula: copula method, 'gau' for gaussian, 't' for student-t. For k <> 'all', 'both' is also possible.
        :param base_exponent: maximum nbr_simulations = 2^base_exponent
        :return: lots and optionally saves a convergence plot of the error in the basket spreads
        """
        print(f"Starting error convergence analysis..")
        prices = {}
        methods = ['sobol', 'pseudo']
        markers = {'sobol': 'o', 'pseudo': 'x'}
        copula_title = {'gau': 'Gaussian ', 't': 'Student-T '}
        print("Calculating prices...")
        if k == 'all':
            for k in range(1, 6):
                prices[k] = {}
                for method in methods:
                    steps, prices[k][method] = self.get_price_basket_cds(
                        copula=copula, k=k, exponent=base_exponent, method=method, iterative=True)
            plt.figure(figsize=(12, 7))
            plt.title(f"{copula_title[copula]}Spread Error Convergence")
            for k in range(1, 6):
                true_value = self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent + 3, method='sobol')
                # true_value = prices[k]['sobol'][-1]
                for method in methods:
                    plt.loglog(steps, np.abs(np.array(prices[k][method]) - true_value),
                               label=f'k={k}, {method}', marker=markers[method])
        else:
            if copula == 'both':
                g_prices = {method: [] for method in methods}
                t_prices = {method: [] for method in methods}
                for method in methods:
                    steps, g_prices[method] = self.get_price_basket_cds(
                        copula='gau', k=k, exponent=base_exponent, method=method, iterative=True)
                    steps, t_prices[method] = self.get_price_basket_cds(
                        copula='t', k=k, exponent=base_exponent, method=method, iterative=True)

                g_true_value = self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent + 3, method='sobol')
                t_true_value = self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent + 3, method='sobol')

                plt.figure(figsize=(12, 7))
                plt.title(f"Spread Error Convergence, default k={k}")
                for method in methods:
                    plt.semilogx(steps, np.abs(np.array(g_prices[method]) - g_true_value), label=f'gaussian, {method}',
                                 marker=markers[method])
                    plt.semilogx(steps, np.abs(np.array(t_prices[method]) - t_true_value), label=f'student-t, {method}',
                                 marker=markers[method])
            else:
                for method in methods:
                    prices[method] = []
                    steps, prices[method] = self.get_price_basket_cds(
                        copula=copula, k=k, exponent=base_exponent, method=method, iterative=True)
                true_value = self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent + 3, method='sobol')
                plt.figure(figsize=(12, 7))
                plt.title(f"{copula_title[copula]}Spread Error Convergence, default k={k}")
                for method in methods:
                    plt.loglog(steps, np.abs(np.array(prices[method]) - true_value), marker='o',
                               label=f'{method}')
        plt.loglog(steps, 15*1/np.sqrt(steps), label=r'$O(N^{-1/2})$')
        plt.legend(loc='best')
        plt.xlabel('Number of Simulations')
        plt.ylabel('Absolute Error (bps)')
        if save:
            plt.savefig('plots/SpreadConvergenceError.png', dpi='figure')
        plt.show()
        print(f"End of error convergence analysis.")

    def compare_gauss_t_prices(self, save=False, base_exponent=20):
        """
        calculate and plot gaussian vs student-t copula basket cds spreads
        :param save: set to True to save the plot
        :param base_exponent: nbr_simulations = 2^base_exponent
        :return: Plots and optionally saves the plot comparing gaussain against studen-t copula basket cds spreads.
        """
        print("Starting comparison between gaussian-copula and t-copula prices")
        # for all k-th to default calc gaussian and t-copula prices, put into table or graph
        k_range = np.arange(1, 6)
        dx = 0.2
        prices = np.zeros((5, 2))
        print("Calculating prices...")
        for k in k_range:
            exponent = base_exponent
            g_price = self.get_price_basket_cds(copula='gau', k=k, exponent=exponent)
            t_price = self.get_price_basket_cds(copula='t', k=k, exponent=exponent)
            prices[k-1, :] = np.array([g_price, t_price])
        plt.figure(figsize=(12, 7))
        rects = plt.bar(k_range-dx, prices[:, 0], label='gaussian copula', width=0.35)
        plt.bar_label(rects, padding=3, fmt=" %.1f ")
        rects = plt.bar(k_range+dx, prices[:, 1], label='student-t copula', width=0.35)
        plt.bar_label(rects, padding=3, fmt=" %.1f ")
        plt.title("Gaussian vs Student-T Copula priced basket CDS")
        plt.xlabel('k-th to default')
        plt.ylabel('Fair Spread (bps)')
        plt.legend(loc='best')
        if save:
            plt.savefig('plots/CompareGaussTPrices.png', dpi='figure')
        plt.show()
        print("End of comparison between gaussian-copula and t-copula prices.")

    def yield_curve_sensitivity(self, base_exponent=20, copula='gau', save=False,
                                diff_type='absolute'):
        """
        calculate and plot sensitivity of the basket cds spread against changes in the yield curve
        :param base_exponent: nbr_simulations = 2^base_exponent
        :param copula: copula method, 'gau' for gaussian, 't' for student-t, 'both' for both methods in one plot
        :param save: set to True to save the plot
        :param diff_type: set to absolute for showing absolute spreads, set to relative for showing relative changes in 
                          spread with respect to the base level.
        :return: Plots and optionally saves the sensitivity plot of the basket spread wrt changes in the yield curve.
        """
        print("Starting yield curve sensitivity..")
        yield_curve_spreads = np.linspace(-4, 4, 10)
        print("Calculating prices..")
        if diff_type == 'relative':
            y_axis = 'Relative Change in Fair Spread'
        else:
            y_axis = 'Fair Spread (bps)'

        if copula == 'both':
            if diff_type == 'relative':
                g_baseline = np.array([self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent)
                                       for k in range(1, 6)])
                t_baseline = np.array([self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent)
                                       for k in range(1, 6)])

            g_prices = {k: [] for k in range(1, 6)}
            t_prices = {k: [] for k in range(1, 6)}
            for spread in yield_curve_spreads:
                def discount_func_shifted(time):
                    return base_discount_factor(time, self.spot_rate_params, spot_rate_shift=spread)
                self.default_func = self._refit_hazard(discount_override=discount_func_shifted)

                for k in range(1, 6):
                    g_prices[k].append(
                        self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent,
                                                  discount_override=discount_func_shifted))
                    t_prices[k].append(
                        self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent,
                                                  discount_override=discount_func_shifted))

            self.default_func = self._refit_hazard()
            plt.figure(figsize=(12, 7))
            if diff_type == 'relative':
                for k in range(1, 6):
                    plt.plot(yield_curve_spreads, (np.array(t_prices[k])-t_baseline[k-1]) / t_baseline[k-1],
                             label=f'k = {k}, student-t', ls='--')
                    plt.plot(yield_curve_spreads, (np.array(g_prices[k])-g_baseline[k-1]) / g_baseline[k-1],
                             label=f'k = {k}, gaussian')
            else:
                for k in range(1, 6):
                    plt.plot(yield_curve_spreads, t_prices[k], label=f'k = {k}, student-t', ls='--')
                    plt.plot(yield_curve_spreads, g_prices[k], label=f'k = {k}, gaussian')
        else:
            if diff_type == 'relative':
                baseline = [self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent)
                            for k in range(1, 6)]
            prices = {k: [] for k in range(1, 6)}
            for spread in yield_curve_spreads:
                def discount_func_shifted(time):
                    return base_discount_factor(time, self.spot_rate_params, spot_rate_shift=spread)
                self.default_func = self._refit_hazard(discount_override=discount_func_shifted)
                for k in range(1, 6):
                    prices[k].append(self.get_price_basket_cds(
                        copula=copula, k=k, exponent=base_exponent, discount_override=discount_func_shifted))

            self.default_func = self._refit_hazard()
            plt.figure(figsize=(12, 7))
            if diff_type == 'relative':
                for k in range(1, 6):
                    plt.plot(yield_curve_spreads, (np.array(prices[k])-baseline[k-1]) / baseline[k-1],
                             label=f'default k = {k}')
            else:
                for k in range(1, 6):
                    plt.plot(yield_curve_spreads, prices[k], label=f'default k = {k}')
        plt.legend(loc='best')
        plt.xlabel(r'Parallel Spot Rate Shift (%)')
        plt.ylabel(y_axis)
        plt.title("Yield Curve Sensitivity")
        if save:
            plt.savefig('plots/YieldCurveSensitivity.png', dpi='figure')
        plt.show()
        print("End of yield curve sensitivity analysis.")

    def maturity_sensitivity(self, base_exponent=20, copula='gau', save=False, diff_type='absolute'):
        """
        calculate and plot sensitivity of the basket cds spread against changes in the CDS maturity
        :param base_exponent: nbr_simulations = 2^base_exponent
        :param copula: copula method, 'gau' for gaussian, 't' for student-t, 'both' for both methods in one plot
        :param save: set to True to save the plot
        :param diff_type: set to absolute for showing absolute spreads, set to relative for showing relative changes in
                          spread with respect to the base level.
        :return: Plots and optionally saves the sensitivity plot of the basket spread wrt changes in the CDS maturity.
        """
        print("Starting maturity sensitivity..")
        maturities = np.linspace(0.5, 10, num=20)
        base_maturity = self.maturity
        print("Calculating prices..")
        if diff_type == 'relative':
            y_axis = 'Relative Change in Fair Spread'
        else:
            y_axis = 'Fair Spread (bps)'

        if copula == 'both':
            if diff_type == 'relative':
                g_baseline = np.array([self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent)
                                       for k in range(1, 6)])
                t_baseline = np.array([self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent)
                                       for k in range(1, 6)])

            g_prices = {k: [] for k in range(1, 6)}
            t_prices = {k: [] for k in range(1, 6)}
            for maturity in maturities:
                self.maturity = maturity
                for k in range(1, 6):
                    g_prices[k].append(
                        self.get_price_basket_cds(copula='gau', k=k, exponent=base_exponent))
                    t_prices[k].append(
                        self.get_price_basket_cds(copula='t', k=k, exponent=base_exponent))

            plt.figure(figsize=(12, 7))
            if diff_type == 'relative':
                for k in range(1, 6):
                    plt.plot(maturities, (np.array(t_prices[k]) - t_baseline[k - 1]) / t_baseline[k - 1],
                             label=f'k = {k}, student-t', ls='--')
                    plt.plot(maturities, (np.array(g_prices[k]) - g_baseline[k - 1]) / g_baseline[k - 1],
                             label=f'k = {k}, gaussian')
            else:
                for k in range(1, 6):
                    plt.plot(maturities, t_prices[k], label=f'k = {k}, student-t', ls='--')
                    plt.plot(maturities, g_prices[k], label=f'k = {k}, gaussian')
        else:
            if diff_type == 'relative':
                baseline = [self.get_price_basket_cds(copula=copula, k=k, exponent=base_exponent)
                            for k in range(1, 6)]
            prices = {k: [] for k in range(1, 6)}
            for maturity in maturities:
                self.maturity = maturity
                for k in range(1, 6):
                    prices[k].append(self.get_price_basket_cds(
                        copula=copula, k=k, exponent=base_exponent))
            plt.figure(figsize=(12, 7))
            if diff_type == 'relative':
                for k in range(1, 6):
                    plt.plot(maturities, (np.array(prices[k]) - baseline[k - 1]) / baseline[k - 1],
                             label=f'default k = {k}')
            else:
                for k in range(1, 6):
                    plt.plot(maturities, prices[k], label=f'default k = {k}')
        self.maturity = base_maturity
        plt.legend(loc='best')
        plt.xlabel(r'Maturity (years)')
        plt.ylabel(y_axis)
        plt.title("Maturity Sensitivity")
        if save:
            plt.savefig('plots/MaturitySensitivity.png', dpi='figure')
        plt.show()
        print("End of maturity sensitivity analysis.")


def run_test():
    """
    :return: test function to test the basket pricer. Instantiates the class and runs a sensitivity plot. 
    """
    pricer = BasketCdsPricer(
        fit=False, plot=False
    )
    # spread = pricer.get_price_basket_cds()
    # pricer.correlation_sensitivity(mode='single')
    # pricer.yield_curve_sensitivity(copula='both', base_exponent=15)
    pricer.maturity_sensitivity(copula='both', base_exponent=15)
    print("DONE")


if __name__ == '__main__':
    run_test()

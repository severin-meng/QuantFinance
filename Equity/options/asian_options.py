"""
This script shows a comparison between Euler-Maruyama and Milstein (and closed-form solution where applicable)
for SDE in the case of asian options.

"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


class MonteCarloOptionBaseClass:
    def __init__(self, s0, rfr, time, volatility, strike=100):
        self.spot_price = s0
        self.risk_free_rate = rfr
        self.time_to_expiry = time
        self.volatility = volatility
        self.asset_paths = None
        self.discount_factor = np.exp(-self.risk_free_rate * self.time_to_expiry)
        self.strike = strike

    def generate_asset_paths(self, *args):
        raise NotImplementedError("The base class does not implement the method \'generate_asset_paths\'.")

    @staticmethod
    def european_payoff(asset_value, strike_value, is_call):
        if is_call:
            return np.maximum(asset_value - strike_value, 0)
        return np.maximum(strike_value - asset_value, 0)

    def arithmetic_mean(self, nbr_samples=None):
        if nbr_samples is None:
            nbr_samples = self.asset_paths.shape[0]
        # number of intervals between prices = nbr samples - 1
        # number of intervals between samples = nbr(samples) - 1
        sampling_interval = int((self.asset_paths.shape[0] - 1) / (nbr_samples - 1))
        residual = (self.asset_paths.shape[0] - 1) % sampling_interval  # used to center discrete sampling period
        sample = self.asset_paths[int(residual/2)::sampling_interval, :]
        return sample.mean(axis=0)

    def geometric_mean(self, nbr_samples=None):
        if nbr_samples is None:
            nbr_samples = self.asset_paths.shape[0]
            # number of intervals between prices = nbr samples - 1
            # number of intervals between samples = nbr(samples) - 1
        sampling_interval = int((self.asset_paths.shape[0] - 1) / (nbr_samples - 1))
        residual = (self.asset_paths.shape[0] - 1) % sampling_interval  # used to center discrete sampling period
        sample = np.log(self.asset_paths[int(residual/2)::sampling_interval, :])
        return np.exp(sample.mean(axis=0))

    def asian_payoff(self, is_call, nbr_samples, averaging_method, strike_type):
        average = averaging_method(nbr_samples)
        if strike_type == 'floating':
            return self.european_payoff(self.asset_paths[-1, :], average, is_call)
        # fixed strike
        return self.european_payoff(average, self.strike, is_call)

    def option_price(self, is_european=False, is_call=True, nbr_samples=None, averaging='arithmetic',
                     strike_type='fixed', return_path_prices=False):
        if self.asset_paths is None:
            self.generate_asset_paths()
        if is_european:
            payoffs = self.european_payoff(self.asset_paths[-1, :], self.strike, is_call)
        else:
            averaging_dict = {'arithmetic': self.arithmetic_mean,
                              'geometric': self.geometric_mean}
            averaging_method = averaging_dict[averaging]
            payoffs = self.asian_payoff(is_call, nbr_samples, averaging_method, strike_type)
        if return_path_prices:
            # skip averaging step
            return payoffs * self.discount_factor
        return np.average(payoffs) * self.discount_factor


class MilsteinMCOption(MonteCarloOptionBaseClass):
    def __init__(self, s0, rfr, time, volatility, time_steps, nbr_simulations, strike=100, rnd_seed=10000):
        self.seed = rnd_seed
        self.nbr_simulations = nbr_simulations
        self.time_steps = time_steps
        super().__init__(s0, rfr, time, volatility, strike)

    def generate_asset_paths(self):
        np.random.seed(self.seed)

        # define dt
        dt = self.time_to_expiry / self.time_steps  # length of time interval

        # simulate #'nbr_simulations' asset price paths with #'time_steps' timesteps
        s = np.zeros((self.time_steps, self.nbr_simulations))
        s[0] = self.spot_price

        for i in range(0, self.time_steps - 1):
            w = np.random.standard_normal(self.nbr_simulations)
            s[i + 1] = s[i] * (
                    1
                    + self.risk_free_rate * dt
                    + self.volatility * w * np.sqrt(dt)
                    + 0.5 * self.volatility ** 2 * (w ** 2 - 1) * dt)
        self.asset_paths = s
        return self.asset_paths


class EulerMaruyamaMCOption(MonteCarloOptionBaseClass):
    def __init__(self, s0, rfr, time, volatility, time_steps, nbr_simulations, strike=100, rnd_seed=10000):
        self.seed = rnd_seed
        self.nbr_simulations = nbr_simulations
        self.time_steps = time_steps
        super().__init__(s0, rfr, time, volatility, strike)

    def generate_asset_paths(self):
        np.random.seed(self.seed)

        # define dt
        dt = self.time_to_expiry / self.time_steps  # length of time interval

        # simulate #'nbr_simulations' asset price paths with #'time_steps' timesteps
        s = np.zeros((self.time_steps, self.nbr_simulations))
        s[0] = self.spot_price

        for i in range(0, self.time_steps - 1):
            w = np.random.standard_normal(self.nbr_simulations)
            s[i + 1] = s[i] * (
                    1
                    + self.risk_free_rate * dt
                    + self.volatility * w * np.sqrt(dt))
        self.asset_paths = s
        return self.asset_paths


class ClosedFormMCOption(MonteCarloOptionBaseClass):
    def __init__(self, s0, rfr, time, volatility, time_steps, nbr_simulations, strike=100, rnd_seed=10000):
        self.seed = rnd_seed
        self.nbr_simulations = nbr_simulations
        self.time_steps = time_steps
        super().__init__(s0, rfr, time, volatility, strike)

    def generate_asset_paths(self):
        np.random.seed(self.seed)

        # define dt
        dt = self.time_to_expiry / self.time_steps  # length of time interval

        # simulate #'nbr_simulations' asset price paths with #'time_steps' timesteps
        s = np.zeros((self.time_steps, self.nbr_simulations))
        s[0] = self.spot_price

        for i in range(0, self.time_steps - 1):
            w = np.random.standard_normal(self.nbr_simulations)
            s[i + 1] = s[i] * np.exp(
                (self.risk_free_rate - 0.5 * self.volatility**2) * dt
                + self.volatility * w * np.sqrt(dt))
        self.asset_paths = s
        return self.asset_paths


def compare_schemes(initial_stock, time, volatility, risk_free_rate):
    # vary nbr of time steps
    simulations = 10000
    time_step_range = [12, 52] + list(np.arange(1, 16) * 252)  # 1 year of trading days, monthly to 15 prices per day
    schemes = [EulerMaruyamaMCOption, MilsteinMCOption]

    single_paths = np.zeros((252, 3))
    time_range = np.linspace(0, time, 252)

    strong_test = np.zeros((len(time_step_range), 2))
    strong_var_test = np.zeros((len(time_step_range), 2))

    weak_test = np.zeros((len(time_step_range), 2))
    weak_var_test = np.zeros((len(time_step_range), 2))

    asian_price_diff = np.zeros((len(time_step_range), 2))
    asian_price_var = np.zeros((len(time_step_range), 2))

    for i, time_step in enumerate(time_step_range):
        closed_form_scheme = ClosedFormMCOption(initial_stock, risk_free_rate, time, volatility, time_step, simulations)
        cf_paths = closed_form_scheme.generate_asset_paths()
        cf_path_prices = closed_form_scheme.option_price(return_path_prices=True)
        for j, scheme in enumerate(schemes):
            mc_scheme = scheme(initial_stock, risk_free_rate, time, volatility, time_step, simulations)
            asset_paths = mc_scheme.generate_asset_paths()
            scheme_path_prices = mc_scheme.option_price(return_path_prices=True)

            # compare single path
            if time_step == 252:
                single_paths[:, 0] = cf_paths[:, 0]
                single_paths[:, j+1] = asset_paths[:, 0]

            # test strong convergence
            random_variable = np.abs(cf_paths-asset_paths)
            expectation_values = np.average(random_variable, axis=1)
            variance_values = np.std(random_variable, axis=1)
            exp_mean_end = expectation_values[-1]
            strong_test[i, j] = exp_mean_end
            strong_var_test[i, j] = variance_values[-1]

            # test weak convergence
            exact_values = cf_paths[-1, :]
            test_values = asset_paths[-1, :]
            weak_conv_value = np.abs(np.average(exact_values) - np.average(test_values))
            weak_conv_variance = np.std(exact_values - test_values)
            weak_test[i, j] = weak_conv_value
            weak_var_test[i, j] = weak_conv_variance

            # test asian option price
            random_variable = cf_path_prices - scheme_path_prices
            price_diff = np.average(random_variable)
            abs_price_diff = np.abs(price_diff)
            asian_price_diff[i, j] = abs_price_diff
            variance_values = np.std(random_variable)
            asian_price_var[i, j] = variance_values

    # compare single path
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Single asset path: Euler-Maruyama vs Milstein vs Closed Form")

    ax[0].set_title("Global asset path")
    ax[0].plot(time_range, single_paths[:, 0], label='Closed Form')
    ax[0].plot(time_range, single_paths[:, 1], label='Euler-Maruyama')
    ax[0].plot(time_range, single_paths[:, 2], label='Milstein')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$S_t$')
    ax[0].legend(loc='best')

    ax[1].set_title("Zoomed-in asset path")
    ax[1].plot(time_range, single_paths[:, 0], label='Closed Form')
    ax[1].plot(time_range, single_paths[:, 1], label='Euler-Maruyama')
    ax[1].plot(time_range, single_paths[:, 2], label='Milstein')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$S_t$')
    ax[1].legend(loc='best')
    ax[1].set_xlim([0.224, 0.230])
    ax[1].set_ylim([127.15, 127.45])
    fig.tight_layout()
    plt.show()

    # strong convergence test
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Strong convergence test")

    ax[0].set_title("Expected absolute asset price difference")
    ax[0].loglog([1/t for t in time_step_range], strong_test[:, 0], 'o', markeredgecolor='none',
                 label='Euler-Maruyama')
    ax[0].loglog([1/t for t in time_step_range], strong_test[:, 1], 'o', markeredgecolor='none',
                 label='Milstein')
    ax[0].loglog([1/t for t in time_step_range], [(1/t)**0.5 for t in time_step_range],
                 label=r'$O(\delta t^{0.5})$')
    ax[0].loglog([1 / t for t in time_step_range], [1/t for t in time_step_range], label=r'$O(\delta t)$')
    ax[0].legend(loc='best')
    ax[0].set_xlabel(r'$\delta t$')

    ax[1].set_title("Std deviation of absolute asset price difference")
    ax[1].loglog([1 / t for t in time_step_range], strong_var_test[:, 0], 'o', markeredgecolor='none',
                 label='Euler-Maruyama')
    ax[1].loglog([1 / t for t in time_step_range], strong_var_test[:, 1], 'o', markeredgecolor='none',
                 label='Milstein')
    ax[1].loglog([1 / t for t in time_step_range], [(1 / t) ** 0.5 for t in time_step_range],
                 label=r'$O(\delta t^{0.5})$')
    ax[1].loglog([1 / t for t in time_step_range], [1 / t for t in time_step_range], label=r'$O(\delta t)$')
    ax[1].legend(loc='best')
    ax[1].set_xlabel(r'$\delta t$')
    fig.tight_layout()
    plt.show()

    # weak convergence test
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Weak convergence test")

    ax[0].set_title("Absolute expectation of asset price difference")
    ax[0].loglog([1 / t for t in time_step_range], weak_test[:, 0], 'o', markeredgecolor='none',
                 label='Euler-Maruyama')
    ax[0].loglog([1 / t for t in time_step_range], weak_test[:, 1], 'o', markeredgecolor='none',
                 label='Milstein')
    ax[0].loglog([1 / t for t in time_step_range], [(1 / t) ** 0.5 for t in time_step_range],
                 label=r'$O(\delta t^{0.5})$')
    ax[0].loglog([1 / t for t in time_step_range], [1 / t for t in time_step_range], label=r'$O(\delta t)$')
    ax[0].legend(loc='best')
    ax[0].set_xlabel(r'$\delta t$')

    ax[1].set_title("Std deviation of asset price difference")
    ax[1].loglog([1 / t for t in time_step_range], weak_var_test[:, 0], 'o', markeredgecolor='none',
                 label='Euler-Maruyama')
    ax[1].loglog([1 / t for t in time_step_range], weak_var_test[:, 1], 'o', markeredgecolor='none',
                 label='Milstein')
    ax[1].loglog([1 / t for t in time_step_range], [(1 / t) ** 0.5 for t in time_step_range],
                 label=r'$O(\delta t^{0.5})$')
    ax[1].loglog([1 / t for t in time_step_range], [1 / t for t in time_step_range], label=r'$O(\delta t)$')
    ax[1].legend(loc='best')
    ax[1].set_xlabel(r'$\delta t$')
    fig.tight_layout()
    plt.show()

    # test asian option price
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Continuously sampled, fixed strike asian call option with arithmetic averaging")

    ax[0].set_title("Absolute price difference")
    ax[0].loglog([1 / t for t in time_step_range], asian_price_diff[:, 0], 'o', markeredgecolor='none',
                 label='Euler-Maruyama')
    ax[0].loglog([1 / t for t in time_step_range], asian_price_diff[:, 1], 'o', markeredgecolor='none',
                 label='Milstein')
    ax[0].loglog([1 / t for t in time_step_range], [(1 / t) ** 0.5 for t in time_step_range],
                 label=r'$O(\delta t^{0.5})$')
    ax[0].loglog([1 / t for t in time_step_range], [1 / t for t in time_step_range], label=r'$O(\delta t)$')
    ax[0].legend(loc='best')
    ax[0].set_xlabel(r'$\delta t$')

    ax[1].set_title("Std deviation of price difference")
    ax[1].loglog([1 / t for t in time_step_range], asian_price_var[:, 0], 'o', markeredgecolor='none',
                 label='Euler-Maruyama')
    ax[1].loglog([1 / t for t in time_step_range], asian_price_var[:, 1], 'o', markeredgecolor='none',
                 label='Milstein')
    ax[1].loglog([1 / t for t in time_step_range], [(1 / t) ** 0.5 for t in time_step_range],
                 label=r'$O(\delta t^{0.5})$')
    ax[1].loglog([1 / t for t in time_step_range], [1 / t for t in time_step_range], label=r'$O(\delta t)$')
    ax[1].legend(loc='best')
    ax[1].set_xlabel(r'$\delta t$')
    fig.tight_layout()
    plt.show()


def get_all_feature_combinations():
    is_call_set = [True, False]
    strike_types = ['fixed', 'floating']
    averaging_methods = ['arithmetic', 'geometric']
    nbr_samples_set = [None, 12]
    features_combos = np.array([[avg, call, samples, strike_type]
                               for call in is_call_set
                               for strike_type in strike_types
                               for avg in averaging_methods
                               for samples in nbr_samples_set])
    return features_combos


def compute_all_asian_option_prices(initial_stock, risk_free_rate, time, volatility, number_of_days,
                                    number_of_simulations, strike, mc_scheme=ClosedFormMCOption):
    # initialize scheme
    scheme = mc_scheme(initial_stock, risk_free_rate, time, volatility, number_of_days, number_of_simulations,
                       strike)
    scheme.generate_asset_paths()
    feature_set = get_all_feature_combinations()
    price_dict = dict()
    for averaging, is_call, nbr_samples, strike_type in feature_set:
        price = scheme.option_price(is_european=False, is_call=is_call, nbr_samples=nbr_samples, averaging=averaging,
                                    strike_type=strike_type, return_path_prices=False)
        key = build_short_key(averaging, is_call, nbr_samples, strike_type)
        price_dict[key] = price
    return price_dict


def build_short_key(averaging_method, is_call, nbr_samples, strike_type):
    sampling = 'Cont' if nbr_samples is None else 'Disc'
    averaging = 'Arith' if averaging_method == 'arithmetic' else 'Geom'
    strike = 'Fxd' if strike_type == 'fixed' else 'Flt'
    put_call = 'Call' if is_call else 'Put'
    key = f"{put_call} {strike} {averaging} {sampling}"
    return key


def vary_parameters(initial_stock, risk_free_rate, time, volatility, strike, days_per_year, variations=5):
    initial_stock_range = np.linspace(0.5*initial_stock, 3*initial_stock, variations)
    strike_range = np.linspace(0.5 * strike, 3 * strike, variations)
    rfr_range = np.linspace(-risk_free_rate, 3*risk_free_rate, variations)
    time_range = np.linspace(0.25*time, 3*time, variations)
    days_range = np.array([int(days_per_year*time_value) for time_value in time_range])
    vola_range = np.linspace(0.25*volatility, 3*volatility, variations)
    prices_p_year_range = np.linspace(0.2*days_per_year, 5*days_per_year, variations).astype(int)
    return initial_stock_range, strike_range, rfr_range, time_range, days_range, vola_range, prices_p_year_range


def vary_asian_option_parameters(initial_stock, risk_free_rate, time, volatility, strike, number_simulations,
                                 days_per_year=252, variations=5):
    initial_stocks, strikes, risk_free_rates, years, days, volatilities, prices_per_year = vary_parameters(
        initial_stock, risk_free_rate, time, volatility, strike, days_per_year, variations=variations)

    base_days = int(days_per_year * time)

    features = get_all_feature_combinations()
    base_dict = {'Put/Call': ['Call' if call else 'Put' for call in features[:, 1]],
                 'Strike Type': features[:, 3],
                 'Averaging': features[:, 0],
                 'Sampling': ['Discrete' if sampling else 'Continuous' for sampling in features[:, 2]]}

    stock_variations_print = base_dict.copy()
    strike_variations_print = base_dict.copy()
    rfr_variations_print = base_dict.copy()
    years_days_variations_print = base_dict.copy()
    vola_variations_print = base_dict.copy()
    price_per_year_variations_print = base_dict.copy()

    stock_variations_plot = defaultdict(list)
    strike_variations_plot = defaultdict(list)
    rfr_variations_plot = defaultdict(list)
    years_days_variations_plot = defaultdict(list)
    vola_variations_plot = defaultdict(list)
    price_per_year_variations_plot = defaultdict(list)

    markerdict = {
        'Put Fxd ': "v",
        'Put Flt ': "^",
        'Call Fxd': "<",
        'Call Flt': ">",
    }

    for init_stock_val in initial_stocks:
        price_dict = compute_all_asian_option_prices(init_stock_val, risk_free_rate, time, volatility, base_days,
                                                     number_simulations, strike)
        stock_variations_print[f'S_0: {init_stock_val}'] = price_dict.values()
        for label, price in price_dict.items():
            stock_variations_plot[label].append(price)
    print("Varying initial stock price")
    print(tabulate(stock_variations_print, headers="keys"))
    print('\n')

    plt.figure(figsize=(9, 7))
    plt.title("Asian option prices with varying initial stock price")
    for label, prices in stock_variations_plot.items():
        plt.plot(initial_stocks, prices, label=label, marker=markerdict[label[:8]])
    plt.legend(loc='best', ncol=2)
    plt.xlabel(r'$S_0$')
    plt.show()

    for strike_value in strikes:
        price_dict = compute_all_asian_option_prices(initial_stock, risk_free_rate, time, volatility, base_days,
                                                     number_simulations, strike_value)
        strike_variations_print[f'E: {strike_value}'] = price_dict.values()
        for label, price in price_dict.items():
            strike_variations_plot[label].append(price)
    print("Varying strike price")
    print(tabulate(strike_variations_print, headers="keys"))
    print('\n')

    plt.figure(figsize=(9, 7))
    plt.title("Asian option prices with varying strike")
    for label, prices in strike_variations_plot.items():
        plt.plot(strikes, prices, label=label, marker=markerdict[label[:8]])
    plt.legend(loc='best', ncol=2)
    plt.xlabel(r'$E$')
    plt.show()

    for rfr_value in risk_free_rates:
        price_dict = compute_all_asian_option_prices(initial_stock, rfr_value, time, volatility, base_days,
                                                     number_simulations, strike)
        rfr_variations_print[f'r: {rfr_value}'] = price_dict.values()
        for label, price in price_dict.items():
            rfr_variations_plot[label].append(price)
    print("Varying risk-free interest rate")
    print(tabulate(rfr_variations_print, headers="keys"))
    print('\n')

    plt.figure(figsize=(9, 7))
    plt.title("Asian option prices with varying risk free rate")
    for label, prices in rfr_variations_plot.items():
        plt.plot(risk_free_rates, prices, label=label, marker=markerdict[label[:8]])
    plt.legend(loc='best', ncol=2)
    plt.xlabel(r'$r$')
    plt.show()

    for time_years, nbr_days in zip(years, days):
        price_dict = compute_all_asian_option_prices(initial_stock, risk_free_rate, time_years, volatility, nbr_days,
                                                     number_simulations, strike)
        years_days_variations_print[f'T-t: {time_years}'] = price_dict.values()
        for label, price in price_dict.items():
            years_days_variations_plot[label].append(price)
    print("Varying time to expiry")
    print(tabulate(years_days_variations_print, headers="keys"))
    print('\n')

    plt.figure(figsize=(9, 7))
    plt.title("Asian option prices with varying time to maturity")
    for label, prices in years_days_variations_plot.items():
        plt.plot(years, prices, label=label, marker=markerdict[label[:8]])
    plt.legend(loc='best', ncol=2)
    plt.xlabel(r'$(T-t)$')
    plt.show()

    for vola_value in volatilities:
        price_dict = compute_all_asian_option_prices(initial_stock, risk_free_rate, time, vola_value, base_days,
                                                     number_simulations, strike)
        vola_variations_print[f'Vola: {vola_value}'] = price_dict.values()
        for label, price in price_dict.items():
            vola_variations_plot[label].append(price)
    print("Varying volatility")
    print(tabulate(vola_variations_print, headers="keys"))
    print('\n')

    plt.figure(figsize=(9, 7))
    plt.title("Asian option prices with varying volatility")
    for label, prices in vola_variations_plot.items():
        plt.plot(volatilities, prices, label=label, marker=markerdict[label[:8]])
    plt.legend(loc='best', ncol=2)
    plt.xlabel(r'$\sigma$')
    plt.show()

    for time_steps in prices_per_year:
        price_dict = compute_all_asian_option_prices(initial_stock, risk_free_rate, time, volatility, time_steps,
                                                     number_simulations, strike)
        price_per_year_variations_print[f'Time steps: {time_steps}'] = price_dict.values()
        for label, price in price_dict.items():
            price_per_year_variations_plot[label].append(price)
    print("Varying number of prices per year")
    print(tabulate(price_per_year_variations_print, headers="keys"))

    plt.figure(figsize=(9, 7))
    plt.title("Asian option prices with varying number of time steps")
    for label, prices in price_per_year_variations_plot.items():
        plt.plot(prices_per_year, prices, label=label, marker=markerdict[label[:8]])
    plt.legend(bbox_to_anchor=[0, 0.55], loc='upper left', ncol=2)
    plt.xlabel(r'Time steps')
    plt.show()


def base_asian_options(initial_stock, risk_free_rate, time, volatility, strike, number_simulations, days_per_year=252):
    features = get_all_feature_combinations()
    base_price_dict = {'Put/Call': ['Call' if call else 'Put' for call in features[:, 1]],
                       'Strike Type': features[:, 3],
                       'Averaging': features[:, 0],
                       'Sampling': ['Discrete' if sampling else 'Continuous' for sampling in features[:, 2]]}
    base_prices = compute_all_asian_option_prices(initial_stock, risk_free_rate, time, volatility, days_per_year,
                                                  number_simulations, strike, mc_scheme=ClosedFormMCOption)
    base_price_dict['Price'] = base_prices.values()
    print(tabulate(base_price_dict, headers="keys"))
    print('\n')


if __name__ == "__main__":
    stock0 = 100
    strike_val = 100
    time_horizon = 1
    nbr_of_days = 252  # number of business days in a year
    vola = 0.2
    risk_free_interest_rate = 0.05

    run_part_1 = True
    run_part_2 = False

    if run_part_1:
        #  Part 1: Compare-EulerMaruyama, Milstein and Exact Scheme
        compare_schemes(stock0, time_horizon, vola, risk_free_interest_rate)

    if run_part_2:
        # Part 2: Compute Asian options with varying input parameters
        nbr_of_simulations = 100000
        base_asian_options(stock0, risk_free_interest_rate, time_horizon, vola, strike_val, nbr_of_simulations,
                           days_per_year=nbr_of_days)
        vary_asian_option_parameters(stock0, risk_free_interest_rate, time_horizon, vola, strike_val,
                                     nbr_of_simulations, days_per_year=nbr_of_days)

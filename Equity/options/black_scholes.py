"""
This module implements the basic black-scholes-merton vanilla put and call option pricing formulae assuming constant
volatility, constant short rate, no dividends, and geometric brownian motion of the underlying.
"""
from scipy.special import ndtr
# from math import sqrt, exp, log, pi
from numpy import sqrt, exp, log, pi
import numpy as np
import matplotlib.pyplot as plt


sqr_twopi = sqrt(2 * pi)


def bs_call(spot, strike, vol, rate, tte):
    """
    returns the black-scholes call option price assuming constant instantaneous short rate, no borrow cost, no dividends
    """
    total_vol = vol * sqrt(tte)
    d1 = (log(spot/strike) + (rate + vol**2/2) * tte) / total_vol
    d2 = d1 - total_vol
    discount_factor = exp(-rate * tte)
    price = spot * ndtr(d1) - strike * discount_factor * ndtr(d2)
    return price


def bs_put(spot, strike, vol, rate, tte):
    """
    returns the black-scholes call option price assuming constant instantaneous short rate, no borrow cost, no dividends
    """
    total_vol = vol * sqrt(tte)
    d1 = (log(spot/strike) + (rate + vol**2/2) * tte) / total_vol
    d2 = d1 - total_vol
    discount_factor = exp(-rate * tte)
    price = -spot * ndtr(-d1) + strike * discount_factor * ndtr(-d2)
    return price


def put_call_parity(spot, strike, vol, rate, tte):
    call = bs_call(spot, strike, vol, rate, tte)
    put = bs_put(spot, strike, vol, rate, tte)
    parity = call - put - (spot - exp(-rate*tte)*strike)
    return parity


class BlackScholesVanilla:
    def __init__(self, spot, strike, vol, rate, tte):
        self.spot = spot
        self.strike = strike
        self.vol = vol
        self.rate = rate
        self.tte = tte
        self.total_vol = self.vol * sqrt(tte)
        self.disc_fact = exp(-self.rate * self.tte)
        self.d1 = (log(spot/strike) + (rate + vol**2/2) * tte) / self.total_vol
        self.d2 = self.d1 - self.total_vol
        self.n1 = ndtr(self.d1)
        self.n2 = ndtr(self.d2)

    @staticmethod
    def get_type():
        return 'Vanilla'

    def price(self, is_call=True):
        if is_call:
            return self.spot * self.n1 - self.disc_fact * self.strike  * self.n2
        return self.disc_fact * self.strike * (1 - self.n2) - self.spot * (1 - self.n1)

    def delta(self, is_call=True):
        if is_call:
            return self.n1
        return self.n1 - 1

    def gamma(self, is_call=True):
        return exp(-self.d1**2/2) / (self.spot * self.total_vol * sqr_twopi)

    def theta(self, is_call=True):
        if is_call:
            return -self.rate * self.strike * self.disc_fact * self.n2 - self.spot * exp(-self.d1**2/2) * self.total_vol / (2 * self.tte * sqr_twopi)
        return self.rate * self.strike * self.disc_fact * (1 - self.n2) - self.spot * exp(-self.d1**2/2) * self.total_vol / (2 * self.tte * sqr_twopi)

    def vega(self, is_call=True):
        # vega per percentage point change in volatility
        return self.spot * exp(-self.d1**2/2) * sqrt(self.tte) / sqr_twopi * 0.01

    def rho(self, is_call=True):
        # rho per percentage point in rate
        if is_call:
            return self.tte * self.strike * self.disc_fact * self.n2 * 0.01
        return -self.tte * self.strike * self.disc_fact * (1 - self.n2) * 0.01


class BlackScholesDigitalCashOrNothing:
    def __init__(self, spot, strike, vol, rate, tte):
        self.spot = spot
        self.strike = strike
        self.vol = vol
        self.rate = rate
        self.tte = tte
        self.total_vol = self.vol * sqrt(tte)
        self.disc_fact = exp(-self.rate * self.tte)
        self.d1 = (log(spot/strike) + (rate + vol**2/2) * tte) / self.total_vol
        self.d2 = self.d1 - self.total_vol
        self.n1 = ndtr(self.d1)
        self.n2 = ndtr(self.d2)
        self.deriv_n2d2 = exp(-self.d2**2/2) / sqr_twopi

    @staticmethod
    def get_type():
        return 'Digital Cash-Or-Nothing'

    def price(self, is_call=True):
        if is_call:
            return self.disc_fact  * self.n2
        return self.disc_fact * (1 - self.n2)

    def delta(self, is_call=True):
        if is_call:
            return 1 / (self.spot * self.total_vol) * self.disc_fact * self.deriv_n2d2
        return - 1 / (self.spot * self.total_vol) * self.disc_fact * self.deriv_n2d2

    def gamma(self, is_call=True):
        if is_call:
            return - self.d1/ (self.spot**2 * self.total_vol**2) * self.disc_fact * self.deriv_n2d2
        return self.d1/ (self.spot**2 * self.total_vol**2) * self.disc_fact * self.deriv_n2d2

    def theta(self, is_call=True):
        if is_call:
            return self.rate * self.disc_fact * self.n2 + self.disc_fact * self.deriv_n2d2 * (self.d1 / (2*self.tte) - self.rate / self.total_vol)
        return self.rate * self.disc_fact * (1 - self.n2) - self.disc_fact * self.deriv_n2d2 * (self.d1 / (2*self.tte) - self.rate / self.total_vol)

    def vega(self, is_call=True):
        # vega per percentage point change in volatility
        if is_call:
            return (- self.disc_fact * self.deriv_n2d2 * self.d1 / self.vol) * 0.01
        return (self.disc_fact * self.deriv_n2d2 * self.d1 / self.vol) * 0.01

    def rho(self, is_call=True):
        # rho per percentage point change in rate
        if is_call:
            return (- self.tte * self.disc_fact * self.n2 + self.disc_fact * self.deriv_n2d2 * self.tte / self.total_vol) * 0.01
        return (- self.tte * self.disc_fact * (1 - self.n2) - self.disc_fact * self.deriv_n2d2 * self.tte / self.total_vol) * 0.01


def run_plots(x_variables, options, label="Spot", ref=0):
    call_prices = options.price(is_call=True)
    put_prices = options.price(is_call=False)
    call_deltas = options.delta(is_call=True)
    put_deltas = options.delta(is_call=False)
    call_gammas = options.gamma(is_call=True)
    put_gammas = options.gamma(is_call=False)
    call_thetas = options.theta(is_call=True)
    put_thetas = options.theta(is_call=False)
    call_rhos = options.rho(is_call=True)
    put_rhos = options.rho(is_call=False)
    call_vegas = options.vega(is_call=True)
    put_vegas = options.vega(is_call=False)


    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    fig.suptitle(f"Black-Scholes {options.get_type()} Options & Greeks vs {label}")

    axes[0, 0].plot(x_variables, call_prices, label='Call')
    axes[0, 0].plot(x_variables, put_prices, label='Put')
    axes[0, 0].plot([ref, ref], [0, max(call_prices)], 'k')
    axes[0, 0].set_title("Price")
    axes[0, 0].legend()
    axes[0, 0].grid()

    axes[0, 1].plot(x_variables, call_deltas, label='Call')
    axes[0, 1].plot(x_variables, put_deltas, label='Put')
    axes[0, 1].plot([ref, ref], [min(put_deltas), max(call_deltas)], 'k')
    axes[0, 1].set_title("Delta")
    axes[0, 1].legend()
    axes[0, 1].grid()

    axes[0, 2].plot(x_variables, call_gammas, label='Call')
    axes[0, 2].plot(x_variables, put_gammas, label='Put')
    axes[0, 2].plot([ref, ref], [0, max(call_gammas)], 'k')
    axes[0, 2].set_title("Gamma")
    axes[0, 2].legend()
    axes[0, 2].grid()

    axes[1, 0].plot(x_variables, call_thetas, label='Call')
    axes[1, 0].plot(x_variables, put_thetas, label='Put')
    axes[1, 0].plot([ref, ref], [min(call_thetas), max(put_thetas)], 'k')
    axes[1, 0].set_title("Theta")
    axes[1, 0].legend()
    axes[1, 0].grid()

    axes[1, 1].plot(x_variables, call_rhos, label='Call')
    axes[1, 1].plot(x_variables, put_rhos, label='Put')
    axes[1, 1].plot([ref, ref], [min(put_rhos), max(call_rhos)], 'k')
    axes[1, 1].set_title("Rho %")
    axes[1, 1].legend()
    axes[1, 1].grid()

    axes[1, 2].plot(x_variables, call_vegas, label='Call')
    axes[1, 2].plot(x_variables, put_vegas, label='Put')
    axes[1, 2].plot([ref, ref], [0, max(call_vegas)], 'k')
    axes[1, 2].set_title("Vega %")
    axes[1, 2].legend()
    axes[1, 2].grid()

    plt.tight_layout()
    plt.show()


def plot_everything(spot, strike, vol, rate, tte, option_class):
    # price vs spot
    spots = np.linspace(0.25*spot, 2*spot, 100)
    options = option_class(spots, strike, vol, rate, tte)
    run_plots(spots, options, label="Spot", ref=strike)

    # price vs strike
    strikes = np.linspace(0.25 * strike, 2 * strike, 100)
    options = option_class(spot, strikes, vol, rate, tte)
    run_plots(spots, options, label="Strike", ref=spot)

    # price vs tte
    ttes = np.linspace(1/250, 4*tte, 100)
    options = option_class(spot, strike, vol, rate, ttes)
    run_plots(ttes, options, label="Time-To-Expiry", ref=tte)

    # price vs rate
    rates = np.linspace(-rate, 3*rate, 100)
    options = option_class(spot, strike, vol, rates, tte)
    run_plots(rates, options, label="Risk-Free Rate", ref=rate)

    # price vs vola
    volas = np.linspace(vol*0.2, vol*4, 100)
    options = option_class(spot, strike, volas, rate, tte)
    run_plots(volas, options, label="Volatility", ref=vol)


if __name__ == '__main__':
    plot_everything(100, 100, 0.2, 0.02, 1, BlackScholesVanilla)
    plot_everything(100, 100, 0.2, 0.02, 1, BlackScholesDigitalCashOrNothing)

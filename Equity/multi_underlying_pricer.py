import numpy as np
from scipy.stats.qmc import Halton, Sobol
from scipy.stats import norm
import matplotlib.pyplot as plt
from volatility.path_constructors import BrownianBridge


def get_random_numbers(exponent=10, dim=5, method='pseudo', distr='gauss', seed=42):
    """
    :param exponent: base 2 logarithm of number of simulations = 2^exponent
    :param dim: dimension of sample
    :param method: number generation method, 'pseudo' for pseudo-random Mersenne-Twister, 'sobol' or 'halton' for
                   respective scrambled low-discrepancy sequence
    :param distr: distribution to sample from, either 'gauss' or 'uniform'
    :return: sampled pseudo-random or low-discrepancy numbe of shape (dim, 2^exponent)
    """

    assert distr in ('gauss', 'uniform')
    assert method in ('sobol', 'halton', 'pseudo')

    # options: pseudo, halton, sobol
    if method == 'pseudo':
        n = pow(2, exponent)
        gen = np.random.default_rng(seed=seed)
        if distr == 'gauss':
            return gen.standard_normal((dim, n))
        if distr == 'uniform':
            return gen.random((dim, n))
    elif method == 'sobol':
        gen = Sobol(dim, bits=64, seed=None)
        nbrs = gen.random_base2(exponent).T
        if distr == 'gauss':
            return norm.ppf(nbrs)
        if distr == 'uniform':
            return nbrs
    elif method == 'halton':
        n = pow(2, exponent)
        gen = Halton(dim)
        nbrs = gen.random(n).T
        if distr == 'gauss':
            return norm.ppf(nbrs)
        if distr == 'uniform':
            return nbrs


def get_correlated_randoms(corr_mat, n_paths, n_timesteps, method='pseudo'):
    dim = corr_mat.shape[0] * n_timesteps
    pseudo_sqrt = np.linalg.cholesky(corr_mat)
    exponent = int(np.ceil(np.log2(n_paths)))
    nbr_samples = int(2**exponent)
    iid_gaussians = get_random_numbers(method=method, distr='gauss', dim=dim, exponent=exponent)
    # iid_gaussians = iid_gaussians[:, :nbr_samples]
    # shape is (underlyings * timesteps, simulations)
    iid_gaussians = iid_gaussians.reshape((corr_mat.shape[0], int(nbr_samples*n_timesteps)), order='F')
    corr_gauss = np.matmul(pseudo_sqrt, iid_gaussians)
    corr_gauss = corr_gauss.reshape((corr_mat.shape[0], n_timesteps, nbr_samples), order='F')
    reshaped = corr_gauss.transpose((2, 1, 0))
    return reshaped


def generate_gbm_paths(expiry, discount_rate, vols, corr_mat, method='pseudo'):
    n_paths = 2**16
    nbr_underlyings = corr_mat.shape[0]
    dt = expiry
    sqrdt = np.sqrt(dt)
    n_timesteps = int(expiry / dt)

    mean = np.zeros_like(vols)

    # gen = np.random.default_rng(seed=42)
    # shape (nbr_simul, nbr_timesteps, nbr_underlings)
    # randoms = np.array(gen.multivariate_normal(mean, corr_mat, (n_paths, n_timesteps)), dtype=np.float16)
    if method == 'pseudo':
        randoms = get_correlated_randoms(corr_mat, n_paths, n_timesteps, method='pseudo')
        # randoms = get_correlated_randoms(corr_mat, n_paths, n_timesteps, method='sobol')
    elif method == 'sobol':
        exponent = int(np.ceil(np.log2(n_paths)))
        brownian_bridge = BrownianBridge(n_timesteps)
        brownian_bridge.create_equidistant_frame()
        randoms = get_random_numbers(method='sobol', distr='gauss', dim=nbr_underlyings * n_timesteps, exponent=exponent)
        randoms = randoms.reshape((nbr_underlyings, n_timesteps, n_paths), order='F')
        wiener_paths = np.zeros_like(randoms)
        brownian_bridge.build_path(wiener_paths, randoms)
        wiener_paths = np.concatenate((np.zeros((nbr_underlyings, 1, n_paths)), wiener_paths), axis=1)
        wiener_increments = np.diff(wiener_paths, axis=1)  # these are still independent
        wiener_increments = wiener_increments.reshape((nbr_underlyings, n_timesteps * n_paths), order='F')
        eig_vals, eig_vecs = np.linalg.eigh(corr_mat)
        pseudo_sqrt = eig_vecs @ np.diag(np.sqrt(eig_vals))
        correlated_increments = pseudo_sqrt @ wiener_increments
        wiener_increments = correlated_increments.reshape((nbr_underlyings, n_timesteps, n_paths))
        randoms = wiener_increments.transpose((2, 1, 0))


    # s_{t+dt}/s_t
    quotient_stock = np.exp((discount_rate - vols**2/2)*dt + vols * randoms * sqrdt)

    stock_init = np.ones((n_paths, 1, corr_mat.shape[0]))
    quotient_stock = np.cumproduct(np.concatenate((stock_init, quotient_stock), axis=-2), axis=-2)
    print("Paths generated")
    return quotient_stock


def discount_factor(discount_rate, time):
    return np.exp(-discount_rate * time)


def price_basket_option(paths, strike, discount_rate, expiry, notional):
    # paths shape is (nbr_simulations, nbr_days, nbr_underlyings)
    worst_of_price = np.mean(paths[:, -1, :], axis=-1)
    option_payoff = discount_factor(discount_rate, expiry) * (
            -notional + notional/strike * np.where(worst_of_price < strike, worst_of_price, strike))

    # payoff = discounted_coupons + notional + option_payoff
    price = np.mean(option_payoff)
    return price


def plot_convergence(payoffs, exact_sol):
    nbr_paths = len(payoffs)
    max_log = np.log10(nbr_paths)
    logspace = np.logspace(3, max_log, 1000)
    logplot = []
    for l in logspace:
        logplot.append(np.mean(payoffs[:int(l)]))
    relative_res = np.abs(logplot-exact_sol)
    plt.loglog(logspace, relative_res)
    plt.scatter(logspace, relative_res)
    plt.loglog(logspace, 80/np.sqrt(logspace))
    plt.loglog(logspace, 4000/ logspace)
    plt.show()


def plot_paths(paths, simul_idx, horizontals=[], verticals=[]):
    plot_paths = paths[simul_idx, :, :]
    times = plot_paths.shape[0]
    underlyings = plot_paths.shape[1]
    for i in range(underlyings):
        plt.plot(np.arange(times), plot_paths[:, i])
    for h in horizontals:
        plt.plot(np.arange(times), h * np.ones(times), '-.')
    for v in verticals:
        plt.plot(250*v * np.ones(2), [np.min(plot_paths), np.max(plot_paths)], '-.')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    """
    strike_perc = 1.1992540061473846
    barrier_perc = 0.5521831959486008
    autocall_barrier = 0.9468936562538147
    discount_rate = 0.012914824485778811
    coupon_rate = 0.057548989057540904
    coupon_freq = 0.5
    autocall_freq = 0.25
    expiry = 2
    vols = np.array([0.149483360350132, 0.3279642671346664, 0.4145426481962204])
    corrs = np.array([-0.7705254024658131, 0.38454342272585446, -0.02522538136951505])
    notional = 1000
    """
    strike_perc = 0.9
    discount_rate = 0.01
    expiry = 1.0
    vols = np.array([0.2])
    corrs = np.array([1])
    notional = 900
    corr_mat = np.array([[1]])
    """corr_mat = np.array([[1, corrs[0], corrs[1]],
                        [corrs[0], 1, corrs[2]],
                        [corrs[1], corrs[2], 1]])"""

    d1 = (-np.log(strike_perc) + (discount_rate + vols[0]**2/2)*expiry) / (vols[0] * np.sqrt(expiry))
    d2 = d1 - vols[0] * np.sqrt(expiry)

    put_price = 1000*(- norm.cdf(-d1) + strike_perc * np.exp(-discount_rate * expiry) * norm.cdf(-d2))

    paths = generate_gbm_paths(expiry, discount_rate, vols, corr_mat, method='pseudo')

    def plot_paths(simul_idx, horizontals=[strike_perc], verticals=[]):
        plot_paths = paths[simul_idx, :, :]
        times = plot_paths.shape[0]
        underlyings = plot_paths.shape[1]
        for i in range(underlyings):
            plt.plot(np.arange(times), plot_paths[:, i])
        for h in horizontals:
            plt.plot(np.arange(times), h * np.ones(times), '-.')
        for v in verticals:
            plt.plot(250 * v * np.ones(2), [np.min(plot_paths), np.max(plot_paths)], '-.')
        plt.grid(True)
        plt.show()

    #  strike, barrier, autocall_barrier, discount_rate, coupon, coupon_freq, autocall_freq, expiry, notional):
    prc = price_basket_option(paths, strike_perc, discount_rate, expiry, notional)
    print(prc)




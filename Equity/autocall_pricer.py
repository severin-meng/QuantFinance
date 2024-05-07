import numpy as np
from scipy.stats.qmc import Halton, Sobol
# from scipy.stats import norm
from scipy.special import ndtri
import time
import matplotlib.pyplot as plt
from volatility.path_constructors import BrownianBridge, build_path_numba


DAYS_PER_YEAR = 256


def get_random_numbers(exponent=10, dim=5, method='pseudo', distr='gauss', seed=42):
    """
    :param exponent: base 2 logarithm of number of simulations = 2^exponent
    :param dim: dimension of sample
    :param method: number generation method, 'pseudo' for pseudo-random Mersenne-Twister, 'sobol' or 'halton' for
                   respective scrambled low-discrepancy sequence
    :param distr: distribution to sample from, either 'gauss' or 'uniform'
    :return: sampled pseudo-random or low-discrepancy number of shape (dim, 2^exponent)
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
            return ndtri(nbrs)
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
    n_paths = int(2**14)
    dt = 1 / DAYS_PER_YEAR
    sqrdt = np.sqrt(dt)
    n_timesteps = int(expiry / dt)
    nbr_underlyings = corr_mat.shape[0]
    mean = np.zeros_like(vols)

    # gen = np.random.default_rng(seed=42)
    # shape (nbr_simul, nbr_timesteps, nbr_underlings)
    # randoms = np.array(gen.multivariate_normal(mean, corr_mat, (n_paths, n_timesteps)), dtype=np.float16)
    # randoms = get_correlated_randoms(corr_mat, n_paths, n_timesteps, method='sobol')

    start = time.time()

    if method == 'pseudo':
        randoms = get_correlated_randoms(corr_mat, n_paths, n_timesteps, method='pseudo')
        random_time = time.time()
    else:
        exponent = int(np.ceil(np.log2(n_paths)))
        brownian_bridge = BrownianBridge(n_timesteps)
        brownian_bridge.create_equidistant_frame()
        bridge_frame = time.time()
        randoms = get_random_numbers(method='sobol', distr='gauss', dim=nbr_underlyings * n_timesteps, exponent=exponent)
        sobol_rand_time = time.time()
        randoms = randoms.reshape((nbr_underlyings, n_timesteps, n_paths), order='F')
        wiener_paths = np.zeros_like(randoms)
        # brownian_bridge.build_path(wiener_paths, randoms)
        build_path_numba(
            brownian_bridge.stddev, brownian_bridge.number_of_steps, brownian_bridge.left_index,
            brownian_bridge.right_index, brownian_bridge.bridge_index, brownian_bridge.left_weight,
            brownian_bridge.right_weight, wiener_paths, randoms)
        wiener_paths = np.concatenate((np.zeros((nbr_underlyings, 1, n_paths)), wiener_paths), axis=1)
        bridge_full_time = time.time()
        wiener_increments = np.diff(wiener_paths, axis=1)  # these are still independent
        wiener_increments = wiener_increments.reshape((nbr_underlyings, n_timesteps * n_paths), order='F')
        increments_time = time.time()
        eig_vals, eig_vecs = np.linalg.eigh(corr_mat)
        pseudo_sqrt = eig_vecs @ np.diag(np.sqrt(eig_vals))
        correlated_increments = pseudo_sqrt @ wiener_increments
        correlated_time = time.time()
        wiener_increments = correlated_increments.reshape((nbr_underlyings, n_timesteps, n_paths), order='F')
        randoms = wiener_increments.transpose((2, 1, 0))
        random_time = time.time()

    # s_{t+dt}/s_t
    quotient_stock = np.exp((discount_rate - vols**2/2)*dt + vols * randoms * sqrdt)

    stock_init = np.ones((n_paths, 1, 3))
    quotient_stock = np.cumproduct(np.concatenate((stock_init, quotient_stock), axis=-2), axis=-2)
    print("Paths generated")
    quotient_time = time.time() - start
    print(f"Randoms time: {random_time-start}")
    print(f"full time: {quotient_time-start}")
    if method == 'sobol':
        print(f"bridge time: {bridge_frame-start}")
        print(f"sobol gen time: {sobol_rand_time-bridge_frame}")
        print(f"brownian bridging: {bridge_full_time-sobol_rand_time}")
        print(f"wiener increments time: {increments_time-bridge_full_time}")
        print(f"correlated time: {correlated_time-increments_time}")
        print(f"final reshaping: {random_time-correlated_time}")
    return quotient_stock


def discount_factor(discount_rate, time):
    return np.exp(-discount_rate * time)


def price_autocall(paths, strike, barrier, autocall_barrier, discount_rate, coupon, coupon_freq, autocall_freq, expiry, notional):
    # paths shape is (nbr_simulations, nbr_days, nbr_underlyings)
    nbr_simul = paths.shape[0]
    nbr_days = paths.shape[1]

    coupon_dates = np.arange(1, expiry//coupon_freq+1) * coupon_freq
    coupon_payments = np.array([coupon * notional * coupon_freq for i in range(int(expiry//coupon_freq))])
    discounted_coupons = np.array([0] + [discount_factor(discount_rate, time) * coupon_payment
                                   for time, coupon_payment in zip(coupon_dates, coupon_payments)])
    # extend discounted coupons by 1 to account for no autocall being index -1
    coupon_date_index = np.rint(coupon_dates * DAYS_PER_YEAR)

    autocall_observ_dates = np.arange(1, expiry//autocall_freq) * autocall_freq
    autocall_date_index = (np.rint(autocall_observ_dates * DAYS_PER_YEAR) + 1).astype(np.int32)
    autocall_observ_and_maturity = np.arange(1, expiry//autocall_freq + 1) * autocall_freq

    autocall_breaches = np.where(np.all(paths[:, autocall_date_index, :] > autocall_barrier, axis=-1), 1, 0)
    # first_autocall is array like [nbr_simulations, int = autocall_date_index]
    is_autocall_breach = np.any(autocall_breaches, axis=-1)
    first_autocall = np.where(is_autocall_breach, np.argmax(autocall_breaches, axis=-1), -1)
    autocall_date = autocall_observ_and_maturity[first_autocall]  #
    last_coupon = first_autocall // 2

    coupons_until_autocall = np.cumsum(discounted_coupons)[last_coupon]
    full_coupons = np.sum(discounted_coupons)

    notional_contribution = notional * discount_factor(discount_rate, autocall_date)

    is_barrier_knocked = np.any(np.min(paths, axis=-1) < barrier, axis=-1)
    worst_of_price = np.min(paths[:, -1, :], axis=-1)
    option_payoff = discount_factor(discount_rate, expiry) * (
            -notional + notional/strike * np.where(worst_of_price < strike, worst_of_price, strike))

    # payoff = discounted_coupons + notional + option_payoff
    payoffs = np.where(is_autocall_breach, coupons_until_autocall + notional_contribution, full_coupons + notional_contribution + np.where(is_barrier_knocked, option_payoff, 0))
    price = np.mean(payoffs)
    return price


def plot_error_convergence(payoffs):
    nbr_paths = len(payoffs)
    max_log = np.log2(nbr_paths)
    logspace = np.logspace(3, max_log-1, 100, base=2)
    logplot = []
    for l in logspace:
        logplot.append(np.mean(payoffs[:int(l)]))
    best_res = np.mean(payoffs)
    relative_res = np.abs(logplot-best_res)
    plt.loglog(logspace, relative_res)
    plt.scatter(logspace, relative_res)
    plt.loglog(logspace, 80/np.sqrt(logspace))
    plt.loglog(logspace, 4000/ logspace)
    plt.show()


def plot_convergence(payoffs):
    nbr_paths = len(payoffs)
    max_log = np.log2(nbr_paths)
    logspace = np.logspace(3, max_log, 1000, base=2)
    logplot = []
    for l in logspace:
        logplot.append(np.mean(payoffs[:int(l)]))
    plt.semilogx(logspace, logplot)
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
    start = time.time()
    strike_perc = 1.0
    barrier_perc = 0.8
    autocall_barrier = 1.0
    discount_rate = 0.01
    coupon_rate = 0.058
    coupon_freq = 0.25
    autocall_freq = 0.25
    expiry = 1.0
    vols = np.array([0.2, 0.2, 0.2])
    corrs = np.array([0.4, 0.3, 0.6])
    notional = 1000
    corr_mat = np.array([[1, corrs[0], corrs[1]],
                        [corrs[0], 1, corrs[2]],
                        [corrs[1], corrs[2], 1]])
    paths = generate_gbm_paths(expiry, discount_rate, vols, corr_mat, method='sobol')
    after_path = time.time()
    print(f"Time to path generation: {after_path-start}")

    def plot_paths(simul_idx, horizontals=[strike_perc, barrier_perc, autocall_barrier], verticals=[]):
        plot_paths = paths[simul_idx, :, :]
        times = plot_paths.shape[0]
        underlyings = plot_paths.shape[1]
        for i in range(underlyings):
            plt.plot(np.arange(times), plot_paths[:, i])
        for h in horizontals:
            plt.plot(np.arange(times), h * np.ones(times), '-.')
        for v in verticals:
            plt.plot(DAYS_PER_YEAR * v * np.ones(2), [np.min(plot_paths), np.max(plot_paths)], '-.')
        plt.grid(True)
        plt.show()

    #  strike, barrier, autocall_barrier, discount_rate, coupon, coupon_freq, autocall_freq, expiry, notional):
    prc = price_autocall(paths, strike_perc, barrier_perc, autocall_barrier, discount_rate, coupon_rate, coupon_freq, autocall_freq, expiry, notional)
    print(prc)
    end = time.time()
    print(f"Time: {end-start}")




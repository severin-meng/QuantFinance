import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
from wiener_path_constructors import Incremental, SpectralSplit, BrownianBridge
from wiener_path_generators import PseudoRandomPathGenerator, SobolPathGenerator
from stock_models import VanillaGBM, ConstantShortRate
from path_dependent_payoffs import autocall_payoff

DAYS_PER_YEAR = 252

np.seterr(invalid='raise')


def error_convergence_equal_nbr_paths(path_exponent=13, samples=10):
    strike_perc = 1.0
    barrier_perc = 0.8
    autocall_barrier = 1.2
    discount_rate = 0.01
    coupon_rate = 0.058
    coupon_freq = 0.25
    autocall_freq = 0.25
    expiry = 1.0
    vols = np.array([0.6, 0.4, 0.2])
    corrs = np.array([0.6, 0.3, 0.4])
    nbr_underlyings = vols.shape[0]
    notional = 1000
    corr_mat = np.array([[1, corrs[0], corrs[1]],
                         [corrs[0], 1, corrs[2]],
                         [corrs[1], corrs[2], 1]])
    dt = 1 / DAYS_PER_YEAR
    sampling_times = dt * np.arange(1, int(expiry) * DAYS_PER_YEAR + 1, 1)
    const_short_rate = ConstantShortRate(discount_rate)

    pseudo_inc = []
    for i in range(samples):
        start = default_timer()
        path_gen = PseudoRandomPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
        path_time = default_timer()
        print(f"Time to incremental pseudo path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        pseudo_inc.append(pseudo_payoffs)
    # pseudo_inc shape is list of samples payoff lists: [samples, path_exponent]
    pseudo_inc = np.array(pseudo_inc)

    pseudo_antithetic = []
    for i in range(samples):
        start = default_timer()
        path_gen = PseudoRandomPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=True).paths
        path_time = default_timer()
        print(f"Time to antithetic pseudo path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        pseudo_antithetic.append(pseudo_payoffs)
    # pseudo_antithetic shape is list of samples payoff lists: [samples, 2*path_exponent]
    pseudo_antithetic = np.array(pseudo_antithetic)
    # average across antithetic samples for improved convergence
    pseudo_antithetic = np.mean(pseudo_antithetic.reshape(samples, -1, 2), axis=2)

    sobol_inc = []
    for i in range(samples):
        start = default_timer()
        path_gen = SobolPathGenerator(
            sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=Incremental)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
        path_time = default_timer()
        print(f"Time to incremental sobol path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        sobol_inc.append(pseudo_payoffs)
    # sobol_inc shape is list of samples payoff lists: [samples, path_exponent]
    sobol_inc = np.array(sobol_inc)

    sobol_bb = []
    for i in range(samples):
        start = default_timer()
        path_gen = SobolPathGenerator(
            sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=BrownianBridge, use_matrix=True)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
        path_time = default_timer()
        print(f"Time to brownian bridge sobol path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        sobol_bb.append(pseudo_payoffs)
    # sobol_bb shape is list of samples payoff lists: [samples, path_exponent]
    sobol_bb = np.array(sobol_bb)

    sobol_spectral = []
    for i in range(samples):
        start = default_timer()
        path_gen = SobolPathGenerator(
            sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=SpectralSplit)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
        path_time = default_timer()
        print(f"Time to spectral split sobol path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        sobol_spectral.append(pseudo_payoffs)
    # sobol_spectral shape is list of samples payoff lists: [samples, path_exponent]
    sobol_spectral = np.array(sobol_spectral)

    results = [pseudo_inc, pseudo_antithetic, sobol_inc, sobol_bb, sobol_spectral]
    label_dict = {0: "PseudoInc", 1: "PseudoAnti", 2: "SobolInc", 3: "SobolBB", 4: "SobolSpec"}

    # plot payoffs
    max_log = path_exponent
    logspace = np.logspace(6, max_log, 100, base=2)
    rolling_mean = []
    for path_nbr in logspace:
        new_entry = []
        for res in results:
            new_entry.append(np.mean(res[:, :int(path_nbr)], axis=1))
        rolling_mean.append(new_entry)
    rolling_mean = np.array(rolling_mean)  # has shape [100, 5, 10]
    pseudo_std = np.std(rolling_mean, ddof=1, axis=-1)

    for i in range(pseudo_std.shape[1]):
        plt.loglog(logspace, pseudo_std[:, i], label=label_dict[i])
        plt.scatter(logspace, pseudo_std[:, i])
    plt.loglog(logspace, pseudo_std[0, 0] * np.sqrt(logspace[0]) / np.sqrt(logspace),
               label=r'$\propto \frac{1}{\sqrt{N}}$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Monte Carlo Error Convergence")
    plt.xlabel('Number of Paths')
    plt.ylabel('Standard Error')
    plt.show()
    print("DONE")


def price_convergence_equal_nbr_paths(path_exponent=13, seed=None):
    strike_perc = 1.0
    barrier_perc = 0.8
    autocall_barrier = 1.2
    discount_rate = 0.01
    coupon_rate = 0.058
    coupon_freq = 0.25
    autocall_freq = 0.25
    expiry = 1.0
    vols = np.array([0.6, 0.4, 0.2])
    nbr_underlyings = vols.shape[0]
    corrs = np.array([0.6, 0.3, 0.4])

    notional = 1000
    corr_mat = np.array([[1, corrs[0], corrs[1]],
                         [corrs[0], 1, corrs[2]],
                         [corrs[1], corrs[2], 1]])
    dt = 1 / DAYS_PER_YEAR
    sampling_times = dt * np.arange(1, int(expiry) * DAYS_PER_YEAR + 1, 1)
    const_short_rate = ConstantShortRate(discount_rate)

    # pseudorandom incremental
    start = default_timer()
    path_gen = PseudoRandomPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
    path_time = default_timer()
    print(f"Time to incremental pseudo path generation: {path_time - start}")
    pseudo_inc = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                 coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)

    # pseudorandom antithetic
    start = default_timer()
    path_gen = PseudoRandomPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=True).paths
    path_time = default_timer()
    print(f"Time to antithetic pseudo path generation: {path_time - start}")
    pseudo_antithetic = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                        coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
    # average across antithetic samples for improved convergence
    pseudo_antithetic = np.mean(pseudo_antithetic.reshape(-1, 2), axis=1)

    # sobol incremental
    start = default_timer()
    path_gen = SobolPathGenerator(
        sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=Incremental, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
    path_time = default_timer()
    print(f"Time to incremental sobol path generation: {path_time - start}")
    sobol_inc = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)

    # sobol brownian bridge
    start = default_timer()
    path_gen = SobolPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat,
                                  path_constructor=BrownianBridge, use_matrix=True, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
    path_time = default_timer()
    print(f"Time to brownian bridge sobol path generation: {path_time - start}")
    sobol_bb = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                               coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)

    # sobol spectral split
    start = default_timer()
    path_gen = SobolPathGenerator(
        sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=SpectralSplit, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
    path_time = default_timer()
    print(f"Time to spectral split sobol path generation: {path_time - start}")
    sobol_spectral = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                     coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)

    results = [pseudo_inc, pseudo_antithetic, sobol_inc, sobol_bb, sobol_spectral]
    label_dict = {0: "PseudoInc", 1: "PseudoAnti", 2: "SobolInc", 3: "SobolBB", 4: "SobolSpec"}

    # plot payoffs
    max_log = path_exponent
    logspace = np.logspace(6, max_log, 100, base=2)
    rolling_mean = []
    for nbr_paths in logspace:
        new_entry = []
        for res in results:
            new_entry.append(np.mean(res[:int(nbr_paths)]))
        rolling_mean.append(new_entry)
    rolling_mean = np.array(rolling_mean)  # has shape [100, 5]

    for i in range(rolling_mean.shape[1]):
        plt.semilogx(logspace, rolling_mean[:, i], label=label_dict[i])
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Monte Carlo Price Convergence")
    plt.xlabel('Number of Paths')
    plt.ylabel('Price')
    plt.show()
    print("DONE")


def error_convergence_equal_time(path_exponent=13, samples=10):
    strike_perc = 1.0
    barrier_perc = 0.8
    autocall_barrier = 1.2
    discount_rate = 0.01
    coupon_rate = 0.058
    coupon_freq = 0.25
    autocall_freq = 0.25
    expiry = 1.0
    vols = np.array([0.6, 0.4, 0.2])
    nbr_underlyings = vols.shape[0]
    corrs = np.array([0.6, 0.3, 0.4])

    notional = 1000
    corr_mat = np.array([[1, corrs[0], corrs[1]],
                         [corrs[0], 1, corrs[2]],
                         [corrs[1], corrs[2], 1]])
    dt = 1 / DAYS_PER_YEAR
    sampling_times = dt * np.arange(1, int(expiry) * DAYS_PER_YEAR + 1, 1)
    const_short_rate = ConstantShortRate(discount_rate)

    pseudo_inc = []
    for i in range(samples):
        start = default_timer()
        path_gen = PseudoRandomPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(int(2 ** (path_exponent + 1.5)), antithetic_sampling=False).paths
        path_time = default_timer()
        print(f"Time to incremental pseudo path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        pseudo_inc.append(pseudo_payoffs)
        final = default_timer()
        pseudo_inc_time = final - start
    # pseudo_inc shape is list of samples payoff lists: [samples, path_exponent]
    pseudo_inc = np.array(pseudo_inc)

    pseudo_antithetic = []
    for i in range(samples):
        start = default_timer()
        path_gen = PseudoRandomPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=True).paths
        path_time = default_timer()
        print(f"Time to antithetic pseudo path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        pseudo_antithetic.append(pseudo_payoffs)
        final = default_timer()
        pseudo_antithetic_time = final - start
    # pseudo_antithetic shape is list of samples payoff lists: [samples, 2*path_exponent]
    pseudo_antithetic = np.array(pseudo_antithetic)
    # average across antithetic samples for improved convergence
    pseudo_antithetic = np.mean(pseudo_antithetic.reshape(samples, -1, 2), axis=2)

    sobol_inc = []
    for i in range(samples):
        start = default_timer()
        path_gen = SobolPathGenerator(
            sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=Incremental)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
        path_time = default_timer()
        print(f"Time to incremental sobol path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        sobol_inc.append(pseudo_payoffs)
        final = default_timer()
        sobol_inc_time = final - start
    # sobol_inc shape is list of samples payoff lists: [samples, path_exponent]
    sobol_inc = np.array(sobol_inc)

    sobol_bb = []
    for i in range(samples):
        start = default_timer()
        path_gen = SobolPathGenerator(
            sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=BrownianBridge, use_matrix=True)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
        path_time = default_timer()
        print(f"Time to brownian bridge sobol path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        sobol_bb.append(pseudo_payoffs)
        final = default_timer()
        sobol_bb_time = final - start
    # sobol_bb shape is list of samples payoff lists: [samples, path_exponent]
    sobol_bb = np.array(sobol_bb)

    sobol_spectral = []
    for i in range(samples):
        start = default_timer()
        path_gen = SobolPathGenerator(
            sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=SpectralSplit)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
        path_time = default_timer()
        print(f"Time to spectral split sobol path generation: {path_time - start}")
        pseudo_payoffs = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                         coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
        sobol_spectral.append(pseudo_payoffs)
        final = default_timer()
        sobol_spectral_time = final - start
    # sobol_spectral shape is list of samples payoff lists: [samples, path_exponent]
    sobol_spectral = np.array(sobol_spectral)

    results = [pseudo_inc, pseudo_antithetic, sobol_inc, sobol_bb, sobol_spectral]
    calc_times = [pseudo_inc_time, pseudo_antithetic_time, sobol_inc_time, sobol_bb_time, sobol_spectral_time]
    label_dict = {0: "PseudoInc", 1: "PseudoAnti", 2: "SobolInc", 3: "SobolBB", 4: "SobolSpec"}

    # plot payoffs
    cpu_times = []
    steps = 1000
    rolling_mean = np.empty((len(results), steps, samples))
    for j, (res, cpu_time) in enumerate(zip(results, calc_times)):
        new_entry = np.empty((steps, samples))
        timescale = np.logspace(-10, np.log2(cpu_time), steps, base=2)
        simul_scale = np.logspace(2, np.log2(res.shape[1]-1), steps, base=2)
        for i, nbr_paths in enumerate(simul_scale):
            new_entry[i, :] = np.mean(res[:, :int(nbr_paths)], axis=1)
        rolling_mean[j, :, :] = new_entry
        cpu_times.append(timescale)
    std_dev = np.std(rolling_mean, ddof=1, axis=-1)

    for i in range(std_dev.shape[0]):
        plt.loglog(cpu_times[i], std_dev[i, :], label=label_dict[i])
    """plt.loglog(cpu_times[0], std_dev[0, 0] * np.sqrt(cpu_times[0][0]) / np.sqrt(logspace),
                  label=r'$\propto \frac{1}{\sqrt{N}}$')"""
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Monte Carlo Error Convergence")
    plt.xlabel('CPU Time')
    plt.ylabel('Standard Error')
    plt.show()
    print("DONE")


def price_convergence_equal_time(path_exponent=17, seed=None):
    strike_perc = 1.0
    barrier_perc = 0.8
    autocall_barrier = 1.2
    discount_rate = 0.01
    coupon_rate = 0.058
    coupon_freq = 0.25
    autocall_freq = 0.25
    expiry = 1.0
    vols = np.array([0.6, 0.4, 0.2])
    nbr_underlyings = vols.shape[0]
    corrs = np.array([0.6, 0.3, 0.4])

    notional = 1000
    corr_mat = np.array([[1, corrs[0], corrs[1]],
                         [corrs[0], 1, corrs[2]],
                         [corrs[1], corrs[2], 1]])
    dt = 1 / DAYS_PER_YEAR
    sampling_times = dt * np.arange(1, int(expiry) * DAYS_PER_YEAR + 1, 1)
    const_short_rate = ConstantShortRate(discount_rate)

    # pseudorandom incremental
    start = default_timer()
    path_gen = PseudoRandomPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(int(2 ** (path_exponent + 1.3)), antithetic_sampling=False).paths
    path_time = default_timer()
    print(f"Time to incremental pseudo path generation: {path_time - start}")
    pseudo_inc = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                 coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
    final = default_timer()
    pseudo_inc_time = final-start

    # pseudorandom antithetic
    start = default_timer()
    path_gen = PseudoRandomPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=True).paths
    path_time = default_timer()
    print(f"Time to antithetic pseudo path generation: {path_time - start}")
    pseudo_antithetic = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                        coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
    # average across antithetic samples for improved convergence
    pseudo_antithetic = np.mean(pseudo_antithetic.reshape(-1, 2), axis=1)
    final = default_timer()
    pseudo_antithetic_time = final - start

    # sobol incremental
    start = default_timer()
    path_gen = SobolPathGenerator(
        sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=Incremental, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
    path_time = default_timer()
    print(f"Time to incremental sobol path generation: {path_time - start}")
    sobol_inc = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
    final = default_timer()
    sobol_inc_time = final - start

    # sobol brownian bridge
    start = default_timer()
    path_gen = SobolPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat,
                                  path_constructor=BrownianBridge, use_matrix=True, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
    path_time = default_timer()
    print(f"Time to brownian bridge sobol path generation: {path_time - start}")
    sobol_bb = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                               coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
    final = default_timer()
    sobol_bb_time = final - start

    # sobol spectral split
    start = default_timer()
    path_gen = SobolPathGenerator(
        sampling_times, nbr_underlyings, correlation=corr_mat, path_constructor=SpectralSplit, seed=seed)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False).paths
    path_time = default_timer()
    print(f"Time to spectral split sobol path generation: {path_time - start}")
    sobol_spectral = autocall_payoff(gbm_paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                     coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
    final = default_timer()
    sobol_spectral_time = final - start

    results = [pseudo_inc, pseudo_antithetic, sobol_inc, sobol_bb, sobol_spectral]
    calc_times = [pseudo_inc_time, pseudo_antithetic_time, sobol_inc_time, sobol_bb_time, sobol_spectral_time]
    label_dict = {0: "PseudoInc", 1: "PseudoAnti", 2: "SobolInc", 3: "SobolBB", 4: "SobolSpec"}

    # plot payoffs
    cpu_times = []
    rolling_mean = []
    for res, cpu_time in zip(results, calc_times):
        new_entry = []
        timescale = np.linspace(0, cpu_time, 1000)
        simul_scale = np.linspace(2, res.shape[0], 1000)
        for nbr_paths in simul_scale:
            new_entry.append(np.mean(res[:int(nbr_paths)]))
        rolling_mean.append(new_entry)
        cpu_times.append(timescale)
    rolling_mean = np.array(rolling_mean)  # has shape [5, 1000]

    """# plot brownian bridge result as "target" line
    best_res = rolling_mean[3, -1]
    start_time = 0
    end_time = cpu_times[3][-1]
    plt.plot([start_time, end_time], [best_res, best_res], label='SobolBB Target', color='k', linestyle='--')
    """
    for i in range(rolling_mean.shape[0]):
        plt.plot(cpu_times[i], rolling_mean[i, :], label=label_dict[i])
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Monte Carlo Price Convergence")
    plt.xlabel('CPU Time (seconds)')
    plt.ylabel('Price')
    plt.show()
    print("DONE")


if __name__ == '__main__':
    # price_convergence_equal_nbr_paths(path_exponent=14, seed=42)
    # error_convergence_equal_nbr_paths(path_exponent=11)
    price_convergence_equal_time(path_exponent=17, seed=42)
    # error_convergence_equal_time(path_exponent=18, samples=20)

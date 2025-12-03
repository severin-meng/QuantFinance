import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from wiener_path_constructors import PathConstructor, Incremental, SpectralSplit, BrownianBridge
from stochastic_path_generators import WienerPathGenerator, SobolPathGenerator, PseudoRandomPathGenerator
from models import EquityModel, EquityPaths, ConstantShortRate, GBM, HestonTruncatedEuler, HestonQuadraticExponential
from models import HestonQuadraticExponentialMartingaleCorrection, DeterministicShortRate
from payoffs import AutocallableBRC, BRC, PathDependentProduct, WorstOfDownOutCall, WorstOfDownInPut, WorstOfPut
from payoffs import WorstOfCall, WorstOfUpOutCall, AverageCall

from scrambling import reset_rng


def test_autocall():
    strike = 1.0
    expiry = 2.0
    knockin = 0.7
    coupon = 0.1
    coupon_freq = 4
    autocall_lvl = 1.0
    autocall_freq = 2

    ABRC = AutocallableBRC(
        strike=strike,
        expiry=expiry,
        knock_in_level=knockin,
        coupon_rate = coupon,
        coupon_frequency=coupon_freq,
        autocall_level=autocall_lvl,
        autocall_frequency=autocall_freq)

    brc = BRC(strike=strike, expiry=expiry, knock_in_level=knockin, coupon_rate=coupon, coupon_frequency=coupon_freq)

    call = WorstOfCall(strike=strike, expiry=expiry)

    avg_call = AverageCall(strike=strike, expiry=expiry)

    prods = {"Autocallable": ABRC, "BRC": brc, "call": call, "avg": avg_call}

    vols = np.array([0.2, 0.3, 0.4])
    spots = np.array([0.9, 0.8, 1.2])
    short_rate = 0.03
    corr_mat = np.array([
        [1., 0.6, 0.7],
        [0.6, 1., 0.8],
        [0.7, 0.8, 1.],
    ])

    model = GBM(volatilities=vols, short_rate_func=ConstantShortRate(short_rate=short_rate), correlation_matrix=corr_mat,
                initial_levels=spots)



    log2_nbr_points = 19
    num_paths = 1<<log2_nbr_points
    plot_points = 1 << np.array(range(10, log2_nbr_points+1))
    seed = 20251203
    # seed = 2
    runs = ["price", "price-variance", "delta", "delta-variance", "gamma", "gamma-variance"]
    nbr_variance_runs = 100
    """
    from here on, play with the rng and path constructor
    compute price, delta, gamma
    reference using pseudorandom?
    also - can save and reuse sampled correlated wiener paths - only generate once per seed, value 3 times for greeks?
    
    ok this is not very convincing - I see no improvement using hash-based owen vs simple bit-shift. Matousek is better but the result is noisy.
    maybe check out greeks next? Or try to replicate the paper plot using asian options?
    Convergence is maybe 1/n**0.6 or 1/n**0.7. But the intercept is much lower - about 2 orders of magnitude compared to pseudorandom, with very similar scaling though.
    """

    modes = ["Autocallable", "BRC", "call", "avg"]
    mode = modes[1]
    assert mode in prods
    prod = prods[mode]
    timeline = prod.simulation_times(frequency=60)

    # ref prcs are for 0 time steps!
    # ref_prcs = {"Autocallable": 881.3967291390097, "BRC": 885.31988337175, "call": 0.03619903998649099, "avg": 0.16378947952403367}
    ref_prcs = {"Autocallable": 881.4193346094362, "BRC": 885.2899872775976}
    if mode in ref_prcs:
        ref_prc = ref_prcs[mode]
        print(f"Reading reference price: {ref_prc}")
    else:
        print("Computing reference price")
        ref_num_paths = 1 << 25
        paths_per_pass = 1 << 17
        nbr_pass = int(ref_num_paths / paths_per_pass)
        # path_gen = PseudoRandomPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, seed=seed)
        path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, path_constructor=SpectralSplit, scrambler=False)
        ref_prc = 0
        for i in range(nbr_pass):
            print(f"{i+1}/{nbr_pass}")
            paths = model.generate_paths(paths_per_pass, path_gen)
            payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                                  is_clean_price=False)
            ref_prc += np.mean(payoffs)/nbr_pass
        print(f"Reference price: {ref_prc}")

    path_constructors = [Incremental, BrownianBridge, SpectralSplit]
    scramblers = [None, "bit-shift", "mato", "nestu", "hash-owen-bb", "hash-owen-alt", "hash-owen-final"]
    scramblers = ["bit-shift", "mato", "hash-owen-final"]
    # pick path_constructor and scrambler
    path_constructor = path_constructors[2]
    plt.figure()
    for scrambler in scramblers:
        print(f"Computing scrambler: {scrambler}")
        reset_rng(seed)
        results = np.empty((nbr_variance_runs, len(plot_points)))
        for i in range(nbr_variance_runs):
            print(f"{i+1}/{nbr_variance_runs}")
            path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, seed=seed,
                                          path_constructor=path_constructor, scrambler=scrambler)
            paths = model.generate_paths(num_paths, path_gen)
            payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True, is_clean_price=False)
            for j, k in enumerate(plot_points):
                results[i, j] = np.mean(payoffs[:k]) - ref_prc

            # cum_mean = np.cumsum(payoffs) / np.array(range(1, num_paths + 1))

        plt.loglog(plot_points, np.mean(results**2, axis=0), label=f"BrownianBridge-{scrambler}")
    # pseudorandom reference
    print("Computing pseudorandom variance")
    results = np.empty((nbr_variance_runs, len(plot_points)))
    for i in range(nbr_variance_runs):
        print(f"{i+1}/{nbr_variance_runs}")
        seed += i
        path_gen = PseudoRandomPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix,
                                             seed=seed)
        paths = model.generate_paths(num_paths, path_gen)
        payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                              is_clean_price=False)
        for j, k in enumerate(plot_points):
            results[i, j] = np.mean(payoffs[:k]) - ref_prc
    pseudo_var = np.mean(results ** 2, axis=0)
    plt.loglog(plot_points, pseudo_var, label=f"Pseudorandom variance")
    plt.loglog(plot_points, pseudo_var[0] / plot_points**2 * plot_points[0] ** 2, label="1 / n^2")
    plt.loglog(plot_points, pseudo_var[0] / plot_points ** 3 * plot_points[0] ** 3, label="1 / n^3")
    plt.loglog(plot_points, pseudo_var[0] / plot_points * plot_points[0], label="1 / n")

    # plt.plot([1025, num_paths], [ref_prc, ref_prc], label=f"Reference price")

    plt.legend()
    plt.grid()
    plt.title(f"Sobol convergence for {mode}")
    plt.xlabel("Path Index")
    plt.ylabel("Mean Squared Error")
    plt.show()


if __name__ == "__main__":
    test_autocall()


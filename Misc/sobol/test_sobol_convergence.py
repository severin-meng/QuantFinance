import numpy as np

from wiener_path_constructors import PathConstructor, Incremental, SpectralSplit, BrownianBridge
from stochastic_path_generators import WienerPathGenerator, SobolPathGenerator, PseudoRandomPathGenerator
from models import EquityModel, EquityPaths, ConstantShortRate, GBM, HestonTruncatedEuler, HestonQuadraticExponential
from models import HestonQuadraticExponentialMartingaleCorrection, DeterministicShortRate
from payoffs import AutocallableBRC, BRC, PathDependentProduct, WorstOfDownOutCall, WorstOfDownInPut, WorstOfPut
from payoffs import WorstOfCall, WorstOfUpOutCall


def test_autocall():
    strike = 1.0
    expiry = 2.0
    knockin = 0.7
    coupon = 0.1
    coupon_freq = 4
    autocall_lvl = 1.0
    autocall_freq = 2

    prod = AutocallableBRC(
        strike=strike,
        expiry=expiry,
        knock_in_level=knockin,
        coupon_rate = coupon,
        coupon_frequency=coupon_freq,
        autocall_level=autocall_lvl,
        autocall_frequency=autocall_freq)

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

    timeline = prod.simulation_times(frequency=0)

    """
    from here on, play with the rng and path constructor
    compute price, delta, gamma
    reference using pseudorandom?
    also - can save and reuse sampled correlated wiener paths - only generate once per seed, value 3 times for greeks?
    """
    path_constructors = [Incremental, BrownianBridge, SpectralSplit]
    scramblers = [None, "bit-shift", "mato", "nestu", "hash-owen-bb", "hash-owen-alt", "hash-owen-final"]

    # pick path_constructor and scrambler
    path_constructor = path_constructors[1]
    scrambler = scramblers[4]

    # loop over seed
    # R = 50 # number of rounds to measure variance
    seed = 20251202
    num_paths = 1<<16

    path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, seed=seed,
                                  path_constructor=path_constructor, scrambler=scrambler)
    paths = model.generate_paths(num_paths, path_gen)
    price = np.mean(prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True, is_clean_price=False))
    print(price)


if __name__ == "__main__":
    test_autocall()


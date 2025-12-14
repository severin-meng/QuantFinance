import numpy as np
import matplotlib
from copy import deepcopy

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from wiener_path_constructors import PathConstructor, Incremental, SpectralSplit, BrownianBridge
from stochastic_path_generators import WienerPathGenerator, SobolPathGenerator, PseudoRandomPathGenerator, SobolPathGeneratorPCA
from models import EquityModel, EquityPaths, ConstantShortRate, GBM, HestonTruncatedEuler, HestonQuadraticExponential
from models import HestonQuadraticExponentialMartingaleCorrection, DeterministicShortRate
from payoffs import AutocallableBRC, BRC, PathDependentProduct, WorstOfDownOutCall, WorstOfDownInPut, WorstOfPut
from payoffs import WorstOfCall, WorstOfUpOutCall, AverageCall, FixedStrikeAsianCall, LookbackCall, Accumulator
from payoffs import LookbackCallFixed, DigitalCall, Pathological

from scrambling import reset_rng


def bump_model_s1(model, dx=0.01):
    model.initial_levels[0] += dx

modes = ["Autocallable", "BRC", "worst_of_call", "avg", "asianArithmetic", "asianGeometric", "dip", "lookback", "accu", "lookback_fix", "digital", "pathological"]

def get_products_models():
    strike = 1.0
    expiry = 2.0
    knockin = 0.7
    coupon = 0.1
    coupon_freq = 4
    autocall_lvl = 1.0
    autocall_freq = 2
    asian_fixing_freq = 8

    ABRC = AutocallableBRC(
        strike=strike,
        expiry=expiry,
        knock_in_level=knockin,
        coupon_rate = coupon,
        coupon_frequency=coupon_freq,
        autocall_level=autocall_lvl,
        autocall_frequency=autocall_freq)

    brc = BRC(strike=strike, expiry=expiry, knock_in_level=knockin, coupon_rate=coupon, coupon_frequency=coupon_freq)

    worst_of_call = WorstOfCall(strike=strike, expiry=expiry)

    #  strike: float, expiry: float, barrier_level: float, initial_fixing: float = 1.,
    #                  quantity: float = 1., barrier_type: str = 'continuous'):
    dip = WorstOfDownInPut(strike=strike, expiry=expiry, barrier_level=knockin, barrier_type='discrete')

    avg_call = AverageCall(strike=strike, expiry=expiry)

    asianGeom = FixedStrikeAsianCall(strike=strike, expiry=expiry, fixing_freq=asian_fixing_freq, avg="geometric")
    asianArith = FixedStrikeAsianCall(strike=strike, expiry=expiry, fixing_freq=asian_fixing_freq, avg="arithmetic")

    accu = Accumulator(strike=0.95, expiry=1.0, fixing_freq=250, ko_level=1.18)

    lookback = LookbackCall(expiry=expiry, current_strike=1.0)

    lookback_fixed = LookbackCallFixed(strike=strike, expiry=expiry)

    digital = DigitalCall(strike, expiry)

    pathological = Pathological(strike, expiry, fixing_freq=4)

    prods = {"Autocallable": ABRC, "BRC": brc, "worst_of_call": worst_of_call, "avg": avg_call, "asianGeometric": asianGeom,
             "asianArithmetic": asianArith, "dip": dip, "lookback": lookback, "accu": accu, "lookback_fix": lookback_fixed,
             "digital": digital, "pathological": pathological}

    vols = np.array([0.3, 0.3, 0.3])
    spots = np.array([1.0, 1.0, 1.0])
    short_rate = 0.03
    corr_mat = np.array([
        [1., 0.8, 0.2],
        [0.8, 1., 0.2],
        [0.2, 0.2, 1.],
    ])

    model_m = GBM(volatilities=vols, short_rate_func=ConstantShortRate(short_rate=short_rate), correlation_matrix=corr_mat,
                initial_levels=spots)

    model_s = GBM(volatilities=np.array([0.2]), short_rate_func=ConstantShortRate(short_rate=short_rate), initial_levels=np.array([1.0]))

    models = {'multi': model_m, 'single': model_s}
    return prods, models

def test_convergence():
    prods, models = get_products_models()

    offset = 5
    log2_nbr_points = offset + 9
    num_paths = 1<<log2_nbr_points
    plot_points = 1 << np.array(range(3, log2_nbr_points+1))
    seed = 20251203
    nbr_variance_runs = 100
    #           0               1       2               3       4                   5               6       7           8       9               10          11
    modes = ["Autocallable", "BRC", "worst_of_call", "avg", "asianArithmetic", "asianGeometric", "dip", "lookback", "accu", "lookback_fix", "digital", "pathological"]
    mode = modes[1]
    compute_pca_full = True
    if mode == 'worst_of_call':
        model = models['multi']
        freq = 12
    elif mode == 'dip':
        model = models['multi']
        freq = 12
    elif mode in ('lookback', 'lookback_fix'):
        compute_pca_full = False
        model = models['single']
        freq = 50
    elif mode == 'asianArithmetic':
        compute_pca_full = False
        model = models['single']
        freq = 0
    elif mode == 'asianGeometric':
        compute_pca_full = False
        model = models['single']
        freq = 0
    elif mode == "accu":
        compute_pca_full = False
        model = models['single']
        freq = 0  # freq already set to 250
    elif mode == 'digital':
        compute_pca_full = False
        model = models['single']
        freq = 12
    elif mode == 'pathological':
        freq = 12
        compute_pca_full = True
        model = models['multi']
    else:
        model = models['multi']
        freq = 12
    assert mode in prods
    prod = prods[mode]
    if mode == 'digital':
        prod.smoothing = 0.01
    timeline = prod.simulation_times(frequency=freq)
    print(timeline)
    compute_delta_gamma = True

    dx = 0.0001
    model_up = deepcopy(model)
    model_down = deepcopy(model)
    bump_model_s1(model_up, dx)
    bump_model_s1(model_down, -dx)


    ref_prcs0 = {"Autocallable": 881.3967291390097, "BRC": 885.31988337175, "worst_of_call": 0.03619903998649099, "avg": 0.16378947952403367}
    ref_prcs60 = {"Autocallable": 881.4193346094362, "BRC": 885.2899872775976}
    all_prcs = {0: ref_prcs0, 60: ref_prcs60}
    ref_prcs = all_prcs.get(freq, ())
    # ref prcs are for 0 time steps!
    ref_prcs = {}
    if mode in ref_prcs:
        ref_prc = ref_prcs[mode]
        print(f"Reading reference price: {ref_prc}")
    else:
        print("Computing reference price")
        log2_nbr_points_ref = offset + 15
        ref_num_paths = 1 << log2_nbr_points_ref
        paths_per_pass = 1 << (log2_nbr_points_ref-5)
        nbr_pass = int(ref_num_paths / paths_per_pass)
        # path_gen = PseudoRandomPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, seed=seed)
        path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, path_constructor=BrownianBridge, scrambler=False)
        ref_prc = 0
        ref_up = 0
        ref_dn = 0
        for i in range(nbr_pass):
            print(f"{i+1}/{nbr_pass}")
            paths = model.generate_paths(paths_per_pass, path_gen, save=True, reuse=False)
            payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                                  is_clean_price=False)
            ref_prc += np.mean(payoffs)/nbr_pass
            if not compute_delta_gamma:
                continue
            paths_up = model_up.generate_paths(paths_per_pass, path_gen, save=False, reuse=True)
            payoffs_up = prod.discounted_path_payoff(paths_up, model.short_rate_func, is_barrier_continuous=True,
                                                  is_clean_price=False)
            ref_up += np.mean(payoffs_up) / nbr_pass
            paths_down = model_down.generate_paths(paths_per_pass, path_gen, save=False, reuse=True)
            payoffs_down = prod.discounted_path_payoff(paths_down, model.short_rate_func, is_barrier_continuous=True,
                                                  is_clean_price=False)
            ref_dn += np.mean(payoffs_down) / nbr_pass
        ref_delta = (ref_up - ref_dn) / (2*dx)
        ref_gamma = (ref_up + ref_dn - 2 * ref_prc) / (dx**2)
        print(f"Reference price: {ref_prc}, delta: {ref_delta}, gamma: {ref_gamma}")
    path_constructors = [Incremental, BrownianBridge, SpectralSplit, None]
    pth_constr_names = ["Incr", "BB", "PCA", "PCA-Full"]
    scramblers = [None, "bit-shift", "mato", "nestu", "hash-owen-bb", "hash-owen-alt", "hash-owen-final"]
    scramblers =["bit-shift", "mato", "hash-owen-final"]
    # pick path_constructor and scrambler
    # path_constructor = path_constructors[2]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Convergence Test for {mode}")
    ax[0].set_title("Price MSE")
    ax[1].set_title("Delta MSE")
    ax[2].set_title("Gamma MSE")

    ax[0].set_xlabel("N")
    ax[1].set_xlabel("N")
    ax[2].set_xlabel("N")

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

    for pth_constr_name, path_constructor in zip(pth_constr_names, path_constructors):
        if not compute_pca_full and pth_constr_name == "PCA-Full":
            print(f"Skipping {pth_constr_name}")
            continue
        print(f"Computing path constructor: {pth_constr_name}")
        for scrambler in scramblers:
            print(f"Computing scrambler: {scrambler}")
            if scrambler == "mato":
                nbr_variance_runs = 100
            else:
                nbr_variance_runs = 100
            # reset_rng(seed)
            results = np.empty((nbr_variance_runs, len(plot_points)))
            results_delta = np.empty((nbr_variance_runs, len(plot_points)))
            results_gamma = np.empty((nbr_variance_runs, len(plot_points)))
            for i in range(nbr_variance_runs):
                print(f"{i+1}/{nbr_variance_runs}")
                if pth_constr_name == "PCA-Full":
                    path_gen = SobolPathGeneratorPCA(timeline, model.random_factors, correlation=model.correlation_matrix, seed=seed, scrambler=scrambler)
                else:
                    path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, seed=seed,
                                                  path_constructor=path_constructor, scrambler=scrambler)
                paths = model.generate_paths(num_paths, path_gen, save=True, reuse=False)
                payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True, is_clean_price=False)
                for j, k in enumerate(plot_points):
                    results[i, j] = np.mean(payoffs[:k]) - ref_prc
                if not compute_delta_gamma:
                    continue
                paths_up = model_up.generate_paths(num_paths, path_gen, save=False, reuse=True)
                payoffs_up = prod.discounted_path_payoff(paths_up, model.short_rate_func, is_barrier_continuous=True, is_clean_price=False)
                paths_dn = model_down.generate_paths(num_paths, path_gen, save=False, reuse=True)
                payoffs_dn = prod.discounted_path_payoff(paths_dn, model.short_rate_func, is_barrier_continuous=True, is_clean_price=False)

                deltas = (payoffs_up - payoffs_dn) / (2*dx)
                gammas = (payoffs_up + payoffs_dn - 2*payoffs) / (dx**2)
                for j, k in enumerate(plot_points):
                    results_delta[i, j] = np.mean(deltas[:k]) - ref_delta
                for j, k in enumerate(plot_points):
                    results_gamma[i, j] = np.mean(gammas[:k]) - ref_gamma

                # cum_mean = np.cumsum(payoffs) / np.array(range(1, num_paths + 1))

            ax[0].loglog(plot_points, np.mean(results**2, axis=0), 'o-', label=f"{pth_constr_name}-{scrambler}")
            if not compute_delta_gamma:
                continue
            ax[1].loglog(plot_points, np.mean(results_delta ** 2, axis=0), 'o-', label=f"{pth_constr_name}-{scrambler}")
            ax[2].loglog(plot_points, np.mean(results_gamma ** 2, axis=0), 'o-', label=f"{pth_constr_name}-{scrambler}")

    # pseudorandom reference
    print("Computing pseudorandom variance")
    results = np.empty((nbr_variance_runs, len(plot_points)))
    results_delta = np.empty((nbr_variance_runs, len(plot_points)))
    results_gamma = np.empty((nbr_variance_runs, len(plot_points)))
    for i in range(nbr_variance_runs):
        print(f"{i+1}/{nbr_variance_runs}")
        seed += i
        path_gen = PseudoRandomPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix,
                                             seed=seed)
        paths = model.generate_paths(num_paths, path_gen, save=True, reuse=False)
        payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                              is_clean_price=False)
        for j, k in enumerate(plot_points):
            results[i, j] = np.mean(payoffs[:k]) - ref_prc
        if not compute_delta_gamma:
            continue
        paths_up = model_up.generate_paths(num_paths, path_gen, save=False, reuse=True)
        payoffs_up = prod.discounted_path_payoff(paths_up, model.short_rate_func, is_barrier_continuous=True,
                                              is_clean_price=False)
        paths_dn = model_down.generate_paths(num_paths, path_gen, save=False, reuse=True)
        payoffs_dn = prod.discounted_path_payoff(paths_dn, model.short_rate_func, is_barrier_continuous=True,
                                              is_clean_price=False)

        deltas = (payoffs_up - payoffs_dn) / (2 * dx)
        gammas = (payoffs_up + payoffs_dn - 2 * payoffs) / (dx ** 2)
        for j, k in enumerate(plot_points):
            results_delta[i, j] = np.mean(deltas[:k]) - ref_delta
        for j, k in enumerate(plot_points):
            results_gamma[i, j] = np.mean(gammas[:k]) - ref_gamma

    pseudo_var = np.mean(results ** 2, axis=0)
    pseudo_delta_var = np.mean(results_delta ** 2, axis=0)
    pseudo_gamma_var = np.mean(results_gamma ** 2, axis=0)
    ax[0].loglog(plot_points, np.mean(results ** 2, axis=0), 'o-', label=f"Pseudorandom")
    ax[0].loglog(plot_points, pseudo_var[0] / plot_points**2 * plot_points[0] ** 2, 'b-', linewidth=2, label="1 / n^2")
    ax[0].loglog(plot_points, pseudo_var[0] / plot_points * plot_points[0], 'g-', linewidth=2, label="1 / n")

    if compute_delta_gamma:
        ax[1].loglog(plot_points, pseudo_delta_var, 'o-', label=f"Pseudorandom")
        ax[1].loglog(plot_points, pseudo_delta_var[0] / plot_points ** 2 * plot_points[0] ** 2, 'b-', linewidth=2,
                     label="1 / n^2")
        ax[1].loglog(plot_points, pseudo_delta_var[0] / plot_points * plot_points[0], 'g-', linewidth=2, label="1 / n")
        ax[2].loglog(plot_points, pseudo_gamma_var, 'o-', label=f"Pseudorandom")
        ax[2].loglog(plot_points, pseudo_gamma_var[0] / plot_points ** 2 * plot_points[0] ** 2, 'b-', linewidth=2,
                     label="1 / n^2")
        ax[2].loglog(plot_points, pseudo_gamma_var[0] / plot_points * plot_points[0], 'g-', linewidth=2, label="1 / n")


    # plt.loglog(plot_points, pseudo_var[0] / plot_points ** 3 * plot_points[0] ** 3, 'r-', linewidth=2, label="1 / n^3")


    # plt.plot([1025, num_paths], [ref_prc, ref_prc], label=f"Reference price")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    # plt.title(f"Sobol convergence for {mode}")
    fig.tight_layout()
    plt.show()


def run_sensi_analysis():
    from SALib.sample.sobol import sample as sobol_sample
    from SALib.analyze import sobol
    from scipy.special import ndtri

    prods, models = get_products_models()
    modes = ["Autocallable", "BRC", "worst_of_call", "avg", "asian"]
    mode = modes[4]
    if mode == 'asian':
        model = models['single']
    else:
        model = models['multi']
    assert mode in prods
    prod = prods[mode]
    freq = 8
    timeline = prod.simulation_times(frequency=freq)

    dx = 0.01
    model_up = deepcopy(model)
    model_down = deepcopy(model)
    bump_model_s1(model_up, dx)
    bump_model_s1(model_down, -dx)

    n_timesteps = len(timeline) - 1
    dim = n_timesteps * model.random_factors
    problem = {
        'num_vars': dim,
        'names': [f'x{i}' for i in range(dim)],
        'bounds': [[0, 1] for i in range(dim)],
    }
    calc_second_order = False
    uniforms = sobol_sample(problem, 1 << 14, calc_second_order=calc_second_order, scramble=False)
    # here comes the evaluate_model part
    n_samples = uniforms.shape[0]
    EPS = 1.e-12
    uniforms = np.clip(uniforms, EPS, 1 - EPS)
    samples = ndtri(uniforms)
    # iid_numbers = np.ascontiguousarray(samples.T).reshape((model.random_factors, n_timesteps, n_samples), order='F')
    iid_numbers = np.ascontiguousarray(samples.T).reshape((model.random_factors, n_timesteps, n_samples), order='C')

    print(f"Computing Sensi analysis with {n_samples} samples")

    path_constructors = [Incremental, BrownianBridge, SpectralSplit]
    # path_constructors = [Incremental]
    for i in range(len(path_constructors)):
        path_constructor = path_constructors[i]
        print(100 * "#")
        print(f"Running {path_constructor}")

        pth_constr = path_constructor(timeline)
        uncorrelated_increments = pth_constr.get_wiener_increments(iid_numbers)
        # path_gen = PseudoRandomPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, seed=seed)
        path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix,
                                      path_constructor=path_constructor, scrambler=False)
        path_gen.uncorrelated_increments = uncorrelated_increments
        paths = model.generate_paths(n_samples, path_gen, save=False, reuse=True)
        payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                              is_clean_price=False)
        print(f"Price: {np.mean(payoffs)}")
        paths_up = model_up.generate_paths(n_samples, path_gen, save=False, reuse=True)
        payoffs_up = prod.discounted_path_payoff(paths_up, model.short_rate_func, is_barrier_continuous=True,
                                              is_clean_price=False)
        paths_dn = model_down.generate_paths(n_samples, path_gen, save=False, reuse=True)
        payoffs_dn = prod.discounted_path_payoff(paths_dn, model.short_rate_func, is_barrier_continuous=True,
                                              is_clean_price=False)

        deltas = (payoffs_up - payoffs_dn) / (2 * dx)
        print(f"Delta: {np.mean(deltas)}")
        gammas = (payoffs_up + payoffs_dn - 2 * payoffs) / (dx**2)
        print(f"Gamma: {np.mean(gammas)}")
        print(f"Analyzing Sensitivity for Value")
        # here is the sobol analysis
        Si = sobol.analyze(problem, payoffs, print_to_console=False, calc_second_order=calc_second_order)
        print("Individual contributions: ", Si['S1'])
        print("Total Sobol Index: ", Si['ST'])
        print("Effective dimension: ", sum(Si['ST']))
        print(50*"#")

        print("Analyzing Sensitivity for Delta")
        # here is the sobol analysis
        Si = sobol.analyze(problem, deltas, print_to_console=False, calc_second_order=calc_second_order)
        print("Individual contributions: ", Si['S1'])
        print("Total Sobol Index: ", Si['ST'])
        print("Effective dimension: ", sum(Si['ST']))
        print(50*"#")

        print("Analyzing Sensitivity for Gamma")
        # here is the sobol analysis
        Si = sobol.analyze(problem, gammas, print_to_console=False, calc_second_order=calc_second_order)
        print("Individual contributions: ", Si['S1'])
        print("Total Sobol Index: ", Si['ST'])
        print("Effective dimension: ", sum(Si['ST']))
        print(50*"#")

    # print(Si)


def test_sensitivity():
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from SALib.test_functions import Ishigami

    problem = {
        'num_vars': 3,
        'names': ['x1','x2','x3'],
        'bounds': [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
    }

    param_values = saltelli.sample(problem, 1<<14)  # shape: (8192, 3)
    # generates N*(2D + 2) samples - D is 3, N is 1024
    # if only first order: N * (D + 2)
    # output container
    """
    Y = np.zeros([param_values.shape[0]])
    for i, X in enumerate(param_values):
        Y[i] = Ishigami.evaluate(X)
    """
    Y = Ishigami.evaluate(param_values)

    Si = sobol.analyze(problem, Y, print_to_console=True)

    print("DONE")
    pass


def test_pca():
    prods, models = get_products_models()
    modes = ["Autocallable", "BRC", "worst_of_call", "avg", "asianArithmetic", "asianGeometric", "dip", "lookback", "accu", "lookback_fix", "digital", "pathological"]
    mode = modes[11]
    print(mode)
    if mode in ('asianArithmetic', 'asianGeometric'):
        model = models['single']
        freq = 0
    elif mode in ('lookback', 'lookback_fix'):
        model = models['single']
        freq = 50
    elif mode == 'accu':
        model = models['single']
        freq = 0
    elif mode == 'pathological':
        model = models['multi']
        freq = 12
    else:
        model = models['multi']
        freq = 0
    assert mode in prods
    prod = prods[mode]

    num_paths = 1 << 18
    timeline = prod.simulation_times(frequency=freq)
    path_gen = SobolPathGeneratorPCA(timeline, model.random_factors, correlation=model.correlation_matrix, scrambler=False)
    paths = model.generate_paths(num_paths, path_gen, save=False, reuse=False)
    payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                          is_clean_price=False)
    value = np.mean(payoffs)
    print(f"PCA: {value}")

    path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, path_constructor=BrownianBridge, scrambler=False)
    paths = model.generate_paths(num_paths, path_gen, save=False, reuse=False)
    payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                          is_clean_price=False)
    value = np.mean(payoffs)
    print(f"BB: {value}")

    path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, path_constructor=SpectralSplit, scrambler=False)
    paths = model.generate_paths(num_paths, path_gen, save=False, reuse=False)
    payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                          is_clean_price=False)
    value = np.mean(payoffs)
    print(f"SS: {value}")


def test_digital():
    prods, models = get_products_models()
    modes = ["Autocallable", "BRC", "worst_of_call", "avg", "asianArithmetic", "asianGeometric", "dip", "lookback", "accu", "lookback_fix", "digital", "pathological"]
    model = models['single']
    mode = modes[10]
    print(mode)
    freq = 12
    prod = prods[mode]
    num_paths = 1 << 12
    timeline = prod.simulation_times(frequency=freq)

    path_gen = SobolPathGenerator(timeline, model.random_factors, correlation=model.correlation_matrix, path_constructor=BrownianBridge, scrambler=False)
    paths = model.generate_paths(num_paths, path_gen, save=False, reuse=False)

    plt.figure()
    for smooth in (0, 0.01, 0.0075, 0.005, 0.0025, 0.001):
        prod.smoothing = smooth
        payoffs = prod.discounted_path_payoff(paths, model.short_rate_func, is_barrier_continuous=True,
                                              is_clean_price=False)
        value = np.mean(payoffs)
        print(f"Smoothing: {smooth} -> value: {value}")

        plt.scatter(paths.paths[0, -1, :], payoffs, label=f'Smoothing = {smooth}')
    plt.xlabel("terminal spot")
    plt.ylabel("payoff")
    plt.title("digital payoff")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_convergence()
    # test_sensitivity()
    # run_sensi_analysis()
    # test_pca()
    # test_digital()
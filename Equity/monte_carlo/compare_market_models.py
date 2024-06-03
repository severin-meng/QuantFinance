import numpy as np
from timeit import default_timer

from wiener_path_generators import PseudoRandomPathGenerator, SobolPathGenerator
from wiener_path_constructors import Incremental, BrownianBridge, SpectralSplit
from market_models import (
    ConstantShortRate,
    HestonTruncatedEuler,
    HestonQuadraticExponential,
    HestonQuadraticExponentialMartingaleCorrection,
    VanillaGBM,
)
from path_dependent_payoffs import AutocallableBRC, BRC


if __name__ == '__main__':
    strike_perc = 1.0
    barrier_perc = 0.8
    autocall_barrier = 1.2
    discount_rate = 0.01
    coupon_rate = 0.058
    coupon_freq = 0.25
    autocall_freq = 0.25
    expiry = 1.0
    notional = 1000

    corrs_eq_eq = np.array([0.6, 0.3, 0.4])  # equity-equity correlations
    corrs_heston = np.array([-0.4, -0.4, -0.4])  # pair-wise equity-variance correlations
    corrs_var_var = np.array([0.4, 0.4, 0.4])
    corrs_eq_var = np.zeros(6, dtype=np.float64)  # off-diagonal equity-variance correlations
    # asset 0, asset 1, asset 2, variance 0, variance 1, variance 2
    corr_mat = np.array([
        [1,                corrs_eq_eq[0],   corrs_eq_eq[1],   corrs_heston[0],   corrs_eq_var[0],   corrs_eq_var[1]],
        [corrs_eq_eq[0],   1,                corrs_eq_eq[2],   corrs_eq_var[2],   corrs_heston[1],   corrs_eq_var[3]],
        [corrs_eq_eq[1],   corrs_eq_eq[2],   1,                corrs_eq_var[4],   corrs_eq_var[5],   corrs_heston[2]],
        [corrs_heston[0],  corrs_eq_var[2],  corrs_eq_var[4],  1,                 corrs_var_var[0],  corrs_var_var[1]],
        [corrs_eq_var[0],  corrs_heston[1],  corrs_eq_var[5],  corrs_var_var[0],  1,                 corrs_var_var[2]],
        [corrs_eq_var[1],  corrs_eq_var[3],  corrs_heston[2],  corrs_var_var[1],  corrs_var_var[2],  1]])

    # corr_mat = np.identity(6)
    const_vol_corr_mat = corr_mat[:3, :3]

    init_vols = np.array([0.2, 0.2, 0.2])
    nbr_underlyings = 2 * init_vols.shape[0]
    long_term_vols = np.array([0.2, 0.2, 0.2])
    mean_reversion_speeds = np.array([0.7, 0.7, 0.7])
    vol_of_vols = np.array([1, 1, 1])

    const_short_rate = ConstantShortRate(discount_rate)

    ContinuousAutoCallable = AutocallableBRC(strike_perc, expiry, const_short_rate, barrier_perc, autocall_barrier,
                                             1/autocall_freq, coupon_rate, 1/coupon_freq, notional=notional,
                                             knock_in_type='continuous')

    path_exponent = 18
    sampling_range = [4, 8, 16, 32, 64, 128, 256]

    gbm = []
    trunc_euler = []
    quadr_exponential = []
    qe_martingale = []

    for sample in sampling_range:
        sampling_times = ContinuousAutoCallable.simulation_times(sample)
        gbm_start = default_timer()
        path_gen = SobolPathGenerator(sampling_times, init_vols.shape[0], correlation=const_vol_corr_mat, seed=None)
        vanilla_model = VanillaGBM(long_term_vols, const_short_rate, path_gen)
        vanilla_paths = vanilla_model.generate_paths(2**path_exponent, antithetic_sampling=False, save=False)
        vanilla_payoff = ContinuousAutoCallable.path_payoff(vanilla_paths)

        euler_start = default_timer()
        path_gen = SobolPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=None)
        euler_model = HestonTruncatedEuler(const_short_rate, long_term_vols, mean_reversion_speeds, vol_of_vols,
                                           path_gen, correlation_matrix=corr_mat, init_vols=init_vols)
        euler_paths = euler_model.generate_paths(2 ** path_exponent, antithetic_sampling=False, save=False)
        payoffs_euler = ContinuousAutoCallable.path_payoff(euler_paths)

        qe_start = default_timer()
        path_gen = SobolPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=None)
        qe_model = HestonQuadraticExponential(const_short_rate, long_term_vols, mean_reversion_speeds, vol_of_vols,
                                              path_gen, correlation_matrix=corr_mat, init_vols=init_vols)
        qe_paths = qe_model.generate_paths(2 ** path_exponent, antithetic_sampling=False, save=False, reuse=True)
        payoffs_qe = ContinuousAutoCallable.path_payoff(qe_paths)

        qem_start = default_timer()
        path_gen = SobolPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=None)
        qem_model = HestonQuadraticExponentialMartingaleCorrection(const_short_rate, long_term_vols, mean_reversion_speeds, vol_of_vols,
                                              path_gen, correlation_matrix=corr_mat, init_vols=init_vols)
        qem_paths = qem_model.generate_paths(2 ** path_exponent, antithetic_sampling=False, save=False,
                                          reuse=True)  # use same random numbers
        payoffs_qem = ContinuousAutoCallable.path_payoff(qem_paths)
        end = default_timer()
        print(10*"#", f"   Time steps: {sample}   ", 10*"#")
        print(f"GBM Calc time: {euler_start - gbm_start}")
        print(f"Euler Calc time: {qe_start-euler_start}")
        print(f"Quadr.Exp. Calc time: {qem_start - qe_start}")
        print(f"Quadr.Exp.-Martingale Calc time: {end - qem_start}")
        gbm.append(np.mean(vanilla_payoff))
        trunc_euler.append(np.mean(payoffs_euler))
        quadr_exponential.append(np.mean(payoffs_qe))
        qe_martingale.append(np.mean(payoffs_qem))
    print(trunc_euler)
    print(quadr_exponential)
    print(qe_martingale)
    print(gbm)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogx(sampling_range, trunc_euler, label='Truncated Euler')
    plt.semilogx(sampling_range, quadr_exponential, label='Quadratic Exponential')
    plt.semilogx(sampling_range, qe_martingale, label='QE Martingale Corr.')
    plt.semilogx(sampling_range, gbm, label='Constant Vol GBM')
    plt.xlabel('Nbr Time Steps')
    plt.ylabel('Autocallable Price')
    plt.title('Heston Discretization Schemes')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    print("DONE")
    print("DONE AGAIN")
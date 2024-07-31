import numpy as np
import matplotlib.pyplot as plt
from functools import cache, cached_property
from scipy.special import ndtr

from wiener_path_generators import WienerPathGenerator


__all__ = ['EquityPaths', 'ConstantShortRate', 'VanillaGBM', 'HestonTruncatedEuler', 'HestonQuadraticExponential',
           'HestonQuadraticExponentialMartingaleCorrection']


np.set_printoptions(linewidth=500, precision=4, suppress=True)


class EquityPaths:
    """
    Dataclass containing simulated asset paths, vols (paths), correlation (paths), sampling_times, time_steps,
    nbr stocks, nbr_paths, nbr_timesteps
    default shape of paths, vols is [nbr_underlyings, nbr_timesteps (optional), nbr_paths (optional)]
    correlations shape is [nbr_underlyings, nbr_underlyings, nbr_timesteps (optional)]
    """
    def __init__(self, paths, variances, corr_mat, sampling_times, time_steps):
        assert len(paths.shape) == 3
        assert paths.shape[0] == variances.shape[0]
        assert corr_mat.shape[0] == corr_mat.shape[1]
        assert (paths.shape[0] == corr_mat.shape[0]) or (paths.shape[0] * 2 == corr_mat.shape[0])
        if len(variances.shape) == 3:
            assert variances.shape[1] == paths.shape[1]
            assert variances.shape[2] == paths.shape[2]
        if len(corr_mat.shape) == 3:
            assert corr_mat.shape[2] == paths.shape[1]
        self.paths = paths
        self.variances = variances
        self.corr_mat = corr_mat
        self.sampling_times = sampling_times
        self.time_steps = time_steps

    def thin_along_paths(self, path_mask):
        # remove paths according to path_mask
        self.paths = self.paths[:, :, path_mask]
        if len(self.variances.shape) == 3:
            self.variances = self.variances[:, :, path_mask]
        return self

    def plot_path(self, simul_index=0):
        plt.figure()
        paths = self.paths[:, :, simul_index]
        for i in range(paths.shape[0]):
            plt.plot(self.sampling_times, paths[i, :], label=f'Underlying {i}')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

    def plot_variance(self, simul_index=0):
        plt.figure()
        if len(self.variances.shape) != 3:
            variances = self.variances
            for i in range(variances.shape[0]):
                plt.plot(self.sampling_times, variances[i] * np.ones_like(self.sampling_times), label=f'Underlying {i}')
        else:
            variances = self.variances[:, :, simul_index]
            for i in range(variances.shape[0]):
                plt.plot(self.sampling_times, variances[i, :], label=f'Underlying {i}')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

class DeterministicShortRate:
    def get_short_rate(self, time=0):
        raise NotImplementedError

    def get_discount_factor(self, end, start=0):
        raise NotImplementedError

    def get_integrated_short_rate(self, end, start=0):
        raise NotImplementedError


class ConstantShortRate(DeterministicShortRate):
    def __init__(self, short_rate):
        # short rate is in continuously compounded format
        self.short_rate = short_rate

    def get_short_rate(self, time=0):
        if isinstance(time, np.ndarray):
            return self.short_rate * np.ones_like(time)
        return self.short_rate

    # TODO: implement caching that can handle numpy arrays
    def get_discount_factor(self, end, start=0):
        return np.exp(-self.get_integrated_short_rate(end, start=start))

    def get_integrated_short_rate(self, end, start=0):
        return (end - start) * self.short_rate


class VanillaGBM:
    """ Constant volatility geometric Brownian motion with deterministic short rate, flat dividend yield and
    flat repo spread"""

    def __init__(self, volatilities: np.ndarray, short_rate_func: DeterministicShortRate,
                 wiener_path_gen: WienerPathGenerator, correlation_matrix=None, dividend_yield=0, repo_spread=0,
                 initial_levels=1):
        if not isinstance(initial_levels, np.ndarray):
            initial_levels = initial_levels * np.ones_like(volatilities)
        self.initial_levels = initial_levels
        self.vols = volatilities
        # TODO: reorder corr_mat, vols and initial levels such that highest vol is first
        self.short_rate_func = short_rate_func

        self.forward_spread = repo_spread - dividend_yield  # TODO: adjust to same dimension as vols
        self.wiener_path_gen = wiener_path_gen
        if correlation_matrix is None:
            correlation_matrix = self.wiener_path_gen.corr_mat
        self.correlation_matrix = correlation_matrix
        self.sampling_times = wiener_path_gen.sampling_times

    def generate_paths(self, nbr_paths, save=False, reuse=True, antithetic_sampling=False, reuse_correlation=True,
                       log_paths=False):
        # wiener increments are time-scaled, but not volatility scaled
        wiener_increments = self.wiener_path_gen.get_correlated_increments(
            nbr_paths, save=save, reuse=reuse,
            correlation_override=self.correlation_matrix if not reuse_correlation else None)
        # shape of wiener_increments is [nbr_underlyings, nbr_timesteps, nbr_paths]
        short_rate_contrib = self.short_rate_func.get_integrated_short_rate(
            start=self.sampling_times[:-1], end=self.sampling_times[1:])
        forward_vol_contrib = (-self.forward_spread - self.vols ** 2 / 2)[..., None] * \
            self.wiener_path_gen.get_time_increments()[None, ...]
        exponent1 = short_rate_contrib[None, ..., None] + forward_vol_contrib[..., None]
        exponent2 = self.vols[:, None, None] * wiener_increments

        if antithetic_sampling:
            log_stock = np.empty(
                (wiener_increments.shape[0], wiener_increments.shape[1] + 1, 2 * nbr_paths), dtype=np.float64)
        else:
            log_stock = np.empty(
                (wiener_increments.shape[0], wiener_increments.shape[1] + 1, nbr_paths), dtype=np.float64)
        log_stock[:, 0, :] = np.log(self.initial_levels[..., None])
        if antithetic_sampling:
            log_stock[:, 1:, ::2] = exponent1 - exponent2
            log_stock[:, 1:, 1::2] = exponent1 + exponent2
            # quotient_stock = np.insert(exponent1 - exponent2, np.arange(nbr_paths), exponent1 + exponent2, axis=2)
            # stock_init = np.log(self.initial_levels) + np.ones((self.vols.shape[0], 1, 2 * nbr_paths))
        else:
            log_stock[:, 1:, :] = exponent1 + exponent2
            # stock_init = np.log(self.initial_levels) + np.zeros((self.vols.shape[0], 1, nbr_paths))
        paths = np.cumsum(log_stock, axis=1)
        if not log_paths:
            paths = np.exp(paths)
        return EquityPaths(paths, self.vols**2, self.correlation_matrix, self.sampling_times,
                           self.wiener_path_gen.get_time_increments())


class MultiAssetHeston:
    def __init__(self, short_rate_func: DeterministicShortRate, long_term_vols, mean_reversion_speeds, vol_of_vols,
                 wiener_path_gen: WienerPathGenerator, correlation_matrix=None, dividend_yield=0, repo_spread=0,
                 initial_levels=1.0, init_vols: np.ndarray = None):
        if init_vols is None:
            init_vols = long_term_vols
        if not isinstance(initial_levels, np.ndarray):
            initial_levels = initial_levels * np.ones_like(init_vols)
        self.initial_stocks = initial_levels
        self.initial_vols = init_vols
        self.short_rate_func = short_rate_func
        self.theta = long_term_vols**2
        self.kappa = mean_reversion_speeds
        self.xi = vol_of_vols
        self.wiener_path_gen = wiener_path_gen
        if correlation_matrix is None:
            correlation_matrix = self.wiener_path_gen.corr_mat
        self.correlation_matrix = correlation_matrix
        self.dividend_yield = dividend_yield
        self.repo_spread = repo_spread
        self.sampling_times = wiener_path_gen.sampling_times
        self.nbr_underlyings = self.correlation_matrix.shape[0] // 2
        self.feller_lhs = 2 * self.kappa * self.theta / self.xi**2

    @cached_property
    def feller_condition(self):
        return self.feller_lhs > 1

    def generate_paths(self, *args, **kwargs):
        raise NotImplementedError


class HestonTruncatedEuler(MultiAssetHeston):
    """ Multi-Asset Heston Model using Truncated Euler Discretization
    Truncated Euler means using Euler-Maruyama discretization with max(0, variance) in place of all the variance terms
    in the SDEs except for one (s.t. v(t+1) = v(t) + f(max(0, v(t))). This still allows negative variance, so it is not
    an absorption scheme.
    See also https://papers.tinbergen.nl/06046.pdf
    """
    # TODO: test this

    def generate_paths(self, nbr_paths, save=False, reuse=True, antithetic_sampling=False, reuse_correlation=True,
                       log_paths=False):
        # wiener increments are time-scaled, but not volatility scaled
        wiener_increments = self.wiener_path_gen.get_correlated_increments(
            nbr_paths, save=save, reuse=reuse,
            correlation_override=self.correlation_matrix if not reuse_correlation else None)
        if antithetic_sampling:
            log_stocks = np.empty((self.nbr_underlyings, self.wiener_path_gen.nbr_timesteps + 1, 2*nbr_paths),
                              dtype=np.float64)
        else:
            log_stocks = np.empty((self.nbr_underlyings, self.wiener_path_gen.nbr_timesteps + 1, nbr_paths),
                              dtype=np.float64)
        log_stocks[:, 0, :] = np.log(self.initial_stocks[..., None])
        variances = np.empty_like(log_stocks)
        variances[:, 0, :] = self.initial_vols[..., None] ** 2

        if antithetic_sampling:
            for idx, time_step in enumerate(self.wiener_path_gen.get_time_increments()):
                log_stocks[:, idx+1, ::2], variances[:, idx+1, ::2] = self._increment_path(
                    log_stocks[:, idx, ::2], variances[:, idx, ::2], time_step, wiener_increments[:, idx, :])
                log_stocks[:, idx + 1, 1::2], variances[:, idx + 1, 1::2] = self._increment_path(
                    log_stocks[:, idx, 1::2], variances[:, idx, 1::2], time_step, -wiener_increments[:, idx, :])
        else:
            for idx, time_step in enumerate(self.wiener_path_gen.get_time_increments()):
                log_stocks[:, idx+1, :], variances[:, idx+1, :] = self._increment_path(
                    log_stocks[:, idx, :], variances[:, idx, :], time_step, wiener_increments[:, idx, :])
        if log_paths:
            stocks = log_stocks
        else:
            stocks = np.exp(log_stocks)
        return EquityPaths(stocks, np.maximum(variances, 0), self.correlation_matrix, self.sampling_times,
                           self.wiener_path_gen.get_time_increments())

    def _increment_path(self, current_log_stock, current_var, dt, wiener_increment):
        # current stock has shape [nbr_underlyings, nbr_paths]; same as current_var
        # wiener_increment has shape [2*nbr_underlyings, nbr_paths], dt is scalar
        floored_var = np.maximum(current_var, 0)
        sqr_var = np.sqrt(floored_var)
        next_log_stocks = current_log_stock + self.short_rate_func.get_integrated_short_rate(dt) + (
                    self.repo_spread - self.dividend_yield - 0.5 * floored_var
            ) * dt + sqr_var * wiener_increment[:self.nbr_underlyings, :]
        next_vars = current_var + self.kappa[..., None] * (
                    self.theta[..., None] - floored_var
            ) * dt + self.xi[..., None] * sqr_var * wiener_increment[self.nbr_underlyings:, :]
        return next_log_stocks, next_vars


# TODO: adaptive class that switches from Full Truncation Euler to Quadratic Exponential if Feller is violated,
#  depending on speed (check if worth it)


class HestonQuadraticExponential(MultiAssetHeston):
    """ Multi-Asset Heston Model using Quadratic-Exponential Discretization
    Sources:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405  (single asset)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2729475 (multi-asset, based on single asset paper)
    """
    # TODO: test this
    def __init__(self, short_rate_func: DeterministicShortRate, long_term_vols, mean_reversion_speeds, vol_of_vols,
                 wiener_path_gen: WienerPathGenerator, correlation_matrix=None, dividend_yield=0, repo_spread=0,
                 initial_levels=1.0, init_vols: np.ndarray = None):
        super().__init__(short_rate_func, long_term_vols, mean_reversion_speeds, vol_of_vols,
                         wiener_path_gen, correlation_matrix=correlation_matrix, dividend_yield=dividend_yield,
                         repo_spread=repo_spread, initial_levels=initial_levels, init_vols=init_vols)
        # value of (conditional variance)/(conditional expectation)**2  where quadratic scheme switches to exponential
        self.switching_value = 1.5  # has to be in interval [1,2]
        self.gamma_1 = self.gamma_2 = 0.5  # weights used in variance quadrature, should add up to 1

        # dissect correlation matrix for easier handling
        # heston_rho[i] = rho[i, nbr_und + i]
        heston_correlation = np.empty(self.nbr_underlyings)
        for i in range(self.nbr_underlyings):
            heston_correlation[i] = self.correlation_matrix[i, self.nbr_underlyings + i]
        # this array contains the pairwise asset - variance correlations in the heston model
        self.heston_correlation = heston_correlation
        assert np.all(self.heston_correlation < 1) and np.all(self.heston_correlation > -1)
        self.time_steps = self.wiener_path_gen.get_time_increments()

    def generate_paths(self, nbr_paths, save=False, reuse=True, antithetic_sampling=False, reuse_correlation=True,
                       log_paths=False):
        # wiener increments are time-scaled, but not volatility scaled
        # remove time scale from wiener increments
        correlated_normals = self.wiener_path_gen.get_correlated_increments(
            nbr_paths, save=save, reuse=reuse,
            correlation_override=self.correlation_matrix if not reuse_correlation else None).copy() / \
            np.sqrt(self.time_steps[None, ..., None])

        # decorrelate asset-variance pairs to feed into single-asset quadratic exponential algorithm
        # TODO: check correlations of this!
        correlated_normals = (self._partial_decorrelator @ correlated_normals.transpose(2, 0, 1)).transpose(1, 2, 0)

        if antithetic_sampling:
            log_stocks = np.empty((self.nbr_underlyings, self.wiener_path_gen.nbr_timesteps + 1, 2*nbr_paths),
                              dtype=np.float64)
        else:
            log_stocks = np.empty((self.nbr_underlyings, self.wiener_path_gen.nbr_timesteps + 1, nbr_paths),
                              dtype=np.float64)
        log_stocks[:, 0, :] = np.log(self.initial_stocks[..., None])
        variances = np.empty_like(log_stocks)
        variances[:, 0, :] = self.initial_vols[..., None] ** 2

        if antithetic_sampling:
            for idx, time_step in self.time_steps:
                log_stocks[:, idx+1, ::2], variances[:, idx+1, ::2] = self._increment_path(
                    log_stocks[:, idx, ::2], variances[:, idx, ::2], time_step, correlated_normals[:, idx, :])
                log_stocks[:, idx + 1, 1::2], variances[:, idx + 1, 1::2] = self._increment_path(
                    log_stocks[:, idx, 1::2], variances[:, idx, 1::2], time_step, -correlated_normals[:, idx, :])
        else:
            for idx, time_step in enumerate(self.wiener_path_gen.get_time_increments()):
                log_stocks[:, idx+1, :], variances[:, idx+1, :] = self._increment_path(
                    log_stocks[:, idx, :], variances[:, idx, :], time_step, correlated_normals[:, idx, :])
        if log_paths:
            stocks = log_stocks
        else:
            stocks = np.exp(log_stocks)
        return EquityPaths(stocks, variances, self.correlation_matrix, self.sampling_times,
                           self.wiener_path_gen.get_time_increments())

    @cached_property
    def _partial_decorrelator(self):
        # this matrix contains the inverses of the pairwise asset variance correlation matrices
        decorrelator = np.zeros((2*self.nbr_underlyings, 2*self.nbr_underlyings))
        for i in range(self.nbr_underlyings):
            corr = self.heston_correlation[i]
            sqr = np.sqrt(1 - corr**2)
            decorrelator[i, i] = 1 / sqr
            decorrelator[i, self.nbr_underlyings + i] = - corr / sqr
            decorrelator[i + self.nbr_underlyings, i + self.nbr_underlyings] = 1
        return decorrelator

    def _increment_path(self, current_log_stock, current_var, dt, correlated_normals):
        # current stock has shape [nbr_underlyings, nbr_paths]; same as current_var
        # std_increment has shape [2*nbr_underlyings, nbr_paths], dt is scalar
        psi, aux_vars, next_var = self._variance_process(current_var, dt, correlated_normals[self.nbr_underlyings:, :])
        next_log_stock = self._stock_process(current_log_stock, current_var, next_var, dt,
                                             correlated_normals[:self.nbr_underlyings], psi, aux_vars)
        return next_log_stock, next_var

    def _variance_process(self, current_var, time_step, std_normal):
        assert current_var.shape == std_normal.shape
        # wiener increment is normally distributed with mean=0 and variance=timestep
        # it has shape [nbr_underlyings, nbr_paths]

        # determine variance scheme: quadratic vs exponential
        cond_mean, cond_var = self._exact_moments_of_sqrt_process(current_var, time_step)
        psi = cond_var / cond_mean**2

        # psi has shape [nbr_underlyings, nbr_paths]
        psi_mask = (psi <= self.switching_value)
        aux_vars = np.empty((psi.shape[0], psi.shape[1], 2), dtype=np.float64)
        aux_vars[psi_mask] = self._match_moments_quadratic(psi[psi_mask], cond_mean[psi_mask])
        aux_vars[~psi_mask] = self._match_moments_exponential(psi[~psi_mask], cond_mean[~psi_mask])
        next_var = np.empty_like(current_var, dtype=np.float64)
        next_var[psi_mask] = self._quadratic_scheme(aux_vars[psi_mask], std_normal[psi_mask])
        next_var[~psi_mask] = self._exponential_scheme(aux_vars[~psi_mask], std_normal[~psi_mask])
        return psi, aux_vars, next_var

    def _stock_process(self, current_log_stock, current_var, next_var, time_step, std_increment, *args):
        k0, k1, k2, k3, k4 = self._constants_of_stock_calculation(time_step)
        return current_log_stock + k0 + k1 * current_var + k2 * next_var + np.sqrt(
            k3 * current_var + k4 * next_var) * std_increment

    @staticmethod
    def _quadratic_scheme(a_and_b, std_normal):
        # a_and_b has shape [std_normal.shape, 2] and a is in the first column, b in the second column
        return a_and_b[:, 0] * (a_and_b[:, 1] + std_normal)**2

    @staticmethod
    def _exponential_scheme(p_and_beta, std_normal):
        # p_and_beta has shape [std_normal.shape, 2] and p is in the first column, beta in the second column
        uniform = ndtr(std_normal)
        assert np.all(uniform < 1)
        uniform_mask = (uniform <= p_and_beta[:, 0])
        result = np.empty_like(uniform, dtype=np.float64)
        result[uniform_mask] = 0
        result[~uniform_mask] = 1 / p_and_beta[:, 1][~uniform_mask] * np.log((1-p_and_beta[:, 0][~uniform_mask])/(1-uniform[~uniform_mask]))
        # alternatively: return np.maximum(1 / p_and_beta[:, 1] * np.log((1-p_and_beta[:, 0])/(1-uniform)), 0)
        return result

    def _exact_moments_of_sqrt_process(self, current_variance, time_step):
        k1, k2, k3, k4 = self._constants_of_moments_calculation(time_step)
        cond_var = k1[..., None] * current_variance + k2[..., None]
        cond_mean = k3[..., None] * current_variance + k4[..., None]
        return cond_mean, cond_var

    @cache
    def _constants_of_stock_calculation(self, time_step, *args):
        k0 = - self.heston_correlation * self.kappa * self.theta / self.xi * time_step + self.short_rate_func.get_integrated_short_rate(time_step)
        k1 = self.gamma_1 * time_step * (self.kappa * self.heston_correlation / self.xi - 0.5) - \
            self.heston_correlation / self.xi
        k2 = self.gamma_2 * time_step * (self.kappa * self.heston_correlation / self.xi - 0.5) + \
            self.heston_correlation / self.xi
        k3 = self.gamma_1 * time_step * (1 - self.heston_correlation ** 2)
        k4 = self.gamma_2 * time_step * (1 - self.heston_correlation ** 2)
        return k0[..., None], k1[..., None], k2[..., None], k3[..., None], k4[..., None]

    @cache
    def _constants_of_moments_calculation(self, time_step):
        exp_kappa_dt = np.exp(-self.kappa * time_step)
        k1 = self.xi**2 * exp_kappa_dt / self.kappa * (1 - exp_kappa_dt)
        k2 = self.theta * self.xi**2 / (2 * self.kappa) * (1 - exp_kappa_dt)**2
        k3 = exp_kappa_dt
        k4 = self.theta * (1 - exp_kappa_dt)
        return k1, k2, k3, k4

    @staticmethod
    def _match_moments_quadratic(psi, cond_mean):
        # psi is a 1D array
        assert np.all(psi <= 2)
        two_over_psi = 2. / psi
        b_squared = two_over_psi - 1 + np.sqrt(two_over_psi * (two_over_psi - 1))
        b = np.sqrt(b_squared)
        a = cond_mean / (1 + b_squared)
        return np.array([a, b]).T  # need shape [psi.shape[0], 2]

    @staticmethod
    def _match_moments_exponential(psi, cond_mean):
        # psi is a 1D array
        assert np.all(psi > 1)
        p = (psi - 1)/(psi + 1)
        beta = (1 - p)/cond_mean
        return np.array([p, beta]).T  # need shape [psi.shape[0], 2]


class HestonQuadraticExponentialMartingaleCorrection(HestonQuadraticExponential):
    """ Multi-Asset Heston Model using Quadratic-Exponential Discretization with Martingale correction
    See source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405  (single asset)
    """
    def __init__(self, short_rate_func: DeterministicShortRate, long_term_vols, mean_reversion_speeds, vol_of_vols,
                 wiener_path_gen: WienerPathGenerator, correlation_matrix=None, dividend_yield=0, repo_spread=0,
                 initial_levels=1, init_vols: np.ndarray = None):
        super().__init__(short_rate_func, long_term_vols, mean_reversion_speeds, vol_of_vols,
                         wiener_path_gen, correlation_matrix=correlation_matrix, dividend_yield=dividend_yield,
                         repo_spread=repo_spread, initial_levels=initial_levels, init_vols=init_vols)
        if np.any(self.heston_correlation > 0):
            lhs = self.heston_correlation / self.xi * (1 + self.kappa * self.gamma_2 * self.time_steps)
            lhs -= 0.5 * self.gamma_2 * self.time_steps * self.heston_correlation**2
            rhs = 2 * self.kappa / (self.xi**2 * (1 - np.exp(-self.kappa * self.time_steps)))
            assert lhs < rhs
            print("Not all Heston correlations are negative. Martingale correction might not be possible."
                  "Proceed with caution.")

    def _stock_process(self, current_log_stock, current_var, next_var, time_step, std_increment, *args):
        psi, aux_vars = args
        capital_a, k1, k2, k3, k4 = self._constants_of_stock_calculation(time_step)
        k0 = self._martingale_correction(psi, k1, k3, capital_a, aux_vars, current_var) + self.short_rate_func.get_integrated_short_rate(time_step)
        return current_log_stock + k0 + k1 * current_var + k2 * next_var + np.sqrt(
            k3 * current_var + k4 * next_var) * std_increment

    @cache
    def _constants_of_stock_calculation(self, time_step):
        k1 = self.gamma_1 * time_step * (self.kappa * self.heston_correlation/self.xi - 0.5) - \
            self.heston_correlation / self.xi
        k2 = self.gamma_2 * time_step * (self.kappa * self.heston_correlation / self.xi - 0.5) + \
            self.heston_correlation / self.xi
        k3 = self.gamma_1 * time_step * (1 - self.heston_correlation ** 2)
        k4 = self.gamma_2 * time_step * (1 - self.heston_correlation ** 2)
        capital_a = k2 + 0.5 * k4
        return capital_a[..., None], k1[..., None], k2[..., None], k3[..., None], k4[..., None]

    def _martingale_correction(self, psi, k1, k3, capital_a, aux_vars, current_var):
        # TODO: check if masking is worth it in terms of speed
        psi_mask = (psi <= self.switching_value)
        k0 = np.empty_like(psi, dtype=np.float64)
        k0_quadratic = self._quadratic_martingale_correction(k1, k3, capital_a, aux_vars, current_var)
        k0_exp = self._exponential_martingale_correction(k1, k3, capital_a, aux_vars, current_var)
        # k0[psi_mask] = self._quadratic_martingale_correction(k1[psi_mask], k3[psi_mask], capital_a[psi_mask], aux_vars[psi_mask], current_var[psi_mask])
        # k0[~psi_mask] = self._exponential_martingale_correction(k1[~psi_mask], k3[~psi_mask], capital_a[~psi_mask], aux_vars[~psi_mask], current_var[~psi_mask])
        k0[psi_mask] = k0_quadratic[psi_mask]
        k0[~psi_mask] = k0_exp[~psi_mask]
        return k0

    @staticmethod
    def _quadratic_martingale_correction(k1, k3, capital_a, aux_vars, current_var):
        a = aux_vars[:, :, 0]
        b = aux_vars[:, :, 1]
        k0 = - capital_a * b ** 2 * a / (1 - 2 * capital_a * a) + 0.5 * np.log(1 - 2 * capital_a * a)
        k0 -= (k1 + 0.5 * k3) * current_var
        return k0

    @staticmethod
    def _exponential_martingale_correction(k1, k3, capital_a, aux_vars, current_var):
        p = aux_vars[:, :,  0]
        beta = aux_vars[:, :, 1]
        assert np.all(capital_a < beta), "Martingale correction impossible for given parameter set"
        k0 = - np.log(p + beta * (1 - p) / (beta - capital_a))
        k0 -= (k1 + 0.5 * k3) * current_var
        return k0


if __name__ == '__main__':
    from wiener_path_generators import PseudoRandomPathGenerator, SobolPathGenerator

    strike_perc = 0.792695078253746
    barrier_perc = 0.785103440284729
    autocall_barrier = 0.9628797680139541
    discount_rate = -0.011309803500771522
    coupon_rate = 0.09402714550495148
    coupon_freq = 0.25
    autocall_freq = 0.25
    expiry = 1.25
    vols = np.array([0.06668279357254506, 0.21892812550067903, 0.23095036298036575])
    nbr_underlyings = vols.shape[0]
    corrs = np.array([-0.2737681568179663, -0.6046816339561372, 0.45873618255231263])

    notional = 1000
    corr_mat = np.array([[1, corrs[0], corrs[1]],
                         [corrs[0], 1, corrs[2]],
                         [corrs[1], corrs[2], 1]])

    const_short_rate = ConstantShortRate(discount_rate)

    from path_dependent_payoffs import AutocallableBRC

    ContinuousAutoCallable = AutocallableBRC(strike_perc, expiry, const_short_rate, barrier_perc, autocall_barrier,
                                             1/autocall_freq, coupon_rate, 1/coupon_freq, notional=notional,
                                             knock_in_type='discrete')
    path_exponent = 10
    sample = 250
    sampling_times = ContinuousAutoCallable.simulation_times(sample)
    print(sampling_times)

    path_gen = SobolPathGenerator(
        sampling_times, nbr_underlyings, correlation=corr_mat, use_matrix=True, scaling='time-scaled', seed=0)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False)
    sobol_bb_uniform = ContinuousAutoCallable.path_payoff(gbm_paths)

    print(np.mean(sobol_bb_uniform) - notional)

"""    
    path_gen = SobolPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=41)
    gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
    gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False, log_paths=False)

    import matplotlib.pyplot as plt
    plt.figure()
    pth = gbm_paths.paths
    avg_path = np.mean(pth, axis=2)
    for i in range(avg_path.shape[0]):
        plt.plot(sampling_times, avg_path[i], label=f'{i}')
    plt.ylim(0.9, 1.1)
    plt.legend(loc='best')
    plt.show()

"""
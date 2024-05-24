import numpy as np
import matplotlib.pyplot as plt

from wiener_path_generators import WienerPathGenerator


__all__ = ['EquityPaths', 'ConstantShortRate', 'VanillaGBM']


class EquityPaths:
    """
    Dataclass containing simulated asset paths, vols (paths), correlation (paths), sampling_times, time_steps, nbr stocks, nbr_paths, nbr_timesteps
    default shape of paths, vols is [nbr_underlyings, nbr_timesteps (optional), nbr_paths (optional)]
    correlations shape is [nbr_underlyings, nbr_underlyings, nbr_timesteps (optional)]
    """
    def __init__(self, paths, vols, corr_mat, sampling_times, time_steps):
        assert len(paths.shape) == 3
        assert paths.shape[0] == vols.shape[0] == corr_mat.shape[0] == corr_mat.shape[1]
        if len(vols.shape) == 3:
            assert vols.shape[1] == paths.shape[1]
            assert vols.shape[2] == paths.shape[2]
        if len(corr_mat.shape) == 3:
            assert corr_mat.shape[2] == paths.shape[1]
        self.paths = paths
        self.vols = vols
        self.corr_mat = corr_mat
        self.sampling_times = sampling_times
        self.time_steps = time_steps

    def thin_along_paths(self, path_mask):
        # remove paths according to path_mask
        self.paths = self.paths[:, :, path_mask]
        if len(self.vols.shape) == 3:
            self.vols = self.vols[:, :, path_mask]
        return self

    def plot_path(self, simul_index=0):
        plt.figure()
        time_range = np.concatenate((np.array([0]), self.sampling_times))
        paths = self.paths[:, :, simul_index]
        for i in range(paths.shape[0]):
            plt.plot(time_range, paths[i, :], label=f'Underlying {i}')
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
        if type(initial_levels) == float:
            self.initial_levels = initial_levels * np.ones_like(volatilities)
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
            start=np.concatenate((np.array([0]), self.sampling_times[:-1])), end=self.sampling_times)
        forward_vol_contrib = (-self.forward_spread - self.vols ** 2 / 2)[..., None] * \
            self.wiener_path_gen.get_time_increments()[None, ...]
        exponent1 = short_rate_contrib[None, ..., None] + forward_vol_contrib[..., None]
        exponent2 = self.vols[:, None, None] * wiener_increments
        if not log_paths:
            if antithetic_sampling:
                exp1 = np.exp(exponent1)
                exp2 = np.exp(exponent2)
                quotient_stock = np.insert(exp1 / exp2, np.arange(nbr_paths), exp1 * exp2, axis=2)
                stock_init = self.initial_levels * np.ones((self.vols.shape[0], 1, 2 * nbr_paths))
            else:
                quotient_stock = np.exp(exponent1 + exponent2)
                stock_init = self.initial_levels * np.ones((self.vols.shape[0], 1, nbr_paths))
            paths = np.cumproduct(np.concatenate((stock_init, quotient_stock), axis=1), axis=1)
        else:
            if antithetic_sampling:
                quotient_stock = np.insert(exponent1 - exponent2, np.arange(nbr_paths), exponent1 + exponent2, axis=2)
                stock_init = np.log(self.initial_levels) + np.ones((self.vols.shape[0], 1, 2 * nbr_paths))
            else:
                quotient_stock = exponent1 + exponent2
                stock_init = np.log(self.initial_levels) + np.zeros((self.vols.shape[0], 1, nbr_paths))
            paths = np.cumsum(np.concatenate((stock_init, quotient_stock), axis=1), axis=1)
        return EquityPaths(paths, self.vols, self.correlation_matrix, self.sampling_times,
                           self.wiener_path_gen.get_time_increments())

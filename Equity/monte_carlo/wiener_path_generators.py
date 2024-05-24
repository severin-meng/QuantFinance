import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from wiener_path_constructors import PathConstructor, Incremental, BrownianBridge
from typing import Type

DAYS_PER_YEAR = 256


__all__ = ['WienerPathGenerator', 'SobolPathGenerator', 'PseudoRandomPathGenerator']


class RandomNumberGenerator:
    def __init__(self, nbr_timesteps, nbr_underlyings, seed):
        self.nbr_timesteps = nbr_timesteps
        self.nbr_underlyings = nbr_underlyings
        self.seed = seed

    def draw_samples(self, nbr_samples, distribution, **kwargs):
        """ return samples of shape (nbr_underlyings, nbr_timesteps, nbr_samples) """
        raise NotImplementedError


class PseudoRandomGenerator(RandomNumberGenerator):
    def __init__(self, nbr_timesteps, nbr_underlyings=1, seed=None):
        super().__init__(nbr_timesteps, nbr_underlyings, seed)
        self.gen = np.random.default_rng(seed=self.seed)

    def draw_samples(self, nbr_samples, distribution, correlation_matrix=None):
        assert distribution in ('uniform', 'gaussian')
        if distribution == 'gaussian':
            if correlation_matrix is not None:
                samples = self.gen.multivariate_normal(np.zeros(self.nbr_underlyings), correlation_matrix,
                                                       (nbr_samples, self.nbr_timesteps)).T
            else:
                samples = self.gen.standard_normal((self.nbr_underlyings, self.nbr_timesteps, nbr_samples))
        else:
            samples = self.gen.random((self.nbr_underlyings, self.nbr_timesteps, nbr_samples))
        return samples


class SobolGenerator(RandomNumberGenerator):
    def __init__(self, nbr_timesteps, nbr_underlyings=1, seed=None, bits=30, scramble=True, optimization=None):
        super().__init__(nbr_timesteps, nbr_underlyings, seed)
        self.dim = int(self.nbr_timesteps * self.nbr_underlyings)
        self.bits = bits
        self.scramble = scramble
        self.optimization = optimization
        self.gen = qmc.Sobol(
            self.dim, scramble=self.scramble, bits=self.bits, seed=self.seed, optimization=self.optimization)

    def draw_samples_power2(self, log2_nbr_samples, distribution, reshape=True):
        """ reshaping takes significant time for log2_nbr_samples > 12 """
        assert distribution in ('uniform', 'gaussian')
        samples = self.gen.random_base2(log2_nbr_samples)
        # shape of this is (2**log2_nbr_samples, nbr_timesteps * nbr_underlyings)
        if distribution == 'gaussian':
            # avoiding zero
            samples += 1 / (1 << (self.bits + 1))
            samples = ndtri(samples)
        if reshape:
            # returns F-contiguous array of shape (nbr_underlyings, nbr_timesteps, nbr_paths)
            return np.array(samples.T, order='C').reshape((
                self.nbr_underlyings, self.nbr_timesteps, 1 << log2_nbr_samples), order='F')
        else:
            return samples.T.reshape((
                self.nbr_underlyings, self.nbr_timesteps, 1 << log2_nbr_samples), order='F')

    def draw_samples(self, nbr_samples, distribution, reshape=True):
        if (nbr_samples & (nbr_samples - 1) == 0) and nbr_samples != 0:  # check if number is a power of 2
            return self.draw_samples_power2(nbr_samples.bit_length() - 1, distribution, reshape)
        raise Exception("Can only draw a power of 2 sample size from Sobol\' sequence.")


class WienerPathGenerator:
    """ Paths generated are of shape [nbr_underlyings, sampling_times, nbr_paths] """
    def __init__(self, sampling_times: np.ndarray, nbr_underlyings, **kwargs):
        self.sampling_times = sampling_times
        self.nbr_timesteps = sampling_times.shape[0]
        self.nbr_underlyings = nbr_underlyings
        self.corr_mat = kwargs.get('correlation')
        if self.corr_mat is None:
            self.corr_mat = np.identity(nbr_underlyings)
        assert self.corr_mat.shape in ((self.nbr_underlyings, self.nbr_underlyings),
                                       (self.nbr_timesteps, self.nbr_underlyings, self.nbr_underlyings))
        assert ((self.corr_mat.T == self.corr_mat).all() and (np.linalg.eigvals(self.corr_mat) >= 0).all())

    def get_correlated_increments(self, nbr_paths, **kwargs):
        raise NotImplementedError

    def get_uncorrelated_increments(self, nbr_paths, **kwargs):
        raise NotImplementedError

    def get_time_increments(self):
        raise NotImplementedError


class SobolPathGenerator(WienerPathGenerator):
    def __init__(self, sampling_times, nbr_underlyings, correlation=None,
                 path_constructor: Type[PathConstructor] = BrownianBridge, seed=None,
                 optimization=None, **kwargs):
        super().__init__(sampling_times, nbr_underlyings, correlation=correlation)
        self.path_constructor = path_constructor(self.sampling_times, **kwargs)
        self.num_gen = SobolGenerator(self.nbr_timesteps, self.nbr_underlyings, seed=seed, optimization=optimization)
        eig_vals, eig_vecs = np.linalg.eigh(self.corr_mat)
        self.correlation_sqrt = eig_vecs @ np.diag(np.sqrt(eig_vals))
        self.correlated_increments = None
        self.uncorrelated_increments = None
        self.reshape = getattr(self.path_constructor, 'use_matrix', True)

    def get_correlated_increments(self, nbr_paths, correlation_override=None, iid_numbers=None, save=False, reuse=True):
        if reuse and self.correlated_increments:
            return self.correlated_increments
        uncorrelated_increments = self.get_uncorrelated_increments(
            nbr_paths, iid_numbers=iid_numbers, save=False, reuse=reuse)
        if correlation_override is not None:
            eig_vals, eig_vecs = np.linalg.eigh(self.corr_mat)
            self.correlation_sqrt = eig_vecs @ np.diag(np.sqrt(eig_vals))
        correlated_increments = self._correlate_increments(uncorrelated_increments)
        if save:
            self.correlated_increments = correlated_increments
        return correlated_increments

    def get_uncorrelated_increments(self, nbr_paths, iid_numbers=None, save=False, reuse=True):
        if reuse and self.uncorrelated_increments:
            return self.uncorrelated_increments
        if iid_numbers is None:
            iid_numbers = self.num_gen.draw_samples(nbr_paths, distribution='gaussian', reshape=self.reshape)
        uncorrelated_increments = self.path_constructor.get_wiener_increments(iid_numbers)
        if save:
            self.uncorrelated_increments = uncorrelated_increments
        return uncorrelated_increments

    def _correlate_increments(self, uncorrelated_increments):
        return (self.correlation_sqrt @ uncorrelated_increments.transpose(2, 0, 1)).transpose(1, 2, 0)

    def get_time_increments(self):
        return self.path_constructor.time_increments


class PseudoRandomPathGenerator(WienerPathGenerator):
    def __init__(self, sampling_times, nbr_underlyings, correlation=None, seed=None):
        super().__init__(sampling_times, nbr_underlyings, correlation=correlation)
        # pseudo-random numbers do not benefit from effective dimension reduction
        self.path_constructor = Incremental(self.sampling_times)
        self.num_gen = PseudoRandomGenerator(self.nbr_timesteps, self.nbr_underlyings, seed=seed)
        self.correlation_sqrt = None
        self.correlated_increments = None
        self.uncorrelated_increments = None

    def get_correlated_increments(self, nbr_paths, correlation_override=None, iid_numbers=None, save=False, reuse=True,
                                  method='direct'):
        if reuse and self.correlated_increments:
            return self.correlated_increments
        if correlation_override:
            self.corr_mat = correlation_override
            self.correlation_sqrt = None
        if method == 'direct':
            # method == 'direct' directly draws correlated samples from numpy
            correlated_unit_time_increments = self.num_gen.draw_samples(nbr_paths, 'gaussian', self.corr_mat)
            correlated_increments = self.path_constructor.get_wiener_increments(correlated_unit_time_increments)
        else:
            uncorrelated_increments = self.get_uncorrelated_increments(
                nbr_paths, iid_numbers=iid_numbers, save=False, reuse=reuse)
            correlated_increments = self._correlate_increments(uncorrelated_increments)
        if save:
            self.correlated_increments = correlated_increments
        return correlated_increments

    def get_uncorrelated_increments(self, nbr_paths, iid_numbers=None, save=False, reuse=True):
        if reuse and self.uncorrelated_increments:
            return self.uncorrelated_increments
        if iid_numbers is None:
            iid_numbers = self.num_gen.draw_samples(nbr_paths, 'gaussian')
        uncorrelated_increments = self.path_constructor.get_wiener_increments(iid_numbers)
        if save:
            self.uncorrelated_increments = uncorrelated_increments
        return uncorrelated_increments

    def _correlate_increments(self, uncorrelated_increments):
        if self.correlation_sqrt is None:
            eig_vals, eig_vecs = np.linalg.eigh(self.corr_mat)
            self.correlation_sqrt = eig_vecs @ np.diag(np.sqrt(eig_vals))
        return self.correlation_sqrt @ uncorrelated_increments

    def get_time_increments(self):
        return self.path_constructor.time_increments

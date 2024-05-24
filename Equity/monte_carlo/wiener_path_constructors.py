""" This module contains different wiener path constructors to be used with low-discrepancy sampling """

import numpy as np
from numba import int32, float32, boolean
from numba.experimental import jitclass


__all__ = ['PathConstructor', 'Incremental', 'SpectralSplit', 'BrownianBridge']


class PathConstructor:
    def __init__(self, sampling_times: np.ndarray, **kwargs):
        """
        :param sampling_times: 1D numpy array containing times at which the wiener path is to be sampled.
        """
        assert len(sampling_times.shape) == 1  # array is 1-dimensional
        assert sampling_times.shape[0] > 1  # for one-step sampling, no path constructor method is required
        self.sampling_times = sampling_times
        self.nbr_steps = sampling_times.shape[0]
        self.time_increments = np.diff(np.concatenate((np.array([0]), self.sampling_times)))
        self.increment = self.time_increments[0]
        self.is_uniform = np.allclose(self.time_increments, self.increment)
        self.pseudo_square_root = None

    def build_path(self, iid_gaussians: np.ndarray):
        """
        :param iid_gaussians: 3D numpy array with shape (nbr_underlyings, nbr_timesteps, nbr_paths)
        :return: wiener path at discretized times sampling_times, starting at 0
        """
        raise NotImplementedError("Subclass must implement method \'build_path\'!")

    def get_wiener_increments(self, iid_gaussians: np.ndarray):
        """
        :param iid_gaussians: 3D numpy array with shape (nbr_underlyings, nbr_timesteps, nbr_paths)
        :return: wiener path increments at discretized times sampling_times
        """
        assert iid_gaussians.shape[1] == self.nbr_steps
        path = self.build_path(iid_gaussians)
        initial_state = np.zeros((iid_gaussians.shape[0], 1, iid_gaussians.shape[2]))
        return np.diff(np.concatenate((initial_state, path), axis=1), axis=1)


class Incremental(PathConstructor):
    def __init__(self, sampling_times: np.ndarray):
        super().__init__(sampling_times)
        self.sqrt_timesteps = None

    def build_path(self, iid_gaussians: np.ndarray):
        return np.cumsum(self.get_wiener_increments(iid_gaussians), axis=1)

    def get_wiener_increments(self, iid_gaussians: np.ndarray):
        assert len(iid_gaussians.shape) == 3 and iid_gaussians.shape[1] == self.nbr_steps
        if self.is_uniform:
            return np.sqrt(self.increment) * iid_gaussians
        if self.sqrt_timesteps is None:
            self.sqrt_timesteps = np.sqrt(self.time_increments)
        return self.sqrt_timesteps[None, ..., None] * iid_gaussians


class SpectralSplit(PathConstructor):
    def __init__(self, sampling_times: np.ndarray):
        super().__init__(sampling_times)
        self._calculate_spectral_split_matrix()

    def _calculate_spectral_split_matrix(self):
        if self.pseudo_square_root is not None:
            return
        if self.is_uniform:
            return self._calculate_spectral_split_matrix_uniform()
        covariance = np.fromfunction(
            lambda i, j: np.array([self.sampling_times[i], self.sampling_times[j]]).min(axis=0),
            (self.nbr_steps, self.nbr_steps), dtype=int)
        eig_val, eig_vec = np.linalg.eigh(covariance)
        self.pseudo_square_root = eig_vec[:, ::-1] @ np.diag(np.sqrt(eig_val[::-1]))  # in decreasing eigenvalue order

    def _calculate_spectral_split_matrix_uniform(self):
        """ With uniform time-spacing there is an analytical solution to the eigenvalue problem.
            See equation 10.73 in chapter 10.9.1 in Monte Carlo Methods in Finance by P. Jäckel """
        assert self.is_uniform
        two_n_plus_1 = 2 * self.nbr_steps + 1
        pi_half = np.pi/2
        prefactor = 1 / np.sqrt(two_n_plus_1)
        pseudo_sqrt = np.fromfunction(
            lambda i, j: prefactor * np.sin((i+1) * np.pi * (2 * j + 1)/two_n_plus_1) /
                         np.sin((2 * j + 1) / two_n_plus_1 * pi_half),
            (self.nbr_steps, self.nbr_steps), dtype=float)
        self.pseudo_square_root = pseudo_sqrt * np.sqrt(self.increment)

    def build_path(self, iid_gaussians: np.ndarray):
        assert len(iid_gaussians.shape) == 3 and iid_gaussians.shape[1] == self.nbr_steps
        if self.pseudo_square_root is None:
            self._calculate_spectral_split_matrix()
        return self.pseudo_square_root @ iid_gaussians


class BrownianBridge(PathConstructor):
    def __init__(self, sampling_times: np.ndarray, use_matrix=True):
        super().__init__(sampling_times)
        self.left_index = np.zeros(self.nbr_steps, dtype=int)  # left side point used in construction
        self.right_index = np.zeros(self.nbr_steps, dtype=int)  # right side point used in construction
        self.bridge_index = np.zeros(self.nbr_steps, dtype=int)  # indicates in which order the bridge is built
        self.left_weight = np.zeros(self.nbr_steps, dtype=float)  # expectation weight of left side point used in construction
        self.right_weight = np.zeros(self.nbr_steps, dtype=float)  # expect. weight of right side point used in construction
        self.stddev = np.zeros(self.nbr_steps, dtype=float)  # standard deviation at constructed point
        self.frame_built = False
        self.use_matrix = use_matrix

        self._create_frame()
        if self.use_matrix:
            self._calculate_path_constructor_matrix()

    def _create_frame(self):
        if self.frame_built:
            return
        if self.is_uniform:
            return self._create_frame_uniform()
        point_map = np.zeros(self.nbr_steps)
        # point_map is used to indicate which points are already constructed. If point_map[i] is zero, path point i is
        # not yet constructed. point_map[i] - 1 is the index of the variate that constructs the path point # i.
        point_map[-1] = 1  # the first point in the construction is the global step
        self.bridge_index[0] = self.nbr_steps - 1  # the global step is constructed from the first variate
        self.stddev[0] = np.sqrt(self.sampling_times[-1])  # the variance of the global step is t_n
        self.left_weight[0] = self.right_weight[0] = 0  # the global step to the last point in time is special.
        next_empty_idx = 0
        for i in range(1, self.nbr_steps):
            while point_map[next_empty_idx]:  # find the next unpopulated entry in the point_map
                next_empty_idx += 1
            next_existing_idx = next_empty_idx
            while not point_map[next_existing_idx]:  # find the next populated entry in the point_map from there.
                next_existing_idx += 1
            next_new_idx = next_empty_idx + (next_existing_idx - 1 - next_empty_idx) // 2
            # next_new_idx is now the index of the point to be constructed next.
            point_map[next_new_idx] = i
            self.bridge_index[i] = next_new_idx
            self.left_index[i] = next_empty_idx
            self.right_index[i] = next_existing_idx
            self.left_weight[i] = (self.sampling_times[next_existing_idx] - self.sampling_times[next_new_idx]) / (
                    self.sampling_times[next_existing_idx] - self.sampling_times[next_empty_idx-1])
            # self.right_weight[i] = (next_new_idx + 1 - next_empty_idx) / (next_existing_idx + 1 - next_empty_idx)
            self.right_weight[i] = 1 - self.left_weight[i]  # this should be faster
            self.stddev[i] = np.sqrt(((self.sampling_times[next_new_idx] - self.sampling_times[next_empty_idx-1]) * (
                    self.sampling_times[next_existing_idx] - self.sampling_times[next_new_idx])) / (
                    self.sampling_times[next_existing_idx] - self.sampling_times[next_empty_idx-1]))
            next_empty_idx = next_existing_idx + 1
            if next_empty_idx >= self.nbr_steps:
                next_empty_idx = 0
        self.frame_built = True

    def _create_frame_uniform(self):
        """ With uniform time-spacing there is a simpler brownian bridge """
        assert self.is_uniform
        point_map = np.zeros(self.nbr_steps)
        # point_map is used to indicate which points are already constructed. If point_map[i] is zero, path point i is
        # not yet constructed. point_map[i] - 1 is the index of the variate that constructs the path point # i.
        point_map[-1] = 1  # the first point in the construction is the global step
        self.bridge_index[0] = self.nbr_steps - 1  # the global step is constructed from the first variate
        self.stddev[0] = np.sqrt(self.nbr_steps)  # the variance of the global step is number_of_steps * 1.0
        self.left_weight[0] = self.right_weight[0] = 0  # the global step to the last point in time is special.
        next_empty_idx = 0
        for i in range(1, self.nbr_steps):
            while point_map[next_empty_idx]:  # find the next unpopulated entry in the point_map
                next_empty_idx += 1
            next_existing_idx = next_empty_idx
            while not point_map[next_existing_idx]:  # find the next populated entry in the point_map from there.
                next_existing_idx += 1
            next_new_idx = next_empty_idx + (next_existing_idx - 1 - next_empty_idx) // 2
            # next_new_idx is now the index of the point to be constructed next.
            point_map[next_new_idx] = i
            self.bridge_index[i] = next_new_idx
            self.left_index[i] = next_empty_idx
            self.right_index[i] = next_existing_idx
            self.left_weight[i] = (next_existing_idx - next_new_idx) / (next_existing_idx + 1 - next_empty_idx)
            # self.right_weight[i] = (next_new_idx + 1 - next_empty_idx) / (next_existing_idx + 1 - next_empty_idx)
            self.right_weight[i] = 1 - self.left_weight[i]  # this should be faster
            self.stddev[i] = np.sqrt(((next_new_idx + 1 - next_empty_idx) * (next_existing_idx - next_new_idx))
                                     / (next_existing_idx + 1 - next_empty_idx))
            next_empty_idx = next_existing_idx + 1
            if next_empty_idx >= self.nbr_steps:
                next_empty_idx = 0

        self.stddev *= np.sqrt(self.increment)  # uniform time step
        self.frame_built = True

    def _calculate_path_constructor_matrix(self):
        if not self.frame_built:
            self._create_frame()
        # find A such that path = A * normal_variates in the time dimension (identical in underlying and nbr paths dim)
        A = np.zeros((self.nbr_steps, self.nbr_steps))
        A[-1, 0] = self.stddev[0]
        B = np.identity(self.nbr_steps)
        for i in range(self.nbr_steps):
            next_empty_idx = self.left_index[i]
            next_existing_idx = self.right_index[i]
            next_new_idx = self.bridge_index[i]
            C = np.identity(self.nbr_steps)
            if next_empty_idx:
                C[next_new_idx, next_empty_idx-1] = self.left_weight[i]
                C[next_new_idx, next_existing_idx] = self.right_weight[i]
                A[next_new_idx, i] = self.stddev[i]
            else:
                C[next_new_idx, next_existing_idx] = self.right_weight[i]
                A[next_new_idx, i] = self.stddev[i]
            B = C @ B
        self.pseudo_square_root = B @ A

    def _build_path_sequential(self, iid_gaussians):
        if not self.frame_built:
            self._create_frame()
        path = np.empty_like(iid_gaussians)
        # path has shape [nbr_underlyings, nbr_timesteps, nbr_simulations]
        assert path.shape[1] == self.nbr_steps
        path[:, -1, :] = self.stddev[0] * iid_gaussians[:, 0, :]
        for i in range(self.nbr_steps):
            next_empty_idx = self.left_index[i]
            next_existing_idx = self.right_index[i]
            next_new_idx = self.bridge_index[i]
            if next_empty_idx:
                path[:, next_new_idx, :] = self.left_weight[i] * path[:, next_empty_idx-1, :] + self.right_weight[i] *\
                                           path[:, next_existing_idx, :] + self.stddev[i] * iid_gaussians[:, i, :]
            else:
                path[:, next_new_idx, :] = self.right_weight[i] * path[:, next_existing_idx, :] + self.stddev[i] * iid_gaussians[:, i, :]
        return path

    def build_path(self, iid_gaussians):
        assert len(iid_gaussians.shape) == 3 and iid_gaussians.shape[1] == self.nbr_steps
        if not self.frame_built:
            self._create_frame()
        if self.use_matrix:
            if self.pseudo_square_root is None:
                self._calculate_path_constructor_matrix()
            return self.pseudo_square_root @ iid_gaussians
        return self._build_path_sequential(iid_gaussians)


@jitclass({'sampling_times': float32[:],
           'nbr_steps': int32,
           'time_increments': float32[:],
           'increment': float32,
           'is_uniform': boolean,
           'left_index': int32[:],
           'right_index': int32[:],
           'bridge_index': int32[:],
           'left_weight': float32[:],
           'right_weight': float32[:],
           'stddev': float32[:],
           'frame_built': boolean})
class BrownianBridgeNumba(object):
    def __init__(self, sampling_times: np.ndarray):
        assert len(sampling_times.shape) == 1  # array is 1-dimensional
        assert sampling_times.shape[0] > 1  # for one-step sampling, no path constructor method is required
        self.sampling_times = sampling_times
        self.nbr_steps = sampling_times.shape[0]
        time_incr = self.sampling_times[1:] - self.sampling_times[:-1]
        zero = np.array([sampling_times[0]], dtype=float32)
        self.time_increments = np.concatenate((zero, time_incr))
        self.increment = self.time_increments[0]
        self.is_uniform = np.allclose(self.time_increments, self.increment)

        self.left_index = np.zeros(self.nbr_steps, dtype=np.int32)  # left side point used in construction
        self.right_index = np.zeros(self.nbr_steps, dtype=np.int32)  # right side point used in construction
        self.bridge_index = np.zeros(self.nbr_steps, dtype=np.int32)  # indicates in which order the bridge is built
        self.left_weight = np.zeros(self.nbr_steps, dtype=np.float32)  # expectation weight of left side point used in construction
        self.right_weight = np.zeros(self.nbr_steps, dtype=np.float32)  # expect. weight of right side point used in construction
        self.stddev = np.zeros(self.nbr_steps, dtype=np.float32)  # standard deviation at constructed point
        self.frame_built = False

        self._create_frame()

    def _create_frame(self):
        if self.frame_built:
            return
        if self.is_uniform:
            return self._create_frame_uniform()
        point_map = np.zeros(self.nbr_steps, dtype=np.int32)
        # point_map is used to indicate which points are already constructed. If point_map[i] is zero, path point i is
        # not yet constructed. point_map[i] - 1 is the index of the variate that constructs the path point # i.
        point_map[-1] = 1  # the first point in the construction is the global step
        self.bridge_index[0] = self.nbr_steps - 1  # the global step is constructed from the first variate
        self.stddev[0] = np.sqrt(self.sampling_times[-1])  # the variance of the global step is t_n
        self.left_weight[0] = self.right_weight[0] = 0  # the global step to the last point in time is special.
        next_empty_idx = 0
        for i in range(1, self.nbr_steps):
            while point_map[next_empty_idx]:  # find the next unpopulated entry in the point_map
                next_empty_idx += 1
            next_existing_idx = next_empty_idx
            while not point_map[next_existing_idx]:  # find the next populated entry in the point_map from there.
                next_existing_idx += 1
            next_new_idx = next_empty_idx + (next_existing_idx - 1 - next_empty_idx) // 2
            # next_new_idx is now the index of the point to be constructed next.
            point_map[next_new_idx] = i
            self.bridge_index[i] = next_new_idx
            self.left_index[i] = next_empty_idx
            self.right_index[i] = next_existing_idx
            self.left_weight[i] = (self.sampling_times[next_existing_idx] - self.sampling_times[next_new_idx]) / (
                    self.sampling_times[next_existing_idx] - self.sampling_times[next_empty_idx-1])
            # self.right_weight[i] = (next_new_idx + 1 - next_empty_idx) / (next_existing_idx + 1 - next_empty_idx)
            self.right_weight[i] = 1 - self.left_weight[i]  # this should be faster
            self.stddev[i] = np.sqrt(((self.sampling_times[next_new_idx] - self.sampling_times[next_empty_idx-1]) * (
                    self.sampling_times[next_existing_idx] - self.sampling_times[next_new_idx])) / (
                    self.sampling_times[next_existing_idx] - self.sampling_times[next_empty_idx-1]))
            next_empty_idx = next_existing_idx + 1
            if next_empty_idx >= self.nbr_steps:
                next_empty_idx = 0
        self.frame_built = True

    def _create_frame_uniform(self):
        """ With uniform time-spacing there is a simpler brownian bridge """
        assert self.is_uniform
        point_map = np.zeros(self.nbr_steps, dtype=np.int32)
        # point_map is used to indicate which points are already constructed. If point_map[i] is zero, path point i is
        # not yet constructed. point_map[i] - 1 is the index of the variate that constructs the path point # i.
        point_map[-1] = 1  # the first point in the construction is the global step
        self.bridge_index[0] = self.nbr_steps - 1  # the global step is constructed from the first variate
        self.stddev[0] = np.sqrt(self.nbr_steps)  # the variance of the global step is number_of_steps * 1.0
        self.left_weight[0] = self.right_weight[0] = 0  # the global step to the last point in time is special.
        next_empty_idx = 0
        for i in range(1, self.nbr_steps):
            while point_map[next_empty_idx]:  # find the next unpopulated entry in the point_map
                next_empty_idx += 1
            next_existing_idx = next_empty_idx
            while not point_map[next_existing_idx]:  # find the next populated entry in the point_map from there.
                next_existing_idx += 1
            next_new_idx = next_empty_idx + (next_existing_idx - 1 - next_empty_idx) // 2
            # next_new_idx is now the index of the point to be constructed next.
            point_map[next_new_idx] = i
            self.bridge_index[i] = next_new_idx
            self.left_index[i] = next_empty_idx
            self.right_index[i] = next_existing_idx
            self.left_weight[i] = (next_existing_idx - next_new_idx) / (next_existing_idx + 1 - next_empty_idx)
            # self.right_weight[i] = (next_new_idx + 1 - next_empty_idx) / (next_existing_idx + 1 - next_empty_idx)
            self.right_weight[i] = 1 - self.left_weight[i]  # this should be faster
            self.stddev[i] = np.sqrt(((next_new_idx + 1 - next_empty_idx) * (next_existing_idx - next_new_idx))
                                     / (next_existing_idx + 1 - next_empty_idx))
            next_empty_idx = next_existing_idx + 1
            if next_empty_idx >= self.nbr_steps:
                next_empty_idx = 0

        self.stddev *= np.sqrt(self.increment)  # uniform time step
        self.frame_built = True

    def _build_path_sequential(self, iid_gaussians):
        if not self.frame_built:
            self._create_frame()
        path = np.empty_like(iid_gaussians, dtype=np.float64)
        # path has shape [nbr_underlyings, nbr_timesteps, nbr_simulations]
        assert path.shape[1] == self.nbr_steps
        path[:, -1, :] = self.stddev[0] * iid_gaussians[:, 0, :]
        for i in range(self.nbr_steps):
            next_empty_idx = self.left_index[i]
            next_existing_idx = self.right_index[i]
            next_new_idx = self.bridge_index[i]
            if next_empty_idx:
                path[:, next_new_idx, :] = self.left_weight[i] * path[:, next_empty_idx-1, :] + self.right_weight[i] *\
                                           path[:, next_existing_idx, :] + self.stddev[i] * iid_gaussians[:, i, :]
            else:
                path[:, next_new_idx, :] = self.right_weight[i] * path[:, next_existing_idx, :] + self.stddev[i] * iid_gaussians[:, i, :]
        return path

    def build_path(self, iid_gaussians):
        assert len(iid_gaussians.shape) == 3 and iid_gaussians.shape[1] == self.nbr_steps
        if not self.frame_built:
            self._create_frame()
        return self._build_path_sequential(iid_gaussians)


if __name__ == '__main__':
    from timeit import default_timer
    from scipy.stats.qmc import Sobol
    from scipy.special import ndtri

    timesteps = 12  # 2**8
    underlyings = 3
    simulations = 2**14
    expiry = 3.0
    dt = expiry / timesteps
    sampling_times = dt * np.arange(1, timesteps + 1, 1)
    randoms = np.random.default_rng(seed=42).standard_normal(underlyings*timesteps*simulations)

    randoms = np.reshape(randoms, (underlyings, timesteps, simulations), order='C')

    bb = BrownianBridge(sampling_times)
    bb_path = bb.build_path(randoms)

    spectral = SpectralSplit(sampling_times)
    spec_path = spectral.build_path(randoms)

    np.set_printoptions(linewidth=500, precision=4, suppress=True)
    print(bb.pseudo_square_root)
    print(spectral.pseudo_square_root)

    """
    start = default_timer()
    bridge = BrownianBridge(sampling_times, use_matrix=False)
    path1 = bridge.build_path(randoms)
    end = default_timer()
    print(f"build path loop: {end-start}")
    start = default_timer()
    bridge = BrownianBridge(sampling_times, use_matrix=True)
    path2 = bridge.build_path(randoms)
    end = default_timer()
    print(f"matrix path time: {end-start}")
    print(np.max(np.abs(path2-path1)))

    print("DONE")

    import matplotlib.pyplot as plt

    for i in range(underlyings):
        for j in range(10):
            plt.plot(path1[i, :, j])
    plt.show()
    print("DONE Again")
    """

import numpy as np
from scipy.stats.qmc import Sobol
from scipy.special import ndtri


class BrownianBridge:
    def __init__(self, number_of_steps):
        self.number_of_steps = number_of_steps
        self.left_index = np.zeros(self.number_of_steps, dtype=int)  # left side point used in construction
        self.right_index = np.zeros(self.number_of_steps, dtype=int)  # right side point used in construction
        self.bridge_index = np.zeros(self.number_of_steps, dtype=int)  # indicates in which order the bridge is built
        self.left_weight = np.zeros(self.number_of_steps)  # expectation weight of left side point used in construction
        self.right_weight = np.zeros(self.number_of_steps)  # expect. weight of right side point used in construction
        self.stddev = np.zeros(self.number_of_steps)  # standard deviation at constructed point

    def create_equidistant_frame(self):
        assert self.number_of_steps  # there must be at least one step
        point_map = np.zeros(self.number_of_steps)  
        # point_map is used to indicate which points are already constructed. If point_map[i] is zero, path point i is 
        # not yet constructed. point_map[i] - 1 is the index of the variate that constructs the path point # i.
        point_map[-1] = 1  # the first point in the construction is the global step
        self.bridge_index[0] = self.number_of_steps - 1  # the global step is constructed from the first variate
        self.stddev[0] = np.sqrt(self.number_of_steps)  # the variance of the global step is number_of_steps * 1.0
        self.left_weight[0] = self.right_weight[0] = 0  # the global step to the last point in time is special.
        next_empty_idx = 0
        for i in range(1, self.number_of_steps):
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
            if next_empty_idx >= self.number_of_steps:
                next_empty_idx = 0

    def build_path_matrix(self):
        # find A such that path = A * normal_variates in the time dimension (identical in underlying and nbr paths dim)
        A = np.zeros((self.number_of_steps, self.number_of_steps))
        A[-1, 0] = self.stddev[0]
        B = np.identity(self.number_of_steps)
        # product_matrices = []
        # path[:, -1, :] = self.stddev[0] * normal_variates[:, 0, :]
        for i in range(self.number_of_steps):
            next_empty_idx = self.left_index[i]
            next_existing_idx = self.right_index[i]
            next_new_idx = self.bridge_index[i]
            C = np.identity(self.number_of_steps)
            if next_empty_idx:
                C[next_new_idx, next_empty_idx-1] = self.left_weight[i]
                C[next_new_idx, next_existing_idx] = self.right_weight[i]
                A[next_new_idx, i] = self.stddev[i]
                """path[:, next_new_idx, :] = self.left_weight[i] * path[:, next_empty_idx - 1, :] + \
                                           self.right_weight[i] * path[:, next_existing_idx, :] + \
                                           self.stddev[i] * normal_variates[:, i, :]"""
            else:
                C[next_new_idx, next_existing_idx] = self.right_weight[i]
                A[next_new_idx, i] = self.stddev[i]
                """path[:, next_new_idx, :] = self.right_weight[i] * path[:, next_existing_idx, :] + self.stddev[
                    i] * normal_variates[:, i, :]"""
            # product_matrices.append(C)
            B = C @ B
        # product_matrices = product_matrices[::-1]
        # product_matrices.append(A)
        # return np.linalg.multi_dot(product_matrices)
        return B @ A

    def build_path_with_matrix(self, normal_variates, path_matrix):
        assert normal_variates.shape[1] == self.number_of_steps
        return path_matrix @ normal_variates

    def build_path(self, path, normal_variates):
        # path has shape [nbr_underlyings, nbr_timesteps, nbr_simulations]
        assert normal_variates.shape == path.shape and path.shape[1] == self.number_of_steps
        path[:, -1, :] = self.stddev[0] * normal_variates[:, 0, :]
        for i in range(self.number_of_steps):
            next_empty_idx = self.left_index[i]
            next_existing_idx = self.right_index[i]
            next_new_idx = self.bridge_index[i]
            if next_empty_idx:
                path[:, next_new_idx, :] = self.left_weight[i] * path[:, next_empty_idx-1, :] + self.right_weight[i] *\
                                           path[:, next_existing_idx, :] + self.stddev[i] * normal_variates[:, i, :]
            else:
                path[:, next_new_idx, :] = self.right_weight[i] * path[:, next_existing_idx, :] + self.stddev[i] * normal_variates[:, i, :]


if __name__ == '__main__':
    import time
    timesteps = 256  # 2**8
    underlyings = 3
    simulations = 2**16
    bridge = BrownianBridge(timesteps)
    bridge.create_equidistant_frame()
    start = time.time()
    path_matrix = bridge.build_path_matrix()
    end = time.time()
    print(f"path matrix: {end-start}")
    randoms = np.random.default_rng(seed=42).standard_normal(underlyings*timesteps*simulations)

    """gen = Sobol(underlyings * timesteps, bits=64, seed=None)
    nbrs = gen.random_base2(int(np.log2(simulations))).T
    randoms = ndtri(nbrs)"""

    randoms = np.reshape(randoms, (underlyings, timesteps, simulations), order='C')

    path1 = np.zeros_like(randoms)
    start = time.time()
    bridge.build_path(path1, randoms)
    end = time.time()
    print(f"build path loop: {end-start}")
    start = time.time()
    path2 = bridge.build_path_with_matrix(randoms, path_matrix)
    end = time.time()
    print(f"matrix path time: {end-start}")
    print(np.max(np.abs(path2-path1)))

    print("DONE")

    import matplotlib.pyplot as plt

    for i in range(underlyings):
        for j in range(10):
            plt.plot(path1[i, :, j])

    print("DONE Again")

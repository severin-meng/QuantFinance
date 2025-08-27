import sobol_initialisation
from sobol_initialisation import bits

from pprint import pprint

class Sobol:
    """
    32-bit sobol sequence generator
    max dimension is 8'129'334
    """
    def __init__(self, dim):
        self.dim = dim
        self.index = 0
        self.dir_nbrs = self.get_direction_nbrs()  # shape: [dim][bit]
        self.state = [0 for i in range(dim)]

    def get_direction_nbrs(self):
        return sobol_initialisation.get_direction_numbers(self.dim)

    def next(self):

        # Gray code: find position j of right-most zero bit in current index n
        n = self.index
        j = 0
        while n & 1 == 1:
            n >>= 1
            j += 1

        for i in range(self.dim):
            self.state[i] ^= self.dir_nbrs[i][j]

        self.index += 1

    def get_uniform(self):
        self.next()
        return [i / 2**bits for i in self.state]

    def random(self, dim):
        # compatibility with numpy's default_rng
        return self.get_uniform()


def print_sobol_nums(dim, count):
    gen = Sobol(dim)
    print([0.0 for i in range(dim)])
    for i in range(count-1):
        print(gen.get_uniform())


def approximate_n_sphere(dim, samples, gen=None):
    # compute the volume of the n-dim unit ball using Monte Carlo as a benchmark.
    # convergence worsens in high dimensions due to curse of dimensionality - the average distance from the origin
    # quickly outgrows the radius of 1, and new points are increasingly unlikely to land inside the sphere.
    from scipy.linalg import norm
    from scipy.special import gamma
    from math import pi
    true_value = pi ** (dim/2) / gamma(dim/2 + 1)
    if gen is None:
        gen = Sobol(dim)
    full_volumes = []
    errors = []
    counter = 0
    for i in range(samples):
        uniform = gen.random(dim)
        if norm(uniform) < 1:
            counter += 1
        orthant_volume = counter / samples
        full_volume = orthant_volume * 2**dim
        full_volumes.append(full_volume)
        errors.append(full_volume - true_value)
    return true_value, full_volumes, errors


if __name__ == '__main__':
    dim = 5
    calls = 10
    print_sobol_nums(dim, calls+1)

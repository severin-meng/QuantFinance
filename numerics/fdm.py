from enum import Enum
import numpy as np
import scipy as sc

from products import ExerciseType
from products import EuropeanOption, BermudanOption, AmericanOption, Product

TIME_ROUNDING = 10

"""
More ideas
- rannacher smoothing: use a couple of implicit steps at discontinuities/payoff for smoother solution
- richardson extrapolation: solve twice, once with coarser grid, subtract weighted value to improve error
- implement toivanen ikonen splitting and PSOR for bermudan and americans
- automatically deduce time or spot grid
- after solving to inception, interpolate solution in spot grid to get value at spot
- non-uniform grids? 
"""


__all__ = ['ExplicitEulerBS', 'ImplicitEulerBS', 'CrankNicolsonBS']


class EarlyExerciseHandler(Enum):
    """
    MAX is taking max of option value and payoff after the numerical step - fastest method.
    IT is Ikonen Toivanen operator splitting. Fast, more accurate.
    PSOR is projected successive over relaxation. Slow (iterative) method, most accurate.
    """
    MAX = 0
    IT = 1
    PSOR = 2


class BlackScholesModel:
    def __init__(self, vol, rate_d, rate_f):
        self.vol = vol
        self.rate_d = rate_d
        self.rate = rate_d
        self.rate_f = rate_f
        self.div_yield = rate_f


class FDMScheme:
    def __init__(self, time_axis, spot_axis, model: BlackScholesModel, early_exercise_handler: EarlyExerciseHandler):
        """
        assuming uniform axes
        """
        # TODO: Idea. Get Time Axis only for relevant dates from product - pay dates, observations etc
        #  enrich with simulation dates for accuracy, also with dividend dates
        # TODO: same with spot axis - enrich with barriers, strike.
        self.time_axis = time_axis
        self.spot_axis = spot_axis
        self.vol = model.vol
        self.rate_d = model.rate_d
        self.rate_f = model.rate_f  # div yield or foreign rate

        self.early_exercise_handler = early_exercise_handler

        self.scheme = 'Base'

    def step_backwards(self, price_t, price_t_plus_dt, time_idx, product, *args, **kwargs):
        raise NotImplementedError("Function step_backwards is implemented on derived schemes.")


class ExplicitEulerBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, model, early_exercise_handler=EarlyExerciseHandler.MAX):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, model, early_exercise_handler)
        self.scheme = 'Explicit'
        self.matrix_dct = {}
        for time_idx in range(time_axis.shape[0] - 1):
            dt = round(time_axis[time_idx + 1] - time_axis[time_idx], TIME_ROUNDING)
            if dt not in self.matrix_dct:
                self.matrix_dct[dt] = self.compute_matrix(dt)

    def compute_matrix(self, dt):
        n = len(self.spot_axis)
        n_range = np.arange(1, n - 1)

        s_minus_term = 0.5 * (self.vol ** 2 * n_range ** 2 - (self.rate_d - self.rate_f) * n_range) * dt
        s_neutral_term = 1 - (self.rate_d + self.vol ** 2 * n_range ** 2) * dt
        s_plus_term = 0.5 * (self.vol ** 2 * n_range ** 2 + (self.rate_d - self.rate_f) * n_range) * dt
        matrix = np.zeros((n - 2, n))  # n-2 rows, n columns: columns are known, rows unknowns

        matrix[:, 1:-1] += np.diag(s_neutral_term)
        matrix[:, :-2] += np.diag(s_minus_term)
        matrix[:, 2:] += np.diag(s_plus_term)
        return matrix

    def step_backwards(self, price_t, price_t_plus_dt, time_idx, product, *args, **kwargs):
        dt = round(self.time_axis[time_idx + 1] - self.time_axis[time_idx], TIME_ROUNDING)
        price_t[1:-1] = self.matrix_dct[dt] @ price_t_plus_dt
        time_step = self.time_axis[time_idx]
        if product.is_exercise_time(time_step):
            price_t[:] = np.maximum(price_t, product.exercise_payoff(self.spot_axis, time_step))
        return price_t


class ImplicitEulerBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, model, early_exercise_handler=EarlyExerciseHandler.MAX):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, model, early_exercise_handler)
        self.scheme = 'Implicit'

        self.lu_factors = {}
        self.lowers = {}
        self.uppers = {}
        for time_idx in range(time_axis.shape[0] - 1):
            dt = round(time_axis[time_idx + 1] - time_axis[time_idx], TIME_ROUNDING)
            if dt not in self.lu_factors:
                matrix, lower, upper = self.compute_matrix(dt)
                lu_factor = sc.linalg.lu_factor(matrix)
                self.lu_factors[dt] = lu_factor
                self.lowers[dt] = lower
                self.uppers[dt] = upper

        # self.matrix, self.lower, self.upper = self.compute_matrix()
        # self.lu_factor = sc.linalg.lu_factor(self.matrix)

    def compute_matrix(self, dt):
        n = len(self.spot_axis)
        n_range = np.arange(1, n - 1)

        sub_diagonal = 0.5 * (-self.vol ** 2 * n_range ** 2 + (self.rate_d - self.rate_f) * n_range) * dt
        diagonal_term = 1 + (self.rate_d + self.vol ** 2 * n_range ** 2) * dt
        sup_diagonal = 0.5 * (-self.vol ** 2 * n_range ** 2 - (self.rate_d - self.rate_f) * n_range) * dt
        matrix = np.zeros((n - 2, n - 2))
        matrix += np.diag(sub_diagonal[1:], -1)
        matrix += np.diag(diagonal_term)
        matrix += np.diag(sup_diagonal[:-1], 1)

        return matrix, sub_diagonal[0], sup_diagonal[-1]

    def step_backwards(self, price_t, price_t_plus_dt, time_idx, product, *args, **kwargs):
        # upper and lower boundary are known
        dt = round(self.time_axis[time_idx + 1] - self.time_axis[time_idx], TIME_ROUNDING)
        residual_vector = np.zeros(price_t.shape[0]-2)
        residual_vector[0] = price_t[0] * self.lowers[dt]
        residual_vector[-1] = price_t[-1] * self.uppers[dt]
        price_t[1:-1] = sc.linalg.lu_solve(self.lu_factors[dt], price_t_plus_dt[1:-1] - residual_vector)
        # TODO: if AMERICAN exercise type: do ikonen toivanen splitting
        # If Bermudan and time step is exercise: PSOR
        # else: LU decomp
        time_step = self.time_axis[time_idx]
        if product.is_exercise_time(time_step):
            price_t[:] = np.maximum(price_t, product.exercise_payoff(self.spot_axis, time_step))
        return price_t


class CrankNicolsonBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, model, early_exercise_handler=EarlyExerciseHandler.MAX):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, model, early_exercise_handler)
        self.model = 'CrankNicolson'
        self.theta = 0.5  # relative weight of explicit scheme

        # self.expl_matrix, self.impl_matrix, self.lower, self.upper = self.compute_matrix()
        # self.lu_factor = sc.linalg.lu_factor(self.impl_matrix)
        self.expl_matrices = {}
        self.impl_matrices = {}
        self.lowers = {}
        self.uppers = {}
        self.lu_factors = {}
        for time_idx in range(time_axis.shape[0] - 1):
            dt = round(time_axis[time_idx + 1] - time_axis[time_idx], TIME_ROUNDING)
            if dt not in self.lu_factors:
                expl_matrix, impl_matrix, lower, upper = self.compute_matrix(dt)
                lu_factor = sc.linalg.lu_factor(impl_matrix)
                self.lu_factors[dt] = lu_factor
                self.lowers[dt] = lower
                self.uppers[dt] = upper
                self.expl_matrices[dt] = expl_matrix
                self.impl_matrices[dt] = impl_matrix

    def compute_matrix(self, dt):
        n = len(self.spot_axis)
        n_range = np.arange(1, n - 1)

        a_n = 0.5 * (self.vol ** 2 * n_range ** 2 - (self.rate_d - self.rate_f) * n_range) * dt
        b_n = (self.rate_d + self.vol ** 2 * n_range ** 2) * dt
        c_n = 0.5 * (self.vol ** 2 * n_range ** 2 + (self.rate_d - self.rate_f) * n_range) * dt
        # explicit part
        expl_matrix = np.zeros((n - 2, n))  # n-2 rows, n columns: columns are known, rows unknowns
        expl_matrix[:, :-2] = + np.diag(self.theta * a_n)
        expl_matrix[:, 1:-1] += np.diag(1 - b_n * self.theta)
        expl_matrix[:, :-2] = + np.diag(self.theta * c_n)

        # implicit matrix
        impl_matrix = np.zeros((n - 2, n - 2))
        impl_matrix += np.diag((1 - self.theta) * -a_n[1:], -1)
        impl_matrix += np.diag(1 + b_n * (1 - self.theta))
        impl_matrix += np.diag((1 - self.theta) * -c_n[:-1], 1)

        lower_residual = (1 - self.theta) * -a_n[0]
        upper_residual = (1 - self.theta) * -c_n[-1]

        return expl_matrix, impl_matrix, lower_residual, upper_residual

    def step_backwards(self, price_t, price_t_plus_dt, time_idx, product, omega=0.1, sor=False):
        dt = round(self.time_axis[time_idx + 1] - self.time_axis[time_idx], TIME_ROUNDING)
        residual_vector = np.zeros(price_t.shape[0] - 2)
        # TODO: implement clamping of residual vector here?
        residual_vector[0] = price_t[0] * self.lowers[dt]
        residual_vector[-1] = price_t[-1] * self.uppers[dt]
        rhs = self.expl_matrices[dt] @ price_t_plus_dt - residual_vector
        # TODO: if AMERICAN exercise type: do ikonen toivanen splitting
        # If Bermudan and time step is exercise: PSOR
        # else: LU decomp
        time_step = self.time_axis[time_idx]
        if product.is_exercise_time(time_step):
            payoff = product.exercise_payoff(self.spot_axis, time_step)
            if self.early_exercise_handler == EarlyExerciseHandler.MAX:
                price_t[1:-1] = sc.linalg.lu_solve(self.lu_factors[dt], rhs)
                price_t[:] = np.maximum(price_t, payoff)
            elif self.early_exercise_handler == EarlyExerciseHandler.IT:
                raise NotImplementedError(f"EarlyExerciseHandler '{self.early_exercise_handler}' not implemented.")
            elif self.early_exercise_handler == EarlyExerciseHandler.PSOR:
                price_t[1:-1] = psor(self.impl_matrices[dt], rhs, price_t_plus_dt[1:-1], payoff, omega=1.)
            else:
                raise NotImplementedError(f"EarlyExerciseHandler '{self.early_exercise_handler}' not implemented.")
        else:
            price_t[1:-1] = sc.linalg.lu_solve(self.lu_factors[dt], rhs)
        return price_t


def successive_over_relaxation(left_matrix, right_vector, init_guess, omega=1., error_tolerance=1.e-6):
    # solve: Ax = b iteratively
    it = 0
    old_sol = init_guess
    residual = np.linalg.norm(left_matrix @ old_sol - right_vector)
    print(f"Residual at step {it} is {residual}")
    while residual < error_tolerance:
        new_sol = np.empty_like(old_sol)
        for i in range(new_sol.shape[0]):
            new_sol[i] = old_sol[i] * (1 - omega) + omega / left_matrix[i, i] * (right_vector[i] - np.dot(left_matrix[i, :i], new_sol[:i]) - np.dot(left_matrix[i, i+1:], old_sol[i+1:]))
        old_sol = new_sol
        # assess error
        residual = np.linalg.norm(left_matrix @ old_sol - right_vector)
        it += 1
        print(f"Residual at step {it} is {residual}")
    print(f"finished after {it} iterations.")
    return old_sol


def psor(left_matrix, right_vector, init_guess, min_value, omega=1.0, error_tolerance=1.e-6):
    # solve: Ax = b iteratively. After each iterator floor solution at min_value
    # TODO: error tolerance does not work here
    it = 0
    old_sol = init_guess
    residual = np.linalg.norm(left_matrix @ old_sol - right_vector)
    print(f"Residual at step {it} is {residual}")
    while residual < error_tolerance:
        new_sol = np.empty_like(old_sol)
        for i in range(new_sol.shape[0]):
            new_sol[i] = old_sol[i] * (1 - omega) + omega / left_matrix[i, i] * (right_vector[i] - np.dot(left_matrix[i, :i], new_sol[:i]) - np.dot(left_matrix[i, i+1:], old_sol[i+1:]))
            # projection step
            new_sol[i] = max(new_sol[i], min_value[i])
        old_sol = new_sol
        # assess error
        residual = np.linalg.norm(left_matrix @ old_sol - right_vector)
        it += 1
        print(f"Residual at step {it} is {residual}")
    print(f"finished after {it} iterations.")
    return old_sol


def toivanen_ikonen():
    # idea: lambda factor
    # do lu factorization
    # clamp with payoff, adjust lambda factor
    pass


def value(prod: Product, model: BlackScholesModel, n_time, n_spot, scheme='explicit'):
    time_axis = prod.get_time_axis(n_time)
    n_time = time_axis.shape[0]
    spot_axis = prod.get_spot_axis(spot, n_spot)
    assert (spot_axis[0] < spot < spot_axis[-1])
    # computation grid
    grid = np.empty((n_time, n_spot))
    # fill final and boundary conditions
    grid[-1, :] = prod.get_final_condition(spot_axis)
    grid[:, 0] = prod.get_lower_boundary_condition(time_axis, min(spot_axis), model)
    grid[:, -1] = prod.get_upper_boundary_condition(time_axis, max(spot_axis), model)

    if scheme == 'explicit':
        solver = ExplicitEulerBS(time_axis, spot_axis, model)
    elif scheme == 'implicit':
        solver = ImplicitEulerBS(time_axis, spot_axis, model)
    elif scheme == 'crank-nicolson':
        solver = CrankNicolsonBS(time_axis, spot_axis, model)
    else:
        raise NotImplementedError(f"FDM scheme {scheme} not implemented.")

    for j in range(0, n_time-1)[::-1]:
        solver.step_backwards(grid[j, :], grid[j+1, :], j, prod)

    # now, grid[0, :] has solution on spot grid
    prev_spot = spot_axis[0]
    for i, spot_i in enumerate(spot_axis):
        if spot_i == spot:
            return grid[0, i]
        if spot_i > spot:
            # linear interpolation of solution on spot grid
            return grid[0, i-1] + (spot - prev_spot) * (grid[0, i] - grid[0, i-1]) / (spot_i - prev_spot)
        prev_spot = spot_i
    raise ValueError("Failed to interpolate solution.")


if __name__ == "__main__":
    vol = 0.3
    rate = 0.05
    spot = 100.
    strikes = np.array([100., 120., 140.])
    exercise_dates = np.array([0.5, 1.0, 2.0])
    prod = BermudanOption(strikes, exercise_dates, True)
    european = EuropeanOption(140., 2.0, True)
    model = BlackScholesModel(vol, rate, 0.0)
    n_time = 4000
    n_spot = 1000
    european_val = value(european, model, n_time, n_spot, scheme='implicit')
    print(european_val)
    bermudan_value = value(prod, model, n_time, n_spot, scheme='implicit')
    print(bermudan_value)
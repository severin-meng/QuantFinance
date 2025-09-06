import numpy as np
import scipy as sc

from products import ExerciseType


__all__ = ['ExplicitEulerBS', 'ImplicitEulerBS', 'CrankNicolsonBS']


class BlackScholesModel:
    def __init__(self, vol, rate_d, rate_f):
        self.vol = vol
        self.rate_d = rate_d
        self.rate = rate_d
        self.rate_f = rate_f
        self.div_yield = rate_f


class FDMScheme:
    def __init__(self, time_axis, spot_axis, model: BlackScholesModel):
        """
        assuming uniform axes
        """
        self.time_axis = time_axis
        self.spot_axis = spot_axis
        self.vol = model.vol
        self.rate_d = model.rate_d
        self.rate_f = model.rate_f  # div yield or foreign rate

        self.scheme = 'Base'

    def step_backwards(self, price_t, price_t_plus_dt, time_idx, product, *args, **kwargs):
        raise NotImplementedError("Function step_backwards is implemented on derived schemes.")


class ExplicitEulerBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, model):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, model)
        self.scheme = 'Explicit'
        self.matrix = self.compute_matrix()

    def compute_matrix(self):
        dt = self.time_axis[1] - self.time_axis[0]
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
        price_t[1:-1] = self.matrix @ price_t_plus_dt
        time_step = self.time_axis[time_idx]
        if product.is_exercise_time(time_step):
            price_t = np.maximum(price_t, product.exercise_payoff(self.spot_axis))
        return price_t


class ImplicitEulerBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, model):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, model)
        self.scheme = 'Implicit'
        self.matrix, self.lower, self.upper = self.compute_matrix()
        self.lu_factor = sc.linalg.lu_factor(self.matrix)

    def compute_matrix(self):
        dt = self.time_axis[1] - self.time_axis[0]
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
        residual_vector = np.zeros(price_t.shape[1]-2)
        residual_vector[0] = price_t[0] * self.lower
        residual_vector[-1] = price_t[-1] * self.upper
        price_t[1:-1] = sc.linalg.lu_solve(self.lu_factor, price_t_plus_dt[1:-1] - residual_vector)
        # TODO: if AMERICAN exercise type: do ikonen toivanen splitting
        # If Bermudan and time step is exercise: PSOR
        # else: LU decomp
        time_step = self.time_axis[time_idx]
        if product.is_exercise_time(time_step):
            price_t = np.maximum(price_t, product.exercise_payoff(self.spot_axis))
        return price_t


class CrankNicolsonBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, model):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, model)
        self.model = 'CrankNicolson'
        self.theta = 0.5  # relative weight of explicit scheme
        self.expl_matrix, self.impl_matrix, self.lower, self.upper = self.compute_matrix()
        self.lu_factor = sc.linalg.lu_factor(self.impl_matrix)

    def compute_matrix(self):
        dt = self.time_axis[1] - self.time_axis[0]
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
        residual_vector = np.zeros(price_t.shape[0] - 2)
        # TODO: implement clamping of residual vector here?
        residual_vector[0] = price_t[0] * self.lower
        residual_vector[-1] = price_t[-1] * self.upper
        rhs = self.expl_matrix @ price_t_plus_dt - residual_vector
        # TODO: if AMERICAN exercise type: do ikonen toivanen splitting
        # If Bermudan and time step is exercise: PSOR
        # else: LU decomp
        time_step = self.time_axis[time_idx]
        if product.exercise_type == ExerciseType.AMERICAN:
            # TODO: Toivanen Ikonen Splitting
            raise NotImplementedError
        elif product.is_exercise_time(time_step):
            payoff = product.exercise_payoff(self.spot_axis, time_step)
            price_t[1:-1] = psor(self.impl_matrix, rhs, price_t_plus_dt[1:-1], payoff, omega=1.0, error_tolerance=1.e-6)
            return price_t
        else:
            price_t[1:-1] = sc.linalg.lu_solve(self.lu_factor, rhs)
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

"""
Toy implementation of least squares monte carlo to price a bermudan option
"""

import numpy as np
from numpy.linalg import svd
from numpy.random import default_rng


class PathSpec:
    spot_times = None
    discount_maturities = None
    use_numeraire = None


class PathStruct:
    path = None
    discounts = None
    numeraires = None


class GBM:
    def __init__(self, spot, vol, rate):
        self.spot = spot
        self.vol = vol
        self.rate = rate

    def init_model(self, path_spec: PathSpec):
        time_steps = np.concatenate((np.zeros(1), path_spec.spot_times))
        n = len(time_steps)

        self.path = np.empty_like(time_steps)
        self.path[0] = spot
        self.drifts = np.empty(n-1)
        self.diffusions = np.empty(n-1)
        for i in range(n - 1):
            dt = time_steps[i + 1] - time_steps[i]
            vol_sqrdt = np.sqrt(dt) * vol
            self.drifts[i] = rate * dt - vol_sqrdt * vol_sqrdt * 0.5
            self.diffusions[i] = vol_sqrdt

        self.discounts = np.empty(n-1)   # contains discount factor from t-1 to t.
        self.numeraires = np.empty(n - 1)  # contains numeraire from 0 to t.
        for i in range(n - 1):
            self.discounts[i] = np.exp(-self.rate * (path_spec.discount_maturities[i] - path_spec.spot_times[i]))
            self.numeraires[i] = np.exp(self.rate * path_spec.spot_times[i])

    def build_path(self, rng):
        scen = PathStruct()
        scen.discounts = self.discounts
        scen.numeraires = self.numeraires
        normals = rng.standard_normal(size=len(self.diffusions))
        for i in range(len(self.diffusions)):
            self.path[i + 1] = self.path[i] * np.exp(self.drifts[i] + normals[i] * self.diffusions[i])
        scen.path = self.path[1:]
        return scen


class European:
    def __init__(self, strike, maturity, is_call):
        self.strike = strike
        self.maturity = maturity
        self.is_call = is_call

    def get_path_spec(self):
        path_spec = PathSpec()
        path_spec.spot_times = np.array([self.maturity])
        path_spec.use_numeraire = [True]
        path_spec.discount_maturities = np.array([self.maturity])
        return path_spec

    def value(self, scen):
        if self.is_call:
            return max(scen.path[0] - self.strike, 0) / scen.numeraires[0]
        return max(self.strike - scen.path[0], 0) / scen.numeraires[0]


class Bermudan:
    def __init__(self, strikes, exercise_dates, is_call):
        self.expriy = exercise_dates[-1]
        self.early_exercise_dates = exercise_dates[:-1]
        self.exercise_dates = exercise_dates
        self.is_call = is_call
        self.strikes = strikes

    def get_path_spec(self):
        path_spec = PathSpec()
        path_spec.spot_times = self.exercise_dates
        path_spec.use_numeraire = [True for _ in range(len(self.exercise_dates))]
        path_spec.discount_maturities = np.empty(len(self.exercise_dates))
        for i in range(len(self.exercise_dates) - 1):
            path_spec.discount_maturities[i] = self.exercise_dates[i + 1]
        path_spec.discount_maturities[-1] = self.exercise_dates[-1]
        return path_spec

    def early_exercise_payoffs(self, scen):
        """ if called, what is the payoff """
        if self.is_call:
            return np.maximum(scen.path[:-1] - self.strikes[:-1], 0)
        return np.maximum(self.strikes[:-1] - scen.path[:-1], 0)

    def spot_on_early_exercise(self, scen):
        return scen.path[:-1]

    def payoff(self, scen):
        """ what is the cashflow if not called """
        if self.is_call:
            return max(scen.path[-1] - self.strikes[-1], 0)
        return max(self.strikes[-1] - scen.path[-1], 0)


class American:
    def __init__(self, strike, tte, is_call):
        self.expriy = exercise_dates[-1]
        self.early_exercise_dates = exercise_dates[:-1]
        self.exercise_dates = exercise_dates
        self.is_call = is_call
        self.strikes = strikes

    def get_path_spec(self):
        path_spec = PathSpec()
        path_spec.spot_times = self.exercise_dates
        path_spec.use_numeraire = [True for _ in range(len(self.exercise_dates))]
        path_spec.discount_maturities = np.empty(len(self.exercise_dates))
        for i in range(len(self.exercise_dates) - 1):
            path_spec.discount_maturities[i] = self.exercise_dates[i + 1]
        path_spec.discount_maturities[-1] = self.exercise_dates[-1]
        return path_spec

    def early_exercise_payoffs(self, scen):
        """ if called, what is the payoff """
        if self.is_call:
            return np.maximum(scen.path[:-1] - self.strikes[:-1], 0)
        return np.maximum(self.strikes[:-1] - scen.path[:-1], 0)

    def spot_on_early_exercise(self, scen):
        return scen.path[:-1]

    def payoff(self, scen):
        """ what is the cashflow if not called """
        if self.is_call:
            return max(scen.path[-1] - self.strikes[-1], 0)
        return max(self.strikes[-1] - scen.path[-1], 0)


def fit_regression(X, y, reg_param=0.0):
    # idea: solve X * beta = y given X and y
    # then use beta on new X to produce new y.
    # SVD: X = U @ S @ Vh
    # where S is diagonal matrix of singular values
    # U is not square but has orthonormal columns: u.T @ u = id_3 (but not: u @ u.T != id_n)
    # Vh is orthogonal, square: V @ Vh = Vh @ V = id_3
    # linear regression: X is sample matrix: rows are samples, columns are features
    # linear regression: (X.T X)^-1 X.T y = beta
    # substitute SVD into this equation:
    # X.T = V @ S @ U.T -> X.T X = V @ S @ U.T @ U @ S @ Vh = V @ S^2 @ Vh
    # (X.T X)^-1 = (V @ S^2 @ Vh) ^-1 = V @ S^-2 Vh (because: Vh @ V = V @ Vh = id_3)
    # and finally: (X.T X)^-1 X.T = V @ S^-2 @ Vh @ V @ S @ U.T = V @ S^-1 U.T
    # TODO: estimate regularization size
    svd_res = svd(X, full_matrices=False)
    uh = svd_res.U.T
    s = svd_res.S
    v = svd_res.Vh.T
    sigma = s / (s ** 2 + reg_param ** 2)
    betas = v * sigma @ uh @ y
    return betas


def build_regression_input_1d(model_state):
    return np.array([1.0, model_state, model_state**2, model_state**3])


def build_regression_input(model_states):
    biases = np.ones_like(model_states)
    squares = model_states **2
    cubes = model_states **3
    return np.stack((biases, model_states, squares, cubes), axis=1)


def value_lsm(product: Bermudan, model: GBM, nbr_paths_presimul, nbr_paths_main, rng):
    model.init_model(product.get_path_spec())

    nbr_regressions = len(product.early_exercise_dates)  # maturity is not a regression date
    payoffs = np.empty(nbr_paths_presimul)  # only one cashflow
    early_exercise_payoffs = np.empty((nbr_regressions, nbr_paths_presimul))
    model_states = np.empty((nbr_regressions, nbr_paths_presimul))  # model dimension implicitly 1
    discounted_payoffs = np.zeros((nbr_regressions, nbr_paths_presimul))
    discount_factors = np.empty((nbr_regressions, nbr_paths_presimul))

    # run presimulations
    for i in range(nbr_paths_presimul):
        scen = model.build_path(rng)
        payoffs[i] = product.payoff(scen)
        # discounted_payoffs[-1, i] = scen.discounts[-1] * payoffs[i]
        early_exercise_payoffs[:, i] = product.early_exercise_payoffs(scen)
        model_states[:, i] = product.spot_on_early_exercise(scen)
        discount_factors[:, i] = scen.discounts[:-1]

    # run regressions; start with final regression because it is special
    regression_betas = np.empty((nbr_regressions, 4))
    # TODO: number of regression variables is determined by
    # number of model states (e.g. 3 stocks, 3 stochastic vols)
    # and some product specifics - e.g. for a swaption, use forward swap rate? Or something like that
    # and then build products
    # e.g. for each state, 3 powers
    # then for mixing states - 1 extra per pair
    # and for mixing states with product stuff?
    # and finally there is a bias

    # now move this product value to the next regression date
    # for this, need to discount it back. Problematic if discounting is stochastic - must be pathwise?
    product_value = payoffs
    for j in range(nbr_regressions)[::-1]:
        # TODO technically, also add in discounted cashflows between previous exercise date and this one
        regression_target = discount_factors[j, :] * product_value  # + discounted_payoffs[j, :]
        regression_input = build_regression_input(model_states[j, :])
        regression_betas[j, :] = fit_regression(regression_input, regression_target)
        regression_value = regression_input @ regression_betas[j, :]
        # proxy on indicator to determine pathwise product value on-path at last early exercise date
        # now move this product value to the next regression date
        product_value = np.where(regression_value > early_exercise_payoffs[j, :],
                                 regression_target, early_exercise_payoffs[j, :])
        print(f"End of Regression step {j}")

    # now we have all the regression betas
    # run main simulations
    value = 0.0
    for k in range(nbr_paths_main):
        scen = model.build_path(rng)
        payoff = product.payoff(scen)
        # discounted_payoffs = scen.discounts[-1] * payoff
        early_exercise_payoff = product.early_exercise_payoffs(scen)
        model_states = product.spot_on_early_exercise(scen)
        discount_factors = scen.discounts[:-1]

        product_value = payoff
        for j in range(nbr_regressions)[::-1]:
            regression_input = build_regression_input_1d(model_states[j])
            proxy_value = regression_input @ regression_betas[j, :]
            regression_target = discount_factors[j] * product_value
            product_value = np.where(proxy_value > early_exercise_payoff[j],
                                     regression_target, early_exercise_payoff[j])

        value += product_value / scen.numeraires[0]

    return value / nbr_paths_main


def value_basic(product: European, model: GBM, nbr_paths_main, rng):
    model.init_model(product.get_path_spec())

    value = 0.0
    for k in range(nbr_paths_main):
        scen = model.build_path(rng)
        pathwise_val = product.value(scen)
        value += pathwise_val
    return value / nbr_paths_main


if __name__ == '__main__':
    rng = default_rng(0)
    vol = 0.3
    rate = 0.05
    spot = 100.
    strikes = np.array([100., 120., 140.])
    exercise_dates = np.array([0.5, 1.0, 2.0])
    prod = Bermudan(strikes, exercise_dates, True)
    european = European(140., 2.0, True)
    model = GBM(spot, vol, rate)
    nbr_paths_presimul = 200_000
    nbr_paths_main = 1_000_000
    european_val = value_basic(european, model, nbr_paths_main, rng)
    print(european_val)
    bermudan_value = value_lsm(prod, model, nbr_paths_presimul, nbr_paths_main, rng)
    print(bermudan_value)

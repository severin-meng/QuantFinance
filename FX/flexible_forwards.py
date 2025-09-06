import numpy as np

np.set_printoptions(linewidth=500, precision=6, suppress=True)
import matplotlib.pyplot as plt
from scipy.special import ndtr

max_scale = 3.0


class FDMScheme:
    def __init__(self, time_axis, spot_axis, vol, rate_d, rate_f):
        """
        assuming uniform axes
        """
        self.time_axis = time_axis
        self.spot_axis = spot_axis
        self.vol = vol
        self.rate_d = rate_d
        self.rate_f = rate_f  # div yield or foreign rate

        self.model = 'Base'

    def step_backwards(self, *args):
        raise NotImplementedError("Function step_backwards is implemented on derived schemes.")


class ExplicitEulerBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, vol, rate_d, rate_f):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, vol, rate_d, rate_f)
        self.model = 'Explicit'
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

    def step_backwards(self, prices, time_idx):
        prices[time_idx, 1:-1] = self.matrix @ prices[time_idx + 1]


class ImplicitEulerBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, vol, rate_d, rate_f):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, vol, rate_d, rate_f)
        self.model = 'Implicit'
        self.matrix, self.lower, self.upper = self.compute_matrix()

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

    def step_backwards(self, prices, time_idx):
        # upper and lower boundary are known
        residual_vector = np.zeros(prices.shape[1]-2)
        residual_vector[0] = prices[time_idx, 0] * self.lower
        residual_vector[-1] = prices[time_idx, -1] * self.upper
        prices[time_idx, 1:-1] = np.linalg.solve(self.matrix, prices[time_idx + 1, 1:-1] - residual_vector)


class CrankNicolsonBS(FDMScheme):
    def __init__(self, time_axis, spot_axis, vol, rate_d, rate_f):
        """
        assuming uniform axes
        """
        super().__init__(time_axis, spot_axis, vol, rate_d, rate_f)
        self.model = 'CrankNicolson'
        self.theta = 0.5  # relative weight of explicit scheme
        self.expl_matrix, self.impl_matrix, self.lower, self.upper = self.compute_matrix()

    def compute_matrix(self):
        dt = self.time_axis[1] - self.time_axis[0]
        n = len(self.spot_axis)
        n_range = np.arange(1, n - 1)

        a_n = 0.5 * (self.vol ** 2 * n_range ** 2 - (self.rate_d - self.rate_f) * n_range) * dt
        b_n =  (self.rate_d + self.vol ** 2 * n_range ** 2) * dt
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

    def step_backwards(self, prices, time_idx, omega=0.1, sor=True):
        residual_vector = np.zeros(prices.shape[1] - 2)
        residual_vector[0] = prices[time_idx, 0] * self.lower
        residual_vector[-1] = prices[time_idx, -1] * self.upper
        rhs = self.expl_matrix @ prices[time_idx + 1] - residual_vector
        if sor:
            prices[time_idx, 1:-1] = successive_over_relaxation(self.impl_matrix, rhs, prices[time_idx + 1], omega)
        else:
            prices[time_idx, 1:-1] = np.linalg.solve(self.impl_matrix, rhs)


def successive_over_relaxation(left_matrix, right_vector, init_guess, omega, error_tolerance=1.e-6):
    # solve: Ax = b iteratively
    iter = 0
    old_sol = init_guess
    residual = np.linalg.norm(left_matrix @ old_sol - right_vector)
    print(f"Residual at step {iter} is {residual}")
    while residual < error_tolerance:
        new_sol = np.empty_like(old_sol)
        for i in range(new_sol.shape[0]):
            new_sol[i] = old_sol[i] * (1 - omega) + omega / left_matrix[i, i] * (right_vector[i] - np.dot(left_matrix[i, :i], new_sol[:i]) - np.dot(left_matrix[i, i+1:], old_sol[i+1:]))
        old_sol = new_sol
        # assess error
        residual = np.linalg.norm(left_matrix @ old_sol - right_vector)
        iter += 1
        print(f"Residual at step {iter} is {residual}")
    print(f"finished after {iter} iterations.")
    return old_sol


class EuropeanOption:
    def __init__(self, strike, tte, is_call):
        self.strike = strike
        self.tte = tte
        self.is_call = is_call
        self.product = 'European ' + 'Call' if is_call else 'Put'

    def get_time_axis(self, n_time, t_start = 0):
        return np.linspace(t_start, self.tte, n_time)

    def get_spot_axis(self, n_spot, spot, spot_min = 0):
        return np.linspace(spot_min, max_scale * max(spot, self.strike), n_spot)

    def get_final_condition(self, spot_axis):
        if self.is_call:
            return np.maximum(spot_axis - self.strike, 0)
        return np.maximum(self.strike - spot_axis, 0)

    def get_upper_boundary(self, time_axis, max_spot, rate, div_yield):
        if self.is_call:
            return max_spot * np.exp(-div_yield * (self.tte - time_axis)) - self.strike * np.exp(-rate * (self.tte - time_axis))
        return np.zeros_like(time_axis)

    def get_lower_boundary(self, time_axis, min_spot, rate, div_yield):
        if self.is_call:
            return np.zeros_like(time_axis)
        return self.strike * np.exp(-rate * (self.tte - time_axis)) - min_spot * np.exp(-div_yield * (self.tte - time_axis))

    def adjust_result(self, prices, time_idx, spot_axis):
        pass

    def analytical_price(self, spot,  vol, rate, div_yield, *args, **kwargs):
        if spot == 0:
            return 0 if self.is_call else np.exp(-rate * self.tte) * self.strike
        sqrt_tte = np.sqrt(self.tte)
        drift = (rate - div_yield + vol ** 2 / 2) * self.tte
        d1 = (np.log(spot / self.strike) + drift) / (vol * sqrt_tte)
        d2 = d1 - (vol * sqrt_tte)
        if self.is_call:
            return spot * np.exp(-div_yield * self.tte) * ndtr(d1) - self.strike * np.exp(-rate * self.tte) * ndtr(d2)
        return -spot * np.exp(-div_yield * self.tte) * ndtr(-d1) + self.strike * np.exp(-rate * self.tte) * ndtr(-d2)


class FlexibleForward:
    def __init__(self, strike, tte):
        self.strike = strike
        self.tte = tte
        self.product = 'Flexible Forward'

    def get_time_axis(self, n_time, t_start = 0):
        return np.linspace(t_start, self.tte, n_time)

    def get_spot_axis(self, n_spot, spot, spot_min = 0.0):
        return np.linspace(spot_min, 1.5 * max(spot, self.strike), n_spot)

    def get_final_condition(self, spot_axis):
        return spot_axis - self.strike

    def get_upper_boundary(self, time_axis, max_spot, rate_d, rate_f):
        return (max_spot - self.strike) * np.ones_like(time_axis)

    def get_lower_boundary(self, time_axis, min_spot, rate_d, rate_f):
        return min_spot * np.exp(-rate_f * (self.tte - time_axis)) - self.strike * np.exp(-rate_d * (self.tte - time_axis))

    def adjust_result(self, prices, time_idx, spot_axis):
        prices[time_idx] = np.maximum(prices[time_idx], spot_axis - self.strike)

    def analytical_price(self, spot,  vol, rate_d, rate_f, method='Simple', theta=0.25, lambd=0.0):
        assert method in ('Simple', 'GJ', 'Severin', 'Richardson GJ')
        imm_payoff = spot - self.strike
        fwd_payoff = spot * np.exp(-rate_f * self.tte) - self.strike * np.exp(-rate_d * self.tte)
        if method == 'Simple':
            return max(imm_payoff, fwd_payoff)
        elif method == 'GJ':
            # theta = 0.42584
            # lambd = 0.44867
            adjusted_tte = self.tte * (1 - theta)
            adjusted_strike = self.strike * (1 - np.exp(-rate_d * adjusted_tte)) / (1 - np.exp(-rate_f * adjusted_tte))
            call = EuropeanOption(adjusted_strike, theta * self.tte, is_call=True)
            call_prc = call.analytical_price(spot, vol, rate_d, rate_f)
            return max(imm_payoff, fwd_payoff + (1 - np.exp(-rate_f * adjusted_tte)) * (1 + lambd) * call_prc)
        elif method == 'Richardson GJ':
            adjusted_tte = self.tte * (1 - theta)
            adjusted_strike = self.strike * (1 - np.exp(-rate_d * adjusted_tte)) / (1 - np.exp(-rate_f * adjusted_tte))
            call = EuropeanOption(adjusted_strike, theta * self.tte, is_call=True)
            call_prc = call.analytical_price(spot, vol, rate_d, rate_f)
            three_step = fwd_payoff + (1 - np.exp(-rate_f * adjusted_tte)) * call_prc

            two_step = fwd_payoff
            one_step = fwd_payoff
            no_step = spot - self.strike
            return max(imm_payoff, three_step + lambd * (three_step - two_step))
            # return three_step + 3.5 * (three_step - two_step) - 0.5 * (two_step - one_step)
            # return three_step + lambd * (three_step - two_step) # + lambd/2 * (two_step - one_step)
        # S-K < Vfwd + opt -> Vfwd * opt * (1 + lambd)
        # S-K > Vfwd + opt -> then this is S-K * (1 + lambd) - lambd * Vfd
        # want: S-K > Vfwd + (1 + lambd)*opt -> S-K
        
        
        # want: If S-K > Vfwd + (1 + lambd) * opt -> S-K, i.e. cancel the Vfwd
        # max(S-K, Vfwd + (1+lambd)*opt) = max(S-K + lambd Vfwd, (1+lambd) (Vfwd + opt)) - lambd Vfwd
        # (1 + lambd) V3 = {
        elif method == 'Put':
            t_theta = self.tte * theta
            adjusted_tte = self.tte - t_theta
            fwd_payoff = spot * np.exp(-rate_f * t_theta) - self.strike * np.exp(-rate_d * t_theta)
            adjusted_strike = self.strike * (1 - np.exp(-rate_d * adjusted_tte)) / (1 - np.exp(-rate_f * adjusted_tte))
            put = EuropeanOption(adjusted_strike, t_theta, is_call=False)
            put_prc = put.analytical_price(spot, vol, rate_d, rate_f)
            return max(imm_payoff, fwd_payoff + (1 - np.exp(-rate_f * adjusted_tte)) * put_prc)

        else:
            raise NotImplementedError(f"Method {method} not implemented.")


def value_black_scholes(scheme='implicit'):
    spot = 1.
    strike = 1.
    rate_d = 0.04
    rate_f = 0.04  # dividend yield
    vol = 0.2
    tte = 1.
    n_spot = 601
    if scheme == 'explicit':
        n_time = 5*n_spot
    else:
        n_time = n_spot

    is_call = True
    plt_min = int(n_spot * 0.5)
    plt_max = int(n_spot * 0.85)

    # prod = EuropeanOption(strike, tte, is_call)
    prod = FlexibleForward(strike, tte)
    time_axis = prod.get_time_axis(n_time)
    spot_axis = prod.get_spot_axis(n_spot, spot)
    # computation grid
    grid = np.empty((n_time, n_spot))
    # fill final and boundary conditions
    grid[-1, :] = prod.get_final_condition(spot_axis)
    grid[:, 0] = prod.get_lower_boundary(time_axis, min(spot_axis), rate_d, rate_f)
    grid[:, -1] = prod.get_upper_boundary(time_axis, max(spot_axis), rate_d, rate_f)

    if scheme == 'explicit':
        model = ExplicitEulerBS(time_axis, spot_axis, vol, rate_d, rate_f)
    elif scheme == 'implicit':
        model = ImplicitEulerBS(time_axis, spot_axis, vol, rate_d, rate_f)
    else:
        raise NotImplementedError(f"FDM scheme {scheme} not implemented.")

    for j in range(0, n_time-1)[::-1]:
        model.step_backwards(grid, j)
        prod.adjust_result(grid, j, spot_axis)

    # plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(spot_axis[plt_min:plt_max], grid[0, plt_min:plt_max], label=f'FDM-{model.model}', linestyle='--', color='blue')
    analytical_prices = np.empty_like(spot_axis)
    for idx, s in enumerate(spot_axis):
        analytical_prices[idx] = prod.analytical_price(s, vol, rate_d, rate_f, method='Simple')
    error_simple = grid[0] - analytical_prices
    if analytical_prices is not None:
        ax1.plot(spot_axis[plt_min:plt_max], analytical_prices[plt_min:plt_max], label='Analytical Simple', color='green')

    # optimal params
    theta_opt = 0.42584
    lambd_opt = 0.44867
    analytical_prices = np.empty_like(spot_axis)
    for idy, s in enumerate(spot_axis):
        analytical_prices[idy] = prod.analytical_price(s, vol, rate_d, rate_f, method='GJ', theta=theta_opt, lambd=lambd_opt)
    error_opt = grid[0] - analytical_prices

    # optimal params
    theta_opt = 0.42584
    lambd_opt = 0.44867
    analytical_prices = np.empty_like(spot_axis)
    for idy, s in enumerate(spot_axis):
        analytical_prices[idy] = prod.analytical_price(s, vol, rate_d, rate_f, method='Richardson GJ', theta=theta_opt, lambd=lambd_opt)
    error_richardson = grid[0] - analytical_prices

    errors = []
    theta_range = np.linspace(0.1, 0.9, 20)
    for idx, theta in enumerate(theta_range):
        analytical_prices = np.empty_like(spot_axis)
        for idy, s in enumerate(spot_axis):
            analytical_prices[idy] = prod.analytical_price(s, vol, rate_d, rate_f, method='GJ', theta=theta)
        errors.append(grid[0] - analytical_prices)
        if analytical_prices is not None and idx == 3:
            ax1.plot(spot_axis[plt_min:plt_max], analytical_prices[plt_min:plt_max], label=fr'GJ $(\theta={theta:.2f})$', color='cyan')

    # ax1.plot(spot_axis, grid[-1], label='Payoff', color='black')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Spot')
    ax1.set_ylabel('Price')

    if error_simple is not None:
        ax2 = ax1.twinx()
        ax2.plot(spot_axis[plt_min:plt_max], error_simple[plt_min:plt_max], label='2-step', color="gray")
        ax2.plot(spot_axis[plt_min:plt_max], error_opt[plt_min:plt_max], label=rf'GJ$(\theta, \lambda)$', color="purple")
        ax2.plot(spot_axis[plt_min:plt_max], error_richardson[plt_min:plt_max], label=rf'GJ Richardson', color="brown")

        colors = plt.cm.tab20.colors
        for idx, theta in enumerate(theta_range):
            err_diff = error_simple - errors[idx]
            rel_err_diff = err_diff / errors[idx]
            rel_err_diff = errors[idx]
            ax2.plot(spot_axis[plt_min:plt_max], rel_err_diff[plt_min:plt_max], label=rf'GJ $(\theta={theta:.2f})$', color=colors[idx])
        # ax2.plot(spot_axis[plt_min:plt_max], errors_sev[plt_min:plt_max], label='Error Severin', color="gray")

        ax2.set_ylabel("Error")
        ax2.legend(loc='upper right')
    # print(grid[0])

    plt.grid(True)

    plt.title(f"{prod.product} under Black-Scholes")
    plt.show()

    """
    rates = np.linspace(rate_d - 0.01, rate_d + 0.01, 61)
    prices = []
    errors_proxy = []
    errors_gj = []
    for rate_f in rates:
        n_spot = 91
        mid = 60
        n_time = n_spot
        strike = spot # * np.exp(-rate_f * tte)
        prod = FlexibleForward(strike, tte)
        time_axis = prod.get_time_axis(n_time)
        spot_axis = prod.get_spot_axis(n_spot, spot)
        # computation grid
        grid = np.empty((n_time, n_spot))
        # fill final and boundary conditions
        grid[-1, :] = prod.get_final_condition(spot_axis)
        grid[:, 0] = prod.get_lower_boundary(time_axis, min(spot_axis), rate_d, rate_f)
        grid[:, -1] = prod.get_upper_boundary(time_axis, max(spot_axis), rate_d, rate_f)

        if scheme == 'explicit':
            model = ExplicitEulerBS(time_axis, spot_axis, vol, rate_d, rate_f)
        elif scheme == 'implicit':
            model = ImplicitEulerBS(time_axis, spot_axis, vol, rate_d, rate_f)
        else:
            raise NotImplementedError(f"FDM scheme {scheme} not implemented.")

        for j in range(0, n_time - 1)[::-1]:
            model.step_backwards(grid, j)
            prod.adjust_result(grid, j, spot_axis)

        prices.append(grid[0, mid])
        simple_price = prod.analytical_price(spot_axis[mid], vol, rate_d, rate_f, method='Simple')
        errors_proxy.append(grid[0, mid] - simple_price)
        gj_price = prod.analytical_price(spot_axis[mid], vol, rate_d, rate_f, method='GJ')
        errors_gj.append(grid[0, mid] - gj_price)

    plt.plot(rates - rate_d, errors_proxy, label='Errors (Simple)', color='purple')
    plt.plot(rates - rate_d, errors_gj, label='Errors (GJ)', color='orange')
    plt.xlabel(r'$r_f - r_d$')
    plt.ylabel("Error")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title(r"ATM Flexi Fwd Error vs $r_f-r_d$")
    plt.show()
    """


if __name__ == '__main__':
    value_black_scholes()

import numpy as np
from discount_factor import discount_factor as discount_func
from math import ceil
import time


def premium_leg(sorted_default_times, interest_time_series, halfway_default_interest, full_premium,
                nbr_payments, interest_matrix, halfway_accrual=True, n=5):
    """
    :param sorted_default_times: nbr_simulations X k simulated default times. Per simulation, the defaults
                                 are already ascendingly ordered, only the first k defaults are provided.
    :param interest_time_series: payment times of all interest payments
    :param halfway_default_interest: list containing the amount of halfway default per interest payment period and
                                     unit spread
    :param full_premium: pre-computed present value of full premium amount (unit spread), assuming no default occurs
    :param nbr_payments: number of interest payments assuming no default occurs at all
    :param interest_matrix: nbr_payments X nbr_payments interest matrix. element (i, j) contains cumulative sum of
                            interest payments i to j (per unit spread).
    :param halfway_accrual: set to True to use accrued interest payments upon default
    :param n: basket size, 5
    :return: list of premium leg present values per unit spread, per simulation
    """
    # for every default before and including k we need to find the interest accrual time + the halfway payment
    # for interest round down, for halway payment round up. This is already implemented into the interest variables.
    payment_time_indices = np.digitize(sorted_default_times, interest_time_series)-1
    premium_pvs = []
    for default_indices in payment_time_indices:
        if default_indices[0] == nbr_payments:
            premium_pvs.append(full_premium)
            continue
        value = 0
        prev_default = 0  # previous default time index
        weight = 1
        for payment_time_index in default_indices:
            value += weight * interest_matrix[prev_default, payment_time_index]

            # halfway_default_interest[index] corresponds to the accrued interest, it is rounded to half of the ongoing
            # cashflow
            value += 1/n * halfway_default_interest[payment_time_index] if halfway_accrual else 0
            weight -= 1./n
            prev_default = payment_time_index

            if prev_default == nbr_payments:
                break
        # if k defaults occur before maturity, then this term adds the remaining halfway accrued interest
        # the last value in halfway_default_interest is zero ensuring no correction if not all defaults occur before
        # maturity.
        value += weight * halfway_default_interest[prev_default] if halfway_accrual else 0
        premium_pvs.append(value)
    return premium_pvs


def default_leg(sorted_final_default_times, interest_time_series, discount_factors, recovery_rate=0.4, n=5):
    """
    :param sorted_final_default_times: nbr_simulations simulated final default times, i.e.
                                       he k-th default time per simulation
    :param interest_time_series: interest and subsequently als default payment times
    :param discount_factors: pre-computed discount factor per possible default payment time. Contains a 0 at the end to
                             account for not default happening until maturity.
    :param recovery_rate: recovery rate of basket CDS, default is 0.4
    :param n: basket size, 5
    :return: pv of default payment per simulation.
    """
    default_payment_indices = np.digitize(sorted_final_default_times, interest_time_series)
    return (1 - recovery_rate) * discount_factors[default_payment_indices] / n


def calculate_spread(k, simulated_default_times, maturity=5, accrued_time=0, interest_freq=0.25, recovery=0.4,
                     halfway_accrual=True, clean_price=False, discount_factor=discount_func, prnt=False,
                     iterative=False, theor_price=False, basket_spread=1):
    """
    entry point for basket spread compuation, requires sampled co-dependent default times
    :param k: k-th to default
    :param simulated_default_times: nbr_simulations X 5 series of simulated co-dependent default times
    :param maturity: maturity of basket cds, default is 5 (years)
    :param accrued_time: how much of the 5 years is already accrued
    :param interest_freq: interest payment frequency
    :param recovery: fixed recovery rate
    :param halfway_accrual: set to True to use accrued interest payment upon default, set to False to only account for
                            completed interest periods.
    :param clean_price: set to True to calculate the clean price, set to False to calculate the dirty price.
    :param discount_factor: callable function that returns to discount factor for given time t
    :param prnt: set to True to log runtime of compuation steps
    :param iterative: set to True to return not just the one spread but the spreads averaged after increasing number of
                      simulations. useful for convergence analysis.
    :param theor_price: set to True to return not the spread but the theoretical price, i.e. the sum of default and
                      premium leg present values
    :param basket_spread: if using theor_price=True, this basket_spread is used as a spread to compute the theor price
    :return: fair spread of basket cds, or theoretical price (assuming theor_price=True and provided basket_spread)
    """
    if accrued_time > 1/252:
        nbr_payments = ceil((maturity - accrued_time) / interest_freq)
    else:
        nbr_payments = round(maturity / interest_freq)

    first_payment1 = interest_freq - accrued_time % interest_freq

    # year fractions between payments
    delta_ts = np.array([0.] + [interest_freq for i in range(nbr_payments)])
    if first_payment1 > 1 / 252:
        delta_ts[1] = first_payment1
    interest_time_series = np.cumsum(delta_ts)
    discount_factors = np.array([discount_factor(t) for t in interest_time_series])

    if not clean_price:
        # if we want the dirty price, then one has to pay full interest for every period,
        # inlcuding accrued interest for the potentially incomplete first one. does not apply for the discount factor
        delta_ts[1] = interest_freq

    # array of all interest payments, discounted, per spread. First payment is zero (delta_ts starts at 0) -> remove
    interest_value = (delta_ts * discount_factors)[1:]

    interest_halfway = np.zeros(len(interest_value) + 1)
    # last halfway interest is zero: no halfway interest paid past maturity
    interest_halfway[:-1] = 0.5 * interest_value
    # in the jp morgan model, the last interest payment happens at same time as default payment, but is only half
    # of the usual interest payments, it models the default to be in the middle of two interest payments
    # all interest payments that accrue fully (i.e. no default in accrue time) are paid in full
    # i.e. interest_leg = np.cumsum(interest_value)[i] + interest_halfway[i] if default happens between t[i-1] and t[i]

    # full premium to be paid assuming no default occurs until maturity
    full_premium = np.sum(interest_value)

    # interest matrix is pre-computed. Until first default, fully weighted interest is paid. After the first default,
    # the future interest payments have reduced weight, until the next default. The matrix here sums interest payments
    # i to j, which will later be weighted according to the default times.
    interest_matrix = np.zeros((len(interest_time_series), len(interest_time_series)))
    for i in range(len(interest_time_series)):
        for j in range(i):
            interest_matrix[j, i] = np.sum(interest_value[j:i])

    # sort default times
    sorted_default_times = np.sort(simulated_default_times, axis=-1)[:, :k]
    sorted_final_default_times = sorted_default_times[:, -1]

    # calculate legs
    time_start = time.perf_counter()
    premium_legs = premium_leg(
        sorted_default_times, interest_time_series, interest_halfway, full_premium, nbr_payments,
        interest_matrix, halfway_accrual=halfway_accrual, n=5)
    time_p = time.perf_counter()
    if prnt:
        print(f"Premium leg calc time: {(time_p - time_start):.2f} ")

    # add zero, when default occurs after maturity.
    discount_factors = np.array([discount_factor(t) for t in interest_time_series] + [0])
    default_legs = default_leg(sorted_final_default_times, interest_time_series, discount_factors,
                               recovery_rate=recovery)

    time_d = time.perf_counter()
    if prnt:
        print(f"Default leg calc time: {(time_d - time_p):.2f} ")

    if theor_price:
        if iterative:
            max_pow = int(np.log2(len(sorted_default_times)))
            min_pow = 7
            steps = np.logspace(min_pow, max_pow, (max_pow - min_pow + 1) * 2, endpoint=True, base=2, dtype=int)
            theor_prices = []
            for step in steps:
                theor_prices.append(np.mean(default_legs[:step]) - basket_spread * np.mean(premium_legs[:step]))
            return steps, np.array(theor_prices)

        theor_price = np.mean(default_legs) - basket_spread * np.mean(premium_legs)
        return theor_price

    if iterative:
        max_pow = int(np.log2(len(sorted_default_times)))
        min_pow = 7
        steps = np.logspace(min_pow, max_pow, (max_pow-min_pow+1)*2, endpoint=True, base=2, dtype=int)
        spreads = []
        for step in steps:
            spreads.append(np.mean(default_legs[:step])/np.mean(premium_legs[:step]))
        return steps, np.array(spreads)
    else:
        premium = np.mean(premium_legs)
        default = np.mean(default_legs)
        spread_calc = default / premium
        return spread_calc


def run_test():
    """
    :return: None; test function to check functionality of the basket pricer, as well as performance
    """
    from copula_fitter import t_copula_fit_dof
    from correlations import get_correlations
    from hazard_rate_bootstrapping import fit_hazard_rates
    from copula_sampling import gaussian_copula_sampling, students_t_copula_sampling
    from uniform_to_default import default_time_func
    start = time.perf_counter()

    max_dof = t_copula_fit_dof(None, from_file=True)

    hazard_df = fit_hazard_rates(None, from_file=True)
    default_func = default_time_func(hazard_df, interp=True)
    correlation = get_correlations(None, corr_type='kendall', density_estimation=True, from_file=True)
    data_fit_time = time.perf_counter()
    print(f"Time to fit data: {(data_fit_time - start):.1f}")
    expo = 24
    # sampled_def_times_g = gaussian_copula_sampling(default_func, correlation, exponent=expo, method='sobol',
    #                                                prnt=True)
    gauss_sample_time = time.perf_counter()
    print(f"Time to sample gauss data: {(gauss_sample_time - data_fit_time):.1f}")
    sampled_def_times_t = students_t_copula_sampling(default_func, correlation, dof=max_dof, exponent=expo,
                                                     method='sobol', prnt=True)
    t_student_time = time.perf_counter()
    print(f"Time to sample t student data: {(t_student_time - gauss_sample_time):.1f}")
    k = 5

    spread_t = 0
    spread = 0

    # spread = calculate_spread(k, sampled_def_times_g, prnt=True) * 10000
    calc_spread_gauss_time = time.perf_counter()
    print(f"Time to calc gauss spread: {(calc_spread_gauss_time - t_student_time):.1f}")

    spread_t = calculate_spread(k, sampled_def_times_t, prnt=True) * 10000
    calc_spread_t_time = time.perf_counter()
    print(f"Time to calc t spread: {(calc_spread_t_time - calc_spread_gauss_time):.1f}")

    end = time.perf_counter()
    print(f"Gaussian: {spread:.4f}, Student-T: {spread_t:.4f}")
    print(f"Elapsed time: {(end - start):.2f} seconds.")


if __name__ == '__main__':
    run_test()

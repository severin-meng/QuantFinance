import numpy as np
from functools import cached_property, cache
from copy import deepcopy

from stock_models import EquityPaths, ConstantShortRate, VanillaGBM

DAYS_PER_YEAR = 252
np.seterr(invalid='raise')

__all__ = ['AutocallableBRC', 'BRC']


class CouponComponent:
    """ Precalculate discounted value of all fixed rate coupons """
    def __init__(self, short_rate_func, expiry, coupon_frequency, coupon_rate, notional=1, return_notional=False):
        # coupon_frequency is a frequency, i.e. unit [1/time]. Quarterly coupons is coupon_frequency = 4.
        self.short_rate_func = short_rate_func
        self.expiry = expiry
        self.coupon_frequency = coupon_frequency
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.return_notional = return_notional
        self.coupon_dates = np.arange(1, int(self.expiry * self.coupon_frequency) + 1) / self.coupon_frequency
        # coupon_dates contains dates on which coupons are paid

    @cached_property
    def discounted_coupon_values(self):
        discount_factors = [self.short_rate_func.get_discount_factor(time) for time in self.coupon_dates]
        coupon_payments = self.coupon_rate * self.notional / self.coupon_frequency * np.ones_like(self.coupon_dates)
        discounted_coupons = np.array([discount_factor * coupon_payment
                                       for discount_factor, coupon_payment in zip(discount_factors, coupon_payments)])
        if self.return_notional:
            discounted_coupons[-1] += self.notional * discount_factors[-1]
        return discounted_coupons


class Barrier:
    def __init__(self, barrier_level, is_up_barrier, is_down_barrier):
        self.barrier_level = barrier_level
        self.is_up_barrier = is_up_barrier
        self.is_down_barrier = is_down_barrier


# TODO: put paths, vols, correlations, timesteps, timedeltas into separate Paths class


class ContinuousWorstDownBarrier(Barrier):
    """ Continuous worst-of down constant barrier monitoring using brownian bridging """
    def __init__(self, barrier_level, observation_end, observation_start=0, include_endpoint=False):
        super().__init__(barrier_level, False, True)
        self.observation_start = observation_start
        self.observation_end = observation_end
        self.include_endpoint = include_endpoint

    def is_breached(self, paths_info: EquityPaths):
        """
        returns array of shape [nbr_paths] containing the probablity that a barrier breach occurred within the
        observation period.
        """
        # prepend 0 to sampling_times
        sampling_times = np.concatenate((np.array([0]), paths_info.sampling_times))
        assert len(paths_info.paths.shape) == 3 and paths_info.paths.shape[1] == sampling_times.shape[0]
        if isinstance(self.barrier_level, np.ndarray):
            assert self.barrier_level.shape[0] == paths_info.paths.shape[0]  # at least one barrier per underlying

        first_time_index = np.argmax(sampling_times >= self.observation_start)
        if self.include_endpoint:
            if sampling_times[-1] == self.observation_end:
                last_time_index = sampling_times.shape[0] + 1
            else:
                last_time_index = np.argmin(sampling_times <= self.observation_end)
        else:
            if sampling_times[-1] < self.observation_end:
                last_time_index = sampling_times.shape[0] + 1
            else:
                last_time_index = np.argmin(sampling_times < self.observation_end)
        if first_time_index == 0 and last_time_index >= sampling_times.shape[0] + 1:
            observation_paths = paths_info.paths  # this does not copy the paths
        else:
            observation_paths = paths_info.paths[:, first_time_index:last_time_index, :]  # this always copies the paths

        # this has shape [nbr_paths], it is the probability of a single barrier breach during the observation period
        return self._brownian_bridge(observation_paths, paths_info.vols, paths_info.time_steps, paths_info.corr_mat)

    def _brownian_bridge(self, paths, volatilities, time_steps, corr_mat, method='frechet'):
        """
        returns probability of conditional Brownian motion with scale volatility crossing threshold (downwards)
        between time_stop and time_start conditional on value_start and value_end
        """
        barrier_level = self.barrier_level
        if isinstance(self.barrier_level, np.ndarray):
            if len(self.barrier_level.shape) == 1:
                barrier_level = self.barrier_level[..., None, None]  # asset dependent barrier
            elif len(self.barrier_level.shape) == 2:
                barrier_level = self.barrier_level[..., None]  # time dependent barrier
            else:
                raise Exception(f"Unreasonable shape of barrier level: {self.barrier_level.shape}")

        if self.is_down_barrier:
            guaranteed_knocks = np.any(paths <= self.barrier_level, axis=(0, 1))
        else:
            guaranteed_knocks = np.any(paths >= barrier_level, axis=(0, 1))
        print(f"Nbr guaranteed knocks: {guaranteed_knocks.sum()}")
        quotient_paths = np.log(paths[:, :, ~guaranteed_knocks] / self.barrier_level)

        if len(volatilities.shape) == 1:  # constant vols
            volatilities = volatilities[..., None, None]
        elif len(volatilities.shape) == 2:  # time dependent vols
            volatilities = volatilities[:, :-1, None]
        else:
            volatilities = volatilities[:, :-1, ~guaranteed_knocks]  # local volatilities

        # TODO: compare frechet bounds method against copula sampling using stock correlations and a copula
        # marginal barrier cross probability, shape [nbr_underlyings, nbr_times, nbr_paths - discrete_knocks.sum()]
        marginal_barrier_cross = np.exp(-2 * quotient_paths[:, 1:, :] * quotient_paths[:, :-1, :] / (
                    volatilities ** 2 * time_steps[None, ..., None]))

        if paths.shape[0] == 1:
            probability_cross = 1 - np.cumproduct(1-marginal_barrier_cross, axis=1).flatten()
        else:
            # multi asset brownian bridging is more complicated - see https://arxiv.org/pdf/0904.1157
            # option price = [1 - Pr(no knock)] * vanilla option price
            # Pr(no knock) = Pr(Union_i^d(No knock on Asset i))
            # Frechet bounds: P_L = max(0, sum_i^d Pr(no knock on asset i) - k -1)
            #                    <= Pr(no Knock)
            #                    <= min_i=1..d {Pr(no knock on asset i)} = P_U
            # it follows that  1 - P_L >=  1 - Pr(no knock) = Pr(at least one knock)  >= 1 - P_U
            # since payoff is linear in no knock proba -> use average of upper and lower bounds
            correlation_values = corr_mat[np.triu_indices(corr_mat.shape[0], 1, corr_mat.shape[1])]
            """if np.all(correlation_values > 0):
                upper = self._frechet_upper_bound(marginal_barrier_cross)
                lower = self._joint_independent_prob(marginal_barrier_cross)
            elif np.all(correlation_values < 0):
                upper = self._joint_independent_prob(marginal_barrier_cross)
                lower = self._frechet_lower_bound(marginal_barrier_cross)
            else:
                upper = self._frechet_upper_bound(marginal_barrier_cross)
                lower = self._frechet_lower_bound(marginal_barrier_cross)"""
            upper = self._frechet_upper_bound(marginal_barrier_cross)
            lower = self._frechet_lower_bound(marginal_barrier_cross)
            mid = self._joint_independent_prob(marginal_barrier_cross)
            upper_proba_no_cross = np.prod(upper, axis=0)
            lower_proba_no_cross = np.prod(lower, axis=0)
            mid_proba_no_cross = np.prod(mid, axis=0)
            print(f"Upper prob: {1 - np.mean(upper_proba_no_cross)}, mid prob: {1-np.mean(mid_proba_no_cross)}, "
                  f" lower prob: {1 - np.mean(lower_proba_no_cross)}")
            # lower and upper are bounds for the 'no knock' probability
            probability_cross = 1 - 0.5 * (upper_proba_no_cross + lower_proba_no_cross)

        barrier_cross_proba = np.empty_like(guaranteed_knocks, dtype=np.float64)
        barrier_cross_proba[guaranteed_knocks] = 1
        barrier_cross_proba[~guaranteed_knocks] = probability_cross
        print(f"Global barrier cross prob. incl guaranteed ones: {np.mean(barrier_cross_proba)}")
        return barrier_cross_proba

    @staticmethod
    def _frechet_upper_bound(marginal_barrier_cross):
        # upper bound on joint probability that no barrier is crossed given marginal barrier cross probabilities
        return np.min(1 - marginal_barrier_cross, axis=0)

    @staticmethod
    def _frechet_lower_bound(marginal_barrier_cross):
        # lower bound on joint probability that no barrier is crossed given marginal barrier cross probabilities
        first_term = 1 - np.sum(marginal_barrier_cross, axis=0)
        return np.maximum(first_term, 0)

    @staticmethod
    def _joint_independent_prob(marginal_barrier_cross):
        # probability that no barrier is crossed given marginal barrier cross probabilities assuming independence
        return np.prod(1 - marginal_barrier_cross, axis=0)


class DiscreteWorstDownBarrier(Barrier):
    """ Discrete worst-of down constant barrier monitoring used to compare against continuous Barrier """
    def __init__(self, barrier_level, observation_times):
        super().__init__(barrier_level, False, True)
        self.observation_times = observation_times

    def is_breached(self, paths_info: EquityPaths):
        """
        shape of paths is [nbr_underlyings, sampling_times, nbr_paths]
        returns boolean array of shape [nbr_paths] telling whether a barrier breach occurred within the
        observation period.
        """
        # prepend 0 to sampling_times
        sampling_times = np.concatenate((np.array([0]), paths_info.sampling_times))
        assert len(paths_info.paths.shape) == 3 and paths_info.paths.shape[1] == sampling_times.shape[0]
        if isinstance(self.barrier_level, np.ndarray):
            assert self.barrier_level.shape[0] == paths_info.paths.shape[0]  # at least one barrier per underlying

        assert np.all(np.isin(self.observation_times, sampling_times))  # make sure all observation times are sampled
        discrete_breaches = np.any(
            paths_info.paths[:, np.isin(sampling_times, self.observation_times), :] < self.barrier_level, axis=(0, 1))
        return discrete_breaches


class DiscreteAllUpBarrier(Barrier):
    """ Discrete All-of up constant barrier monitoring using brownian bridging """
    def __init__(self, barrier_level, observation_times):
        super().__init__(barrier_level, True, False)
        self.observation_times = observation_times

    def breach_time(self, paths_info: EquityPaths):
        """
        shape of paths is [nbr_underlyings, sampling_times, nbr_paths]
        volatilities has to be of shape [nbr_underlyings]  (constant vol)
            or [nbr_underlyings, sampling_times]  (stock-homogeneous local vol)
            or [nbr_underlyings, sampling_times, nbr_paths]  (local vol)
        returns array of shape [nbr_paths] containing the autocall breach time (np.inf if no breach occurs)
        """
        assert np.all(np.isin(self.observation_times, paths_info.sampling_times))  # make sure all observation times are sampled
        # prepend 0 to sampling_times
        # TODO figure out prepending of 0 to sampling times
        sampling_times = np.concatenate((np.array([0]), paths_info.sampling_times))
        discrete_breaches = np.all(
            paths_info.paths[:, np.isin(sampling_times, self.observation_times), :] > self.barrier_level, axis=0)
        is_breached = np.any(discrete_breaches, axis=0)
        first_breach = np.where(is_breached, np.argmax(discrete_breaches, axis=0), -1)
        breach_dates = np.concatenate((self.observation_times, np.array([np.inf])))[first_breach]
        return breach_dates


class WorstOfPutPayoff:
    def __init__(self, strike, expiry, short_rate_func, initial_fixing=1, quantity=1):
        self.initial_fixing = initial_fixing
        self.rel_strike = strike / self.initial_fixing
        self.expiry = expiry
        self.short_rate_func = short_rate_func
        self.quantity = quantity

    def path_payoff(self, paths: EquityPaths):
        """
        paths has shape [nbr_underlyings, nbr_timesteps, nbr_simulations]
        """
        rel_worst_of_price = np.min(paths.paths[:, -1, :] / self.initial_fixing, axis=0)
        option_payoff = self.short_rate_func.get_discount_factor(self.expiry) * self.quantity * np.maximum(
            self.rel_strike-rel_worst_of_price, 0)
        return option_payoff


class BestOfCallPayoff:
    def __init__(self, strike, expiry, short_rate_func, initial_fixing=1, quantity=1):
        self.initial_fixing = initial_fixing
        self.rel_strike = strike / self.initial_fixing
        self.expiry = expiry
        self.short_rate_func = short_rate_func
        self.quantity = quantity

    def path_payoff(self, paths: EquityPaths):
        """
        paths has shape [nbr_underlyings, nbr_timesteps, nbr_simulations]
        """
        rel_worst_of_price = np.min(paths.paths[:, -1, :] / self.initial_fixing, axis=0)
        option_payoff = self.short_rate_func.get_discount_factor(self.expiry) * self.quantity * np.maximum(
            self.rel_strike-rel_worst_of_price, 0)
        return option_payoff


class WorstOfDownOutCallPayoff:
    def __init__(self, strike, expiry, short_rate_func, barrier_level, initial_fixing=1, notional=1000,
                 barrier_type='continuous'):
        # initial fixing is only used to determine the worst performer
        # for everyting else, use strike: quantity = notional / abs_strike such that quantity * strike = notional
        assert strike > 0
        assert expiry > 0
        assert initial_fixing > 0
        self.strike = strike
        self.expiry = expiry
        self.short_rate_func = short_rate_func
        self.barrier_level = barrier_level
        self.initial_fixing = initial_fixing
        self.notional = notional
        self.barrier_type = barrier_type
        self.rainbow_option = WorstOfPutPayoff(
            self.strike, self.expiry, self.short_rate_func, self.initial_fixing, self.notional / self.strike)

    def path_payoff(self, paths_info: EquityPaths):
        if self.barrier_type == 'discrete':
            barrier = DiscreteWorstDownBarrier(self.barrier_level, paths_info.sampling_times)
            is_path_breached = barrier.is_breached(paths_info)
            # the extra cpu time used to calculate the option payoff for non-breached paths is probably less of an
            # issue compared to copying the breached paths
        elif self.barrier_type == 'continuous':
            barrier = ContinuousWorstDownBarrier(self.barrier_level, observation_end=paths_info.sampling_times[-1],
                                                 observation_start=0, include_endpoint=True)
            is_path_breached = barrier.is_breached(paths_info)
        else:
            raise Exception(f"Unknown Barrier Type {self.barrier_type}")
        option_payoffs = self.rainbow_option.path_payoff(paths_info)
        return option_payoffs * is_path_breached


class WorstOfDownInPutPayoff:
    def __init__(self, strike, expiry, short_rate_func, barrier_level, initial_fixing=1, notional=1000,
                 barrier_type='continuous'):
        # initial fixing is only used to determine the worst performer
        # for everyting else, use strike: quantity = notional / abs_strike such that quantity * strike = notional
        assert strike > 0
        assert expiry > 0
        assert initial_fixing > 0
        self.strike = strike
        self.expiry = expiry
        self.short_rate_func = short_rate_func
        self.barrier_level = barrier_level
        self.initial_fixing = initial_fixing
        self.notional = notional
        self.barrier_type = barrier_type
        self.rainbow_option = WorstOfPutPayoff(
            self.strike, self.expiry, self.short_rate_func, self.initial_fixing, self.notional / self.strike)

    def path_payoff(self, paths_info: EquityPaths):
        if self.barrier_type == 'discrete':
            barrier = DiscreteWorstDownBarrier(self.barrier_level, paths_info.sampling_times)
            is_path_breached = barrier.is_breached(paths_info)
            # the extra cpu time used to calculate the option payoff for non-breached paths is probably less of an
            # issue compared to copying the breached paths
        elif self.barrier_type == 'continuous':
            barrier = ContinuousWorstDownBarrier(self.barrier_level, observation_end=paths_info.sampling_times[-1],
                                                 observation_start=0, include_endpoint=True)
            is_path_breached = barrier.is_breached(paths_info)
        else:
            raise Exception(f"Unknown Barrier Type {self.barrier_type}")
        option_payoffs = self.rainbow_option.path_payoff(paths_info)
        return option_payoffs * is_path_breached


class BRC:
    def __init__(self, strike, expiry, short_rate_func, barrier_level, coupon_rate, coupon_frequency,
                 initial_fixing=1, notional=1000, barrier_type='continuous'):
        self.strike = strike
        self.expiry = expiry
        self.short_rate_func = short_rate_func
        self.barrier_level = barrier_level
        self.coupon_rate = coupon_rate
        self.coupon_frequency = coupon_frequency
        self.initial_fixing = initial_fixing
        self.notional = notional
        self.barrier_type = barrier_type

        self.bond_weight = 1
        self.option_weight = -1
        self.option_component = WorstOfDownInPutPayoff(
            self.strike, self.expiry, self.short_rate_func, self.barrier_level, self.initial_fixing,
            self.notional, self.barrier_type)
        self.coupon_component = CouponComponent(self.short_rate_func, self.expiry, self.coupon_frequency,
                                                self.coupon_rate, self.notional, return_notional=True)

    def path_payoff(self, paths_info: EquityPaths):
        discounted_coupons = self.coupon_component.discounted_coupon_values
        option_payoffs = self.option_component.path_payoff(paths_info)
        return self.bond_weight * sum(discounted_coupons) + self.option_weight * option_payoffs

    @cache
    def simulation_times(self, frequency):
        coupon_dates = self.coupon_component.coupon_dates
        uniform_knock_in_dates = np.arange(1, int(self.expiry * frequency) + 1) / frequency
        return np.array(sorted(set(coupon_dates).union(set(uniform_knock_in_dates))))


class AutocallableBRC:
    def __init__(self, strike, expiry, short_rate_func, knock_in_level, autocall_level, autocall_frequency,
                 coupon_rate, coupon_frequency, initial_fixing=1, notional=1000, knock_in_type='continuous'):
        # initial fixing is only used to determine the worst performer
        # for everyting else, use strike: quantity = notional / abs_strike such that quantity * strike = notional
        assert strike > 0
        assert expiry > 0
        assert initial_fixing > 0
        self.strike = strike
        self.expiry = expiry
        self.short_rate_func = short_rate_func
        self.knock_in_level = knock_in_level
        self.autocall_level = autocall_level
        self.autocall_frequency = autocall_frequency
        self.coupon_rate = coupon_rate
        self.coupon_frequency = coupon_frequency
        self.initial_fixing = initial_fixing
        self.notional = notional
        self.knock_in_type = knock_in_type
        # no autocall check on expiry
        self.autocall_dates = np.arange(1, int(self.expiry * self.autocall_frequency)) / self.autocall_frequency
        self.autocall_barrier = DiscreteAllUpBarrier(self.autocall_level, self.autocall_dates)
        self.brc = BRC(
            self.strike, self.expiry, self.short_rate_func, self.knock_in_level, self.coupon_rate,
            self.coupon_frequency, self.initial_fixing, self.notional, self.knock_in_type)

    def path_payoff(self, paths_info: EquityPaths):
        # calculate autocall paths and payoffs
        autocall_dates = self.autocall_barrier.breach_time(paths_info)
        is_autocalled = autocall_dates < self.expiry
        max_coupon_date_index = np.argmax(
            self.brc.coupon_component.coupon_dates[..., None] >= autocall_dates[None, is_autocalled], axis=0)
        autocall_contribution = np.cumsum(self.brc.coupon_component.discounted_coupon_values)[max_coupon_date_index]
        autocall_contribution += self.notional * self.short_rate_func.get_discount_factor(autocall_dates[is_autocalled])

        # calculate BRC payoffs
        not_called_paths_info = deepcopy(paths_info)
        not_called_paths_info.thin_along_paths(~is_autocalled)
        brc_payoffs = self.brc.path_payoff(not_called_paths_info)

        # stitch together
        payoffs = np.empty_like(is_autocalled, dtype=np.float64)
        payoffs[is_autocalled] = autocall_contribution
        payoffs[~is_autocalled] = brc_payoffs
        return payoffs

    @cache
    def simulation_times(self, frequency):
        return np.array(sorted(set(self.brc.simulation_times(frequency)).union(set(self.autocall_dates))))


def autocall_payoff(paths, strike_perc, barrier_perc, autocall_barrier, short_rate_func,
                    coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=False):
    # paths shape is (nbr_underlyings, nbr_days, nbr_paths)

    coupon_dates = np.arange(1, expiry // coupon_freq + 1) * coupon_freq
    coupon_payments = coupon_rate * notional * coupon_freq * np.ones(int(expiry // coupon_freq))
    discounted_coupons = np.array([0] + [short_rate_func.get_discount_factor(time) * coupon_payment
                                         for time, coupon_payment in zip(coupon_dates, coupon_payments)])
    # extend discounted coupons by 1 to account for no autocall being index -1

    autocall_observ_dates = np.arange(1, expiry // autocall_freq) * autocall_freq
    autocall_date_index = (np.rint(autocall_observ_dates * DAYS_PER_YEAR)).astype(np.int32)
    autocall_observ_and_maturity = np.arange(1, expiry // autocall_freq + 1) * autocall_freq

    autocall_breaches = np.where(np.all(paths[:, autocall_date_index, :] > autocall_barrier, axis=0), 1, 0)
    # first_autocall is array like [int = autocall_date_index, nbr_simulations]
    is_autocall_breach = np.any(autocall_breaches, axis=0)
    first_autocall = np.where(is_autocall_breach, np.argmax(autocall_breaches, axis=0), -1)
    autocall_date = autocall_observ_and_maturity[first_autocall]  #
    last_coupon = first_autocall + 1

    coupons_until_autocall = np.cumsum(discounted_coupons)[last_coupon]
    full_coupons = np.sum(discounted_coupons)

    notional_contribution = notional * short_rate_func.get_discount_factor(autocall_date)

    # daily trigger barrier check
    is_barrier_knocked = np.any(np.min(paths, axis=0) < barrier_perc, axis=0)
    worst_of_price = np.min(paths[:, -1, :], axis=0)
    option_payoff = short_rate_func.get_discount_factor(expiry) * (
            -notional + notional / strike_perc * np.where(worst_of_price < strike_perc, worst_of_price, strike_perc))

    # payoff = discounted_coupons + notional + option_payoff
    payoffs = np.where(is_autocall_breach, coupons_until_autocall + notional_contribution,
                       full_coupons + notional_contribution + np.where(is_barrier_knocked, option_payoff, 0))
    if return_array:
        return payoffs
    price = np.mean(payoffs)
    return price


if __name__ == '__main__':
    from wiener_path_generators import PseudoRandomPathGenerator, SobolPathGenerator
    from timeit import default_timer
    strike_perc = 1.0
    barrier_perc = 0.8
    autocall_barrier = 1.2
    discount_rate = 0.01
    coupon_rate = 0.058
    coupon_freq = 0.25
    autocall_freq = 0.25
    expiry = 1.0
    vols = np.array([0.6, 0.4, 0.2])
    nbr_underlyings = vols.shape[0]
    corrs = np.array([0.6, 0.3, 0.4])

    notional = 1000
    corr_mat = np.array([[1, corrs[0], corrs[1]],
                         [corrs[0], 1, corrs[2]],
                         [corrs[1], corrs[2], 1]])

    const_short_rate = ConstantShortRate(discount_rate)

    DiscreteAutocallable = AutocallableBRC(strike_perc, expiry, const_short_rate, barrier_perc, autocall_barrier,
                                           1/autocall_freq, coupon_rate, 1/coupon_freq, notional=notional,
                                           knock_in_type='discrete')
    ContinuousAutoCallable = AutocallableBRC(strike_perc, expiry, const_short_rate, barrier_perc, autocall_barrier,
                                           1/autocall_freq, coupon_rate, 1/coupon_freq, notional=notional,
                                           knock_in_type='continuous')

    path_exponent = 17
    """
    payoffs_old = autocall_payoff(gbm_paths.paths, strike_perc, barrier_perc, autocall_barrier, const_short_rate,
                                  coupon_rate, coupon_freq, autocall_freq, expiry, notional, return_array=True)
    """

    sampling_range = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    disc_payoffs = []
    cont_payoffs = []
    for sample in sampling_range:
        start = default_timer()
        sampling_times = DiscreteAutocallable.simulation_times(sample)
        path_gen = SobolPathGenerator(sampling_times, nbr_underlyings, correlation=corr_mat, seed=42)
        gbm_model = VanillaGBM(vols, const_short_rate, path_gen)
        gbm_paths = gbm_model.generate_paths(2 ** path_exponent, antithetic_sampling=False)
        paths_end = default_timer()
        payoffs_disc = DiscreteAutocallable.path_payoff(gbm_paths)
        time_disc = default_timer()
        payoffs_cont = ContinuousAutoCallable.path_payoff(gbm_paths)
        time_cont = default_timer()
        print(f"Path sampling for {sample} time steps took {paths_end-start}")
        print(f"Discrete payoff calculation: {time_disc-paths_end}")
        print(f"Continuous payoff calculation: {time_cont - time_disc}")
        disc_payoffs.append(np.mean(payoffs_disc))
        cont_payoffs.append(np.mean(payoffs_cont))
    print(disc_payoffs)
    print(cont_payoffs)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogx(sampling_range, disc_payoffs, label='Discrete Barrier')
    plt.semilogx(sampling_range, cont_payoffs, label='Continuous Barrier')
    plt.xlabel('Nbr Time Steps')
    plt.title('Autcallable Price vs Time Discretization')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    print("DONE")
    print("DONE AGAIN")
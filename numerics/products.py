from scipy.special import ndtr
import numpy as np
from enum import Enum


__all__ = ['ExerciseType', 'EuropeanOption', 'BermudanOption', 'AmericanOption', 'FlexibleForward']


class ExerciseType(Enum):
    EUROPEAN = 1
    AMERICAN = 2
    BERMUDAN = 3


class Product:
    def __init__(self, expiry, exercise_type):
        self.spot_scale = 4
        self.expiry = expiry
        self.exercise_type = exercise_type

    def get_time_axis(self, n_time, t_start=0):
        return np.linspace(t_start, self.expiry, n_time)

    def get_spot_axis(self, spot, n_spots, min_spot=0):
        return np.linspace(min_spot, self.spot_scale * spot, n_spots)

    def is_exercise_time(self, time):
        raise NotImplementedError

    def get_final_condition(self, *args):
        raise NotImplementedError

    def get_upper_boundary_condition(self, *args):
        raise NotImplementedError

    def get_lower_boundary_condition(self, *args):
        raise NotImplementedError


class EuropeanOption(Product):
    def __init__(self, strike, tte, is_call):
        super().__init__(tte, ExerciseType.EUROPEAN)
        self.strike = strike
        self.tte = tte
        self.is_call = is_call
        self.product = 'European ' + 'Call' if is_call else 'Put'

    def is_exercise_time(self, time):
        return False

    def get_final_condition(self, spot_axis):
        if self.is_call:
            return np.maximum(spot_axis - self.strike, 0)
        return np.maximum(self.strike - spot_axis, 0)

    def get_upper_boundary_condition(self, time_axis, max_spot, model):
        if self.is_call:
            return max_spot * np.exp(-model.div_yield * (self.tte - time_axis)) - self.strike * np.exp(-model.rate * (self.tte - time_axis))
        return np.zeros_like(time_axis)

    def get_lower_boundary_condition(self, time_axis, min_spot, model):
        if self.is_call:
            return np.zeros_like(time_axis)
        return self.strike * np.exp(-model.rate * (self.tte - time_axis)) - min_spot * np.exp(-model.div_yield * (self.tte - time_axis))

    def exercise_payoff(self, spot_axis, time):
        if time == self.tte:
            return spot_axis - self.strike if self.is_call else self.strike - spot_axis
        else:
            return np.zeros_like(spot_axis)

    def analytical_price(self, spot,  model):
        sqrt_tte = np.sqrt(self.tte)
        drift = (model.rate - model.div_yield + model.vol ** 2 / 2) * self.tte
        d1 = (np.log(spot / self.strike) + drift) / (model.vol * sqrt_tte)
        d2 = d1 - (model.vol * sqrt_tte)
        if self.is_call:
            call_prc = spot * np.exp(-model.div_yield * self.tte) * ndtr(d1) - self.strike * np.exp(-model.rate * self.tte) * ndtr(d2)
            return np.where(spot == 0, 0, call_prc)
        put_prc = -spot * np.exp(-model.div_yield * self.tte) * ndtr(-d1) + self.strike * np.exp(-model.rate * self.tte) * ndtr(-d2)
        return np.where(spot == 0, np.exp(-model.rate * self.tte) * self.strike, put_prc)


class BermudanOption(Product):
    def __init__(self, strikes, exercise_dates, is_call):
        super().__init__(exercise_dates[-1], ExerciseType.BERMUDAN)
        self.strikes = strikes
        self.exercise_dates = exercise_dates
        self.is_call = is_call

        self.product = "Bermudan " + "Call" if is_call else "Put"

    def is_exercise_time(self, time):
        return time in self.exercise_dates

    def get_time_axis(self, n_time, t_start=0):
        max_dt = (self.expiry - t_start) / n_time
        min_dt = 1.e-10
        time_axis = [t_start]
        t_current = t_start
        exercise_idx = 0
        while exercise_idx < len(self.exercise_dates):
            next_exercise = self.exercise_dates[exercise_idx]
            t_current = time_axis[-1]
            if next_exercise - t_current > max_dt:
                # need to insert time steps
                add_n_steps = int((next_exercise - t_current) / max_dt - 1.e-16) + 1
                dt = (next_exercise - t_current) / add_n_steps
                t = t_current + dt
                while t < next_exercise - min_dt:
                    time_axis.append(t)
                    t += dt
            time_axis.append(next_exercise)
            exercise_idx += 1
        return np.array(time_axis)

    def get_final_condition(self, spot_axis):
        if self.is_call:
            return np.maximum(spot_axis - self.strikes[-1], 0)
        return np.maximum(self.strikes[-1] - spot_axis, 0)

    def get_upper_boundary_condition(self, time_axis, max_spot, model):
        # use european asymptotics, handle early exercise during backwards stepping
        if not self.is_call:
            return np.zeros_like(time_axis)
        time_to_expiry = self.expiry - time_axis
        return max_spot * np.exp(-model.div_yield * time_to_expiry) - self.strikes[-1] * np.exp(-model.rate * time_to_expiry)

    def get_lower_boundary_condition(self, time_axis, min_spot, model):
        if self.is_call:
            return np.zeros_like(time_axis)
        time_to_expiry = self.expiry - time_axis
        return self.strikes[-1] * np.exp(-model.rate * time_to_expiry) - min_spot * np.exp(-model.div_yield * time_to_expiry)

    def exercise_payoff(self, spot_axis, time):
        matches = np.where(self.exercise_dates == time)[0]
        idx = matches[0]
        return spot_axis - self.strikes[idx] if self.is_call else self.strikes[idx] - spot_axis


class AmericanOption(Product):
    def __init__(self, strike, exercise_date, is_call):
        super().__init__(exercise_date, ExerciseType.AMERICAN)
        self.strike = strike
        self.is_call = is_call

        self.product = "American " + "Call" if is_call else "Put"

    def is_exercise_time(self, time):
        return time <= self.expiry

    def get_final_condition(self, spot_axis):
        if self.is_call:
            return np.maximum(spot_axis - self.strike, 0)
        return np.maximum(self.strike - spot_axis, 0)

    def get_upper_boundary_condition(self, time_axis, max_spot, model):
        if not self.is_call:
            return np.zeros_like(time_axis)
        time_to_expiry = self.expiry - time_axis
        return max_spot * np.exp(-model.div_yield * time_to_expiry) - self.strike * np.exp(-model.rate * time_to_expiry)

    def get_lower_boundary_condition(self, time_axis, min_spot, model):
        if self.is_call:
            return np.zeros_like(time_axis)
        # Find indices of the first exercise_date >= time_axis element
        time_to_expiry = self.expiry - time_axis
        return self.strike * np.exp(-model.rate * time_to_expiry) - min_spot * np.exp(-model.div_yield * time_to_expiry)

    def exercise_payoff(self, spot_axis, time):
        return spot_axis - self.strike if self.is_call else self.strike - spot_axis


class FlexibleForward(Product):
    def __init__(self, strike, tte):
        super().__init__(tte, ExerciseType.AMERICAN)
        self.strike = strike
        self.product = 'Flexible Forward'
        self.spot_scale = 1.5

    def is_exercise_time(self, time):
        return time < self.expiry

    def get_final_condition(self, spot_axis):
        return spot_axis - self.strike

    def get_upper_boundary(self, time_axis, max_spot, model):
        return (max_spot - self.strike) * np.ones_like(time_axis)

    def get_lower_boundary(self, time_axis, min_spot, model):
        return min_spot * np.exp(-model.rate_f * (self.expiry - time_axis)) - self.strike * np.exp(
            -model.rate_d * (self.tte - time_axis))

    def exercise_payoff(self, spot_axis, time):
        return spot_axis - self.strike

    def analytical_price(self, spot, model, method='GJ', theta=0.42, lambd=0.45):
        imm_payoff = spot - self.strike
        fwd_payoff = spot * np.exp(-model.rate_f * self.expiry) - self.strike * np.exp(-model.rate_d * self.expiry)
        if method == '2-step':
            return max(imm_payoff, fwd_payoff)
        elif method in ('3-step', 'GJ'):
            adjusted_tte = self.expiry * (1 - theta)
            adjusted_strike = self.strike * (1 - np.exp(-model.rate_d * adjusted_tte)) / (1 - np.exp(-model.rate_f * adjusted_tte))
            call = EuropeanOption(adjusted_strike, theta * self.expiry, is_call=True)
            call_prc = call.analytical_price(spot, model)
            if method == '3-step':
                return max(imm_payoff, fwd_payoff + (1 - np.exp(-model.rate_f * adjusted_tte)) * call_prc)
            return max(imm_payoff, fwd_payoff + (1 - np.exp(-model.rate_f * adjusted_tte)) * (1 + lambd) * call_prc)
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

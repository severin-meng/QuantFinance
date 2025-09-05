"""
Idea is to have a MultiCurve, MultiCurrency interface for pricing and curve calibration
"""

import math


class MulticurveProviderInterface:
    def __init__(self, currencies, discount_curves, ibor_indices, ibor_curves, on_indices, on_curves, fx_matrix):
        self.discount_curves = dict(zip(currencies, discount_curves))
        self.ibor_curves = dict(zip(ibor_indices, ibor_curves))
        self.on_curves = dict(zip(on_indices, on_curves))
        self.fx_matrix = fx_matrix

    def get_discount_factor(self, currency, time):
        """
        return exp(-rate * time) where rate is the zero rate for the currency
        """
        return self.discount_curves[currency].df(time)

    def get_forward_rate(self, ibor_index, start, end, accrual_factor):
        """
        return simple forward rate for ibor index, from start to end
        """
        return self.ibor_curves[ibor_index].fwd(start, end, accrual_factor)

    def get_forward_rate(self, on_index, start, end, accrual_factor):
        """
        return simple forward rate for on index, from start to end
        """
        return self.on_curves[on_index].fwd(start, end, accrual_factor)

    def get_fx_rate(self, curr1, curr2):
        """
        return fx spot rate
        """


class MultipleCurrencyAmount:
    def __init__(self, currencies, amounts):
        self.currencies = currencies
        self.amounts = amounts
        if isinstance(currencies, float):
            assert isinstance(amounts, float)
        elif isinstance(amounts, float):
            raise Exception("Invalid MultiCurrency Amount")
        else:
            assert len(currencies) == len(amounts)


class IborCoupon:
    def __init__(self, currency, notional, ibor_index, fixing_start, fixing_end, fixing_accrual, payment_accrual, pay_date):
        self.currency = currency
        self.notional = notional
        self.ibor_index = ibor_index
        self.fixing_start = fixing_start
        self.fixing_end = fixing_end
        self.fixing_accrual = fixing_accrual
        self.payment_accrual = payment_accrual
        self.pay_date = pay_date

    def getCurrency(self):
        return self.currency

    def getIndex(self):
        return self.ibor_index

    def getFixingPeriodStartTime(self):
        return self.fixing_start

    def getFixingPeriodEndTime(self):
        return self.fixing_end

    def getFixingAccrualFactor(self):
        return self.fixing_accrual

    def getPaymentAccrualFactor(self):
        return self.payment_accrual

    def getPaymentTime(self):
        return self.pay_date

    def getNotional(self):
        return self.notional


def ibor_coupon_value(cpn: IborCoupon, multicurve: MulticurveProviderInterface) -> MultipleCurrencyAmount:
    fwd = multicurve.get_forward_rate(cpn.getIndex(), cpn.getFixingPeriodStartTime(), cpn.getFixingPeriodEndTime(), cpn.getFixingAccrualFactor())
    df = multicurve.get_discount_factor(cpn.getCurrency(), cpn.getPaymentTime())
    pv = cpn.getNotional() * cpn.getPaymentAccrualFactor() * fwd * df
    return MultipleCurrencyAmount(cpn.getCurrency(), pv)


class CurveBlock:
    """
    Idea: curves that depend on each other are grouped in blocks
    That way, one knows what curve's benchmarks another curves parameters depend on

    List of curves in the block as a map:
    Curve Name <-> pair of integers
    1) start index of the curve in the list of parameters
    2) number of parameters in this curve
    """

    def __init__(self, curve_names, start_param_idxs, num_param_idxs):
        self.curve_map = dict(zip(curve_names, zip(start_param_idxs, num_param_idxs)))

    def getStart(self, curve_name):
        return self.curve_map[curve_name][0]

    def getNumParams(self, curve_name):
        return self.curve_map[curve_name][1]


class CurveBlockBundle:
    """
    map of curve blocks together with jacobians
    """
    def __init(self, block_names, blocks, jacobians):
        self.bundle = dict(zip(block_names, zip(blocks, jacobians)))

    def add(self, block_name, block, jacobian):
        self.bundle[block_name] = (block, jacobian)

    def getBlock(self, block_name):
        return self.bundle[block_name]


class Curve:
    def __init__(self, curve_dates, curve_values, calc_type='disc', interpolator='linear', tenor='1d'):
        assert calc_type in ('spot_rate', 'forward_rate')
        assert interpolator in ('linear', 'linear_RT')
        self.curve_dates = curve_dates
        self.curve_values = curve_values
        self.is_zero_rates = True if calc_type=='spot_rate' else False
        self.interpolator = interpolator
        self.tenor = tenor


    def df(self, end):
        assert self.is_zero_rates
        if end <= self.curve_dates[0]:
            # flat extrapolation on short end
            return math.exp(-self.curve_values[0] * end)
        if end in self.curve_dates:
            idx = 0
            for i, dt in enumerate(self.curve_dates):
                if dt == end:
                    idx = i
                    break
            return math.exp(- self.curve_values[idx] * end)
        idx_low = 0
        idx_high = len(self.curve_dates)
        for i, dt in enumerate(self.curve_dates):
            if dt > end:
                idx_low = i-1
                idx_high = i
                break
        delta_t = end - self.curve_dates[idx_low]
        if self.interpolator == 'linear':
            slope = (self.curve_values[idx_high] - self.curve_values[idx_low]) / (self.curve_dates[idx_high] - self.curve_dates[idx_low])
            rate_time = end * (self.curve_values[idx_low] + delta_t * slope)
        else:  # interpolator = linear RT
            slope = (self.curve_dates[idx_high] * self.curve_values[idx_high] - self.curve_dates[idx_low] * self.curve_values[idx_low]) / (self.curve_dates[idx_high] - self.curve_dates[idx_low])
            rate_time = slope * delta_t
        return math.exp(-rate_time)


    def fwd(self, start, stop, accrual):
        """
        return simple forward rate
        relation to discount factors:
        1/df(start) * (1 + Fwd*accrual) = 1/df(stop)

        """
        if self.is_zero_rates:
            df_end = self.df(stop)
            df_start = self.df(start)
            return 1/accrual * (df_start / df_end - 1)

        # if stop-start = fwd tenor, then read from curve/interpolate
        # if this does not match, then get discount factors from fwd curve - check in paper what the impact of this is.
        if start in self.curve_dates:
            # idea:
            pass


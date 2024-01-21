from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


days_per_year = 20
coupon_period = 10

min_ytm = 0.00001
max_ytm = 1


def yield_to_maturity(market_price,
                      coupon_rate,
                      last_coupon=5,
                      n_full_coupons=20,
                      principal_factor=1.):
    coupon_payments = [
        ((n+1)*coupon_period,
         coupon_rate * coupon_period / days_per_year) for n in range(n_full_coupons)]
    coupon_payments.append((n_full_coupons * coupon_period + last_coupon, last_coupon/days_per_year * coupon_rate))

    def ytm_func(yield_to_maturity):
        LHS = market_price
        RHS = 0
        for days, coupon in coupon_payments:
            RHS += coupon / (1 + yield_to_maturity)**(days/days_per_year)
        RHS += principal_factor / (1 + yield_to_maturity)**((n_full_coupons * coupon_period + last_coupon)/days_per_year)

        return LHS - RHS

    result = root_scalar(
        ytm_func, method='bisect', bracket=(min_ytm, max_ytm)
    )
    return result


if __name__ == '__main__':
    # print(yield_to_maturity(market_price=marketprice, coupon_rate=coupon_rate))
    day_range = range(int(40*coupon_period))
    ytms = []
    pfactor = 1.04
    mkt_prc = 1.02
    n_coupons_base = 10
    n_coupons = n_coupons_base
    days = 0
    coupon_rate = 0.12
    for day in day_range:
        days = day
        if day // coupon_period >= 1:
            extra_day = day // int(coupon_period)
            pfactor = 1.04 - extra_day//4 * 0.01
            days = day - extra_day*int(coupon_period)
            n_coupons = n_coupons_base + extra_day
        ytm = yield_to_maturity(market_price=mkt_prc, coupon_rate=coupon_rate, last_coupon=days,
                                principal_factor=pfactor, n_full_coupons=n_coupons).root
        ytms.append(ytm)

    plt.figure()
    plt.plot(day_range, ytms)
    plt.title("Days vs YTM")
    plt.xlabel('Days in last Coupon')
    plt.ylabel('yield to maturity')
    plt.show()

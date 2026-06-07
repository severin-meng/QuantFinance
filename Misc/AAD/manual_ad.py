"""
Manual AD example
"""

from math import exp, log, sqrt, pi
from scipy.special import ndtr  # ndtr = N


def normal_density(x):
    # normal_density = N'
    return exp(-x**2/2) / sqrt(2*pi)


def ad_black_scholes(S, t, K, T, r, vol):
    # formula: C = S * N(d1) - exp(-r*(T-t)) * K * N(d2)
    # d1 = [log(S/K) * (r + vol**2/2) * (T-t)] / [vol * sqrt(T-t)]
    # d2 = d1 - vol * sqrt(T-t)

    # forward pass
    # 0. y_0 = (S, t, r, vol)
    # 1.
    y_1 = log(S/K) + (r + vol**2/2) * (T-t)

    # 2.
    y_2 = vol * sqrt(T-t)

    # 3.
    y_3 = y_1 / y_2

    # 4.
    y_4 = y_3 - y_2

    # 5.
    y_5 = ndtr(y_3)

    # 6.
    y_6 = ndtr(y_4)

    # 7.
    y_7 = exp(-r * (T-t))

    # 8.
    y_8 = S * y_5 - y_7 * K * y_6

    # 9. Output:
    z = y_8


    # initialize adjoints
    S_bar, t_bar, r_bar, vol_bar = 0, 0, 0, 0
    y_1_bar, y_2_bar, y_3_bar, y_4_bar, y_5_bar, y_6_bar, y_7_bar, y_8_bar = 0, 0, 0, 0, 0, 0, 0, 0

    # backward pass
    # 9. y_8 = z (seed)
    y_8_bar = 1

    # 8. y_8 = S * y_5 - y_7 * K * y_6
    S_bar += y_8_bar * y_5
    y_5_bar += y_8_bar * S
    y_7_bar += y_8_bar * (-K) * y_6
    y_6_bar += y_8_bar * (-y_7) * K

    # 7. y_7 = exp(-r * (T-t))
    r_bar += y_7_bar * (-y_7) * (T-t)
    t_bar += y_7_bar * y_7 * r

    # 6. y_6 = N(y_4)
    y_4_bar += y_6_bar * normal_density(y_4)

    # 5. y_5 = N(y_3)
    y_3_bar += y_5_bar * normal_density(y_3)

    # 4. y_4 = y_3 - y_2
    y_3_bar += y_4_bar * 1
    y_2_bar += y_4_bar * -1

    # 3. y_3 = y_1 / y_2
    y_1_bar += y_3_bar / y_2
    y_2_bar += y_3_bar * (-y_3)/y_2

    # 2. y_2 = vol * sqrt(T-t)
    vol_bar += y_2_bar * sqrt(T-t)
    t_bar += y_2_bar * vol * (-1)/(2*sqrt(T-t))

    # 1. y_1 = log(S/K) + (r + vol^2/2) * (T-t)
    S_bar += y_1_bar / S
    r_bar += y_1_bar * (T-t)
    vol_bar += y_1_bar * vol * (T-t)
    t_bar += y_1_bar * (-1) * (r + vol**2/2)

    return z, S_bar, t_bar, r_bar, vol_bar


def black_scholes_val(S, t, K, T, r, vol):
    # formula: C = S * N(d1) - exp(-r*(T-t)) K N(d2)
    d1 = (log(S/K) + (r + vol**2/2) * (T-t)) / (vol * sqrt(T-t))
    d2 = d1 - vol * sqrt(T-t)
    return S * ndtr(d1) - K * exp(-r * (T-t)) * ndtr(d2)


def bump_reprice_black_scholes(S, t, K, T, r, vol, eps=1.e-6):
    val = black_scholes_val(S, t, K, T, r, vol)

    val_s = black_scholes_val(S+eps, t, K, T, r, vol)
    val_t = black_scholes_val(S, t+eps, K, T, r, vol)
    val_r = black_scholes_val(S, t, K, T, r+eps, vol)
    val_vol = black_scholes_val(S, t, K, T, r, vol+eps)

    delta = (val_s - val) / eps
    theta = (val_t - val) / eps
    rho = (val_r - val) / eps
    vega = (val_vol - val) / eps
    return val, delta, theta, rho, vega


def black_scholes_analytical(S, t, K, T, r, vol):
    d1 = (log(S/K) + (r + vol**2/2) * (T-t)) / (vol * sqrt(T-t))
    d2 = d1 - vol * sqrt(T-t)
    val = S * ndtr(d1) - K * exp(-r * (T-t)) * ndtr(d2)

    delta = ndtr(d1)
    vega = S * normal_density(d1) * sqrt(T-t)
    theta = - S * normal_density(d1) * vol / (2 * sqrt(T-t)) - r * K * exp(-r * (T-t)) * ndtr(d2)
    rho = K * (T-t) * exp(-r * (T-t)) * ndtr(d2)
    return val, delta, theta, rho, vega


if __name__ == "__main__":
    S = 100
    t = 0
    K = 100
    T = 1
    r = 0.02
    vol = 0.2
    import pandas as pd
    data = [["Analytisch"] + list(black_scholes_analytical(S, t, K, T, r, vol)),
            ["Adjoint"] + list(ad_black_scholes(S, t, K, T, r, vol)),
            ["Bump&Reprice"] + list(bump_reprice_black_scholes(S, t, K, T, r, vol))]
    df = pd.DataFrame(data, columns=["Type", "Val", "Delta", "Theta", "Rho", "Vega"])
    pd.set_option("display.precision", 12, "display.width", None)
    print(df)

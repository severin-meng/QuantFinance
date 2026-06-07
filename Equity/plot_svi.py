import numpy as np
import matplotlib.pyplot as plt
from numpy import log, exp, sqrt, pow


class SVISkew:
    def __init__(self, a, b, rho, m, sigma, atm_vol, atm_ref):
        # this is the Front Arena raw variance parameterization:
        # vol^2(x) = a + b[rho*(x-m) + sqrt{(x-m)^2 + sigma^2 }]
        # for x=m we have vol^2 = a + b*sigma
        # atm vol: x=0 -> vol^2 = a + b(-rho*m + sqrt(m^2 + sigma^2))
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma
        self.atm_vol = atm_vol
        self.atm_ref = atm_ref

    def compute_atm_vol(self):
        return sqrt(self.variance(0))

    def variance(self, moneyness):
        return self.a + self.b * (self.rho * (moneyness - self.m) + np.sqrt((moneyness - self.m) ** 2 + self.sigma ** 2))

    def plot(self):
        strikes = np.linspace(0.1*self.atm_ref, 10 * self.atm_ref, 10000)
        moneyness = np.log(strikes/self.atm_ref)
        vars = self.variance(moneyness)
        plt.figure()
        plt.ylim(-0.15, 0.60)
        plt.xlim(np.log(0.2), np.log(5))
        plt.ylabel(r"Variance $\sigma^2$")
        plt.xlabel("Log-Moneyness")
        plt.plot(moneyness, vars, label="SVI Variance", linewidth=3)
        left_x = np.log(0.0001)
        left_x2 = np.log(0.0002)
        left_y = self.variance(left_x)
        left_y2 = self.variance(left_x2)
        slope_left = (left_y2-left_y) / (left_x2 - left_x)
        right_x = np.log(1000)
        right_x2 = np.log(1001)
        right_y = self.variance(right_x)
        right_y2 = self.variance(right_x2)
        slope_right = (right_y2-right_y) / (right_x2 - right_x)
        plt.plot([left_x, right_x], [left_y, left_y + (right_x-left_x)*slope_left], "g--", linewidth=1)
        plt.plot([left_x, right_x], [right_y - (right_x - left_x) * slope_right, right_y], "g--", linewidth=1)
        plt.scatter(self.m, self.a+0.005, facecolors='none', edgecolors='r', s=50)
        plt.legend()

        plt.grid(True)
        plt.title("SVI " + rf"$a={self.a:.2f}, b={self.b:.2f}, \rho={self.rho:.2f}, m={self.m:.2f}, \sigma={self.sigma:.2f}$")
        plt.show()
"""
class SVIJWSkew(SVISkew):
    def __init__(self, atm_vol, skew, min_vol, call_wing, put_wing, atm_ref=1, t=1):
        total_var = atm_vol*atm_vol*t
        min_var = min_vol * min_vol
        beta = ...
        alpa = ...
        b = np.sqrt(total_var) / 2 * (call_wing + put_wing)
        rho = 1 - put_wing * np.sqrt(total_var) / b
        m = ...
        sigma = alpha * m
        a = min_var * t - b * ...
        super().__init__(a, b, rho, m, sigma, atm_vol, atm_ref)


class SSVISurface:
    def __init__(self, spot, vol, rho, eta, gamma):
        # only the powerlaw parameterisation
        assert (eta > 0 and eta * (1 + abs(rho)) <= 2.0)
        assert (0 < gamma < 1)
        self.spot = spot
        self.vol = vol
        self.rho = rho
        self.eta = eta
        self.gamma = gamma

    def phi(self, theta):
        # theta = vol*vol*t
        return self.eta * pow(theta, -self.gamma) * pow(1 + theta, self.gamma - 1)

    def iv(self, strike, mat):
        theta = self.vol * self.vol * mat
        phivalk = self.phi(theta) * log(strike/self.spot)
        totalvar = theta / 2 * (1 + self.rho * phivalk + sqrt(pow(phivalk + self.rho, 2) + 1 - self.rho*self.rho))
        return sqrt(totalvar / mat)

    def plot_ivs(self, strikes, mats):
        ivs = self.iv(strike, mat)

"""

def plot_skews(skew1, skew2, points = []):
    strikes = np.linspace(0.1 * skew1.atm_ref, 10 * skew1.atm_ref, 10000)
    moneyness = np.log(strikes / skew1.atm_ref)
    vars1 = skew1.variance(moneyness)
    vars2 = skew2.variance(moneyness)
    plt.figure()
    plt.ylim(-0.15, 0.60)
    plt.xlim(np.log(0.2), np.log(5))
    plt.ylabel(r"Variance $\sigma^2$")
    plt.xlabel("Log-Moneyness")
    plt.plot(moneyness, vars1, label="Base", linewidth=2)
    plt.plot(moneyness, vars2, label="Shift", linewidth=2)
    for point in points:
        plt.scatter(point[0], point[1], label=point[2])
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_svi():
    a = -0.101154
    b = 0.418812
    rho = -0.349378
    m = -0.100222
    sigma = 0.334297
    atm_vol = 0.17419942
    atm_ref = 1.0
    skew = SVISkew(a, b, rho, m, sigma, atm_vol, atm_ref)
    skew.plot()


def scenario_atm():
    a = -0.101154
    b = 0.418812
    rho = -0.349378
    m = -0.100222
    sigma = 0.334297
    atm_vol = 0.17419942
    atm_ref = 1.0
    skew = SVISkew(a, b, rho, m, sigma, atm_vol, atm_ref)

    atm_volc = skew.compute_atm_vol()
    print(atm_volc, atm_vol)

    """
    # THIS IS ONLY THE ATM Shift on A
    shift = 1.1
    new_atm_vol = 1.1 * atm_volc

    # vol^2(x) = a + b[rho*(x-m) + sqrt{(x-m)^2 + sigma^2 }]
    new_a = new_atm_vol ** 2 - b * (-m*rho + sqrt(m**2 + sigma**2))
    print(new_a)
    newskew = SVISkew(new_a, b, rho, m, sigma, new_atm_vol, atm_ref)
    print(new_atm_vol, newskew.compute_atm_vol())
    # skew.plot()
    plot_skews(skew, newskew)
    """

    # NOW FOLLOWS THE FULL 3-Point Shift
    itm_shift = 1.0
    atm_shift = 1.0
    otm_shift = 1.4
    itm_point = 0.6
    atm_strk = 1.0
    otm_strk = 1.2

    itm_x = log(itm_point)
    atm_x = log(atm_strk)
    otm_x = log(otm_strk)
    var_atm = skew.compute_atm_vol() **2 * atm_shift**2
    var_itm = skew.variance(itm_x) * itm_shift**2
    var_otm = skew.variance(otm_x) * otm_shift**2
    points = [[itm_x, var_itm, "ITM"], [otm_x, var_otm, "OTM"], [atm_x, var_atm, "ATM"]]
    # square roots
    f1_itm = sqrt((itm_x - m)**2 + sigma**2)
    f1_atm = sqrt((atm_x - m) **2 + sigma**2)
    f1_otm = sqrt((otm_x - m)**2 + sigma**2)
    # sqrt vs var
    diff_f1_d1 = (f1_itm - f1_atm) * (var_atm - var_otm) - (f1_atm - f1_otm) * (var_itm - var_atm)
    # x vs var
    diff_x1_d1 = - (itm_x - atm_x) * (var_atm - var_otm) + (atm_x - otm_x) * (var_itm - var_atm)
    # new parameter rho
    new_rho = diff_f1_d1 / diff_x1_d1
    # n3
    diff_f1_low_atm = new_rho * (itm_x - atm_x) + f1_itm - f1_atm
    # new parameter b
    new_b = (var_itm - var_atm) / diff_f1_low_atm
    # new parameter a - fit to itm point
    new_a = var_itm - new_b * (new_rho * (itm_x - m) + f1_itm)
    new_skew = SVISkew(new_a, new_b, new_rho, m, sigma, sqrt(var_atm), atm_ref)

    plot_skews(skew, new_skew, points)





if __name__ == "__main__":
    """
    a = -0.101154
    b = 0.418812
    rho = -0.349378
    m = -0.100222
    sigma = 0.334297
    atm_vol = 0.17419942
    atm_ref = 79.945677
    """
    # plot_svi()
    scenario_atm()
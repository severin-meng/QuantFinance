import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Interpolator:
    def __init__(self, x_left, x_right, y_left, y_right, gamma=1.0):
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right

        self.asymmetry = gamma

        assert self.x_left < self.x_right

    def smoothstep(self, x):
        assert self.x_right >= x >= self.x_left
        t = ((x - self.x_left)/ (self.x_right - self.x_left)) ** self.asymmetry
        return self.y_left + (self.y_right - self.y_left) * t**2 * (3 - 2*t)

    def smootherstep(self, x):
        assert self.x_right >= x >= self.x_left
        t = ((x - self.x_left)/ (self.x_right - self.x_left))**self.asymmetry
        # second derivative of smootherstep is
        # 60t - 180t^2 + 120t^3
        return self.y_left + (self.y_right - self.y_left) * t**3 * (10 - 15 * t + 6 * t**2)


class Bumper:
    def __init__(self, x_mid, x_width, height, asymmetry=0.0):
        self.x_mid = x_mid
        self.x_width = x_width
        self.height = height
        self.asymmetry = asymmetry

    def gaussian(self, x):
        return np.exp(- (x - self.x_mid)**2 / (2 * self.x_width**2)) * self.height * (1 + self.asymmetry * (x - self.x_mid))



class VarianceSlice:
    def __init__(self, tte=1):
        self.label='Base'
        self.tte=tte
        self.k_space = np.linspace(-3, 3, 500)

    def variance(self, k):
        raise NotImplementedError

    def slope(self, k, eps=1e-5):
        return (self.variance(k + eps) - self.variance(k-eps)) / (2 * eps)

    def convexity(self, k, eps=1e-5):
        return (self.variance(k + eps) + self.variance(k-eps) - 2*self.variance(k)) / (eps * eps)

    def density_sign(self, k):
        # this is the g-function from gatheral's paper
        dwdk = self.slope(k)
        w = self.variance(k)
        term1 = (1 - k * dwdk / (2 * w)) ** 2
        term2 = - dwdk**2 / 4 * (1 / w + 1/4)
        term3 = self.convexity(k) / 2
        return term1 + term2 + term3

    def density(self, k):
        w = self.variance(k)
        d_minus_k = -(k + w/2) / np.sqrt(w)
        return self.density_sign(k) / np.sqrt(2 * np.pi * w) * np.exp(-d_minus_k**2/2)

    def plot_variance(self):
        # variance is function of log-moneyness k = log(Fwd/Strike)
        variances = self.variance(self.k_space)

        plt.figure()
        plt.title("Total implied variance")
        plt.plot(self.k_space, variances, label=self.label)
        plt.legend()
        plt.xlabel("Log-moneyness log(F/K)")
        plt.grid()
        plt.show()

    def plot_vol(self):
        # variance is function of log-moneyness k = log(Fwd/Strike)
        vols = np.sqrt(self.variance(self.k_space))

        plt.figure()
        plt.title("Total implied vol")
        plt.plot(self.k_space, vols, label=self.label)
        plt.legend()
        plt.xlabel("Log-moneyness log(F/K)")
        plt.grid()
        plt.show()

    def plot_density_sign(self):
        # density sign is function of log-moneyness k = log(Fwd/Strike)
        density_signs = self.density_sign(self.k_space)

        plt.figure()
        plt.title("Sign of marginal density")
        plt.plot(self.k_space, density_signs, label=self.label)
        plt.legend()
        plt.xlabel("Log-moneyness log(F/K)")
        plt.grid()
        plt.show()

    def plot_density(self):
        # density sign is function of log-moneyness k = log(Fwd/Strike)
        density = self.density(self.k_space)

        plt.figure()
        plt.title("Marginal density")
        plt.plot(self.k_space, density, label=self.label)
        plt.legend()
        plt.xlabel("Log-moneyness log(F/K)")
        plt.grid()
        plt.show()

    def plot_slope(self):
        slope = self.slope(self.k_space)
        plt.figure()
        plt.title("Slope of Variance")
        plt.plot(self.k_space, slope, label=self.label)
        plt.legend()
        plt.xlabel("Log-moneyness log(F/K)")
        plt.grid()
        plt.show()

    def plot_convexity(self):
        convexity = self.convexity(self.k_space)
        plt.figure()
        plt.title("Convexity of Variance")
        plt.plot(self.k_space, convexity, label=self.label)
        plt.legend()
        plt.xlabel("Log-moneyness log(F/K)")
        plt.grid()
        plt.show()


class SVIParams(VarianceSlice):
    def __init__(self, a, b, rho, m, s, tte=1/365):
        super().__init__(tte)
        self.a = a  # vertical level of variance
        self.b = b  # overall slope of both put and call wings
        self.rho = rho  # rotation of smile: high rho means high right wing slope
        self.m = m  # log moneyness translation
        self.s = s  # high s reduces atm curvature

        self.label = 'SVI'
        self.tte = 1/365

        assert self.b >= 0
        assert abs(self.rho) < 1
        assert self.s > 0
        assert self.a + self.b * self.s * np.sqrt(1 - self.rho**2) >= 0

    def variance(self, k):
        # k is log moneyness: log(forward/strike)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.s**2))


class SVILocalBump(SVIParams):
    def __init__(self, a, b, rho, m, s, k_start, k_mid, k_end, bump_size):
        super().__init__(a, b, rho, m, s)
        self.label = 'SVI with Bump'

        self.k_start = k_start
        self.k_mid = k_mid
        self.k_end = k_end
        self.bump_size = bump_size
        assert self.k_start < self.k_mid < self.k_end

        self.left_interpolator = Interpolator(self.k_start, self.k_mid, 0, self.bump_size, 1.1)
        self.right_interpolator = Interpolator(self.k_mid, self.k_end, self.bump_size, 0, 0.9)
        self.global_bumper = Bumper(self.k_mid, self.k_end - self.k_start, self.bump_size)

    def variance2(self, k):
        if isinstance(k, np.ndarray):
            ret_val = []
            for k_value in k:
                ret_val.append(self._variance_scalar(k_value))
            return np.array(ret_val)
        else:
            ret_val = self._variance_scalar(k)
        return ret_val

    def variance(self, k):
        base_var = super().variance(k)
        add_var = self.global_bumper.gaussian(k)
        return base_var + add_var

    def _variance_scalar(self, k):
        base_var = super().variance(k)
        if k <= self.k_start or k >= self.k_end:
            return base_var
        elif self.k_start < k < self.k_mid:
            return base_var + self.left_interpolator.smoothstep(k)
        else:
            return base_var + self.right_interpolator.smoothstep(k)


class SviBump(SVIParams):
    def __init__(self, a, b, rho, m, s, bumper):
        super().__init__(a, b, rho, m, s)
        self.label = 'SVI with Bump'
        self.bumper = bumper

    def variance(self, k):
        base_var = super().variance(k)
        add_var = self.bumper.gaussian(k)
        return base_var + add_var



if __name__ == '__main__':
    params = SVIParams(0.1, 0.3, -0.5, 0, 0.2)
    # params.plot_variance()
    # params.plot_density_sign()

    # params = SVILocalBump(0.1, 0.1, -0.5, 0, 0.2, 0.12, 0.15, 0.18, 0.004)

    class NoBump:
        def gaussian(self, x):
            return 0
    bumper = NoBump()
    # bumper = Bumper(0.5, 0.7, 0.15, asymmetry=0)
    params = SviBump(0.04, 0.1, -0.3, 0, 0.2, bumper)
    # params.plot_variance()
    params.plot_vol()
    # params.plot_density_sign()
    params.plot_density()
    # params.plot_slope()
    # params.plot_convexity()

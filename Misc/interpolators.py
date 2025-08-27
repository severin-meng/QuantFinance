import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly


def plot_smoothsteps(x1=0, x2=1, x3 = 2, y1=0, y2=1, y3 = 2.3):
    t_space = np.linspace(0, 1, 100)
    y1_smoothstep = y1 + (y2 - y1) * t_space**2 * (3 - 2*t_space)
    y2_smoothstep = y2 + (y3 - y2) * t_space**2 * (3 - 2*t_space)
    y1_smootherstep = y1 + (y2 - y1) * (6 * t_space**5 - 15 * t_space**4 + 10 * t_space**3)
    y2_smootherstep = y2 + (y3 - y2) * (6 * t_space ** 5 - 15 * t_space ** 4 + 10 * t_space ** 3)
    plt.figure()
    plt.plot(x1 + t_space*(x2-x1), y1_smoothstep, label='smoothstep')
    plt.plot(x2 + t_space*(x3-x2), y2_smoothstep, label='smoothstep')
    plt.plot(x1 + t_space*(x2-x1), y1_smootherstep, label='smoothERstep')
    plt.plot(x2 + t_space*(x3-x2), y2_smootherstep, label='smoothERstep')
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.scatter(x3, y3)
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def plot(x, y, label=''):
    plt.figure()
    plt.plot(x, y, label=label)
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def plot_cutoff(mid, left, right):
    x_space = np.linspace(left, right, 300)
    denominator = 1 - (x_space-mid)**2
    res = np.exp(-1 / denominator)
    plot(x_space, res, label='Cutoff')

def plot_second_deriv_cutoff(mid, left, right, eps=1e-6):
    x_space = np.linspace(left, right, 300)
    denominator_l = 1 - (x_space-eps - mid) ** 2
    denominator_m = 1 - (x_space - mid) ** 2
    denominator_r = 1 - (x_space+eps - mid) ** 2
    res = np.exp(-1/denominator_l) + np.exp(-1/denominator_r) - 2 * np.exp(-1/denominator_m)
    plot(x_space, res/eps**2, label='Cutoff Convexity')


def plot_gaussian(mid, width, eps=1e-6):
    x_space = np.linspace(mid-5*width, mid + 5*width, 300)
    def func(x):
        return np.exp(-(x - mid) ** 2 / (2 * width ** 2))
    res = func(x_space)
    plot(x_space, res, label='Gaussian')
    res_r = func(x_space + eps)
    res_l =  func(x_space - eps)
    plot(x_space, (res_r + res_l - 2*res)/eps**2, label='Gaussian Convexity')


def plot_skewed_gaussian(mid, width, skew, eps=1e-6):
    x_space = np.linspace(mid - 5 * width, mid + 5 * width, 300)
    def func(x):
        return np.exp(-(x - mid) ** 2 / (2 * width ** 2)) * (1 + skew * (x - mid))
    res = func(x_space)
    plot(x_space, res, label='Skewed Gaussian')
    res_r = func(x_space + eps)
    res_l = func(x_space - eps)
    plot(x_space, (res_r + res_l - 2 * res) / eps ** 2, label='Skewed Gaussian Convexity')


def plot_logistic_gaussian(mid, width, skew, eps=1e-6):
    x_space = np.linspace(mid - 5 * width, mid + 5 * width, 300)
    def func(x):
        return  np.exp(-(x - mid) ** 2 / (2 * width ** 2)) / (1 + np.exp(-skew * (x - mid)))
    res = func(x_space)
    plot(x_space, res, label='Logistic Gaussian')
    res_r = func(x_space + eps)
    res_l = func(x_space - eps)
    plot(x_space, (res_r + res_l - 2 * res) / eps ** 2, label='Logistic Gaussian Convexity')


def plot_polynomial_gaussian(mid, width, skew1, skew2, eps=1e-6):
    x_space = np.linspace(mid - 5 * width, mid + 5 * width, 300)
    def func(x):
        return  np.exp(-(x - mid) ** 2 / (2 * width ** 2)) * (1 + skew1 * (x - mid) + skew2 * (x - mid)**2)
    res = func(x_space)
    plot(x_space, res, label='Poly Gaussian')
    res_r = func(x_space + eps)
    res_l = func(x_space - eps)
    plot(x_space, (res_r + res_l - 2 * res) / eps ** 2, label='Poly Gaussian Convexity')


if __name__ == '__main__':
    # plot_smoothsteps()
    # plot_cutoff(0, -1, 1)
    # plot_second_deriv_cutoff(0, -1, 1)
    # plot_gaussian(0, 2)
    # plot_skewed_gaussian(0, 2, -0.6)
    plot_logistic_gaussian(0, 2, 1)
    plot_polynomial_gaussian(0, 2, 1, 0.2)
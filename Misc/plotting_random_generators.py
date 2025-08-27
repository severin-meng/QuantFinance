"""
Exploring various very simple (and some very bad) pseudo-random number generators
"""

import matplotlib.pyplot as plt
import numpy as np


def logistic_func(x, mu=4):
    return mu * x * (1 - x)


def plot_logistic_map():
    x_arr = np.linspace(0, 1, 1000)
    y_arr = logistic_func(x_arr)

    plt.figure()
    plt.plot(x_arr, y_arr)
    plt.grid(True)
    plt.show()


def run_logistic_map(x_0=0.7, runs=1000):
    y = [x_0]
    for i in range(runs):
        y.append(logistic_func(y[i]))

    x = list(range(runs+1))

    plt.figure()
    plt.scatter(x, y, s=1)
    # plt.plot(x[:100], y[:100])
    plt.grid(True)
    plt.show()


def congruental_map(m, a=5**17, M=2**40, c=0):
    return (a*m + c) % M


def ran0(m):
    return congruental_map(m, a=7**5, M=2**31-1, c=0)


def plot_congruental_map(m0=1, runs=1000):
    y = [m0]
    x = list(range(runs+1))
    for i in range(runs):
        y.append(ran0(y[i]))

    plt.figure()
    plt.scatter(x, y, s=1)
    # plt.plot(x[:100], y[:100])
    plt.grid(True)
    plt.show()


def mrg32k3a(y1, y2,)



if __name__ == '__main__':
    # plot_logistic_map()
    # run_logistic_map()
    plot_congruental_map()

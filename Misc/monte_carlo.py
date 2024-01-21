import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_integrator(func, sample_size):
    samples = np.random.rand(sample_size)
    return np.average(func(samples))


def ex_i(x):
    # exact result: 26/3
    return 2 * (2 * x + 1)**2


def ex_ii(x):
    # exact result: sqrt(pi) / 2
    return 1/x**2 * np.exp(-(1/x - 1)**2)


def ex_iii(x):
    # exact result: 3
    return np.sqrt(2) / np.sqrt(np.pi) * (1-x)**4 / x**6 * np.exp(- (1/x - 1)**2 / 2)


if __name__ == "__main__":
    n_range = np.logspace(1, 8, dtype=int)

    exi_mc = []
    exii_mc = []
    exiii_mc = []
    for n in n_range:
        exi_mc.append(monte_carlo_integrator(ex_i, n))
        exii_mc.append(monte_carlo_integrator(ex_ii, n))
        exiii_mc.append(monte_carlo_integrator(ex_iii, n))

    expected_error = [1 / np.sqrt(n) for n in n_range]
    exi_mc_err = [np.abs(26./3 - mc_res) for mc_res in exi_mc]
    exii_mc_err = [np.abs(np.sqrt(np.pi)/2 - mc_res) for mc_res in exii_mc]
    exiii_mc_err = [np.abs(3 - mc_res) for mc_res in exiii_mc]

    plt.loglog(n_range, expected_error, label=r'$O(N^{-1/2})$')
    plt.loglog(n_range, exi_mc_err, '.', markeredgecolor='none', label='Monte Carlo')
    plt.legend(loc='best')
    plt.xlabel(r"$N$")
    plt.ylabel("Error")
    plt.title(r"Monte Carlo Integration of $x^2$")
    plt.show()

    plt.loglog(n_range, expected_error, label=r'$O(N^{-1/2})$')
    plt.loglog(n_range, exii_mc_err, '.', markeredgecolor='none', label='Monte Carlo')
    plt.legend(loc='best')
    plt.xlabel(r"$N$")
    plt.ylabel("Error")
    plt.title(r"Monte Carlo Integration of $e^{-x^2}$")
    plt.show()

    plt.loglog(n_range, expected_error, label=r'$O(N^{-1/2})$')
    plt.loglog(n_range, exiii_mc_err, '.', markeredgecolor='none', label='Monte Carlo')
    plt.legend(loc='best')
    plt.xlabel(r"$N$")
    plt.ylabel("Error")
    plt.title(r"Monte Carlo Integration of $x^4e^{-x^2/2}$")
    plt.show()

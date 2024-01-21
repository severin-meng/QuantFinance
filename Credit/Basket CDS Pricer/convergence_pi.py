import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (7, 7)})

from copula_sampling import get_random_numbers


def calc_pi(method='pseudo', exponent=10, dim=2, iterative=False):
    """
    :param method: method used to sample (random) numbers for Monte Carlo. 'pseudo' for numpy pseudo-random number
                   generation (Mersenne-Twister), 'sobol' or 'halton' for respective scrambled low-discrepancy sequence
    :param exponent: base 2 exponent of number of samples to draw
    :param dim: dimension of samples, 2 for calculating pi
    :param iterative: set to True to return iterative results with increasing simulations,
                      set to False to return single result only
    :return: simulation steps if iterative is True, result of Monte Carlo calculation of pi
    """
    random_nbrs = np.array(get_random_numbers(exponent=exponent, dim=dim, method=method, distr='uniform'),
                           dtype=np.float32)
    eligible_nbrs = np.where(np.linalg.norm(random_nbrs, axis=0) <= 1, 1, 0)
    if iterative:
        max_pow = exponent
        min_pow = 7
        steps = np.logspace(min_pow, max_pow, (max_pow - min_pow + 1) * 2, endpoint=True, base=2, dtype=int)
        my_pies = []
        for step in steps:
            my_pies.append(np.mean(eligible_nbrs[:step]))
        return steps, my_pies
    else:
        my_pi = np.mean(eligible_nbrs)
        return my_pi


def plot_pi_calc(exponent=9, dim=2):
    sobol_nbrs = get_random_numbers(exponent=exponent, dim=dim, method='sobol', distr='uniform')
    mersenne_nbrs = get_random_numbers(exponent=exponent, dim=dim, method='pseudo', distr='uniform')
    # inside_nbrs = np.where(np.linalg.norm(sobol_nbrs, axis=0) <= 1, 1, 0)
    # outside_nbrs = np.where(np.linalg.norm(sobol_nbrs, axis=0) <= 1, 1, 0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    c_sobol = np.where(np.linalg.norm(sobol_nbrs-0.5, axis=0) <= 0.5, 'g', 'r')
    c_pseudo = np.where(np.linalg.norm(mersenne_nbrs-0.5, axis=0) <= 0.5, 'g', 'r')

    angles = np.linspace(0, 2*np.pi, 1000)
    circle_x = np.cos(angles)*0.5 + 0.5
    circle_y = np.sin(angles)*0.5 + 0.5

    ax[0].scatter(sobol_nbrs[0], sobol_nbrs[1], c=c_sobol, marker='o')
    ax[0].plot(circle_x, circle_y, linewidth=5)
    ax[0].set_title("Sobol low-discrepancy sampling", fontdict={'fontsize': 'large', 'fontweight': 'bold'})

    ax[1].scatter(mersenne_nbrs[0], mersenne_nbrs[1], c=c_pseudo, marker='o')
    ax[1].plot(circle_x, circle_y, linewidth=5)
    ax[1].set_title("Mersenne-Twister pseudo-random sampling", fontdict={'fontsize': 'large', 'fontweight': 'bold'})

    plt.show()


def plot_convergence(min_exp=5, max_exp=15, iterative=False):
    """
    :param min_exp: minimum base 2 exponent of number of simulations
    :param max_exp: maximum base 2 exponent of number of simulations
    :param iterative: set to True to plot iterative results with increasing simulations (faster),
                      set to False to return single result only
    :return: None, plot convergence of sobol, halton and pseudo-random Monte Carlo computation of pi
    """
    if iterative:
        steps, sobols = calc_pi(method='sobol', exponent=max_exp, iterative=True)
        # _, haltons = calc_pi(method='halton', exponent=max_exp, iterative=True)
        _, pseudos = calc_pi(method='pseudo', exponent=max_exp, iterative=True)
    else:
        exp_range = np.arange(min_exp, max_exp, 1)
        sobols = []
        pseudos = []
        # haltons = []
        steps = 2 ** exp_range
        for exponent in exp_range:
            sobols.append(calc_pi(method='sobol', exponent=exponent))
            # haltons.append(calc_pi(method='halton', exponent=exponent))
            pseudos.append(calc_pi(method='pseudo', exponent=exponent))

    exact_pi = np.pi/4
    plt.figure(figsize=(12, 7))
    plt.loglog(steps, np.abs(np.array(sobols) - exact_pi), marker='o', label='sobol')
    # plt.loglog(steps, np.abs(np.array(haltons) - exact_pi), marker='o', label='halton')
    plt.loglog(steps, np.abs(np.array(pseudos) - exact_pi), marker='o', label='pseudo')
    plt.loglog(steps, 0.3 * 1 / np.sqrt(steps), label=r'$O(N^{-1/2})$')
    plt.loglog(steps, 0.1*np.log(steps)**2 / steps, label=r'$O(log(N)^2 / N)$')
    plt.legend(loc='best')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Absolute Error')
    plt.title(r'Errors in calculating $\pi$')
    plt.show()


if __name__ == '__main__':
    plot_convergence(min_exp=5, max_exp=30, iterative=True)
    # plot_pi_calc()

import numpy as np
from scipy.stats.qmc import Halton, Sobol
from scipy.stats import norm
from scipy.special import stdtr, stdtrit
from uniform_to_default import default_time_func
import time
from itertools import combinations
import matplotlib.pyplot as plt


__all__ = ['gaussian_copula_sampling',
           'students_t_copula_sampling',
           'get_random_numbers']


def gaussian_copula_sampling(default_times_func, linear_correlation, n=5, method='pseudo', exponent=10, plot=False,
                             prnt=False, cholesky=False):
    """
    :param default_times_func: inverse cumulative default time function, input is 2**exponent X n uniforms
    :param linear_correlation: correlation matrix
    :param n: basket size, 5
    :param method: number generation method for sampling, either 'pseudo' for pseudo-random Mersenne-Twister
                   or 'sobol' or 'halton' for respective low-discrepancy sequence (scrambled)
    :param exponent: base 2 logarithm of number of simulations = 2^exponent
    :param plot: set to true to plot various scatter plots and histograms
    :param prnt: set to true to print time taken for various steps in the sampling process
    :param cholesky: set to true to use cholesky pseudo square-root of correlation,
                     set to false to use spectral decomposition of correlation
    :return: (2^exponent, n) co-dependent default times, sampled from a gaussian copula
    """
    tickers = linear_correlation.columns
    start = time.perf_counter()
    if cholesky:
        pseudo_sqrt = get_cholesky(linear_correlation)
    else:
        eigvals, eigvec = np.linalg.eigh(linear_correlation)
        pseudo_sqrt = np.matmul(eigvec, np.diag(np.sqrt(eigvals)))

    t0 = time.perf_counter()
    # draw normal uncorrelated random variates z
    iid_gaussians = get_random_numbers(method=method, distr='gauss', dim=n, exponent=exponent)
    t1 = time.perf_counter()

    # compute z_bar = A z -> these are now linearly correlated
    corr_gauss = np.matmul(pseudo_sqrt, iid_gaussians)
    t2 = time.perf_counter()

    # map z_bar to uniform via normal cdf
    corr_uniform = norm.cdf(corr_gauss)
    t3 = time.perf_counter()

    # map uniform to default times using marginal inverse cdf
    corr_defaults = default_times_func(corr_uniform)
    t4 = time.perf_counter()

    if plot:
        combos = list(combinations(np.arange(5), 2))
        nbr_plots = len(combos)
        fig, ax = plt.subplots(2, 5, figsize=(13, 5))
        for i in range(nbr_plots):
            a, b = combos[i]
            ax[i % 2, int(i/2)].scatter(norm.cdf(iid_gaussians[:][a]), norm.cdf(iid_gaussians[:][b]), marker='.', s=5)
        title_method = "(Pseudo-Random)" if method == 'pseudo' else f'({method})'
        fig.suptitle(f"Uniform Copula {title_method}")
        plt.show()

        fig, ax = plt.subplots(1, 5, figsize=(12, 2.5))
        for i in range(5):
            ax[i].hist(corr_uniform[i], density=True, bins=30)
        fig.suptitle('Correlated Uniform')
        plt.show()

        combos = list(combinations(np.arange(5), 2))
        nbr_plots = len(combos)
        fig, ax = plt.subplots(2, 5, figsize=(13, 5), constrained_layout=True)
        for i in range(nbr_plots):
            a, b = combos[i]
            ax[i % 2, int(i/2)].scatter(corr_uniform[a], corr_uniform[b], marker='.', s=5)
            ax[i % 2, int(i / 2)].set_xlabel(tickers[a])
            ax[i % 2, int(i / 2)].set_ylabel(tickers[b])
        fig.suptitle("Gaussian Copula")
        plt.show()

    if prnt:
        print(f"Cholesky decomp time: {(t0 - start):.2f}")
        print(f"iid Gaussians time: {(t1 - t0):.2f}")
        print(f"Corr Gaussians time: {(t2 - t1):.2f}")
        print(f"Corr uniforms time: {(t3 - t2):.2f}")
        print(f"corr defaults time: {(t4 - t3):.2f}")
    return corr_defaults.T


def students_t_copula_sampling(default_times_func, linear_correlation, dof=4, n=5, method='pseudo', exponent=10,
                               plot=False, prnt=False, direct_method=False, chi_pseudo=False, cholesky=False):
    """
    :param default_times_func: inverse cumulative default time function, input is 2**exponent X n uniforms
    :param linear_correlation: correlation matrix
    :param dof: degrees of freedom parameter for student-t copula
    :param n: basket size, 5
    :param method: number generation method for sampling, either 'pseudo' for pseudo-random Mersenne-Twister
                   or 'sobol' or 'halton' for respective low-discrepancy sequence (scrambled)
    :param exponent: base 2 logarithm of number of simulations = 2^exponent
    :param plot: set to true to plot various scatter plots and histograms
    :param direct_method: set to true to generate uncorrelated student-t from uniforms via student-t inverse cdf and
                          apply correlation to student-t numbers. not recommended, increases runtime
                          set to false to generate uncorrelated gaussians, correlate them, sample chi-squared number to
                          convert to student-t variates. recommended, much faster than student-t cdf
    :param chi_pseudo: set to true to use a pseudo-random number for the chi-squared variate. set to false to use
                       the same method as the "method" variable to generate the chi-squared variate.
    :param prnt: set to true to print time taken for various steps in the sampling process
    :param cholesky: set to true to use cholesky pseudo square-root of correlation,
                     set to false to use spectral decomposition of correlation
    :return: (2^exponent, n) co-dependent default times, sampled from a student-t copula
    """
    tickers = linear_correlation.columns
    start = time.perf_counter()
    if cholesky:
        pseudo_sqrt = get_cholesky(linear_correlation)
    else:
        eigvals, eigvec = np.linalg.eigh(linear_correlation)
        pseudo_sqrt = np.matmul(eigvec, np.diag(np.sqrt(eigvals)))
    t0 = time.perf_counter()
    if prnt:
        print(f"Cholesky decomp time: {(t0 - start):.2f}")

    if direct_method:
        # draw uniform uncorrelated random variates u
        iid_uniforms = get_random_numbers(method=method, distr='uniform', dim=n, exponent=exponent)
        t1 = time.perf_counter()

        # compute uncorrelated t-
        iid_t_student = stdtrit(dof, iid_uniforms)
        t2 = time.perf_counter()

        if prnt:
            print(f"iid uniforms time: {(t1 - t0):.2f}")
            print(f"iid t student time: {(t2 - t1):.2f}")

        corr_student_t = np.matmul(pseudo_sqrt, iid_t_student)
    else:
        if not chi_pseudo:
            # sampling 5 + 4 iid gaussians to account for the required chi squared variable, improves convergence
            iid_gaussians = get_random_numbers(method=method, distr='gauss', dim=n+dof, exponent=exponent)
            t1 = time.perf_counter()

            # compute z_bar = A z -> these are now linearly correlated
            corr_gauss = np.matmul(pseudo_sqrt, iid_gaussians[:n, :])
            iid_chisquared = iid_gaussians[n:, :]
            t2 = time.perf_counter()
        else:
            # draw normal uncorrelated random variates z
            iid_gaussians = get_random_numbers(method=method, distr='gauss', dim=n, exponent=exponent)
            t1 = time.perf_counter()

            # compute z_bar = A z -> these are now linearly correlated
            corr_gauss = np.matmul(pseudo_sqrt, iid_gaussians)
            t2 = time.perf_counter()

            iid_chisquared = get_random_numbers(method='pseudo', distr='gauss', dim=dof, exponent=exponent)

        iid_chisquared = np.sum(iid_chisquared**2, axis=0)
        chi_square_factor = np.sqrt(dof/iid_chisquared)
        t3 = time.perf_counter()

        if plot:
            combos = list(combinations(np.arange(5), 2))
            nbr_plots = len(combos)
            fig, ax = plt.subplots(2, 5, figsize=(13, 5))
            for i in range(nbr_plots):
                a, b = combos[i]
                ax[i % 2, int(i / 2)].scatter(norm.cdf(iid_gaussians[a]), norm.cdf(iid_gaussians[b]), marker='.', s=5)
            title_method = "(Pseudo-Random)" if method == 'pseudo' else f'({method})'
            fig.suptitle(f"Uniform Copula {title_method}")
            plt.show()

            plt.hist(corr_gauss[0], density=True, bins=30)
            plt.title('corr gauss')
            plt.show()

            plt.hist(iid_chisquared, density=True, bins=30)
            plt.title(f'chi squared (pseudo={chi_pseudo})')
            plt.show()

            plt.hist(chi_square_factor, density=True, bins=30)
            plt.title(f'factor from chi squared (pseudo={chi_pseudo})')
            plt.show()

        if prnt:
            print(f"iid Gaussians time: {(t1 - t0):.2f}")
            print(f"iid gaussian shape: {iid_gaussians.shape}")
            print(f"Corr Gaussians time: {(t2 - t1):.2f}")
            print(f"iid chi time: {(t3 - t2):.2f}")
            print(f"chi factor shape: {chi_square_factor.shape}")

        # student-t number x = sqrt(nu/s) * z_bar
        corr_student_t = corr_gauss * chi_square_factor
    t4 = time.perf_counter()

    # map x to uniform via the cdf of student ts distribution
    corr_uniform = stdtr(dof, corr_student_t)
    t5 = time.perf_counter()

    # map uniform to default times using marginal inverse cdf
    corr_defaults = default_times_func(corr_uniform)
    t6 = time.perf_counter()

    if plot:
        fig, ax = plt.subplots(1, 5, figsize=(12, 2.5))
        for i in range(5):
            ax[i].hist(corr_student_t[i], density=True, bins=30)
        fig.suptitle('Student T')
        plt.show()

        fig, ax = plt.subplots(1, 5, figsize=(12, 2.5))
        for i in range(5):
            ax[i].hist(corr_uniform[i], density=True, bins=30)
        fig.suptitle('Correlated Uniform')
        plt.show()

        combos = list(combinations(np.arange(5), 2))
        nbr_plots = len(combos)
        fig, ax = plt.subplots(2, 5, figsize=(13, 5), constrained_layout=True)
        for i in range(nbr_plots):
            a, b = combos[i]
            ax[i % 2, int(i/2)].scatter(corr_uniform[a], corr_uniform[b], marker='.', s=5)
            ax[i % 2, int(i / 2)].set_xlabel(tickers[a])
            ax[i % 2, int(i / 2)].set_ylabel(tickers[b])
        fig.suptitle("Student-t Copula")
        plt.show()

    if prnt:
        print(f"Corr t time: {(t4 - t2):.2f}")
        print(f"Corr uniforms time: {(t5 - t4):.2f}")
        print(f"Corr defaults time: {(t6 - t5):.2f}")

    return corr_defaults.T


def get_random_numbers(exponent=10, dim=5, method='pseudo', distr='gauss'):
    """
    :param exponent: base 2 logarithm of number of simulations = 2^exponent
    :param dim: dimension of sample
    :param method: number generation method, 'pseudo' for pseudo-random Mersenne-Twister, 'sobol' or 'halton' for
                   respective scrambled low-discrepancy sequence
    :param distr: distribution to sample from, either 'gauss' or 'uniform'
    :return: sampled pseudo-random or low-discrepancy numbe of shape (dim, 2^exponent)
    """

    assert distr in ('gauss', 'uniform')
    assert method in ('sobol', 'halton', 'pseudo')

    # options: pseudo, halton, sobol
    if method == 'pseudo':
        n = pow(2, exponent)
        gen = np.random.default_rng()
        if distr == 'gauss':
            return gen.standard_normal((dim, n))
        if distr == 'uniform':
            return gen.random((dim, n))
    elif method == 'sobol':
        gen = Sobol(dim, bits=64, seed=None)
        nbrs = gen.random_base2(exponent).T
        if distr == 'gauss':
            return norm.ppf(nbrs)
        if distr == 'uniform':
            return nbrs
    elif method == 'halton':
        n = pow(2, exponent)
        gen = Halton(dim)
        nbrs = gen.random(n).T
        if distr == 'gauss':
            return norm.ppf(nbrs)
        if distr == 'uniform':
            return nbrs


def get_cholesky(linear_correlation):
    """
    :param linear_correlation: linear correlation matrix M
    :return: cholesky decomposition A such that M = A A.T
    """
    return np.linalg.cholesky(linear_correlation)


def run_test():
    """
    :return: test function to test the speed and functionality of gaussian and student-t copula sampling
    """
    from copula_fitter import t_copula_fit_dof
    from correlations import get_correlations
    from hazard_rate_bootstrapping import fit_hazard_rates
    max_dof = t_copula_fit_dof(None, from_file=True)
    hazard_df = fit_hazard_rates(None, from_file=True)
    default_func = default_time_func(hazard_df, interp=True)
    correlation_t = get_correlations(None, corr_type='kendall', density_estimation=True, from_file=True)
    correlation_g = get_correlations(None, corr_type='pearson', density_estimation=True, from_file=True)
    expo = 10
    # sampled_def_times = gaussian_copula_sampling(
    #     default_func, correlation_g, exponent=expo, method='pseudo', plot=True, prnt=True)
    sampled_def_times_t = students_t_copula_sampling(
        default_func, correlation_t, dof=max_dof, exponent=expo, method='sobol', plot=True, direct_method=False,
        prnt=True, chi_pseudo=False)
    print("DONE")


if __name__ == '__main__':
    run_test()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.special import gamma
import pandas as pd

import os

__all__ = ['t_copula_fit_dof']


def plot_mle_result(parameter, values, max_val, max_param, offset=False, save=False):
    """
    :param parameter: list of all degrees of freedom parameters tested for maxmium likelihood
    :param values: log likelihood of t-copula per degree of freedom parameter evaluated on historical pseudo-uniforms
    :param max_val: maxium over all log likelihoods
    :param max_param: degrees of freedom parameter that maximises the log likelihood
    :param offset: set to True to adjust y-axis
    :param save: set to True to save the plot of the max log likelihood
    :return: None, plot log likelihood to estimate optimal degrees of freedom parameter
    """
    if offset:
        asympt_val = values[-1]
        max_val = max_val - asympt_val
        values = values - asympt_val
    plt.figure(1, figsize=(7, 4))
    plt.plot(parameter, values)
    plt.scatter(max_param, max_val, marker='x', label=rf'$\nu = {max_param}$')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('log Likelihood')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title("Max Likelihood Estimation for t-Copula")
    if save:
        plt.savefig('plots/MaxLikelihoodEstiCopulaTDof.png', dpi='figure')
    plt.show()


def t_copula_fit_dof(uniform_array, correlation=None, n=5, dof_range=np.arange(1, 50), plot=False, save=False,
                     from_file=False):
    """
    :param uniform_array: 5xN array of historical pseudo-uniforms to fit copula to, obtained from historical log
                          returns via kernel smoothing.
    :param correlation: 5x5 symmetric positive-semidefinite rank correlation matrix
    :param n: basket size, 5
    :param dof_range: range of degrees of freedom parameter to check for maximum log likelihood
    :param plot: if True, then it plots a graph of the log likelihood per degree of freedom parameter
    :param save: if True save the plot and the optimal degrees of freedom to a csv
    :param from_file: if True: read optimal degrees of freedom from file instead of performimg max log likelihood
    :return: degrees of freedom for which the student-t copula achieves maximum log likelihood
    """
    if from_file:
        path = os.sep.join([os.getcwd(), rf'\data\MLE_DOF.csv'])
        max_dof = pd.read_csv(path, index_col=0)
        return max_dof['MLE_DOF'][0]

    if correlation is None:
        correlation = np.identity(n)
    sqrt_det = np.sqrt(np.linalg.det(correlation))
    inv_corr = np.linalg.inv(correlation)

    likelihoods = []
    for dof in dof_range:
        likelihood = t_copula_likelihood(dof, uniform_array, sqrt_det, inv_corr, n)
        likelihoods.append(likelihood)

    argmax = np.argmax(likelihoods)
    max_val = likelihoods[argmax]
    max_dof = dof_range[argmax]
    if plot:
        plot_mle_result(dof_range, likelihoods, max_val, max_dof, save=save)
    if save:
        dct = {'MLE_DOF': max_dof}
        df = pd.DataFrame(dct, index=[0])
        df.to_csv('data/MLE_DOF.csv')
    return max_dof


def t_copula_likelihood(dof, uniform_array, sqrt_det, inv_corr, n=5):
    """
    :param dof: degrees of freedom of student-t copula
    :param uniform_array: array of historical pseudo-uniforms (kernel smoothing applied to historical log returns)
    :param sqrt_det: square root of the determinant of the correlation matrix.
    :param inv_corr: inverse correlation matrix
    :param n: basket size, 5
    :return: log likelihood of student-t copula summed over all historical pseudo-uniforms
    """
    gamma_dof_n = gamma(0.5*(dof + n))
    gamma_dof = gamma(0.5*dof)
    gamma_dof_1 = gamma(0.5 * dof + 0.5)

    prefactor = 1 / sqrt_det * gamma_dof_n / gamma_dof * (gamma_dof/gamma_dof_1) ** n

    likelihood = 0
    t_variates = t.ppf(uniform_array, dof)
    for t_variate in t_variates:
        likelihood += np.log(t_copula_density(dof, t_variate, prefactor, inv_corr, n))
    return likelihood


def t_copula_density(dof, t_variates, prefactor, inv_correlation, n=5):
    """
    :param dof: student-t degrees of freedom parameter
    :param t_variates: student-t distributed variates (inferred from pseudo-uniforms)
    :param prefactor: constant prefactor, see t_copula_likelihood, precomputed to optimize performance
    :param inv_correlation: inverse correlation matrix
    :param n: basket size, 5
    :return: copula density evaluated at t_variates
    """
    top_fraction = np.matmul(t_variates.T, np.matmul(inv_correlation, t_variates)) / dof
    numerator = (1 + top_fraction) ** (-(dof + n)/2)
    denominator = np.prod(1 + t_variates**2/dof) ** (-(dof + 1)/2)
    return prefactor * numerator / denominator


def test_run():
    """
    :return: tests the functionality of this module by calculating the max log likelihood
    """
    from returns_data import get_returns
    from kernel_density import fit_kernel_density
    from correlations import get_correlations
    returns_dat = get_returns(inspect=False, save=False)
    uniform_rets = fit_kernel_density(returns_dat, from_file=True)
    corr = get_correlations(returns_dat, corr_type='kendall', density_estimation=True, from_file=True)
    # can also scale correlation to see influence on optimal dof
    # ones = np.ones_like(corr)
    # corr_scaled = -0.3*(ones - corr) + corr
    max_df = t_copula_fit_dof(uniform_rets, correlation=corr, n=5, dof_range=np.arange(1, 50), plot=True, save=True,
                              from_file=True)
    print(max_df)


def test_run_kde_vs_gau():
    """
    :return: tests the functionality of this module by calculating the max log likelihood
    """
    from returns_data import get_returns
    from kernel_density import fit_kernel_density
    from correlations import get_correlations

    from scipy.stats import norm, zscore

    n = 5
    returns_dat = get_returns(inspect=False, save=False)
    uniform_rets = fit_kernel_density(returns_dat, from_file=True)
    dof_range = np.arange(1, 50)
    correlation = get_correlations(returns_dat, corr_type='kendall', density_estimation=True, from_file=True)

    sqrt_det = np.sqrt(np.linalg.det(correlation))
    inv_corr = np.linalg.inv(correlation)
    likelihoods_s = []
    for dof in dof_range:
        likelihood = t_copula_likelihood(dof, uniform_rets, sqrt_det, inv_corr, n)
        likelihoods_s.append(likelihood)
    argmax = np.argmax(likelihoods_s)
    max_val_s = likelihoods_s[argmax]
    max_dof_s = dof_range[argmax]

    tickers = list(returns_dat.columns)
    z_scores = zscore(returns_dat, ddof=1)
    pseudo_uniforms = {}
    for index, ticker in enumerate(tickers):
        data = z_scores[ticker]
        pseudo_uniform = norm.cdf(data)
        pseudo_uniforms[ticker] = pseudo_uniform
    gau_uniforms = pd.DataFrame(pseudo_uniforms)

    sqrt_det = np.sqrt(np.linalg.det(correlation))
    inv_corr = np.linalg.inv(correlation)
    likelihoods_g = []
    for dof in dof_range:
        likelihood = t_copula_likelihood(dof, gau_uniforms, sqrt_det, inv_corr, n)
        likelihoods_g.append(likelihood)
    argmax = np.argmax(likelihoods_g)
    max_val_g = likelihoods_g[argmax]
    max_dof_g = dof_range[argmax]

    plt.figure(1, figsize=(7, 4))
    plt.plot(dof_range, likelihoods_s, label='MLE using KDE inverse CDF', c='red')
    plt.plot(dof_range, likelihoods_g, label='MLE using Gaussian inverse CDF', c='blue')
    plt.scatter(max_dof_s, max_val_s, marker='x', label=rf'$\nu = {max_dof_s}$', c='red')
    plt.scatter(max_dof_g, max_val_g, marker='x', label=rf'$\nu = {max_dof_g}$', c='blue')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('log Likelihood')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title("Max Likelihood Estimation for t-Copula")
    plt.show()


def plot_t_v_gau():
    from scipy.stats import norm, t
    import matplotlib.pyplot as plt
    import numpy as np
    x_range = np.linspace(-7.5, 7.5, 500)
    gau = norm.pdf(x_range)
    dof_range = [1, 3, 6, 10]
    t_dict = {}
    for dof in dof_range:
        t_dict[dof] = t.pdf(x_range, dof)
    plt.figure()
    plt.plot(x_range, gau, label='normal', linewidth=2)
    for dof in dof_range:
        plt.plot(x_range, t_dict[dof], label=f"t, dof = {dof}")
    plt.legend(loc='best')
    plt.title("Gaussian vs Student-t PDF")
    plt.show()


if __name__ == '__main__':
    plot_t_v_gau()

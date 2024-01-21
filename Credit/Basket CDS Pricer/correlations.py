import numpy as np
import pandas as pd
from scipy.stats import zscore, norm as std_normal
from numpy.linalg import norm
import time

import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (7, 7)})


__all__ = [
    'get_correlations']


def get_correlations(log_returns, uniform_returns=None, corr_type='kendall', density_estimation=True, save=False,
                     from_file=False, plot=False):
    """
    :param log_returns: dataframe containing daily log returns of all corporates
    :param uniform_returns: historical pseudo uniforms (from log returns via kernel smoothing)
    :param corr_type: type of correlation, choose from pearson, spearman, kendall. Will always be converted to gaussian
    :param density_estimation: set to True to calculate correlations from uniform_returns, else calculate on
                               z-scores of log_returns
    :param save: set to True to save the computed correlations as csv
    :param from_file: set to True to read correlations from file instead of calculating them
    :param plot: set to True to plot the computed correlations
    :return: correlation matrix (nearest positive semidefinite symmetric real matrix with unit diagonal)
    """
    assert corr_type in ('pearson', 'spearman', 'kendall')

    if from_file:
        smoothed = 'smoothed' if density_estimation else 'direct'
        path = os.sep.join([os.getcwd(), rf'\data\correlations_{corr_type}_{smoothed}.csv'])
        near_corr = pd.read_csv(path, index_col=0)
        return near_corr
    if density_estimation:
        gaussian_returns = pd.DataFrame(zscore(std_normal.ppf(uniform_returns)), index=uniform_returns.index,
                                        columns=uniform_returns.columns)
    else:
        # convert to z-scores
        gaussian_returns = zscore(log_returns)

    base_corr = estimate_correlations(gaussian_returns, estimation_method=corr_type)
    near_corr = nearest_correlation(base_corr)
    if save:
        smoothed = 'smoothed' if density_estimation else 'direct'
        path = os.sep.join([os.getcwd(), rf'\data\correlations_{corr_type}_{smoothed}.csv'])
        near_corr.to_csv(path)
    if plot:
        smoothed = 'smoothed' if density_estimation else 'direct'
        sns.heatmap(near_corr, annot=True, annot_kws={"size": 11}, fmt=f".3f", linewidth=.5, cmap="coolwarm",
                    cbar=False)
        plt.title(f'Linear Correlation ({smoothed}, {corr_type})')
        if save:
            plt.savefig(f'plots/LinearCorrelation_{smoothed}_{corr_type}.png', dpi='figure')
        plt.show()
    return near_corr


def nearest_correlation(base_correlation, method='alternating'):
    """
    :param base_correlation: matrix estimated from historical time series data - may not be positive semidefinite
    :param method: choose between alternating and spectral
    :return: positive semidefinite correlation matrix obtained using either alternating projections or
             spectral projection
    """
    if method == 'spectral':
        return project_correlation_matrix(base_correlation)
    elif method == 'alternating':
        return alternating_projections_algoritm(base_correlation)


def estimate_correlations(returns_data, estimation_method='pearson'):
    """
    :param returns_data: z-scores of log returns or historical pseudo-uniforms
    :param estimation_method: choose between pearson, spearman, kendall correlation types
    :return: linearized correlation
    """
    estimated_correlations = convert_correlations(
        returns_data.corr(method=estimation_method),
        estimation_method=estimation_method)
    print(f"Finished estimation of gaussian correlation matrix.")
    return estimated_correlations


def convert_correlations(df, estimation_method='pearson'):
    """
    :param df: correlation matrix
    :param estimation_method: estimation method of correlation matrix (pearson, kendall, spearman)
    :return: return linearized correlation matrix used to correlate normally distributed variables
    """
    if estimation_method == 'pearson':
        return df
    if estimation_method == 'spearman':
        return 2.0 * np.sin(np.pi/6.0 * df)
    if estimation_method == 'kendall':
        return np.sin(np.pi/2.0 * df)
    else:
        raise NotImplementedError(f"Correlation estimation method {estimation_method} not implemented!")


def check_positive_semi_definite(matrix):
    """
    :param matrix: correlation matrix
    :return: True if correlation matrix is positive semidefinite, else False
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def _spectral_projection(matrix, tol=1E-12, cutoff=1E-12):
    """
    :param matrix: matrix to project onto non-negative spectrum
    :param tol: minimum threshold for spectrum
    :param cutoff: replacement eigenvalue for eigenvalues below tolerance
    :return: correlation matrix after spectral projection method
    """
    eigen_vals, eigen_vecs = np.linalg.eigh(matrix)
    truncated_eigen_vals = [x if x >= tol else cutoff for x in eigen_vals]
    s_matrix = eigen_vecs
    d_matrix = np.diag(truncated_eigen_vals)
    t_matrix = np.diag(1.0 / np.dot(s_matrix**2, truncated_eigen_vals))
    b_matrix = np.matmul(np.matmul(np.sqrt(t_matrix), s_matrix), np.sqrt(d_matrix))
    c_matrix = np.matmul(b_matrix, b_matrix.T)
    return c_matrix


def project_correlation_matrix(
        correlation_matrix,
        projection_tol=1.0E-12,
        projection_cutoff=1.0E-12):
    """
    :param correlation_matrix: matrix to project onto non-negative spectrum
    :param projection_tol: minimum threshold for spectrum
    :param projection_cutoff: replacement eigenvalue for eigenvalues below tolerance
    :return: correlation matrix after spectral projection method if it is not positive semidefinite,
             else original matrix
    """
    # call to project correlation matrix using Rebonato Jaeckel spectral projection algorithm
    if check_positive_semi_definite(correlation_matrix):
        print(f"Matrix is alread positive semidefinite. No further processing applied.")
        return correlation_matrix
    print(f"Negative eigenvalues detected. Applying projection to correlation matrix ...")
    start_time = time.process_time()
    projected_correlation_matrix = _spectral_projection(
        correlation_matrix, tol=projection_tol, cutoff=projection_cutoff)
    print(f"Frobenius distance due to projection: "
          f"{norm(projected_correlation_matrix - correlation_matrix, 'fro'):.2f}")
    projected_correlation_matrix = pd.DataFrame(
        projected_correlation_matrix,
        columns=correlation_matrix.columns.values,
        index=correlation_matrix.columns.values)
    print(f"Finished projection of correlation matrix. Used time: {time.process_time() - start_time:.2f}")
    return projected_correlation_matrix


def _nearest_correlation_matrix(A, rel_tol=(1.0E-4,), eig_tol=1.0E-4, flag=0, max_iterations=1000, weights=None,
                                except_on_too_many_iterations=True):
    """
    Algorithm by Nick Higham: https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/
    Python code adapted from Mike Croucher: https://github.com/mikecroucher/nearest_correlation
    :param A: input matrix
    :param rel_tol: relative tolerance - stop if a single iteration changes the matrix by less than relative tolerance
    :param eig_tol: minimum positive eigenvalue that is not projected to zero
    :param flag: set to 0 to ignore eig_tol, else set to 1
    :param max_iterations: stop algorithm after max_iterations nbr of iterations unless rel_tol is reached
    :param weights: optional symmetric positive semidefinite matrix - use for weighted Frobenius norm
    :param except_on_too_many_iterations: set to True to raise an Exception if algorithm does not reach rel_tol within
                                          max_iterations
    :return: positive semidefinite correlation matrix, close to input matrix with respect to (weighted) Frobenius norm
    """
    if isinstance(A, ExceededMaxIterationsError):
        ds = A.ds.copy()
        A = A.matrix.copy()
    else:
        ds = np.zeros(np.shape(A))

    eps = np.spacing(1)
    if not np.all(A.T == A):
        raise ValueError("Input matrix is not symmetric.")
    if not rel_tol:
        rel_tol = eps * np.shape(A)[0] * np.array([1, 1])
    if weights is None:
        weights = np.ones(np.shape(A)[0])
    X = A.copy()
    Y = A.copy()
    rel_diffY = np.inf
    rel_diffX = np.inf
    rel_diffXY = np.inf

    Whalf = np.sqrt(np.outer(weights, weights))

    iteration = 0
    while max(rel_diffX, rel_diffY, rel_diffXY) > rel_tol[0]:
        iteration += 1
        if iteration > max_iterations:
            msg = f'No solution found in {str(max_iterations)} iterations.'
            if except_on_too_many_iterations:
                raise ExceededMaxIterationsError(msg, X, iteration, ds)
            else:
                print(f"{msg} Result returned anyway.")

        Xold = X.copy()
        R = X - ds
        R_wtd = Whalf * R
        if flag == 0:
            X = _proj_spectral(R_wtd)
        elif flag == 1:
            X = _proj_spectral_cutoff(R_wtd, eig_tol)
        X = X / Whalf
        ds = X - R
        Yold = Y.copy()
        Y = X.copy()
        _proj_unitdiag(Y)
        normY = norm(Y, 'fro')
        rel_diffX = norm(X - Xold, 'fro') / norm(X, 'fro')
        rel_diffY = norm(Y - Yold, 'fro') / normY
        rel_diffXY = norm(Y - X, 'fro') / normY

        X = Y.copy()
    print(f"Alternating projections algorithm finished after {iteration} iterations.")
    return X


def _proj_unitdiag(A):
    """
    :param A: input matrix
    :return: matrix projected onto set of symmetric matrices with unit diagonal
    """
    if not np.all(A.T == A):
        raise ValueError("Input matrix is not symmetric.")
    np.fill_diagonal(A, 1)


def _proj_spectral(A):
    """
    :param A: input matrix
    :return: matrix projected onto positve spectrum, without renormalisation of diagonal
    """
    if not np.all(A.T == A):
        raise ValueError("Input matrix is not symmetric.")
    eig_val, eig_vec = np.linalg.eigh(A)
    truncated_eigen_values = np.maximum(eig_val, 0)
    A = np.matmul(np.matmul(eig_vec, np.diag(truncated_eigen_values)), eig_vec.T)
    A = (A + A.T) / 2
    return A


def _proj_spectral_cutoff(A, abs_tol):
    """
    :param A: input matrix
    :param abs_tol: eigenvalue tolerance
    :return: matrix projected onto positve spectrum, without renormalisation of diagonal
    """
    if not np.all(A.T == A):
        raise ValueError("Input matrix is not symmetric.")
    eig_val, eig_vec = np.linalg.eigh(A)
    truncated_eigen_values = [x if x >= abs_tol else 0 for x in eig_val]
    A = np.matmul(np.matmul(eig_vec, np.diag(truncated_eigen_values)), eig_vec.T)
    A = (A + A.T) / 2
    return A


class ExceededMaxIterationsError(Exception):
    def __init__(self, msg, matrix, iteration, ds):
        self.msg = msg
        self.matrix = matrix
        self.iteration = iteration
        self.ds = ds

    def __str__(self):
        return repr(self.msg)


def alternating_projections_algoritm(correlation_matrix, relative_tolerance=1.0E-4, max_iterations=1000,
                                     eigenvalue_tolerance=1.0E-12, eigenvalue_cutoff=1.0E-12):
    """
    :param correlation_matrix: input correlation matrix
    :param relative_tolerance: stop alternating projections after relative matrix changes are below this threshold
    :param max_iterations: stop alternating projections after this many iterations
    :param eigenvalue_tolerance: absolute eigenvalue tolerance in alternating projections and subsequent spectral proj.
    :param eigenvalue_cutoff: cutoff value for small eigenvalues in final spectral projection, possibly required
                              because alternating projections ends with a unit-diagonal projection step.
    :return: positive semidefinite correlation matrix, Frobenius close to input matrix
    """
    if check_positive_semi_definite(correlation_matrix):
        print(f"Matrix is alread positive semidefinite. No further processing applied.")
        return correlation_matrix
    print("Negative eigenvalues found. Applying alternating projections algorithm to correlation matrix...")
    start_time = time.process_time()
    projected_correlation_matrix = _nearest_correlation_matrix(
        correlation_matrix,
        rel_tol=[relative_tolerance],
        eig_tol=eigenvalue_tolerance,
        max_iterations=max_iterations,
        except_on_too_many_iterations=False,
        flag=0 if eigenvalue_tolerance == 0 else 1)
    if not check_positive_semi_definite(projected_correlation_matrix):
        print("Negative eigenvalues found after alternating projections. Applying spectral projection...")
        alternating_matrix = projected_correlation_matrix.copy()
        projected_correlation_matrix = _spectral_projection(
            projected_correlation_matrix,
            tol=eigenvalue_tolerance,
            cutoff=eigenvalue_cutoff)
        print(f"Spectral projection applied to alternating projections method results in a Frobenius distance of"
              f" {norm(projected_correlation_matrix - alternating_matrix, 'fro'):.2f}")
    else:
        print(f"Alternating projections successful.")

    print(f"Frobenius distance: {norm(projected_correlation_matrix - correlation_matrix, 'fro'):.2f}")
    projected_correlation_matrix = pd.DataFrame(
        projected_correlation_matrix,
        columns=correlation_matrix.columns.values,
        index=correlation_matrix.columns.values)
    print(f"Finished projection of correlation matrix. Used time: {time.process_time() - start_time:.2f}")
    return projected_correlation_matrix


def test_run():
    """
    :return: plot and possibly save different types of correlation matrices
    """
    from returns_data import get_returns
    from kernel_density import fit_kernel_density
    ln_returns = get_returns(inspect=False, save=False)
    uniform_returns = fit_kernel_density(None, from_file=True)
    """get_correlations(None, uniform_returns=uniform_returns, corr_type='pearson', density_estimation=True, save=False,
                     from_file=False, plot=True)
    get_correlations(None, uniform_returns=uniform_returns, corr_type='spearman', density_estimation=True, save=False,
                     from_file=False, plot=True)
    get_correlations(None, uniform_returns=uniform_returns, corr_type='kendall', density_estimation=True, save=False,
                     from_file=False, plot=True)"""
    get_correlations(ln_returns, corr_type='pearson', density_estimation=False, save=False, from_file=False, plot=True)
    # get_correlations(ln_returns, corr_type='spearman', density_estimation=False, save=False, from_file=False, plot=True)
    # get_correlations(ln_returns, corr_type='kendall', density_estimation=False, save=False, from_file=False, plot=True)


if __name__ == '__main__':
    test_run()

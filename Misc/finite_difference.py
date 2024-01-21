import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def solve_ode(x0, xn, steps, p_func, q_func, f_func, y0, yn):
    x_range, dx = np.linspace(x0, xn, steps + 1, retstep=True)
    p_range = np.array([p_func(x_val) for x_val in x_range[1:-1]])
    q_range = np.array([q_func(x_val) for x_val in x_range[1:-1]])
    f_range = np.array([f_func(x_val) for x_val in x_range[1:-1]])
    a_mat = a_matrix(steps, p_range, q_range, dx)
    b_vec = np.ones(steps+1)
    b_vec[1:-1] = f_range*dx**2
    b_vec[0] = y0
    b_vec[-1] = yn
    solution = np.linalg.solve(a_mat, b_vec)
    return x_range, solution


def a_matrix(steps, p_range, q_range, dx):
    mat = np.zeros((steps+1, steps+1))
    mat[0, 0] = 1
    mat[-1, -1] = 1
    for i in np.arange(1, steps):
        mat[i, i-1] = 1 - p_range[i-1]*dx/2
        mat[i, i] = -2 + q_range[i-1] * dx**2
        mat[i, i+1] = 1 + p_range[i-1] * dx/2
    return mat


def exact_solution(x_val):
    e = np.exp(1)
    u = 3*np.exp(2) + 5/(e-1) + 3*e + 5
    v = -np.exp(3) * (3*e + 2) / (e-1)
    return u * np.exp(-x_val) + v * np.exp(-2*x_val) + 2*x_val**2 - 6*x_val + 7


def error_plot(x0, xn, p_func, q_func, f_func, y0, yn):
    n_range = np.array([10, 50, 100, 200, 400, 800, 1600, 3200])
    error_array = np.zeros((len(n_range)))
    for i, steps in enumerate(n_range):
        x_val, y_val = solve_ode(x0, xn, steps, p_func, q_func, f_func, y0, yn)
        exact_sol = exact_solution(x_val)
        errors = np.abs(y_val - exact_sol)
        error_array[i] = max(errors)

    plt.title("Maximum error in numerical finite difference solution")
    plt.loglog([1/n_val for n_val in n_range], error_array, 'o', markeredgecolor='none', label='Maximum absolute error')
    plt.loglog([1/n_val for n_val in n_range], [1/n_val**2 for n_val in n_range], label=r'$O(\delta x^2)$')
    plt.xlabel(r'$\delta x$')
    plt.ylabel(r'Error')
    plt.legend(loc='best')
    plt.show()


def plot_solution(x0, xn, p_func, q_func, f_func, y0, yn):
    n_range = [10, 50, 100]
    colors = ['r', 'g', 'b']
    markers = ['o', '*', "1"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Numerical and analytical solution of ODE")

    ax[0].set_title("Global solution")
    ax[1].set_title("Zoomed in around maximum error")
    for i, steps in enumerate(n_range):
        x, y = solve_ode(x0, xn, steps, p_func, q_func, f_func, y0, yn)
        exact_val = exact_solution(x)
        error = np.abs(y - exact_val)

        header = ['x_i', 'y_i', 'y_exact', 'error']
        table_data = list(zip(x, y, exact_val, error))
        print(f"Finite difference approximation with {steps} steps:")
        print(tabulate(table_data, headers=header))
        print(f"max error for n={steps}: {max(error)} at x_i = {x[np.argmax(error)]}")
        print("\n")

        ax[0].plot(x, y, label=rf'$n={steps}$', c=colors[i])
        ax[1].plot(x, y, label=rf'$n={steps}$', c=colors[i])
        if steps == min(n_range):
            ax[0].scatter(x, y, marker=markers[i], c=colors[i])
            error = np.abs(y - exact_val)
            error_loc = x[np.argmax(error)]
            error_y = y[np.argmax(error)]
            ax[1].set_xlim([error_loc - 0.002, error_loc + 0.002])
            ax[1].set_ylim([exact_solution(error_loc - 0.001), error_y + 0.003])
        ax[1].scatter(x, y, marker=markers[i], c=colors[i])
        if steps == max(n_range):
            ax[0].plot(x, exact_val, label='exact solution', c='k')
            ax[1].plot(x, exact_val, label='exact solution', c='k')
    ax[0].set_xlabel(r'$x$')
    ax[1].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y(x)$')
    ax[1].set_ylabel(r'$y(x)$')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    a = 1
    b = 2
    alpha = 1
    beta = 6

    def p(k):
        return 3

    def q(k):
        return 2

    def f(k):
        return 4 * k ** 2

    plot_solution(a, b, p, q, f, alpha, beta)
    error_plot(a, b, p, q, f, alpha, beta)

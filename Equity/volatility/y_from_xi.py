import numpy as np
from scipy.special import roots_jacobi, gamma


def yFromXi(params, xi):
    """
    Python translation of the R function yFromXi.
    Requires:
        params: dict with keys 'al', 'nu', 'lam', 'c'
        xi: callable xi(u)
    """

    al = params["al"]
    H = al - 0.5
    nu = params["nu"]
    lam = params["lam"]
    c = params["c"]

    # Gauss–Jacobi quadrature nodes and weights
    # alpha = 2H - 1, beta = 0
    nodes, weights = roots_jacobi(10, 2 * H - 1, 0)

    def y_raw(u):
        if u <= 0:
            return 0.0

        # integrand prefactor
        prefactor = (u / 2) ** (2 * H) * nu**2 / (gamma(al) ** 2)

        # evaluate f(x) at Gauss–Jacobi nodes
        # argument for xi: u/2 * (1 + x)
        xi_vals = xi(u / 2 * (1 + nodes))
        exp_vals = np.exp(-lam * u * (1 - nodes))

        integrand = prefactor * xi_vals * exp_vals
        int_u = np.sum(weights * integrand)

        yu2 = xi(u) - c - int_u
        return np.sqrt(max(yu2, 0.0))

    # Return a vectorized callable
    return np.vectorize(y_raw)

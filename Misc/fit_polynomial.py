import numpy as np
from numpy.polynomial import Polynomial

# Define function
def f(u):
    return np.sqrt(-2 * np.log(u))

# Domain limits
u_min = 1e-8
u_max = 0.02425

# Map u to t in [-1,1]
def u_to_t(u):
    return 2 * (u - u_min) / (u_max - u_min) - 1

# Generate sample points (Chebyshev nodes for better conditioning)
n_samples = 200
t_samples = np.cos(np.pi * (2*np.arange(n_samples) + 1) / (2 * n_samples))  # Chebyshev nodes in [-1,1]
u_samples = u_min + (t_samples + 1) * (u_max - u_min) / 2

# Evaluate function at sample points
y_samples = f(u_samples)

# Fit polynomial in t
degree = 7
p = Polynomial.fit(t_samples, y_samples, degree)

# Print coefficients in ascending order (c0 + c1 t + c2 t^2 + ...)
print("Coefficients:", p.convert().coef)

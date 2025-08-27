import numpy as np
from scipy.optimize import least_squares

# Target function
def f(u):
    return np.sqrt(-2 * np.log(u))

# Polynomial evaluator
def polyval(coeffs, x):
    return np.polyval(coeffs, x)

# Rational approximation function
def rational_approx(x, p_coeffs, q_coeffs):
    return polyval(p_coeffs, x) / polyval(q_coeffs, x)

# Residual function to minimize
def residuals(params, x, y, p_deg, q_deg):
    p_coeffs = params[:p_deg+1]
    q_coeffs = np.concatenate(([1.0], params[p_deg+1:]))  # Fix q0=1 for identifiability
    approx = rational_approx(x, p_coeffs, q_coeffs)
    return approx - y

# Domain (avoid 0 exactly)
u_min = 1e-8
u_max = 0.02425
n_samples = 3000
u_samples = np.linspace(u_min, u_max, n_samples)
y_samples = f(u_samples)

# Degrees of numerator and denominator
p_deg = 4
q_deg = 4

# Initial guess
init_params = np.ones(p_deg + q_deg)

# Fit
result = least_squares(residuals, init_params, args=(u_samples, y_samples, p_deg, q_deg))

p_coeffs = result.x[:p_deg+1]
q_coeffs = np.concatenate(([1.0], result.x[p_deg+1:]))

print("Numerator coeffs:", ",".join([str(coef) for coef in p_coeffs]))
print("Denominator coeffs:", ",".join([str(coef) for coef in q_coeffs]))
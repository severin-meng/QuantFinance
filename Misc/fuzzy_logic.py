import matplotlib.pyplot as plt
import numpy as np

def fabs_diff(x, eps=1.0e-1):
    soft_fabs = x / np.sqrt(x**2 + eps)
    fabs = np.where(x > 0.0, 1, -1)
    linear_fabs = np.where(x < -eps, -1, np.where(x > eps, 1, x * 1 / eps ))
    plt.figure()
    plt.plot(x, soft_fabs, label='Soft fABS')
    plt.plot(x, fabs, label='fABS')
    plt.plot(x, linear_fabs, label='Linear fABS')
    plt.legend()
    plt.grid(True)
    plt.show()


def sqrt_2_log_u(u):
    results = np.sqrt(-2.0 * np.log(u))

    t_range = 2 * u / 0.02425 - 1
    t_range = u
    coeffs = [5.1845554, -57.369457,  227.73917, -350.96344, 199.82236]
    numerator_coeffs = [5.1845554, -57.369457,  227.73917, -350.96344, 199.82236]
    denominator_coeffs = [1.0, -22.243182, 160.89896, -413.23785, 349.16787]
    approx = ((np.ones_like(t_range) * numerator_coeffs[0] + t_range *numerator_coeffs[1] +
               t_range**2 * numerator_coeffs[2] + t_range**3 * numerator_coeffs[3] + t_range**4 * numerator_coeffs[4])
              / (np.ones_like(t_range) * denominator_coeffs[0] + t_range *denominator_coeffs[1] +
                 t_range**2 * denominator_coeffs[2] + t_range**3 * denominator_coeffs[3] + t_range**4 * denominator_coeffs[4]))

    plt.figure()
    plt.plot(u, results, label='sqrt(-2.0 * log(u))')
    plt.plot(u, approx, label='approx')
    plt.xlabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # x_range = np.linspace(-3.0, 3.0, 1000)
    # fabs_diff(x_range, eps=3.0e-1)
    u_range = np.linspace(1.0e-8, 0.002425, 1000)
    sqrt_2_log_u(u_range)

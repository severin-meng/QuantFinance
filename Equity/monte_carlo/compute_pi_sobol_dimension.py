from wiener_path_generators import SobolGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer

ref_pi =3.14159265358979323846264338327950288419716939937510  # wikipedia
ref_val = ref_pi/4


sns.set(rc={'figure.figsize': (7, 7)})


def error_convergence_sobol_dimension(path_exponent=13, samples=10):
    sobol_12 = []

    sobol_910 = []
    sobol_mid = []
    for i in range(samples):
        start = default_timer()
        gen = SobolGenerator(1, 50, seed=i)
        randoms = gen.draw_samples_power2(path_exponent, distribution='uniform')
        pi_val_12 = calc_pi(randoms[:2, 0, :])  # first two dimensions
        sobol_12.append(pi_val_12)
        pi_val_910 = calc_pi(randoms[[10, 20], 0, :])  # last two dimensions
        sobol_910.append(pi_val_910)

        pi_val_mid = calc_pi(randoms[[1, -1], 0, :])
        sobol_mid.append(pi_val_mid)
        end = default_timer()
        print(f"Pi calculation {i} took {end-start} seconds.")

    # sobol_12 shape is list of samples payoff lists: [samples, path_exponent]
    sobol_12 = np.array(sobol_12)
    # sobol_910 shape is list of samples payoff lists: [samples, path_exponent]
    sobol_910 = np.array(sobol_910)
    sobol_mid = np.array(sobol_mid)

    results = [sobol_12, sobol_910, sobol_mid]
    label_dict = {0: "Dimensions 1&2", 1: "Dimensions 49&50", 2: "Dimensions 1 & 50"}

    # plot payoffs
    max_log = path_exponent
    logspace = np.logspace(6, max_log, 1000, base=2)
    rolling_mean = []
    for path_nbr in logspace:
        new_entry = []
        for res in results:
            new_entry.append(np.mean(res[:, :int(path_nbr)], axis=1))
        rolling_mean.append(new_entry)
    rolling_mean = np.array(rolling_mean)-ref_pi  # has shape [1000, 2, 10]
    pseudo_std = np.std(rolling_mean, ddof=1, axis=-1)

    for i in range(pseudo_std.shape[1]):
        plt.loglog(logspace, pseudo_std[:, i], label=label_dict[i])
        # plt.scatter(logspace, pseudo_std[:, i])
    plt.loglog(logspace, pseudo_std[0, 0] * np.sqrt(logspace[0]) / np.sqrt(logspace),
               label=r'$\propto \frac{1}{\sqrt{N}}$')
    plt.loglog(logspace, pseudo_std[0, 0] * logspace[0] / logspace, label=r'$\propto \frac{1}{N}$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Monte Carlo Error Convergence")
    plt.xlabel('Number of Paths')
    plt.ylabel('Standard Error')
    plt.show()
    print("DONE")


def value_convergence_sobol_dimension(path_exponent=13):
    start = default_timer()
    gen = SobolGenerator(1, 50)
    randoms = gen.draw_samples_power2(path_exponent, distribution='uniform')
    pi_val_12 = calc_pi(randoms[:2, 0, :])  # first two dimensions
    sobol_12 = pi_val_12
    pi_val_910 = calc_pi(randoms[-2:, 0, :])  # last two dimensions
    sobol_910 = pi_val_910
    end = default_timer()
    print(f"Pi calculation took {end-start} seconds.")

    results = [sobol_12, sobol_910]
    label_dict = {0: "Dimensions 1&2", 1: "Dimensions 9&10"}

    # plot payoffs
    max_log = path_exponent
    logspace = np.logspace(6, max_log, 1000, base=2)
    rolling_mean = []
    for nbr_paths in logspace:
        new_entry = []
        for res in results:
            new_entry.append(np.mean(res[:int(nbr_paths)]))
        rolling_mean.append(new_entry)
    rolling_mean = np.abs(np.array(rolling_mean)-ref_val)  # has shape [1000, 2]

    for i in range(rolling_mean.shape[1]):
        plt.loglog(logspace, rolling_mean[:, i], label=label_dict[i])
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Monte Carlo Price Convergence")
    plt.xlabel('Number of Paths')
    plt.ylabel('Price')
    plt.show()
    print("DONE")


def calc_pi(uniform_randoms):
    eligible_nbrs = np.where(np.linalg.norm(uniform_randoms, axis=0) <= 1, 1, 0)
    return eligible_nbrs


if __name__ == '__main__':
    error_convergence_sobol_dimension(path_exponent=25, samples=20)
    # value_convergence_sobol_dimension(path_exponent=20)

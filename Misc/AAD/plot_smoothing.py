import numpy as np
import matplotlib.pyplot as plt


def plot_digital():
    fig = plt.figure(figsize=(2, 2))
    lhs_y = np.zeros(2)
    lhs_x = np.array([-10, 0])
    rhs_y = np.ones(2)
    rhs_x = np.array([0, 10])
    mid_x = np.zeros(2)
    mid_y = np.array([-0.2, 1.2])
    plt.plot(lhs_x, lhs_y, "black", linewidth=3)
    plt.plot(rhs_x, rhs_y, "black", linewidth=3)
    # plt.plot(mid_x, mid_y, "g:")
    plt.xlim(-10, 10)
    plt.ylim(-0.2, 1.2)
    # plt.title("Digital")
    plt.xticks([0], [r"$K$"])
    plt.yticks([0, 1])
    plt.grid(True)
    plt.show()


def plot_call_spread(eps=2.0):
    fig = plt.figure(figsize=(2,2))
    lhs_y = np.zeros(2)
    lhs_x = np.array([-10, -eps])
    mid_x = np.array([-eps, eps])
    mid_y = 0 + (mid_x - (-eps)) / (2 * eps)

    rhs_y = np.ones(2)
    rhs_x = np.array([eps, 10])
    plt.plot(lhs_x, lhs_y, "black", linewidth=3)
    plt.plot(mid_x, mid_y, "black", linewidth=3)
    plt.plot(rhs_x, rhs_y, "black", linewidth=3)

    eps_left_x = np.ones(2) * -eps
    eps_right_x = np.ones(2) * eps
    eps_y = np.array([-0.2, 1.2])
    # plt.plot(eps_left_x, eps_y, "g:")
    # plt.plot(eps_right_x, eps_y, "g:")

    plt.xlim(-10, 10)
    plt.ylim(-0.2, 1.2)
    plt.xticks([-eps, eps], [r"$K-\epsilon$", r"$K+\epsilon$"])
    plt.yticks([0, 1])
    plt.grid(True)
    # plt.title("Call Spread")
    plt.show()


if __name__ == "__main__":
    plot_digital()
    plot_call_spread()

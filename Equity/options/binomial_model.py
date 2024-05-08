import matplotlib.pyplot as plt
import numpy as np


def payoff(stock_val, strike_val, iscall=True):
    return max([(stock_val - strike_val), 0]) if iscall else max([(strike_val - stock_val), 0])


def binomial_option(nbr_steps, volatility, is_call):
    # value option using binomial model
    dt = period / nbr_steps
    sqrt_dt = np.sqrt(dt)
    discount_factor = np.exp(-rate * dt)

    # CRR parameterisation
    u = np.exp(volatility * sqrt_dt)
    v = np.exp(-volatility * sqrt_dt)
    p_prime = (1 / discount_factor - v) / (u-v)

    # build binomial tree and fill with stock prices
    # for each time step there is an ordered list of dictionaries, the order is descending in the nbr of up-moves.
    # Each dictionary contains a position value and a stock price value. The position value describes how many
    # up and down moves are required to get here.
    tree = [[{'position': '', 'stock': stock}]]  # starting point, no up/down moves -> empty position
    for i in range(nbr_steps):
        positions = tree[i]
        new_nodes = []

        if not positions:
            break

        # to cover all possible paths, we have to make 1 up move from the top position,
        # and 1 down move each for all positions.

        pos = positions[0]  # top position
        su = pos['stock'] * u
        pu = 'u' + pos['position']
        new_nodes.append({'position': pu, 'stock': su})

        for pos in positions:
            sv = pos['stock'] * v
            pv = pos['position'] + 'v'
            new_nodes.append({'position': pv, 'stock': sv})
        tree.append(new_nodes)

    # add option values into the binomial tree, go in reverse order.
    tree_rev = tree.copy()[::-1]  # revert tree order
    for tree in tree_rev[0]:
        tree.update({'option': payoff(tree['stock'], strike, iscall=is_call)})  # option payoff (value) at maturity
    for i in range(nbr_steps):
        source_trees = tree_rev[i]  # timestep we want to find the option value at
        target_trees = tree_rev[i + 1]  # all positions at t+1 -> required to value option
        for target in target_trees:
            pos = target['position']
            pos_u = 'u' + pos
            pos_v = pos + 'v'
            source_u = [s for s in source_trees if s['position'] == pos_u][0]  # find relevant target node
            source_v = [s for s in source_trees if s['position'] == pos_v][0]  # find relevant target node
            option = discount_factor * (p_prime * source_u['option'] + (1 - p_prime) * source_v['option'])
            target.update({'option': option})

    option = tree_rev[-1][0]['option']
    return option


if __name__ == "__main__":
    stock = 1000
    strike = 900
    rate = 0.01
    period = 1.
    is_call = False

    volatility_range = np.linspace(0.05, 0.8, 25)
    time_steps = 10
    option_values = [binomial_option(time_steps, vola, is_call) for vola in volatility_range]

    # part 1
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Binomial Call Option')

    ax[0].scatter(volatility_range, option_values)
    ax[0].plot(volatility_range, option_values)
    ax[0].set_xlabel('Volatility')
    ax[0].set_ylabel('Option Value')
    ax[0].set_title("Fig. 3.1: Option Value vs Volatility")
    ax[0].grid()

    # part 2
    volatility = 0.2
    step_range = list(range(4, 51))
    option_values = [binomial_option(steps, volatility, is_call) for steps in step_range]

    ax[1].plot(step_range, option_values)
    ax[1].scatter(step_range, option_values)
    ax[1].set_xlabel('Number of Steps')
    ax[1].set_ylabel('Option Value')
    ax[1].set_title("Fig. 3.2: Option Value vs Simulation Steps")
    ax[1].grid()
    plt.show()

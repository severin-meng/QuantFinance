import numpy as np
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
from tabulate import tabulate


class PortfolioOptimisation:
    def __init__(self, returns, vols, correlation):
        self.returns = returns
        self.vols = vols
        self.covariance = multi_dot([np.diag(vols), correlation, np.diag(vols)])

    def portfolio_simulation(self, number_of_portfolios, num_of_asset=4):
        # initialize containers
        rets = np.array([])
        vols = np.array([])
        wts = np.array([])

        for i in range(number_of_portfolios):
            # generate weights
            weights = np.random.random(num_of_asset)[:, np.newaxis] - 0.5  # allow short selling
            while abs(np.sum(weights)) < 1e-10:  # numerical stability as we divide by the sum later on to normalize
                weights = np.random.random(num_of_asset)[:, np.newaxis] - 0.5
            weights /= np.sum(weights)  # normalize weights, i.e. budget equation
            rets = np.append(rets, multi_dot([weights.T, self.returns]))
            vols = np.append(vols, np.sqrt(multi_dot([weights.T, self.covariance, weights])))
            wts = np.append(wts, weights.flatten())

        sharpe_ratio = rets / vols

        portdf = {
            'port_rets': rets,
            'port_vols': vols,
            'weights': wts,
            'sharpe_ratio': sharpe_ratio}
        return portdf

    def efficient_frontier(self, max_return, min_return, steps=100):
        # plot efficient portfolios
        ret_range = np.linspace(min_return, max_return, steps)
        inv_cov = np.linalg.inv(self.covariance)
        unit_vector = np.ones((inv_cov.shape[0]))
        A = multi_dot([unit_vector.T, inv_cov, unit_vector])
        B = multi_dot([unit_vector.T, inv_cov, self.returns])
        C = multi_dot([self.returns.T, inv_cov, self.returns])
        risk_range = np.array([])
        for r in ret_range:
            opt_weights = 1 / (A * C - B ** 2) * np.matmul(inv_cov, (A * self.returns - B * unit_vector) * r + (
                        C * unit_vector - B * self.returns))
            risk = np.sqrt(multi_dot([opt_weights.T, self.covariance, opt_weights]))
            risk_range = np.append(risk_range, risk)
        return risk_range, ret_range

    def tangency_portfolio(self, risk_free_rate, A, B, C, inv_cov, unit_vector):
        expected_return = (C - B*risk_free_rate) / (B - A*risk_free_rate)
        weights = np.matmul(inv_cov, self.returns - risk_free_rate * unit_vector) / (B - A*risk_free_rate)
        std_deviation = np.sqrt((C- 2*risk_free_rate*B + risk_free_rate**2 * A) / (B - A*risk_free_rate)**2)
        sharpe_ratio = (expected_return - risk_free_rate) / std_deviation
        return [risk_free_rate, *weights, expected_return, std_deviation, sharpe_ratio]

    def efficient_frontier_riskfree_rate(self, risk_free_rate, A, B, C, inv_cov, unit_vector, efficient=False):
        tangency_pf = self.tangency_portfolio(risk_free_rate, A, B, C, inv_cov, unit_vector)
        tangency_std = tangency_pf[-2]
        tangency_ret = tangency_pf[-3]
        max_short = -2
        short_weights = np.linspace(max_short, 1, 100)
        if efficient and B/A < risk_free_rate:
            # borrow tangency portfolio, buy risk free rate
            pf_returns = short_weights * tangency_ret + (1-short_weights) * risk_free_rate
            risks = - short_weights * tangency_std
        else:
            # borrow risk free rate, buy tangency portfolio
            pf_returns = (1 - short_weights) * tangency_ret + short_weights * risk_free_rate
            risks = (1 - short_weights) * tangency_std

        return {'port_rets': pf_returns,
                'port_vols': risks}


def minimum_variance_portfolio():
    # initialize parameters
    returns = np.array([0.02, 0.07, 0.15, 0.20])
    vols = np.array([0.05, 0.12, 0.17, 0.25])
    correlation = np.array([[1, 0.3, 0.3, 0.3],
                            [0.3, 1, 0.6, 0.6],
                            [0.3, 0.6, 1, 0.6],
                            [0.3, 0.6, 0.6, 1]])

    optimisation = PortfolioOptimisation(returns, vols, correlation)

    # covariance = multi_dot([np.diag(vols), correlation, np.diag(vols)])
    nbr_portfolios = 700

    simulated_pfs = optimisation.portfolio_simulation(nbr_portfolios, 4)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Optimal Portfolio Allocation 1')

    p1 = ax[0].scatter(x=simulated_pfs['port_vols'], y=simulated_pfs['port_rets'], c=simulated_pfs['sharpe_ratio'],
                       marker='x')
    ax[0].set_title("Fig. 1.1: Monte Carlo Simulated Portfolios")
    ax[0].set_xlabel("Risk")
    ax[0].set_ylabel("Expected Return")
    fig.colorbar(p1, label='Sharpe Ratio', ax=ax[0])
    ax[0].set_xlim(0, 0.5)
    ax[0].set_ylim(-0.3, 0.6)
    ax[0].grid()

    # plot boundary of opportunity set
    boundary_risk, boundary_return = optimisation.efficient_frontier(max_return=1, min_return=-1, steps=1000)

    p2 = ax[1].scatter(x=simulated_pfs['port_vols'], y=simulated_pfs['port_rets'], c=simulated_pfs['sharpe_ratio'],
                       marker='x')
    ax[1].set_title("Fig. 1.2: Simulated Portfolios vs Boundary Set")
    ax[1].set_xlabel("Risk")
    ax[1].set_ylabel("Expected Return")
    fig.colorbar(p2, label='Sharpe Ratio', ax=ax[1])
    ax[1].set_xlim(0, 0.5)
    ax[1].set_ylim(-0.3, 0.6)
    ax[1].plot(boundary_risk, boundary_return, label='Frontier of risky assets', linewidth=3)

    ax[1].legend(loc='best')
    ax[1].grid()
    plt.show()


def maximum_sharpe_ratio_portfolio():
    # Optimal Portfolio Allocation
    returns = np.array([0.02, 0.07, 0.15, 0.20])
    vols = np.array([0.05, 0.12, 0.17, 0.25])
    correlation = np.array([[1, 0.3, 0.3, 0.3],
                            [0.3, 1, 0.6, 0.6],
                            [0.3, 0.6, 1, 0.6],
                            [0.3, 0.6, 0.6, 1]])
    covariance = multi_dot([np.diag(vols), correlation, np.diag(vols)])

    optimisation = PortfolioOptimisation(returns, vols, correlation)

    # part 1
    risk_free_rates = [0.005, 0.01, 0.015, 0.0175]
    inv_cov = np.linalg.inv(covariance)
    unit_vector = np.ones((inv_cov.shape[0]))
    A = multi_dot([unit_vector.T, inv_cov, unit_vector])
    B = multi_dot([unit_vector.T, inv_cov, returns])
    C = multi_dot([returns.T, inv_cov, returns])

    table_data = []
    for r in risk_free_rates:
        table_data.append(optimisation.tangency_portfolio(r, A, B, C, inv_cov, unit_vector))
    header = ['Risk Free Rate', 'Weight1', 'Weight2', 'Weight3', 'Weight4', 'Exp. Return', 'Std Deviation',
              'Sharpe Ratio']
    print(tabulate(table_data, headers=header))

    # Optimization fails for the last example, because we are optimizing the squared slope!

    # part 2
    # efficient frontier when RFR is present
    # lecture 2 module 2 page 79
    risk_free_rates = [0.01, 0.0175]

    risky_frontier = optimisation.efficient_frontier(2, -2, 1000)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Optimal Portfolio Allocation 2')

    efficient_frontiers = [{'rate': rfr, 'efficient_frontier': optimisation.efficient_frontier_riskfree_rate(
        rfr, A, B, C, inv_cov, unit_vector,)} for rfr in risk_free_rates]
    for frontier in efficient_frontiers:
        rfr = frontier['rate']
        efficient_frontier = frontier['efficient_frontier']
        ax[0, 0].plot(efficient_frontier['port_vols'], efficient_frontier['port_rets'],
                      linewidth=2, label=r'$r_f=$ %.2f ' % (100 * rfr) + '%')
    # ax[0, 0].plot(risky_frontier[0], risky_frontier[1], label='risky frontier')
    ax[0, 0].legend(loc='best')
    ax[0, 0].set_xlim(0, 0.5)
    ax[0, 0].set_ylim(-0.5, 0.5)
    ax[0, 0].set_title("Fig. 2.1: Efficient Frontier")
    ax[0, 0].grid()

    min_variance_return = B/A

    for frontier in efficient_frontiers:
        rfr = frontier['rate']
        efficient_frontier = frontier['efficient_frontier']
        ax[0, 1].plot(efficient_frontier['port_vols'], efficient_frontier['port_rets'],
                   linewidth=2, label=r'$r_f=$ %.2f ' % (100 * rfr) + '%')
    ax[0, 1].plot([0, 1], [min_variance_return, min_variance_return],
                  label=r'$\frac{B}{A}=$ %.2f' % (100*min_variance_return) + '%', linestyle='dashed')
    ax[0, 1].legend(loc='best')
    ax[0, 1].set_xlim(0, 0.02)
    ax[0, 1].set_ylim(-0.005, 0.035)
    ax[0, 1].set_title("Fig. 2.2: Efficient Frontier (zoomed in)")
    for label in ax[0, 1].get_xticklabels()[::2]:
        label.set_visible(False)
    ax[0, 1].grid()

    # simulate some risky portfolios
    simulated_pfs = optimisation.portfolio_simulation(700)

    ax[1, 0].scatter(simulated_pfs['port_vols'], simulated_pfs['port_rets'],
                     marker='+', c='g', label='risky portfolios')
    ax[1, 0].set_xlim(0, 0.5)
    ax[1, 0].set_ylim(-0.5, 0.5)
    ax[1, 0].set_title("Fig. 2.3: Efficient Frontier & Risky Portfolios")
    for frontier in efficient_frontiers:
        rfr = frontier['rate']
        efficient_frontier = frontier['efficient_frontier']
        ax[1, 0].plot(efficient_frontier['port_vols'], efficient_frontier['port_rets'],
                      linewidth=2, label=r'$r_f=$ %.2f' % (100 * rfr) + '%')
    ax[1, 0].legend(loc='lower left')
    ax[1, 0].grid()

    ax[1, 1].set_xlim(0, 0.26)
    ax[1, 1].set_ylim(-0.03, 0.28)
    ax[1, 1].set_title("Fig. 2.4: Efficient Frontier")
    efficient_frontiers = [{'rate': rfr, 'efficient_frontier': optimisation.efficient_frontier_riskfree_rate(
        rfr, A, B, C, inv_cov, unit_vector, efficient=True)} for rfr in risk_free_rates]
    for frontier in efficient_frontiers:
        rfr = frontier['rate']
        efficient_frontier = frontier['efficient_frontier']
        ax[1, 1].plot(efficient_frontier['port_vols'], efficient_frontier['port_rets'],
                      linewidth=2, label=r'$r_f=$ %.2f ' % (100 * rfr) + '%')
    ax[1, 1].plot(risky_frontier[0], risky_frontier[1], label='risky frontier', linewidth=1.0, c='k')
    ax[1, 1].legend(loc='best')
    ax[1, 1].grid()

    plt.show()


if __name__ == '__main__':
    maximum_sharpe_ratio_portfolio()
    # minimum_variance_portfolio()

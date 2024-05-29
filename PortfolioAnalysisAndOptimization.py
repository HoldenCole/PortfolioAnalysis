import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import inv

# Step 1: Data Collection
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Adj Close']

# Step 2: Data Preprocessing
returns = data.pct_change().dropna()

# Portfolio performance calculation
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_stddev, portfolio_return

# Including transaction costs
def transaction_costs(weights, prev_weights, cost_per_trade=0.001):
    return np.sum(np.abs(weights - prev_weights)) * cost_per_trade

def negative_sharpe_ratio_with_costs(weights, returns, prev_weights, risk_free_rate=0.01, cost_per_trade=0.001):
    p_std, p_return = portfolio_performance(weights, returns)
    costs = transaction_costs(weights, prev_weights, cost_per_trade)
    return -((p_return - costs) - risk_free_rate) / p_std

def optimize_portfolio_with_costs(returns, prev_weights, cost_per_trade=0.001):
    num_assets = len(returns.columns)
    args = (returns, prev_weights, cost_per_trade)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio_with_costs, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Assume an initial equal-weighted portfolio
prev_weights = np.array([1./len(tickers)]*len(tickers))
optimized_result_with_costs = optimize_portfolio_with_costs(returns, prev_weights)
optimized_weights_with_costs = optimized_result_with_costs.x

# Black-Litterman Model
# Mean and covariance of returns
mu = returns.mean()
cov = returns.cov()

# Market weights (for simplicity, assume equal weights)
market_weights = np.array([1./len(tickers)]*len(tickers))
market_return = np.dot(market_weights, mu)
market_cov = np.dot(np.dot(market_weights, cov), market_weights.T)

# Investor views (for simplicity, assume expected returns)
P = np.eye(len(tickers))
Q = np.array([0.10, 0.12, 0.15, 0.08, 0.14])  # Investor's expected returns

# Black-Litterman model
tau = 0.025
omega = np.diag(np.diag(np.dot(np.dot(P, tau*cov), P.T)))

# Combine market equilibrium and investor views
middle_term = inv(inv(tau*cov) + np.dot(np.dot(P.T, inv(omega)), P))
posterior_mean = np.dot(middle_term, np.dot(inv(tau*cov), mu) + np.dot(np.dot(P.T, inv(omega)), Q))
posterior_cov = cov + middle_term

def negative_sharpe_ratio(weights, mu, cov, risk_free_rate=0.01):
    portfolio_return = np.dot(weights, mu)
    portfolio_volatility = np.sqrt(np.dot(np.dot(weights.T, cov), weights))
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_portfolio(mu, cov):
    num_assets = len(mu)
    args = (mu, cov)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

optimized_result_bl = optimize_portfolio(posterior_mean, posterior_cov)
optimized_weights_bl = optimized_result_bl.x

# Efficient Frontier Calculation
def calculate_efficient_frontier(returns, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_stddev, portfolio_return = portfolio_performance(weights, returns)
        results[0,i] = portfolio_stddev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - 0.01) / portfolio_stddev
    return results, weights_record

results, weights_record = calculate_efficient_frontier(returns)
max_sharpe_idx = np.argmax(results[2])
sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
max_sharpe_allocation = weights_record[max_sharpe_idx]

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.scatter(sdp, rp, marker='*', color='r', s=500)
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Returns')
plt.colorbar(label='Sharpe Ratio')
plt.show()

# Print optimized weights
print(f"Optimized Weights with Costs: {optimized_weights_with_costs}")
print(f"Black-Litterman Optimized Weights: {optimized_weights_bl}")

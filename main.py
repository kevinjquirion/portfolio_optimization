from sklearn.preprocessing import StandardScaler
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import scipy.optimize as sp_opt
from pprint import pprint
import yfinance as yf
import pandas as pd
import numpy as np
import palettable
import functions

## choose tickers for your portfolio ##
tickers = ['AAPL','AMZN','MSFT','SQ']
num_holdings = len(tickers)

## choose dates from which to get data ##
end_date = datetime.today()
start_date = end_date - timedelta(365//2)

data = functions.asset_data(tickers,start_date,end_date)
pprint(data.prices.head())

print("Initial Weights: {}".format(data.weights))
print("Initial Expected return: {:.2f}".format(data.exp_returns))
print("Initial Expected Volatility: {:.2f}".format(data.volatility))
print("Initial Sharpe Ratio: {:.4f}".format(data.sharpe_ratio) )

print('\n', '#' * 100, '\n')

min_weight = 0.0
max_weight = 1.0
initial_weights = [1/num_holdings for _ in range(num_holdings)]
optimal_weights = functions.optimize_sharpe_ratio(data,initial_weights,min_weight,max_weight)
data.set_weights(optimal_weights)
print("Optimal Weights: {}".format(data.weights))
print("Optimized Expected return: {:.2f}".format(data.exp_returns))
print("Optimized Expected Volatility: {:.2f}".format(data.volatility))
print("Optimized Sharpe Ratio: {:.4f}".format(data.sharpe_ratio) )

print('\n', '#' * 100, '\n')

num_iterations = 10000


mc,max_sharpe,min_volatility = functions.sharpe_monte_carlo(num_iterations,data)

print("Results for max Sharpe Ratio: \,")
print(max_sharpe)

print('\n', '#' * 100, '\n')

print("Results for minimum volatility:")
print(min_volatility)

plt.tight_layout()
plt.show()
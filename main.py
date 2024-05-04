from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import pandas as pd
import functions

## choose tickers for your portfolio ##
tickers = ['AAPL','AMZN','MSFT','SQ']
num_holdings = len(tickers)

## choose dates from which to get data ##
end_date = datetime.today()
start_date = end_date - timedelta(365//2)

## create an initial portfolio with evenly weighted positions ##
data = functions.asset_data(tickers,start_date,end_date)

## print some initial values of metrics ##
print("Initial Weights: {}".format(data.weights))
print("Initial Expected return: {:.2f}".format(data.exp_returns))
print("Initial Expected Volatility: {:.2f}".format(data.volatility))
print("Initial Sharpe Ratio: {:.4f}".format(data.sharpe_ratio) )

print('\n', '#' * 100, '\n')

print("## Optimize Sharpe Ratio Using scipy.optimize.minimize ## \n")

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

print("## Optimize Volatility Using scipy.optimize.minimize ## \n")

min_weight = 0.0
max_weight = 1.0
initial_weights = [1/num_holdings for _ in range(num_holdings)]
safe_weights = functions.optimize_volatility(data,initial_weights,min_weight,max_weight)
data.set_weights(safe_weights)
print("Safe Weights: {}".format(data.weights))
print("Safe Expected return: {:.2f}".format(data.exp_returns))
print("Safe Expected Volatility: {:.2f}".format(data.volatility))
print("Safe Sharpe Ratio: {:.4f}".format(data.sharpe_ratio) )

print('\n', '#' * 100, '\n')

print("## Optimize Sharpe Ratio Using Monte Carlo Simulation ## \n")

num_iterations = 1000
mc_ax,max_sharpe,min_volatility = functions.sharpe_monte_carlo(num_iterations,data)

print("## Results for max Sharpe Ratio: ## \n")
print(max_sharpe)

print('\n', '#' * 100, '\n')

print("## Optimize Volatility Using Monte Carlo Simulation ## \n")

print("Results for minimum volatility:")
print(min_volatility)

fig,ax = plt.subplots(1,1,figsize=(11,7))
plot_data = pd.DataFrame({tickers[i]:[max_sharpe.weights[i],min_volatility.weights[i]] for i in range(num_holdings)}).T
plot_data.columns = ['Sharpe Optimized', 'Volatility Optimized']
subp1 = plot_data.plot.bar(ax=ax,rot=0)
ax.tick_params(which='both',labelsize=20)
ax.legend(fontsize=18)


## show the plot produced by the monte carlo simulation ##
plt.tight_layout()
plt.show()
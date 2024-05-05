from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

num_iterations = 10000

mc_sims = functions.mc_portfolios(num_iterations,data)
max_sharpe = mc_sims.get_max_sharpe()
min_risk = mc_sims.get_min_risk()

mc_returns = mc_sims.sim_data['returns']
mc_risks = mc_sims.sim_data['volatility']
sharpe_ratios = mc_sims.sim_data['sharpe ratio']
returns_max = mc_returns.max()
returns_min = mc_returns.min()
risk_min = mc_risks.min()
allowed_risk = 0.3
tolerated = mc_sims.get_max_return(max_risk=allowed_risk)


fig,ax = plt.subplots(1,1,figsize=(10,6))
plot = ax.scatter(mc_risks,mc_returns,c=sharpe_ratios,cmap='jet',s=20)
ax.fill_betweenx(
    y=np.linspace(0,1.5*returns_max,100),x1=[0.99*risk_min for _ in range(100)],
    x2=[allowed_risk for _ in range(100)],color='b',alpha=0.1,
    label=r'Tolerated Risk: $\sigma$ = {:.2f}'.format(allowed_risk)
    )
ax.scatter(max_sharpe['volatility'],max_sharpe['returns'],c='b',marker='X',s=150,label='Maximum Sharpe Ratio')
ax.scatter(min_risk['volatility'],min_risk['returns'],c='r',marker='X',s=150,label='Minimum Risk')
ax.scatter(x=tolerated['volatility'],y=tolerated['returns'],c='limegreen',marker='X',s=150,label='Maximum Return for Tolerated Risk')
ax.set_ylim(0.5*returns_min,1.1*returns_max)
ax.set_title('Risk vs. Return',fontsize=22)
ax.set_xlabel('Volatility',fontsize=18,labelpad=20)
ax.set_ylabel('Expected Returns',fontsize=18,labelpad=20)
ax.tick_params(which='both',direction='in',labelsize=18)
cbar = fig.colorbar(plot)
cbar.set_label(label='Sharpe Ratio',size=20,labelpad=20)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
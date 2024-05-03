from datetime import datetime,timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf
import pandas as pd
import numpy as np


class asset_data:
    rf_rate = yf.download(
        '^TNX',start=datetime.today() - timedelta(1), end = datetime.today()
        )['Adj Close'].iloc[0] / 100
    prices = pd.DataFrame()
    pct_change = pd.DataFrame()
    log_return = pd.DataFrame()
    weights = np.array([])
    exp_returns = 0
    covariance = np.array([])
    volatility = 1
    sharpe_ratio = 0
    ## initialize everything ##
    def __init__(self,tickers,start_date,end_date,price_column = 'Adj Close'):
        print(self.rf_rate)
        for ticker in tickers:
            data = yf.download(ticker,start=start_date,end=end_date)
            self.prices[ticker] = data[price_column]
        self.log_return = np.log(1 + self.prices.pct_change())
        self.covariance = self.log_return.cov()
        self.weights = np.array([1/len(tickers) for _ in range(len(tickers))])
        self.volatility = np.sqrt( np.dot(self.weights.T, np.dot(self.covariance * 252, self.weights) ) )
        self.exp_returns = np.sum( (self.log_return.mean() * self.weights) * 252 )
        self.sharpe_ratio = (self.exp_returns - self.rf_rate) / self.volatility
    ## change the weights, and as a result, change the volatility ##
    def set_weights(self,weights):
        self.weights = np.array(weights)
        self.volatility = np.sqrt( np.dot(self.weights.T, np.dot(self.covariance * 252, self.weights) ) )
        self.exp_returns = np.sum( (self.log_return.mean() * self.weights) * 252 )
        self.sharpe_ratio = ( self.exp_returns - self.rf_rate) / self.volatility

def reweight_portfolio(new_weights,asset_data):
    asset_data.set_weights(new_weights)
    return asset_data.sharpe_ratio

def optimize_sharpe_ratio(asset_data,initial_guess,min_weight,max_weight):
    def min_fun(new_weights,asset_data):
        return - reweight_portfolio(new_weights,asset_data)
    bounds = ([min_weight,max_weight] for _ in range(len(initial_guess)) )
    constraints = {'type':'eq','fun':lambda w: sum(w) - 1}
    result = minimize(
        min_fun, x0=initial_guess, args=(asset_data), method='SLSQP', bounds=bounds, constraints=constraints,tol=1e-8
        ).x
    return result

def sharpe_monte_carlo(num_iterations,asset_data):
    num_holdings = len(asset_data.weights)
    all_weights = np.zeros((num_iterations,num_holdings))
    all_returns = np.zeros((num_iterations))
    all_volatilities = np.zeros((num_iterations))
    all_sharpe_ratios = np.zeros((num_iterations))
    for ind in range(num_iterations):
        temp_weights = np.random.random(size = num_holdings) 
        temp_weights = temp_weights / np.sum(temp_weights)
        all_weights[ind] = temp_weights
        asset_data.set_weights(temp_weights)
        all_returns[ind] = asset_data.exp_returns
        all_volatilities[ind] = asset_data.volatility
        all_sharpe_ratios[ind] = asset_data.sharpe_ratio
    mc_results = pd.DataFrame()
    mc_results['weights'] = pd.Series(list(all_weights))
    mc_results['returns'] = pd.Series(all_returns)
    mc_results['volatility'] = pd.Series(all_volatilities)
    mc_results['sharpe ratio'] = pd.Series(all_sharpe_ratios)

    max_sharpe = mc_results.loc[mc_results['sharpe ratio'].idxmax()]
    min_volatility = mc_results.loc[mc_results['volatility'].idxmin()]

    fig,ax = plt.subplots(1,1,figsize=(10,6))
    im = ax.scatter(x=mc_results['volatility'],y=mc_results['returns'],c=mc_results['sharpe ratio'],cmap='jet')
    ax.set_xlabel('Volatility',fontsize=18,labelpad=20)
    ax.set_ylabel('Expected Return',fontsize=18,labelpad=20)
    cbar = plt.colorbar(im,ax=ax)
    cbar.set_label(label='Sharpe Ratio',fontsize=18,labelpad=20)
    ax.tick_params(which='both',labelsize=16,direction='in')
    return fig,max_sharpe,min_volatility

    
    
    
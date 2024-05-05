from datetime import datetime,timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

## class to hold price data and some other metrics ##
class asset_data:
    ## load in the 10 ys treasury note yield as the risk free rate ##
    rf_rate = yf.download(
        '^TNX',start=datetime.today() - timedelta(5), end = datetime.today()
        )['Adj Close'].iloc[-1] / 100
    ## initialize variables ##
    prices = pd.DataFrame()
    log_return = pd.DataFrame()
    weights = np.array([])
    exp_returns = 0
    covariance = np.array([])
    volatility = 1
    sharpe_ratio = 0
    ## set initial values based on chosen tickers and chosen timeframe ##
    def __init__(self,tickers,start_date,end_date,price_column = 'Adj Close'):
        for ticker in tickers:
            data = yf.download(ticker,start=start_date,end=end_date)
            self.prices[ticker] = data[price_column]
        self.log_return = np.log(1 + self.prices.pct_change())
        self.covariance = self.log_return.cov()
        self.weights = np.array([1/len(tickers) for _ in range(len(tickers))])
        self.volatility = np.sqrt( 
            np.dot(self.weights.T, np.dot(self.covariance * 252, self.weights) ) 
            )
        self.exp_returns = np.sum( (self.log_return.mean() * self.weights) * 252 )
        self.sharpe_ratio = (self.exp_returns - self.rf_rate) / self.volatility
    ## change the weights, and as a result, change the volatility ##
    def set_weights(self,weights):
        self.weights = np.array(weights)
        self.volatility = np.sqrt( 
            np.dot(self.weights.T, np.dot(self.covariance * 252, self.weights) ) 
            )
        self.exp_returns = np.sum( (self.log_return.mean() * self.weights) * 252 )
        self.sharpe_ratio = ( self.exp_returns - self.rf_rate) / self.volatility

## class to create, store, and retrive MC simulation data about portfolios ##
class mc_portfolios:
    sim_data = pd.DataFrame()
    def __init__(self,num_simulations,asset_data):
        progress = ['|','\\','-','/']
        num_holdings = len(asset_data.weights)
        all_weights = np.zeros((num_simulations,num_holdings))
        all_returns = np.zeros((num_simulations))
        all_volatilities = np.zeros((num_simulations))
        all_sharpe_ratios = np.zeros((num_simulations))
        for ind in range(num_simulations):
            if ind % 300 == 0:
                i = (ind//300) % 4
                print(
                     ' '*7 + progress[i] + '  generating random weights  ' + progress[i],end='\r'
                     )
            temp_weights = np.random.random(size = num_holdings) 
            temp_weights = temp_weights / np.sum(temp_weights)
            all_weights[ind] = temp_weights
            asset_data.set_weights(temp_weights)
            all_returns[ind] = asset_data.exp_returns
            all_volatilities[ind] = asset_data.volatility
            all_sharpe_ratios[ind] = asset_data.sharpe_ratio
        self.sim_data['weights'] = pd.Series(list(all_weights))
        self.sim_data['returns'] = pd.Series(all_returns)
        self.sim_data['volatility'] = pd.Series(all_volatilities)
        self.sim_data['sharpe ratio'] = pd.Series(all_sharpe_ratios)
    def get_max_sharpe(self):
        max_sharpe = self.sim_data.loc[self.sim_data['sharpe ratio'].idxmax()]
        return max_sharpe
    def get_min_risk(self):
        min_risk = self.sim_data.loc[self.sim_data['volatility'].idxmin()]
        return min_risk
    def get_max_return(self,max_risk = -1):
        if max_risk < 0:
            return self.sim_data.loc[self.sim_data['returns'].idxmax()]
        else:
            allowed = self.sim_data.loc[self.sim_data['volatility'] <= max_risk]
            return allowed.loc[allowed['returns'].idxmax()]



## get the sharpe ratio of the reweighted portfolio ##
def reweighted_sharpe(new_weights,asset_data):
    asset_data.set_weights(new_weights)
    return asset_data.sharpe_ratio

## get the volatiity of the reweighted portfolio ##
def reweighted_volatility(new_weights,asset_data):
    asset_data.set_weights(new_weights)
    return asset_data.volatility

## minimize the negative sharpe ratio and return the associated weights ##
def optimize_sharpe_ratio(asset_data,initial_guess,min_weight,max_weight):
    def min_fun(new_weights,asset_data):
        return - reweighted_sharpe(new_weights,asset_data)
    bounds = ([min_weight,max_weight] for _ in range(len(initial_guess)) )
    constraints = {'type':'eq','fun':lambda w: sum(w) - 1}
    result = minimize(
        min_fun, x0=initial_guess, args=(asset_data), method='SLSQP',
        bounds=bounds, constraints=constraints,tol=1e-8
        ).x
    return result

## minimize the volatility and return the associated weights ##
def optimize_volatility(asset_data,initial_guess,min_weight,max_weight):
    def min_fun(new_weights,asset_data):
        return  reweighted_volatility(new_weights,asset_data)
    bounds = ([min_weight,max_weight] for _ in range(len(initial_guess)) )
    constraints = {'type':'eq','fun':lambda w: sum(w) - 1}
    result = minimize(
        min_fun, x0=initial_guess, args=(asset_data), method='SLSQP',
        bounds=bounds, constraints=constraints,tol=1e-8
        ).x
    return result



    
    
    
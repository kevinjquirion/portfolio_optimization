from scipy.optimize import minimize
from fredapi import Fred
import yfinance as yf
import pandas as pd
import numpy as np


def adj_close(tickers,start_date,end_date):
    df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker,start=start_date,end=end_date)
        df[ticker] = data['Adj Close']
    return df

def log_daily_change(df):
    return np.log(df / df.shift(1))

def std_dev(weights,log_returns):
    weights = np.array(weights)
    cov = log_returns.cov()
    var = weights.T @ cov @ weights
    return np.sqrt(var)

def expected_returns(weights,log_returns):
    return np.sum(weights * log_returns.mean()) * 252

def sharpe_ratio(weights,log_returns):
    fred = Fred(api_key='5aded4853821613438645e66c9b39603')
    rf_rate = fred.get_series_latest_release('GS10').iloc[-1] / 100
    return (expected_returns(weights,log_returns) - rf_rate) / std_dev(weights,log_returns)

def loss_function(weights,log_returns):
    return - sharpe_ratio(weights,log_returns)

def minimizer(initial_weights, log_returns, min_weight, max_weight):
    const = {'type':'eq', 'fun':lambda weights: np.sum(weights) - 1}
    bounds = [ (min_weight,max_weight) for _ in range(len(initial_weights)) ]
    res = minimize(loss_function,initial_weights,args=(log_returns),method='SLSQP',constraints=const,bounds=bounds).x


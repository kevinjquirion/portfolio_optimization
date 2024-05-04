
# Current Status #

Overview: This repository contains two files: functions, which contains all of the classes and functions of the repository, and main, which provides an example of what the functions can do.


# functions.py #


## class: functions.asset_data(tickers, start_date, end_date, price_column = 'Adj Close')

  Holds the details of the portfolio, as well as the ability to update the relative weights of the holdings.

  ### Parameters:
  
    tickers: list
      list of tickers to pull from market data using yahoo finance
      
    start_date: datetime object
      first day to pull market data to fill your portfolio
      
    end_date: datetime object
      last day to pull market data to dill your portfolio
      
    price_column: str
      column name to use for calculating returns, volitility, and other metrics.

  ### Attributes:
  
    rf_rate: the risk free rate taken as the 10 year US treasury note yield
    
    prices: daily prices of 'tickers' determined by 'price_column'
    
    log_return: daily change of each portfolio element measured as log(1 + pct_change)
    
    weights: relative weight of each portfolio element normalized to sum(weights) = 1
    
    exp_returns: annualized scalar product of the weights and the mean log_returns for each portfolio element
    
    covariance: covariance calculated using weights and log_returns
    
    volatility: square root of the variance of the annualized log_returns
    
    sharpe_ratio: excess returns over the risk free rate divided by the volatility

  ### Methods:
  
    set_weights(weights):
    
      recalculate volatility, expected returns, and sharpe ratio given new weights for the portfolio
      

## function: reweighted_sharpe(new_weights, asset_data)

  return the sharpe ratio of the asset data with new weights

  ### parameters:

    new_weights: values to assign the weights attribute of the asset_data class

    asset_data: an instance of the asset_data class to update

  ### returns:

    reweighted Sharpe ratio

## function: reweighted_volatility(new_weights, asset_data)

  return the volatility of the asset data with new weights

  ### parameters:

    new_weights: values to assign the weights attribute of the asset_data class

    asset_data: an instance of the asset_data class to update

  ### returns:

    reweighted volatility

## function: optimize_sharpe_ratio(asset_data, initial_guess, min_weight, max_weight)

  ### parameters 
    
    

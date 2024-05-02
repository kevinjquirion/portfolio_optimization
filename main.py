from datetime import datetime,timedelta
from functions import adj_close,log_daily_change,sharpe_ratio,minimizer
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

tickers = ['SPY','QQQ','BND','GLD','VTI']

years = 1
end_date = datetime.today()
start_date = end_date - timedelta(days=365*years)

print("Start Date: {}, End Date: {}".format(start_date,end_date))

closes = adj_close(tickers,start_date,end_date)
log_norms = log_daily_change(closes)
log_norms = log_norms.dropna()

cov = log_norms.cov()*252
initial_guess = [ 1/len(tickers) for _ in tickers ]
weights = minimizer(initial_guess, log_norms, 0., 1.0)
print(weights)


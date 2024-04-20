import numpy as np
from scipy.stats import norm
import os
import csv
import math
cwd = os.getcwd()

#question 11,
# When the volatility increases, the call option price will also increase. However, the call option price is capped by
# the current stock price, because PV_call = E[max(S_T - K)]
# So the answer is: the call option price is the current stock price when the volatility is infinity.
# I also buiit a simple Black-Scholes call option price function which verified my answer

def BS_call(S, K, T, r, v):
    """
    S : Current stock price
    K : Strike price
    T : Time to maturity
    r : Constat Risk-free interest rate
    v : Constat Volatility of the underlying stock
    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)

    # Calculate the call price
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

# set input
S = 100
K = 90
T =1
r= 0.05
volatility_list =[0.1, 1, 10, 100, 1000, 10000]
call_price_list =[]

for v in volatility_list:
    call_price = BS_call(S, K, T, r, v)
    call_price_list.append(call_price)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------

# question 12

# get the 1 year historical data of S&P500, there are 252 observations

SP500_file = cwd + "//input_data//SP500.csv"

SP500_raw_data = []

SP500_historical_data = []

with open(SP500_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        SP500_raw_data.append(row)

# clean the data, range start with 1 because the first row is not data

for i in range(1, len(SP500_raw_data)):
    SP500_historical_data.append(float(SP500_raw_data[i][4]))

# volatility estimation
number_observation = len(SP500_historical_data)
number_return = number_observation -1

# calculate the log return
log_return = []
for i in range(1, len(SP500_historical_data)):
    log_return.append(math.log(SP500_historical_data[i]/math.log(SP500_historical_data[i-1])))

log_return_np = np.array(log_return)

# calculate the standard deviation of sp500

sp500_volatility = np.std(log_return_np)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------

# question 14
# The delta of a put option on ETF SPY is always negative, the delta will increase and move asymptotically to 0.


# The put option is to hedge the portfolio value when the ETF SPY index changes.

# Instead of holding the put option (in this case we need to pay put option premium). We could also use Delta hedging.
# The strategy is dynamically short the Delta(ETF SPY index) number of ETF SPY stock,
# The change in portfolio and the change in the ETF SPY stock position will perfectly cancel out each other.


# I also buiit a simple Black-Scholes put option delta value function which verified show the relation between stock price and delta value

def BS_Put_Delta(S, K, T, r, v):
    """
     S : Current stock price
     K : Strike price
     T : Time to maturity
     r : Constat Risk-free interest rate
     v : Constat Volatility of the underlying stock
     """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)

    # Calculate put option delta
    put_delta = - np.exp(-r * T) * norm.cdf(-d1)
    return put_delta

Stock_Price_list = [100, 110, 120, 130, 150, 200, 300, 400, 500]
K = 100
T =1
r= 0.05
v = 0.1
call_price_list =[]
put_delta_list = []
for S in Stock_Price_list:
    put_delta = BS_Put_Delta(S, K, T, r, v)
    put_delta_list .append(put_delta)



print("ok")


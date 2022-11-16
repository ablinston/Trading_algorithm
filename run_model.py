import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s
import random as r
import statistics as st
import os
from datetime import datetime
from itertools import product

# Read in data

raw_prices = pd.read_csv("C:/Users/Andy/Documents/Trading_algorithm/^VIX.csv")

# Change format of the date column
raw_prices[["Year", "Month", "Day"]] = raw_prices["Date"].str.split("-", expand = True)
raw_prices["Year"] = pd.to_numeric(raw_prices["Year"])
raw_prices["Month"] = pd.to_numeric(raw_prices["Month"])

raw_prices.head()
raw_prices["Date_ft"] = raw_prices["Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

prices_by_month_year = raw_prices.groupby(by = ["Year", "Month"]).mean()
prices_by_month_year.head()

plt.plot(list(prices_by_month_year["Open"]))


# Initial exploratory analysis
len(raw_prices.index)

raw_prices["Low"].min()
raw_prices["High"].max()


##############################################
# Run algorithm

init_balance = 12000
max_exposure = 50000
global_end_loss = True
global_overnight_rate = 0.065 / 365

initial_buy_prices = list(np.linspace(10, 30, 8))
initial_sell_prices = list(np.linspace(20, 80, 8))

# len(initial_buy_prices) * len(initial_sell_prices)

train_data = raw_prices[raw_prices["Year"] < 2008].reset_index(drop = True)
test_data = raw_prices[raw_prices["Year"] >= 2010].reset_index(drop = True)


# Get first set of results
results = best_trading_results(train_data,
                              initial_buy_prices,
                              initial_sell_prices,
                              max_exposure = max_exposure,
                              initial_balance = init_balance,
                              end_loss = global_end_loss,
                              overnight_rate = global_overnight_rate)
    
                       
print(results.loc[0])
overall_results = results.loc[0]

# Try these results on random cuts of the training data
r.seed(4234)

for i in range(1,20):
    
    # We want at least 5 years of data
    n_years = r.randrange(5, 15)
    start_year = r.randrange(min(train_data["Year"]),
                             max(train_data["Year"]) - n_years)
    start_month = r.randrange(1, 13)
    
    train_data_filtered = train_data[(train_data.Year >= start_year) &
                                     (train_data.Year <= (start_year + n_years))]
    
    train_data_filtered = train_data_filtered[~((train_data_filtered.Year == start_year) &
                                              (train_data_filtered.Month < start_month))]
    
    train_data_filtered = train_data_filtered[~((train_data_filtered.Year == train_data_filtered.Year.max()) &
                                              (train_data_filtered.Month > start_month))]
    
    train_data_filtered = train_data_filtered.reset_index(drop = True)
    
    # Now work out the best buy and sell
    results_filt = best_trading_results(train_data_filtered,
                                          initial_buy_prices,
                                          initial_sell_prices,
                                          max_exposure = max_exposure,
                                          initial_balance = init_balance,
                                          end_loss = global_end_loss,
                                          overnight_rate = global_overnight_rate)
    
    print(f"Starting in {start_month}/{start_year} for {n_years} years:")
    print(results_filt.loc[0])

    del results_filt
    del train_data_filtered
    
##########################################


# Vectorised
initial_buy_prices = list(np.linspace(10, 50, 500))
initial_sell_prices = list(np.linspace(20, 80, 500))

results = pd.DataFrame(list(product(initial_buy_prices, initial_sell_prices)),
                       columns = ["Buy", "Sell"])
# Remove where buy > sell
results = results[results.Sell > results.Buy]

results["profit"] = calculate_profit_vector(train_data,
                              results["Buy"],
                              results["Sell"],
                              max_exposure = max_exposure,
                              initial_balance = init_balance,
                              end_loss = global_end_loss,
                              overnight_rate = global_overnight_rate)

results[results.profit > 0]["Buy"].min()
results[results.profit > 0]["Buy"].max()
results[results.profit > 0]["Sell"].min()
results[results.profit > 0]["Sell"].max()

results = results.sort_values(by = "profit", ascending = False)




calculate_profit_yearly(train_data, 17.1, 48.4, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)
calculate_profit_yearly(train_data, 20.5, 27, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)

calculate_profit(train_data, 17.1, 48.4, max_exposure = 2e4, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(train_data, 21, 23.7, max_exposure = 5e4, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(train_data, 21, 24, max_exposure = 5e4, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)



calculate_profit_yearly(test_data, 20, 27, 
                        max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit_yearly(test_data, 21, 23.7, 
                        max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)

calculate_profit(test_data, 21, 24, 
                 max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(test_data, 21, 23.7, 
                 max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(test_data, 20, 27, 
                 max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)






##############################################
##############################################
##############################################
# For testing the algorithm to converge on max profits

init_balance = 12000
max_exposure = 50000
global_end_loss = True
global_overnight_rate = 0.065 / 365

initial_buy_prices = list(np.linspace(10, 30, 300))
initial_sell_prices = list(np.linspace(20, 80, 400))

len(initial_buy_prices) * len(initial_sell_prices)

train_data = raw_prices[raw_prices["Year"] < 2008].reset_index(drop = True)
test_data = raw_prices[raw_prices["Year"] >= 2010].reset_index(drop = True)

# We want at least 5 years of data
n_years = r.randrange(5, 15)
start_year = r.randrange(min(train_data["Year"]),
                         max(train_data["Year"]) - n_years)
start_month = r.randrange(1, 13)

train_data_filtered = train_data[(train_data.Year >= start_year) &
                                 (train_data.Year <= (start_year + n_years))]

train_data_filtered = train_data_filtered[~((train_data_filtered.Year == start_year) &
                                          (train_data_filtered.Month < start_month))]

train_data_filtered = train_data_filtered[~((train_data_filtered.Year == train_data_filtered.Year.max()) &
                                          (train_data_filtered.Month > start_month))]
   
train_data_filtered = train_data_filtered.reset_index(drop = True)

# Get first set of results
results_full = list_trading_profit(train_data_filtered,
                              initial_buy_prices,
                              initial_sell_prices,
                              max_exposure = max_exposure,
                              initial_balance = init_balance,
                              end_loss = global_end_loss,
                              overnight_rate = global_overnight_rate)
    
                       
print(results_full.loc[0])

##############################################
##############################################
##############################################
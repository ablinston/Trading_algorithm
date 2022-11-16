import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s
import random as r
import statistics as st
import os
from datetime import datetime

# Read in data

raw_prices = pd.read_csv("C:/Users/Andy/Documents/Trading_algorithm/^VIX.csv")

# Change format of the date column
raw_prices[["Year", "Month", "Day"]] = raw_prices["Date"].str.split("-", expand = True)
raw_prices["Year"] = pd.to_numeric(raw_prices["Year"])
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

initial_buy_prices = list(range(10, 30, 5))
initial_sell_prices = list(range(20, 80, 10))

results = pd.DataFrame([[0,0,0]], columns = ["buy_price", "sell_price", "profit"])

# len(initial_buy_prices) * len(initial_sell_prices)

train_data = raw_prices[raw_prices["Year"] < 2008].reset_index(drop = True)
test_data = raw_prices[raw_prices["Year"] >= 2010].reset_index(drop = True)


# First run through the initial prices and populate the results for the Monte Carlo runs to work with
count = 0
for buy_price in initial_buy_prices:
    for sell_price in initial_sell_prices:
        # Add values to end of the dataframe
        if buy_price < sell_price:
            results.loc[-1] = [buy_price,
                               sell_price,
                               calculate_profit(train_data, buy_price, sell_price, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)]
            results.index += 1
            count += 1
            print(f"Run {count} done.")
        

# Now run the main simulation and wait for it to converge on an answer
r.seed(1021)
results = mcmc_profit(results, train_data, max_exposure = max_exposure, initial_balance = init_balance, min_iterations = 10, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
                       
print(results)

results2 = results
##########################################



calculate_profit_yearly(train_data, 17.1, 48.4, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)
calculate_profit_yearly(train_data, 20.5, 27, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)

calculate_profit(train_data, 17.1, 48.4, max_exposure = 2e4, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(train_data, 21, 23.7, max_exposure = 5e4, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(train_data, 21, 24, max_exposure = 5e4, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)



calculate_profit_yearly(test_data, 21, 24, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit_yearly(test_data, 21, 23.7, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)

calculate_profit(test_data, 21, 24, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, , overnight_rate = global_overnight_rate)
calculate_profit(test_data, 21, 23.7, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(test_data, 20, 27, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)

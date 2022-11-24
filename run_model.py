import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s
import random as r
import statistics as st
import os
import time
from datetime import datetime
from itertools import product
import polars as pl

os.chdir("C:/Users/Andy/Documents/VIX_trading_algorithm")

# Read in data

raw_prices = pd.read_feather("Processed_data.feather")


# Convert to a polars data frame for speed
#raw_prices = pl.DataFrame(raw_prices)

prices_by_month_year = raw_prices.groupby(by = ["Year", "Month"]).mean(numeric_only = True)
prices_by_month_year.head()

#plt.plot(list(prices_by_month_year["Open"]))

raw_prices["Low"].min()
raw_prices["High"].max()


##############################################
# Run algorithm

init_balance = 20000
max_exposure = 0.5
global_end_loss = True

# =============================================================================
# initial_buy_prices = list(np.linspace(10, 30, 8))
# initial_sell_prices = list(np.linspace(20, 80, 8))
# =============================================================================

# len(initial_buy_prices) * len(initial_sell_prices)

train_data = raw_prices[raw_prices["Year"] < 2017].reset_index(drop = True)
test_data = raw_prices[raw_prices["Year"] >= 2017].reset_index(drop = True)

# =============================================================================
# 
# # Get first set of results
# results = best_trading_results(train_data,
#                               initial_buy_prices,
#                               initial_sell_prices,
#                               max_exposure = max_exposure,
#                               initial_balance = init_balance,
#                               end_loss = global_end_loss,
#                               overnight_rate = global_overnight_rate)
#     
#                        
# print(results.loc[0])
# overall_results = results.loc[0]
# 
# =============================================================================
# =============================================================================
# # Try these results on random cuts of the training data
# r.seed(4234)
# 
# for i in range(1,20):
#     
#     # We want at least 5 years of data
#     n_years = r.randrange(5, 15)
#     start_year = r.randrange(min(train_data["Year"]),
#                              max(train_data["Year"]) - n_years)
#     start_month = r.randrange(1, 13)
#     
#     train_data_filtered = train_data[(train_data.Year >= start_year) &
#                                      (train_data.Year <= (start_year + n_years))]
#     
#     train_data_filtered = train_data_filtered[~((train_data_filtered.Year == start_year) &
#                                               (train_data_filtered.Month < start_month))]
#     
#     train_data_filtered = train_data_filtered[~((train_data_filtered.Year == train_data_filtered.Year.max()) &
#                                               (train_data_filtered.Month > start_month))]
#     
#     train_data_filtered = train_data_filtered.reset_index(drop = True)
#     
#     # Now work out the best buy and sell
#     results_filt = best_trading_results(train_data_filtered,
#                                           initial_buy_prices,
#                                           initial_sell_prices,
#                                           max_exposure = max_exposure,
#                                           initial_balance = init_balance,
#                                           end_loss = global_end_loss,
#                                           overnight_rate = global_overnight_rate)
#     
#     print(f"Starting in {start_month}/{start_year} for {n_years} years:")
#     print(results_filt.loc[0])
# 
#     del results_filt
#     del train_data_filtered
# =============================================================================
    
##########################################

start_time = time.time()

# Vectorised
initial_buy_prices = list(np.linspace(10, 40, 100))
initial_sell_prices = list(np.linspace(20, 50, 100))
initial_stop_losses = list(np.linspace(0, 40, 100))

results = pd.DataFrame(list(product(initial_buy_prices, 
                                    initial_sell_prices,
                                    initial_stop_losses)),
                       columns = ["Buy", "Sell", "Stop"])

# Remove where buy > sell
results = results[results.Sell > (results.Buy + 0.15)] # spread added
results = results[results.Buy > (results.Stop + 0.15)] # spread added

results["profit"] = calculate_profit_vector(train_data,
                                              results["Buy"],
                                              results["Sell"],
                                              results["Stop"],
                                              max_exposure = max_exposure,
                                              initial_balance = init_balance,
                                              end_loss = global_end_loss)

print("--- %s seconds ---" % (time.time() - start_time))

print(results.sort_values("profit", ascending = False))

results["max_profit"] = results["profit"].max()
# Work out the range within 5%
best_results = results[results.profit > 0.90 * results.max_profit]

print(best_results["Buy"].min())
print(best_results["Buy"].max())
print(best_results["Sell"].min())
print(best_results["Sell"].max())
print(best_results["Stop"].min())
print(best_results["Stop"].max())

##########################################

start_time = time.time()

# Vectorised
initial_buy_prices = list(np.linspace(8, 18, 200))
initial_sell_prices = list(np.linspace(8, 30, 300))

results = pd.DataFrame(list(product(initial_buy_prices, initial_sell_prices)),
                       columns = ["Buy", "Sell"])
# Remove where buy > sell
results = results[results.Sell > (results.Buy + 0.15)] # spread added

results["profit"] = calculate_profit_vector(train_data,
                                              results["Buy"],
                                              results["Sell"],
                                              max_exposure = max_exposure,
                                              initial_balance = init_balance,
                                              end_loss = global_end_loss)

print("--- %s seconds ---" % (time.time() - start_time))

print(results.sort_values("profit", ascending = False))

results["max_profit"] = results["profit"].max()
# Work out the range within 5%
best_results2 = results[results.profit > 0.95 * results.max_profit]

print(best_results2["Buy"].min())
print(best_results2["Buy"].max())
print(best_results2["Sell"].min())
print(best_results2["Sell"].max())




##########################################

calculate_profit_yearly(test_data, [13.8], [20],[12], max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)

# Test the model with a monte carlo

one_year_monte_carlo = monte_carlo_test_runs(data = test_data,
                                             n_iterations = 1000,
                                             n_years = 1,
                                             buy_prices = [16.1], 
                                             sell_prices = [28.3], 
                                             max_exposure = max_exposure, 
                                             initial_balance = init_balance, 
                                             end_loss = True, 
                                             overnight_rate = global_overnight_rate)

# Plot the results
one_year_monte_carlo["Percent_profit"].plot.hist(grid = True,
                                                 bins = 20)

loser_info(one_year_monte_carlo)


##########################################

# Test combo of models

calculate_profit_yearly(test_data, 11.6, 21.1, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)

one_year_monte_carlo_combo = monte_carlo_test_runs(data = test_data,
                                                 n_iterations = 1000,
                                                 n_years = 1,
                                                 buy_prices = [11.6, 21.1], 
                                                 sell_prices = [13.9, 23.9], 
                                                 max_exposure = max_exposure, 
                                                 initial_balance = init_balance, 
                                                 end_loss = True, 
                                                 overnight_rate = global_overnight_rate
                                                 ).groupby("mc_run").sum()


# Plot the results
one_year_monte_carlo_combo["Percent_profit"].plot.hist(grid = True, bins = 20)

loser_info(one_year_monte_carlo_combo)

##########################################

calculate_profit_yearly(test_data, 
                        [11.6, 21.1], 
                        [13.9, 23.9], 
                        max_exposure = max_exposure, 
                        initial_balance = init_balance, 
                        end_loss = True, 
                        overnight_rate = global_overnight_rate)

two_year_monte_carlo_combo = monte_carlo_test_runs(data = test_data,
                                                 n_iterations = 1000,
                                                 n_years = 2,
                                                 buy_prices = [11.6, 21.1], 
                                                 sell_prices = [13.9, 23.9], 
                                                 max_exposure = max_exposure, 
                                                 initial_balance = init_balance, 
                                                 end_loss = True, 
                                                 overnight_rate = global_overnight_rate
                                                 ).groupby("mc_run").sum()

# Plot the results
two_year_monte_carlo_combo["Percent_profit"].plot.hist(grid = True, bins = 20)

loser_info(two_year_monte_carlo_combo)




















one_year_monte_carlo.head(1000)

calculate_profit_yearly(train_data, 21.3, 23.3, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)
calculate_profit_yearly(train_data, 21, 24, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)

calculate_profit(train_data, 21.137124, 22.909699, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(train_data, 21, 23.7, max_exposure = 5e4, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(train_data, 21, 24, max_exposure = 5e4, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)



calculate_profit_yearly(test_data, 21.1, 23.9, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit_yearly(test_data, 21, 24, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)

calculate_profit(test_data, 21, 24, 
                 max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(test_data, 21, 23.7, 
                 max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)
calculate_profit(test_data, 20, 27, 
                 max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)



calculate_profit_yearly(raw_prices[raw_prices.Year > 1991].reset_index(drop = True), 21.3, 23.3, max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss, overnight_rate = global_overnight_rate)


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


    

##############################################
##############################################
##############################################
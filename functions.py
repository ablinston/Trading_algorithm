import numpy as np
import pandas as pd
import random as r
import statistics as st
from datetime import datetime

# =============================================================================
# def calculate_profit(data, buy_price, sell_price, max_exposure = 1e10, initial_balance = 1e4, end_loss = True, overnight_rate = 0):
#  
#     # Initial variables
#     hold = False
#     shares = 0
#     balance = initial_balance
#     
#     # Loop through each day
#     for i in range(0, len(data.index)):
#         # Check if we're on the last day and sell if so
#         if ((i == len(data.index) - 1) & hold & end_loss):
#             balance += (data["Open"][i] - buy_price) * shares
#         # If we're already holding, look to sell
#         elif hold:
#             if data["High"][i] >= sell_price:
#                 # Calculate the profit
#                 balance += (sell_price - buy_price) * shares
#                 # No long holding
#                 hold = False
#             else:
#                 # We have to charge the overnight rate
#                 pass
#         else:
#             # Check if there is an opportunity to buy
#             if data["Low"][i] <= buy_price:
#                 # Work out how many shares you have
#                 shares = min(max_exposure, balance) / buy_price
#                 hold = True
#     return balance - initial_balance
# =============================================================================

# The overnight rate is a daily % based on sonia + 2.5% usually
# SONIA has been between 4-8% until post 2008

def calculate_profit(data, 
                     buy_price, 
                     sell_price, 
                     max_exposure = 1e10, 
                     initial_balance = 1e4, 
                     end_loss = False, 
                     overnight_rate = (0.025 / 365)):
 
    # Initial variables
    hold = False
    bet_per_pt = 0
    balance = initial_balance
    
    # Loop through each day
    for i in range(1, len(data.index)):
        # Check if we're on the last day and sell if so
        if ((i == len(data.index) - 1) & hold & end_loss):
            balance += (data["Open"][i] - buy_price) * bet_per_pt
        # If we're already holding, look to sell
        elif hold:
            if data["High"][i] >= sell_price:
                # Calculate the profit
                balance += (sell_price - buy_price) * bet_per_pt
                # No long holding
                hold = False
            else:
                # We have to charge the overnight rate
                balance -= (data["Date_ft"][i] - data["Date_ft"][i - 1]).days * data["Open"][i] * bet_per_pt * overnight_rate
        else:
            # Check if there is an opportunity to buy
            if data["Low"][i] <= buy_price:
                # Work out how many shares you have
                bet_per_pt = min(max_exposure, balance * 0.8) / buy_price # always leave some balance to pay financing for a few years
                hold = True
    return balance - initial_balance



# This function runs four further runs from a sorted data frame

def add_markov_chain_runs(results, 
                          raw_prices, 
                          max_exposure, 
                          initial_balance, 
                          end_loss, 
                          overnight_rate):
    
    # We first do a run at the midpoint of the highest two profit points
    buy_price = np.mean(results["buy_price"][0:2])
    sell_price = np.mean(results["sell_price"][0:2])
    
    results.loc[-1] = [buy_price,
                       sell_price,
                       calculate_profit(raw_prices, buy_price, sell_price, max_exposure, initial_balance, end_loss = end_loss, overnight_rate = overnight_rate)]
    results.index += 1
    # Now do two random walks from the last data point and see whether it improves
    
    for run in range(1,10):
        
        # Add random walks to the prices based on how close the top two entries are
        buy_price = results["buy_price"][0] + r.gauss(0, max(1, st.stdev(results["buy_price"][0:2])))
        sell_price = results["sell_price"][0] + r.gauss(0, max(1, st.stdev(results["sell_price"][0:2])))
        
        if 0 < buy_price < sell_price:
            results.loc[-1] = [buy_price,
                               sell_price,
                               calculate_profit(raw_prices, buy_price, sell_price, max_exposure, initial_balance, end_loss = end_loss, overnight_rate = overnight_rate)]
            results.index += 1
    
    return results




def mcmc_profit(results, 
                raw_prices, 
                max_exposure, 
                initial_balance, 
                min_iterations, 
                end_loss, 
                overnight_rate):   
    
    # Sort all the results and reset the index
    results = results.sort_values(by = "profit", ascending = False)
    results = results.reset_index(drop = True)
    
    # Perform this many iterations as a minimum
    for n in range(0, min_iterations):
        
        # Now we need to use a Markov Chain Monte Carlo to narrow in on local minima
        results = add_markov_chain_runs(results, raw_prices, max_exposure, initial_balance, end_loss = end_loss, overnight_rate = overnight_rate)
        
        # Sort all the results and reset the index
        results = results.sort_values(by = "profit", ascending = False)
        results = results.reset_index(drop = True)
        
    # Now continue runs until we are within a 0.1 price range
    while ((max(results["buy_price"][0:4]) - min(results["buy_price"][0:4])) > 0.1) | ((max(results["sell_price"][0:4]) - min(results["sell_price"][0:4])) > 0.1):
       # Continue the Markov Chain
       results = add_markov_chain_runs(results, raw_prices, max_exposure, initial_balance, end_loss = end_loss, overnight_rate = overnight_rate)
       
       # Sort all the results and reset the index
       results = results.sort_values(by = "profit", ascending = False)
       results = results.reset_index(drop = True)
    
    
    return results

# This function calculates profits for single years only
def calculate_profit_yearly(data, 
                            buy_price, 
                            sell_price, 
                            max_exposure, 
                            initial_balance, 
                            end_loss = False, 
                            overnight_rate = (0.025 / 365)):

    # Create a results dataframe
    min_year = data["Year"].min()
    max_year = data["Year"].max()
    yearly_results = pd.DataFrame({"Year": range(min_year, max_year + 1),
                                   "Balance": ([0] * (max_year - min_year + 1))})
    
    # Initial variables
    hold = False
    bet_per_pt = 0
    balance = initial_balance
    
    # Loop through each day
    for i in range(1, len(data.index)):
        # Check if we're on the last day and sell if so
        if ((i == len(data.index) - 1)):
            if hold & end_loss:
                balance += (data["Open"][i] - buy_price) * bet_per_pt
            yearly_results.loc[yearly_results["Year"] == data["Year"][i], "Balance"] += balance
            
        # If we're already holding, look to sell
        elif hold:
            if data["High"][i] >= sell_price:
                # Calculate the profit
                balance += (sell_price - buy_price) * bet_per_pt
                # No long holding
                hold = False
            else:
                # We have to charge the overnight rate
                balance -= (data["Date_ft"][i] - data["Date_ft"][i - 1]).days * data["Open"][i] * bet_per_pt * overnight_rate
        else:
            # Check if there is an opportunity to buy
            if data["Low"][i] <= buy_price:
                # Work out how many shares you have
                bet_per_pt = min(max_exposure, balance * 0.8) / buy_price # always leave some balance to pay financing for a few years
                hold = True
            # If we're on the first day of the year, record the balance
        if data["Year"][i] > data["Year"][i - 1]:
            yearly_results.loc[yearly_results["Year"] == (data["Year"][i - 1]), "Balance"] = balance
                
    return yearly_results


def iterative_trading_profit(training_data, 
                             buy_list, 
                             sell_list, 
                             max_exposure, 
                             initial_balance, 
                             end_loss, 
                             overnight_rate):
    
    results_data = pd.DataFrame({"Buy":, "Sell":, "Profit":})
    
    for buy_price in buy_list:
        for sell_price in sell_list:
            # Add values to end of the dataframe
            if buy_price < sell_price:
                results.loc[-1] = [buy_price,
                                   sell_price,
                                   calculate_profit(train_data, 
                                                    buy_price, 
                                                    sell_price, 
                                                    max_exposure = max_exposure, 
                                                    initial_balance = initial_balance, 
                                                    end_loss = end_loss, 
                                                    overnight_rate = overnight_rate)]
                results.index += 1
    
    return results
    
    
    
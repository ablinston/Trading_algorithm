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
       
        # If we're already holding, look to sell
        if hold:
            if data["High"][i] >= sell_price:
                # Calculate the profit
                balance += (sell_price - buy_price) * bet_per_pt
                # No long holding
                hold = False
                # Check if we're on the last day and sell if so
            elif ((i == len(data.index) - 1) & end_loss):
                balance += (data["Open"][i] - buy_price) * bet_per_pt
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

# This function calculates the profits for vectors of buy prices and corresponding sell prices
def calculate_profit_vector(data, 
                             buy_prices, 
                             sell_prices, 
                             max_exposure = 1e10, 
                             initial_balance = 1e4, 
                             end_loss = False, 
                             overnight_rate = (0.025 / 365)):
 
    # For debugging
    data = raw_prices
    buy_prices = [10, 20, 30]
    sell_prices = [20, 30, 40]
    max_exposure = 1e5
    initial_balance = 1e4
    end_loss = False
    overnight_rate = (0.025 / 365)
    
    # Initial variables
    results_data = pl.DataFrame({"buy_price": buy_prices,
                                 "sell_price": sell_prices,
                                 "hold": [0] * len(buy_prices),
                                 "bet_per_pt": [0] * len(buy_prices),
                                 "balance": [initial_balance] * len(buy_prices)})
       
    results_data = results_data.with_column((pl.col("sell_price") - pl.col("buy_price") - 0.15).alias("buy_sell_diff")) # includes the spread paid deducted
    
    # Loop through each day
    for i in range(1, (len(data.index) - 1)):
       
        # Calculate overnight costs
        results_data = results_data.with_column((pl.col("bet_per_pt") * 
                                                 (data["Date_ft"][i] - data["Date_ft"][i - 1]).days * 
                                                 data["Open"][i] *
                                                 overnight_rate
                                                ).alias("overnight_costs"))

        # Update balances with costs
        results_data = results_data.with_column((pl.col("balance") - pl.col("overnight_costs")).alias("balance"))
        
        # Check if we've hit a selling opportunity in the day
        
        results_data = results_data.with_column(((pl.col("sell_price") > data["High"][i]) &
                                                 (pl.col("sell_price") < data["Low"][i])
                                                 ).alias("sell_ind"))
        
        # Sell out holding for those where it's true
        results_data = results_data.with_column((pl.col("balance") + 
                                                 pl.col("sell_ind") *
                                                 pl.col("bet_per_pt") *
                                                 pl.col("buy_sell_diff")
                                                 ).alias("balance"))
                
        # Check if we've hit a day for buying
        results_data["buy_ind"] = ((results_data["buy_price"] < data["High"][i]) & 
                                   (results_data["buy_price"] > data["Low"][i]))
        
        # Work out the size of bet available
        results_data["size_of_bet"] = results_data["balance"].apply(lambda x: min(x * 0.8, max_exposure))
        
        # Now bet on the shares where appropriate
        results_data["bet_per_pt"] = ~results_data["sell_ind"] * results_data["buy_ind"] * results_data["size_of_bet"] / results_data["buy_price"] # Add in opposite of sell so no buying and selling on same day
        

    # On the last day, sell out if necessary
    if end_loss:
        results_data["balance"] += results_data["bet_per_pt"] * (data["Open"][len(data.index) - 1] - results_data["buy_price"])
    
    return results_data["balance"] - initial_balance


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


def list_trading_profit(training_data, 
                        buy_list, 
                        sell_list, 
                        max_exposure, 
                        initial_balance, 
                        end_loss, 
                        overnight_rate):
    
    results = pd.DataFrame({"Buy":[0], "Sell":[80], "profit":[0]})
    
    for buy_price in buy_list:
        for sell_price in sell_list:
            # Add values to end of the dataframe
            if buy_price < sell_price:
                results.loc[-1] = [buy_price,
                                   sell_price,
                                   calculate_profit(training_data, 
                                                    buy_price, 
                                                    sell_price, 
                                                    max_exposure = max_exposure, 
                                                    initial_balance = initial_balance, 
                                                    end_loss = end_loss, 
                                                    overnight_rate = overnight_rate)]
                results.index += 1
    
    return results
    
    
def best_trading_results(training_data, 
                        buy_list, 
                        sell_list, 
                        max_exposure, 
                        initial_balance, 
                        end_loss, 
                        overnight_rate):
    
    # Get first set of results
    results = list_trading_profit(training_data,
                                  buy_list,
                                  sell_list,
                                  max_exposure = max_exposure,
                                  initial_balance = initial_balance,
                                  end_loss = end_loss,
                                  overnight_rate = overnight_rate)

    # Sort all the results and reset the index
    results = results.sort_values(by = "profit", ascending = False)
    results = results.reset_index(drop = True)

    buy_range = results[0:9]["Buy"].max() - results[0:9]["Buy"].min()
    sell_range = results[0:9]["Sell"].max() - results[0:9]["Sell"].min()


    # Now do an iterative list of computations to narrow down the values
    it = 0
    
    while (buy_range > 0.2 or sell_range > 0.2):
        
        it += 1
        
        new_buy_list = np.linspace(results[0:9]["Buy"].min() + r.gauss(0, 0.2 * st.stdev(results[0:9]["Buy"])), # add random to avoid getting stuck in infinite loop
                                   results[0:9]["Buy"].max() + r.gauss(0, 0.2 * st.stdev(results[0:9]["Buy"])),
                                   8)

        new_sell_list = np.linspace((results[0:9]["Sell"].min() + r.gauss(0, 0.2 * st.stdev(results[0:9]["Sell"]))), 
                                   (results[0:9]["Sell"].max() + r.gauss(0, 0.2 * st.stdev(results[0:9]["Sell"]))),
                                   8)
        
        results = pd.concat([results,
                            list_trading_profit(training_data,
                                                new_buy_list, #  ignore first and last as they've already been calc'd
                                                new_sell_list,
                                                max_exposure = max_exposure,
                                                initial_balance = initial_balance,
                                                end_loss = end_loss,
                                                overnight_rate = overnight_rate)])
        
        # Sort all the results and reset the index
        results = results.sort_values(by = "profit", ascending = False)
        results = results.reset_index(drop = True)
        
        buy_range = results[0:9]["Buy"].max() - results[0:9]["Buy"].min()
        sell_range = results[0:9]["Sell"].max() - results[0:9]["Sell"].min()
        
        print(f"Converging: current buy range is {buy_range} and sell range is {sell_range}")
        
        if it == 10:
            print(f"Not converged. Exiting. Current buy range is {buy_range} and sell range is {sell_range}")
            break
        
    return results

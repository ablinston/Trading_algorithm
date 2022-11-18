import numpy as np
import pandas as pd
import random as r
import statistics as st
from datetime import datetime
import polars as pl

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
                     max_exposure = 1, # a proportion of balance to be bet
                     initial_balance = 1e4, 
                     end_loss = False, 
                     overnight_rate = (0.025 / 365),
                     daily_balances = False):
 
    # Initial variables
    hold = False
    bet_per_pt = 0
    balance = initial_balance
    
    daily_balance_data = pd.DataFrame({"Date":["1900-01-01"],
                                       "High":[0.1],
                                       "Low":[0.1],
                                       "bet_per_pt":[0.1],
                                       "balance":[0]})
    
    # Loop through each day
    for i in range(1, len(data.index)):
       
        # If we're already holding, look to sell
        if hold:
            # We have to charge the overnight rate
            balance -= (data["Date_ft"][i] - data["Date_ft"][i - 1]).days * data["Open"][i] * bet_per_pt * overnight_rate
            if data["High"][i] > sell_price:
                # Calculate the profit
                balance += (sell_price - buy_price - 0.15) * bet_per_pt
                # No long holding
                bet_per_pt = 0
                hold = False
                # Check if we're on the last day and sell if so
            elif ((i == len(data.index) - 1) & end_loss):
                balance += (data["Open"][i] - buy_price - 0.15) * bet_per_pt
                
        else:
            # Check if there is an opportunity to buy
            if (data["Low"][i] < buy_price) & (data["High"][i] > buy_price):
                # Work out how many shares you have
                bet_per_pt = max_exposure * balance / buy_price # always leave some balance to pay financing for a few years
                hold = True
        
        if daily_balances:
            # Add balance to the data frame
            daily_balance_data = pd.concat([daily_balance_data,
                                        pd.DataFrame({"Date":[str(data["Date"][i])],
                                                      "High":[data["High"][i]],
                                                      "Low" :[data["Low"][i]],
                                                      "bet_per_pt": [bet_per_pt],
                                                      "balance":[balance]})],
                                       ignore_index = True)
    if daily_balances:
        
        # Convert Date column to proper date
        
        return daily_balance_data
    else:
        return balance - initial_balance

# This function calculates the profits for vectors of buy prices and corresponding sell prices
def calculate_profit_vector(data, 
                             buy_prices, 
                             sell_prices, 
                             max_exposure = 1, 
                             initial_balance = 1e4, 
                             end_loss = False, 
                             overnight_rate = (0.025 / 365),
                             daily_balances = False):
 
# =============================================================================
#     # For debugging
#     data = raw_prices
#     buy_prices = [10.1, 20.1, 20.1]
#     sell_prices = [20.1, 30.1, 40.1]
#     max_exposure = 0.5
#     initial_balance = 20000
#     end_loss = False
#     overnight_rate = (0.065 / 365)
#     daily_balances = True
# =============================================================================
    
    # Initial variables
    results_data = pl.DataFrame({"buy_price": buy_prices,
                                 "sell_price": sell_prices,
                                 "hold": [0] * len(buy_prices),
                                 "bet_per_pt": [0] * len(buy_prices),
                                 "balance": [initial_balance] * len(buy_prices)})
       
    results_data = results_data.with_column((pl.col("sell_price") - pl.col("buy_price") - 0.15).alias("buy_sell_diff")) # includes the spread paid deducted
    
    daily_balance_data = pl.DataFrame({"Date": ["1900-01-01"],
                                       "High": [0.1],
                                       "Low": [0.1],
                                       "buy_price": np.array([0], dtype = float),
                                       "sell_price": np.array([0], dtype = float),
                                       "bet_per_pt":[0.1],
                                       "balance": np.array([0], dtype = float)})
        
    
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
        results_data = results_data.with_column(((pl.col("bet_per_pt") > 0) &
                                                 (pl.col("sell_price") < data["High"][i])
                                                 ).alias("sell_ind"))
        
        # Sell out holding for those where it's true
        results_data = results_data.with_column((pl.col("balance") + 
                                                 pl.col("sell_ind") *
                                                 pl.col("bet_per_pt") *
                                                 pl.col("buy_sell_diff")
                                                 ).alias("balance"))
                        
        # Check if we've hit a day for buying before reseting bet (so we don't buy and sell on same day)
        results_data = results_data.with_column(((pl.col("bet_per_pt") == 0) &
                                                 (pl.col("buy_price") < data["High"][i]) &
                                                 (pl.col("buy_price") > data["Low"][i]))
                                                .alias("buy_ind"))
        
        # Set the bet to 0 if sell indicator flagged
        results_data = results_data.with_column((pl.when(pl.col("sell_ind")).
                                                 then(0).
                                                 otherwise(pl.col("bet_per_pt")))
                                                 .alias("bet_per_pt"))
        
        # Work out the size of bet available                
        # Now bet on the shares where appropriate
        results_data = results_data.with_column((pl.when(pl.col("buy_ind"))
                                                 .then(pl.col("balance") *
                                                       max_exposure /
                                                       pl.col("buy_price"))
                                                 .otherwise(pl.col("bet_per_pt")))
                                                .alias("bet_per_pt"))
        
        if daily_balances:
            # Add balance to the data frame
            daily_balance_data = pl.concat([daily_balance_data,
                                            pl.concat([pl.DataFrame({"Date":[str(data["Date"][i])] * len(results_data),
                                                                     "High":[data["High"][i]] * len(results_data),
                                                                     "Low":[data["Low"][i]] * len(results_data)}),
                                                                   results_data.select(["buy_price", 
                                                                                        "sell_price", 
                                                                                        "bet_per_pt",
                                                                                        "balance"])],
                                                      how = "horizontal")])
        
    # On the last day, sell out if necessary
    if end_loss:
        results_data = results_data.with_column((pl.col("balance") +
                                                 pl.col("bet_per_pt") *
                                                 (data["Open"][len(data.index) - 1] - pl.col("buy_price") - 0.15)
                                                 ).alias("balance"))
        
    if daily_balances:
        return daily_balance_data
    else:        
        return results_data["balance"] - initial_balance



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
            if (data["Low"][i] <= buy_price) & (data["High"][i] > buy_price):
                # Work out how many shares you have
                bet_per_pt = min(max_exposure, balance * 0.8) / buy_price # always leave some balance to pay financing for a few years
                hold = True
            # If we're on the first day of the year, record the balance
        if data["Year"][i] > data["Year"][i - 1]:
            yearly_results.loc[yearly_results["Year"] == (data["Year"][i - 1]), "Balance"] = balance
                
    return yearly_results


   
    
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

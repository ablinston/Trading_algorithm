


################################


first = calculate_profit(train_data, 20.1, 30.1, max_exposure = max_exposure, initial_balance = 20000.1, end_loss = global_end_loss, overnight_rate = global_overnight_rate, daily_balances = True)


# Vectorised
initial_buy_prices = [20.1]
initial_sell_prices = [30.1]

results = pd.DataFrame(list(product(initial_buy_prices, initial_sell_prices)),
                       columns = ["Buy", "Sell"])
# Remove where buy > sell

second = calculate_profit_vector(train_data,
                                              results["Buy"],
                                              results["Sell"],
                                              max_exposure = max_exposure,
                                              initial_balance = 20000.1,
                                              end_loss = global_end_loss,
                                              overnight_rate = global_overnight_rate,
                                              daily_balances = True)

first.head()
second = second.select(["Date", "bet_per_pt", "balance"]).to_pandas()
second.head()

first.merge(second, on = "Date").to_csv("C:/Users/Andy/Documents/Trading_algorithm/te.csv")


##############################################

te.to_pandas().to_csv("C:/Users/Andy/Documents/Trading_algorithm/te.csv")

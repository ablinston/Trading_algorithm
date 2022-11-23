import pandas as pd
import glob
from datetime import datetime, date, timedelta

data = pd.DataFrame(columns=["Trade Date",
                             "Futures",
                             "Open",
                             "High",
                             "Low",
                             "Close",
                             "Settle",
                             "Change",
                             "Total Volume",
                             "EFP",
                             "Open Interest",
                             "Expiry Date"])

# Loop through the directories and merge all the data together
for file_name in glob.glob("C:/Users/Andy/Documents/VIX_trading_algorithm/VIX futures data/2010s/" + "*.csv"):
    
    temp = pd.read_csv(file_name)
    
    # Format the date column
    temp["Trade Date"] = temp["Trade Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    
    # Add the expiry date as a date
    temp["Expiry Date"] = temp["Trade Date"].max()
    
    # Add to dataset
    data = pd.concat([data, temp], axis = 0)
    del(temp)

for file_name in glob.glob("C:/Users/Andy/Documents/VIX_trading_algorithm/VIX futures data/2020s/" + "*.csv"):
    
    temp = pd.read_csv(file_name)
    
    # Format the date column
    temp["Trade Date"] = temp["Trade Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    
    # Add the expiry date as a date
    temp["Expiry Date"] = temp["Trade Date"].max()
    
    # Add to dataset
    data = pd.concat([data, temp], axis = 0)
    del(temp)
    
for file_name in glob.glob("C:/Users/Andy/Documents/VIX_trading_algorithm/VIX futures data/Archived/" + "*.csv"):
    
    temp = pd.read_csv(file_name)
    
    # Format the date column
    temp["Trade Date"] = temp["Trade Date"].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    
    # Add the expiry date as a date
    temp["Expiry Date"] = temp["Trade Date"].max()
    
    # Add to dataset
    data = pd.concat([data, temp], axis = 0)
    del(temp)
    
# Filter out any zero rows and add string date to better matching
data["Date"] = data["Trade Date"].apply(lambda x: str(x.date()))
data = data[data.Open > 0].sort_values(["Trade Date", "Expiry Date", "Total Volume"],
                                       ascending = False)
data = data.drop_duplicates(subset = ["Date", "Expiry Date"], ).copy()
    
all_futures = pd.Series(data["Expiry Date"].unique()).sort_values().reset_index(drop = True)

# Create a lookup table to find the closest expiring future

date_lookup = pd.DataFrame(columns = ["Date_ft", "Date", "Next future", "Front future"])
start_date = all_futures.min()


for day in range(0, (all_futures.max() - start_date).days):
    
    # Save the date we're looking at
    todays_date = start_date + timedelta(day)
    
    # Find the closest expiring future by looping through
    for i in range(0, len(all_futures) - 1):
        if all_futures[i].date() < todays_date.date():
            continue
        else:
            # we've reached the end
            expiry = all_futures[i]
            next_expiry = all_futures[i + 1]
            break
            
    # Create the entry for this date
    date_lookup = pd.concat([date_lookup,
                             pd.DataFrame({"Date_ft": todays_date,
                                           "Date": str(todays_date.date()),
                                           "Next future": expiry,
                                           "Front future": next_expiry},
                                          index = [0])],
                            axis = 0)

date_lookup.head()

# Read in VIX ticker info

raw_prices = pd.read_csv("C:/Users/Andy/Documents/VIX_trading_algorithm/^VIX.csv")

# Remove data where we don't have high and low
raw_prices = raw_prices[raw_prices.High != raw_prices.Low]

# Change format of the date column
raw_prices[["Year", "Month", "Day"]] = raw_prices["Date"].str.split("-", expand = True)
raw_prices["Year"] = pd.to_numeric(raw_prices["Year"])
raw_prices["Month"] = pd.to_numeric(raw_prices["Month"])

raw_prices.head()
raw_prices["Date_ft"] = raw_prices["Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

# Add the columns we want to merge onto price data
data["Next_future_open"] = data["Open"]
data["Front_future_open"] = data["Open"]
data["Next_future_close"] = data["Close"]
data["Front_future_close"] = data["Close"]

# Now add on the columns
raw_prices_with_futures = pd.merge(raw_prices,
                                   date_lookup.drop("Date_ft", axis = 1),
                                   on = "Date")

raw_prices_with_futures = pd.merge(raw_prices_with_futures,
                                   data[["Date", "Expiry Date", "Next_future_open", "Next_future_close"]],
                                   left_on = ["Date", "Next future"],
                                   right_on = ["Date", "Expiry Date"]).drop("Expiry Date", axis = 1)

raw_prices_with_futures = pd.merge(raw_prices_with_futures,
                                   data[["Date", "Expiry Date", "Front_future_open", "Front_future_close"]],
                                   left_on = ["Date", "Front future"],
                                   right_on = ["Date", "Expiry Date"]).drop("Expiry Date", axis = 1)

# Finally, compute the overnight cost for a spreadbet
raw_prices_with_futures["overnight_cost_per_pt"] = ((raw_prices_with_futures["Front_future_open"] - 
                                                    raw_prices_with_futures["Next_future_open"]) / 31 +
                                                    (0.025 / 365 * raw_prices_with_futures["Open"]))

raw_prices_with_futures["projected_overnight_cost_per_pt"] = ((raw_prices_with_futures["Front_future_close"] - 
                                                    raw_prices_with_futures["Next_future_close"]) / 31 +
                                                    (0.025 / 365 * raw_prices_with_futures["Close"]))

raw_prices_with_futures.to_csv("C:/Users/Andy/Documents/VIX_trading_algorithm/Processed_data.csv",
                               index=False)

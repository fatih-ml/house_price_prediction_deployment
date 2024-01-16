import pandas as pd


def convert_location(new_data):
    new_data = {key: [value] for key, value in new_data.items()}
    X = pd.DataFrame(new_data)
    median_house_prices = pd.read_csv("data/neighborhood_median_prices.csv")
    MHO_new_data = (
        median_house_prices[
            median_house_prices["Neighborhood"] == new_data["MedianHousePrice"][0]
        ]
        .iloc[:, 1]
        .values
    )
    X["MedianHousePrice"] = MHO_new_data
    return X

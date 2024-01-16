from flask import Flask, render_template, request
from utils.new_data_preprocess import convert_location
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models
with open("models/base_model.pkl", "rb") as file:
    model_base = pickle.load(file)

with open("models/meta_model.pkl", "rb") as file:
    model_meta = pickle.load(file)


# function for retreiving dropdown neighborhood names fro data folder
def get_locations_from_csv():
    neighborhood_path = "data/neighborhood_median_prices.csv"
    neighborhoods_df = pd.read_csv(neighborhood_path)
    locations = [i for i in neighborhoods_df["Neighborhood"]]
    return locations


# Add this line to get locations
locations = get_locations_from_csv()


# Function to preprocess input data
def preprocess_input(data):
    X = convert_location(data)
    return X


# Function to get predictions
def get_predictions(data):
    X = preprocess_input(data)
    base_prediction = model_base.predict(X)
    X["Blended_Predictions"] = base_prediction
    final_prediction = np.exp(model_meta.predict(X))
    return final_prediction


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_data = {
            "ExterQual": float(request.form["ExterQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "TotRmsAbvGrd": float(request.form["TotRmsAbvGrd"]),
            "Total_Bathrooms": float(request.form["Total_Bathrooms"]),
            "MedianHousePrice": request.form["MedianHousePrice"],
            "OverallQual": float(request.form["OverallQual"]),
        }

        prediction = get_predictions(input_data)
        return render_template("index.html", prediction=prediction, locations=locations)

    return render_template("index.html", prediction=None, locations=locations)


if __name__ == "__main__":
    app.run(debug=True)

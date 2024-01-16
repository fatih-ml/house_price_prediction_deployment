from utils.new_data_preprocess import convert_location
import joblib
import pickle
import numpy as np

new_data = {
    "ExterQual": 10,
    "GrLivArea": 1000,
    "TotRmsAbvGrd": 5,
    "Total_Bathrooms": 3,
    "MedianHousePrice": "StoneBr",
    "OverallQual": 7,
}


X = convert_location(new_data)


with open("models/base_model.pkl", "rb") as file:
    model_base = pickle.load(file)

with open("models/meta_model.pkl", "rb") as file:
    model_meta = pickle.load(file)

base_prediction = model_base.predict(X)
X["Blended_Predictions"] = base_prediction

final_prediction = np.exp(model_meta.predict(X))
print(final_prediction)

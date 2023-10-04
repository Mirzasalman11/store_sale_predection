import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-store_nbr', type=int, default=1)
parser.add_argument('-family', type=str, default="AUTOMOTIVE")
parser.add_argument('-onpromotion', type=int, default=0)
parser.add_argument('-dcoilwtico', type=float, default=67.714366)
parser.add_argument('-city', type=str, default='Quito')
parser.add_argument('-cluster', type=int, default=13)
parser.add_argument('-type', type=str, default="Holiday")
parser.add_argument('-transferred', type=bool, default=False)  # Change the type to bool
parser.add_argument('-year', type=int, default=2013)  # Change the type to int
parser.add_argument('-month', type=int, default=1)
parser.add_argument('-day', type=int, default=1)

args = parser.parse_args()

# Convert the relevant arguments to their respective types
X_predict = [[args.store_nbr, args.family, args.onpromotion, args.dcoilwtico, args.city, args.cluster, args.type, args.transferred, args.year, args.month, args.day]]

# Create the DataFrame
X_predict = pd.DataFrame(X_predict, columns=['store_nbr', 'family', 'onpromotion', 'dcoilwtico', 'city', 'cluster', 'type', 'transferred', 'year', 'month', 'day'])
print(X_predict)

import joblib
# Load the pre-processing transformer from the saved file
loaded_transformer = joblib.load('C:/salman/ML/New_Task/preprocessing_transformer.pkl')
X_predict_transformed = loaded_transformer.transform(X_predict)
X_predict_transformed=pd.DataFrame(X_predict_transformed)
model = joblib.load('C:/salman/ML/New_Task/model.pkl')
y_pred = model.predict(X_predict_transformed)
print(y_pred[0])
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jinja2

from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from saveModels import save_model, load_model

# This is a simple linear regression model that predicts the closing price of Tesla stock on a given day.
url = "TSLA.csv"
df = pd.read_csv(url)
df['Date'] = pd.to_datetime(df['Date'])  # Convert date to datetime

# Adding features - all of the relevant columns to close price
# Volatility would be a good feature to add to the model
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
df['Volatility'] = df['High'] - df['Low']

features = ['Days', 'Open', 'High', 'Low', 'Volume', 'Volatility']

# Scaling features
df[features] = preprocessing.MinMaxScaler().fit_transform(df[features])

X = pd.DataFrame(df[features])
y = pd.DataFrame(df['Close'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MSE/R^2 = "+str(metrics.mean_squared_error(y_test, y_pred)))
# MSE = 6.796428292046082 which is worse than without features. This can be a case of overfitting.

def predict_price(day, open_price, high, low, volume, volatility):
    scaled_input = preprocessing.MinMaxScaler().fit_transform([[day, open_price, high, low, volume, volatility]])
    predicted_price = model.predict(scaled_input)
    return predicted_price[0]

# Example usage
# day = 10  # 10th day from the start of dataset
# open_price = 5.0
# high = 5.5
# low = 4.8
# volume = 100000
# volatility = high - low
#
# predicted_close = predict_price(day, open_price, high, low, volume, volatility)
# print(f"Predicted Close Price: {predicted_close}")

# Plots of each feature
for feature in features:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature], df['Close'], alpha=0.5)
    plt.title(f"Scatter plot of {feature} vs Close")
    plt.xlabel(feature)
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.show()

# Save model
save_model(model, 'regression_with_features.sav')

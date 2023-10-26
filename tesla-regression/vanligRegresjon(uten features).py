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

url = "TSLA.csv"
df = pd.read_csv(url)
df['Date'] = pd.to_datetime(df['Date'])  # Convert date to datetime

# Training the model
# Convert the 'Date' column to a numerical value
# the number of days since the start
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Scale the data except for Date
# then integrate them back into the dataframe
cols_to_scale = ['Days', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
scaled_values = preprocessing.MinMaxScaler().fit_transform(df[cols_to_scale])
df[cols_to_scale] = scaled_values

df = pd.DataFrame(df.values, columns=df.columns)
df.head()

X = pd.DataFrame(df['Days']) # Date
y = pd.DataFrame(df['Close']) # Close

#Split the set in training and testing set
# test_size = 0.33 is 1/3 of values put in test array
# Random state is a variable that seeds the random generator.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
Y_pred = model.predict(X_train)  # predict the close price

plt.scatter(X_train, y_train, color='blue', label='Real Data')  # Plot blue dots with real data
plt.plot(X_train, Y_pred, color='red', label='Linear Regression Prediction')  # Plot red line with prediction

# Adding descriptive labels and title
plt.xlabel('Days since the start')  # X-axis label
plt.ylabel('Close Price')    # Y-axis label
plt.title('Tesla Stock Price Prediction using Linear Regression')
plt.legend()  # Display legend
plt.show()

# This is a score on how well the model fits the data, there's room for improvement.
print("MSE/R^2 = "+str(metrics.mean_squared_error(y_train,Y_pred))) # Calculate MSE

# Function to predict price on a specific date
def predict_price(date_str):
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    days_since_start = (date_obj - df['Date'].min()).days
    print('Predicting price on ' + date_str + ' (' + str(days_since_start) + ' days since start)')
    print('Predicted price: ' + str(model.predict([[days_since_start]])[0]))
    return model.predict([[days_since_start]])[0]

date_to_predict = "2023-10-27"
predict_price(date_to_predict)

save_model(model, 'regression_without_features.sav')
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jinja2

from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

url = "TSLA.csv"
df = pd.read_csv(url)
df['Date'] = pd.to_datetime(df['Date'])  # Convert date to datetime

# Heatmap code
figure, axis = plt.subplots(figsize=(8,8)) # Create a figure and an axis

corrMatrix = df.drop('Date', axis=1).corr()  # Drop date column because it's not numeric
plt.xticks(range(len(corrMatrix.columns)), corrMatrix.columns, rotation=90)  # Add xticks and yticks names
plt.yticks(range(len(corrMatrix.columns)), corrMatrix.columns)

for i in range(len(corrMatrix.columns)):
    for j in range(len(corrMatrix.columns)):
        # Add text to each cell with the correlation value rounded to 4 decimal places
        # iloc is used to select the cell at row i and column j because csv uses column names as index
        text = axis.text(j, i, round(corrMatrix.iloc[i, j], 7), ha="center", va="center", color="w")

# Standard corr plot matrix
axis.matshow(corrMatrix, cmap='coolwarm')



# scatterplot code
df = df.drop(columns=['Date'])
x = df.values  # returns a numpy array
scaler = preprocessing.MinMaxScaler().fit(x)
df[list(df.columns)] = scaler.transform(df)

# Scatterplot all columns against last column
target_column = 'Close'

# Prepare the subplots
figure, axis = plt.subplots(len(df.columns) - 1, figsize=(15, 15))  # Excluding 'Close' column

# Iterate over each column and scatter plot against 'Close'
for i, col_name in enumerate(df.columns):
    if col_name != target_column:
        axis[i].scatter(x=df[col_name], y=df[target_column])
        axis[i].set_xlabel(col_name)
        axis[i].set_ylabel(target_column)

# Adjust layout and display
figure.tight_layout()
plt.show()


# Scatterplots
# df.plot.scatter(x='Open', y='Close')
ax1 = df.plot(kind='scatter', x='Open', y='Close', color='r')
ax2 = df.plot(kind='scatter', x='High', y='Close', color='g', ax=ax1)
ax3 = df.plot(kind='scatter', x='Low', y='Close', color='b', ax=ax1)
ax4 = df.plot(kind='scatter', x='Volume', y='Close', color='y', ax=ax1)

plt.show()
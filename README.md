# Assignment 2 for DAVE3625
## Tesla stock price prediction based on date task
For this task, I have chosen to use a linear regression model to predict the stock market price of tesla. This is because 
the task requires predicting a continuous value (stock price) rather than categorizing the data into distinct classes, which can be the case for facial recognition technology, to give an example.
These model will train on how to predict the price based on historical, continuous data.
There are 5 files in this repository:
 - correlationPlots.py which contains the plots for the correlation matrix
 - vanligRegresjon.py which contains the model and the prediction function for a model only trained on date and close.
 - regresjonMedFeatures.py which contains the model and the prediction function for a model trained on more features - Date(Day), Open, Volatility(High-Low), Volume
- TSLA.csv, the data for this task
- saveModels.py which contains the functions for saving/loading a model into a file.

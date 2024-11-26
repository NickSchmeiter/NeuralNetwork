# NeuralNetwork
With this Experiment I try to use a normal lstm which is selfmade to predict the price of one stock

Dataset: Daily adjusted close stock price from 1985-2020 from one stock listed on NASDAQ throughout this period.

Features: For a given stock the following features are calculated and normalized over the last 20 days: close of all stocks

Target: Price increase (normalized) of given stock one day ahead.

Modeling architecture: 
LSTM

Performance criteria: 
MSE of true vs. predicted stock price increase for test data

Baseline: 
MSE of true vs. average price increase over all stocks for test data
Results
model which is overfitting? bc of loss = 0.000
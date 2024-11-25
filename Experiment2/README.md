# NeuralNetwork
With this Experiment I try to use a normal lstm which is selfmade to predict stock prices

Dataset: Daily adjusted close stock prices from 1985-2020 from 250 stocks listed on NASDAQ throughout this period.

Features: For a given stock the following features are calculated and normalized over the last 20 days: close of all stocks and stockname as id

Target: Price increase (normalized) of given stock one day ahead.

Modeling architecture: 
LSTM

Performance criteria: 
MSE of true vs. predicted stock price increase for test data

Baseline: 
MSE of true vs. average price increase over all stocks for test data
Results
model which is overfitting? bc of loss = 0.000
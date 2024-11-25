# NeuralNetwork
With this Experiment I try to use a multi input lstm which is selfmade to predict stock prices

Dataset: Daily adjusted close stock prices from 1985-2020 from all stocks listed on NASDAQ throughout this period.

Features: For a given stock the following features are calculated and normalized over the last 100 days: HLOC, volume, close of 20 most correlated stocks

Target: Price increase (percentage) of given stock one day ahead.

Modeling architecture: 
Attention-based multi-input LSTM

Performance criteria: 
MSE of true vs. predicted stock price increase for test data

Baseline: 
MSE of true vs. average price increase over all stocks for test data
Results
model not running because of buggy code
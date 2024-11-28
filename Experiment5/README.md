# NeuralNetwork
With this Experiment I try to use a normal RNN which is selfmade to predict stock prices

Dataset: Daily adjusted close stock prices from 1985-2020 from 250 stocks listed on NASDAQ throughout this period.

Features: For a given stock the following features are calculated and normalized over the last 20 days: close of all stocks and stockname as id

Target: Close (normalized) of given stock one day ahead.

Modeling architecture: 
RNN

Performance criteria: 
MSE of true vs. predicted stock close for test data

Baseline: 
MSE of true vs. average close over all stocks for test data
Results
good performance bc of loss = 0.000
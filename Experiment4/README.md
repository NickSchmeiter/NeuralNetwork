# NeuralNetwork
With this Experiment I try to use a normal RNN which is selfmade to predict the price of one stock

Dataset: Daily adjusted close stock price from 1985-2020 from one stock listed on NASDAQ throughout this period.

Features: For a given stock the following features are calculated and normalized over the last 20 days: close of stock

Target: close (normalized) of given stock one day ahead.

Modeling architecture: 
RNN

Performance criteria: 
MSE of true vs. predicted close for test data

Baseline: 
MSE of true vs. predicted close of stock for test data
Results
good performance bc of loss = 0.001
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from os import listdir
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn import preprocessing
from datetime import timedelta
import custom_rnn as rnn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Define the path to the data folder
data_folder = '/Users/nickschmeiter/Downloads/NeuralNetwork/Experiment4/Data'

# Initialize an empty list to store individual dataframes
dataframes = []

# Loop through all files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate all dataframes into one
all_data_df = pd.concat(dataframes, ignore_index=True)


all_data_df= all_data_df[["Date","Close"]]

#print(all_data_df.head())

all_data_df['Date'] = pd.to_datetime(all_data_df['Date'])

#plt.plot(all_data_df['Date'], all_data_df['Close'])
#plt.show()


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 20
shifted_df = prepare_dataframe_for_lstm(all_data_df, lookback)
print(shifted_df.head())

#shifted_df = pd.pivot_table(shifted_df, index=['Date'], columns=['Ticker'])

shifted_df_as_np = shifted_df.to_numpy()

scaler = preprocessing.MinMaxScaler()
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
print(shifted_df_as_np)
X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

print(X.shape) 
print(y.shape)
print(X)  

X = dc(np.flip(X, axis=1))

split_index = int(len(X) * 0.9)

print(split_index)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.layer_1 = rnn.CustomRNN(1, 5) # input_size, hidden_size
        self.layer_2 = rnn.CustomRNN(5, 100) # input_size, hidden_size
        self.layer_3 = nn.Linear(100, 1) # input_size, output_size

    def forward(self, x):
        out, hidden = self.layer_1(x) # returns tuple consisting of output and sequence
        out, hidden = self.layer_2(hidden)
        output = torch.relu(self.layer_3(out))
        return output
model= RNN().to(device)

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

learning_rate = 0.001
num_epochs = 5
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
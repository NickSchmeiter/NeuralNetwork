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
data_folder = '/Users/nickschmeiter/Downloads/NeuralNetwork/Experiment2/Data'

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


all_data_df= all_data_df[["Date","Close","Ticker"]]

#print(all_data_df.head())

all_data_df['Date'] = pd.to_datetime(all_data_df['Date'])
# Create a mapping of tickers to numeric IDs
ticker_to_id = {ticker: idx for idx, ticker in enumerate(all_data_df['Ticker'].unique())}

# Add the numeric ID to the dataframe
all_data_df['Ticker_ID'] = all_data_df['Ticker'].map(ticker_to_id)
all_data_df.drop(columns=['Ticker'], inplace=True)
print(all_data_df.head())
grouped = all_data_df.groupby('Ticker_ID')

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
prepared_data = []
for Ticker_ID, group in grouped:
    group = group.sort_values('Date')  # Ensure data is sorted by date
    processed_group = prepare_dataframe_for_lstm(group[['Date', 'Close']], lookback)
    processed_group['Ticker_ID'] = Ticker_ID  # Add ticker column back after processing
    prepared_data.append(processed_group)

shifted_df = pd.concat(prepared_data, ignore_index=True)
print(shifted_df.head())

#shifted_df = pd.pivot_table(shifted_df, index=['Date'], columns=['Ticker'])



scalers = {}
normalized_data = []

for Ticker_ID, group in shifted_df.groupby('Ticker_ID'):
    scaler = preprocessing.MinMaxScaler()
    scaled_values = scaler.fit_transform(group.drop(columns=['Ticker_ID']).to_numpy())
    scalers[Ticker_ID] = scaler
    normalized_df = pd.DataFrame(scaled_values, columns=group.columns[:-1])  # Exclude Ticker
    normalized_df['Ticker_ID'] = Ticker_ID
    normalized_data.append(normalized_df)

# Combine normalized data
normalized_df = pd.concat(normalized_data, ignore_index=True)
print(normalized_df.head())


#shifted_df_as_np = shifted_df.to_numpy()

train_data = []
test_data = []

for ticker_ID, group in normalized_df.groupby('Ticker_ID'):
    X = group.iloc[:, 1:].values  # Use shifted Close columns
    y = group.iloc[:, 0].values  # Target is the Close column
    X = dc(np.flip(X, axis=1))
    split_index = int(len(X) * 0.95)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    num_samples_train = (X_train.shape[0] // lookback) * lookback
    X_train = X_train[:num_samples_train]
    y_train = y_train[:num_samples_train]

    num_samples_test = (X_test.shape[0] // lookback) * lookback
    X_test = X_test[:num_samples_test]
    y_test = y_test[:num_samples_test]
    """
    print(f"X_train shape before reshaping: {X_train.shape}")
    print(f"X_test shape before reshaping: {X_test.shape}")
    print(f"y_train shape before reshaping: {y_train.shape}")
    print(f"y_test shape before reshaping: {y_test.shape}")
    """
    X_train = X_train.reshape((-1, lookback+1, 1))
    X_test = X_test.reshape((-1, lookback+1,1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    """
    print(f"Ticker_ID: {ticker_ID}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    """
    train_data.append((torch.tensor(X_train).float(), torch.tensor(y_train).float()))
    test_data.append((torch.tensor(X_test).float(), torch.tensor(y_test).float()))

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(
    torch.cat([data[0] for data in train_data]),
    torch.cat([data[1] for data in train_data])
)

test_dataset = TimeSeriesDataset(
    torch.cat([data[0] for data in test_data]),
    torch.cat([data[1] for data in test_data])
)


batch_size = 100

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
            #print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
           #                                         avg_loss_across_batches))
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
num_epochs = 2
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    X_train_tensor = torch.tensor(X_train).float().to(device)
    predicted = model(X_train_tensor.to(device)).to('cpu').numpy()

plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

"""time = min(all_data_df.Date)
end = max(all_data_df.Date)

T = timedelta(days=20)

while time < end - T:
    print(time)
    subset = all_data_df[ (all_data_df['Date'] >= time) & (all_data_df['Date'] < time + T + timedelta(days=1))]
    table = pd.pivot_table(subset, index=['Date'], columns=['Ticker'])
    table.columns = table.columns.get_level_values(1)
    table = table.dropna(axis=1)
    table = table.loc[:,table.nunique()!=1]

    #normalize
    x = table.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    table = pd.DataFrame(x_scaled, columns=table.columns)

    targets = table.iloc[T]
    table = table[0:T]
    print(table)
    time = time + timedelta(days=1)
"""
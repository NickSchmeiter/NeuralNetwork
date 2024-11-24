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
import custom_lstm as lstm
import multi_input_lstm_layers as milstm


class Net(nn.Module):

    def __init__(self, seq_size, hidden_size):
        super(Net, self).__init__()
        self.seq_size = seq_size
        self.Y_layer = lstm.CustomLSTM(10, hidden_size) 
        self.X_p_layers = nn.ModuleList()
        self.X_n_layers = nn.ModuleList()
        for i in range(self.seq_size):
            self.X_p_layers.append(lstm.CustomLSTM(1, hidden_size))
            self.X_n_layers.append(lstm.CustomLSTM(1, hidden_size))

        self.MI_LSTM_layer = milstm.MultiInputLSTM(hidden_size, hidden_size)
        self.Attention_layer = milstm.Attention(hidden_size, hidden_size)

        self.lin_layer = nn.Linear(hidden_size, 1)

    def forward(self, Y, X_p, X_n):
        # Ensure Y has the correct shape
        if Y.dim() == 3:
            bs, seq_sz, input_sz = Y.shape
        else:
            raise ValueError("Expected input tensor Y to have 3 dimensions, but got {}".format(Y.dim()))

        # Continue with the rest of the forward pass
        Y_tilde, Y_tilde_hidden = self.Y_layer(Y)

        X_p_list = list()
        X_n_list = list()
        for i in range(self.seq_size):
            X_p_slice = X_p[:, :, i:i+1]  # Shape: [batch_size, seq_len, 1]
            X_n_slice = X_n[:, :, i:i+1]  # Shape: [batch_size, seq_len, 1]

        # Skip empty slices
            if X_p_slice.size(2) == 0 or X_n_slice.size(2) == 0:
                print(f"Empty feature slice at index {i}: X_p_slice.shape={X_p_slice.shape}, X_n_slice.shape={X_n_slice.shape}") 
            X_p_out, X_p_hidden = self.X_p_layers[i](X_p[:,:,i:i+1])
            X_p_list.append(X_p_hidden)
            X_n_out, X_n_hidden = self.X_n_layers[i](X_n[:,:,i:i+1])
            X_n_list.append(X_n_hidden)
        X_p_tensor = torch.stack(X_p_list)
        P_tilde = torch.mean(X_p_tensor, 0)
        X_n_tensor = torch.stack(X_n_list)
        N_tilde = torch.mean(X_n_tensor, 0)
        Y_tilde_prime_out, Y_tilde_prime_hidden = self.MI_LSTM_layer(Y_tilde_hidden, P_tilde, N_tilde)

        y_tilde = self.Attention_layer(Y_tilde_prime_hidden)
        output = torch.relu(self.lin_layer(y_tilde))
        return output


class CustomDataset(Dataset):

    def __init__(self, max_length=100, feature_size=10):
        
        # Define the correct path to the Samples directory
        self.base_path = '/Users/nickschmeiter/Downloads/NeuralNetwork/Experiment1/Data/Samples'
        
        # Make a list containing the paths to all your .pkl files
        self.paths = [os.path.join(self.base_path, file) for file in os.listdir(self.base_path) if file.endswith('.pkl')]
        
        # make a list containing the path to all your pkl files
        #self.paths = listdir('c:/data/htw/2021_SS/AKI/Samples/')
        random.shuffle(self.paths)
        self.max_length = max_length
        self.feature_size = feature_size
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        
        
        #with open('c:/data/htw/2021_SS/AKI/Samples/' + self.paths[idx], 'rb') as f:
        with open(self.paths[idx], 'rb') as f:
            item = pickle.load(f)

        print(f"Y shape: {item['Y'].shape}")
        print(f"X_p shape: {item['X_p'].shape}")
        print(f"X_n shape: {item['X_n'].shape}")
        print(f"y shape: {item['y'].shape}")
        Y = item['Y'].to_numpy()
        y = item['y'].to_numpy()
        X_p = item['X_p'].to_numpy()
        X_n = item['X_n'].to_numpy()

        Y = self._pad_or_truncate(Y, feature_size=self.feature_size)
        y = self._pad_or_truncate(y, feature_size=self.feature_size)
        X_p = self._pad_or_truncate(X_p, feature_size=self.feature_size)
        X_n = self._pad_or_truncate(X_n, feature_size=self.feature_size)

        print(f"convert Y shape: {Y.shape}")
        print(f"convert X_p shape: {y.shape}")
        print(f"convert X_n shape: {X_p.shape}")
        print(f"convert y shape: {X_n.shape}")

        return Y, y, X_p, X_n
    
    def _pad_or_truncate(self, array, feature_size=None):
        """
        Pads or truncates an array to the desired max_length.
        """
        if array.ndim == 1:
            array = array[:, None]  # Convert to 2D with single feature dimension

        # Handle feature dimension
        if feature_size is not None and array.shape[1] != feature_size:
            padded_array = np.zeros((array.shape[0], feature_size), dtype=array.dtype)
            # Truncate or pad features
            feature_length = min(array.shape[1], feature_size)
            padded_array[:, :feature_length] = array[:, :feature_length]
            array = padded_array

        if len(array) > self.max_length:
            # Truncate along the first dimension
            return array[:self.max_length]
        else:
            # Pad along the first dimension
            padding_shape = (self.max_length,) + array.shape[1:]  # New shape
            padded = np.zeros(padding_shape, dtype=array.dtype)  # Initialize with zeros
            padded[:len(array)] = array  # Copy the original data
            return padded

"""def collate_fn(batch):
    Y, y, X_p, X_n = zip(*batch)
    
    # Find the maximum lengths for padding
    max_len_Y = max([len(t) for t in Y])  # Maximum length of Y
    max_len_X_p = max([len(t) for t in X_p])  # Maximum length of X_p
    max_len_X_n = max([len(t) for t in X_n])  # Maximum length of X_n
    
    # Pad the sequences to the maximum length
    Y_padded = [np.pad(t, ((0, max_len_Y - len(t)), (0, 0)), 'constant') for t in Y]
    #X_p_padded = [np.pad(t, ((0, max_len_X_p - len(t)), (0, 0)), 'constant') for t in X_p]
    X_p_padded = [torch.nn.functional.pad(torch.tensor(t), (0, 0, 0, max_len_X_p - len(t))) for t in X_p]
    #X_p_padded = torch.cat( (torch.zeros(max_len_X_p-len(X_p), 0), X_p) )
    X_n_padded = [np.pad(t, ((0, max_len_X_n - len(t)), (0, 0)), 'constant') for t in X_n]
    
    # Convert to PyTorch tensors
    # Y has shape (batch_size, seq_len, input_size=1)
    Y_tensor = torch.tensor(Y_padded).float()  # Input size is already 1, no need to unsqueeze
    X_p_tensor = torch.tensor(X_p_padded).float()  # Input size is already 1, no need to unsqueeze
    X_n_tensor = torch.tensor(X_n_padded).float()  # Input size is already 1, no need to unsqueeze
    
    # Target labels `y`
    y_tensor = torch.tensor(y).float()  # Shape: (batch_size,)
    
    return Y_tensor, y_tensor, X_p_tensor, X_n_tensor 
"""
T = 100
batch_size = 512
q = 64

dataset = CustomDataset()
loader = DataLoader(dataset=dataset, batch_size=batch_size)

net = Net(T, q)

criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):

    running_loss = 0

    for Y, labels, X_p, X_n in loader:
        print(Y.shape)
        optimizer.zero_grad()
        outputs = net(Y.float(), X_p.float(), X_n.float())
        
        # Ensure `outputs` has the shape [batch_size]
        outputs = outputs.squeeze()

        # Ensure `labels` has the shape [batch_size]
        labels = labels.squeeze()

        # Calculate the loss
        loss = criterion(outputs, labels.float())
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(loss.item())
    if len(loader) > 0:
        print('Epoch loss: ' + str(running_loss / len(loader)))

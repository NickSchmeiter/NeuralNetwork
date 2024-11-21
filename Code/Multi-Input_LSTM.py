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
        self.Y_layer = lstm.CustomLSTM(5, hidden_size)
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

    def __init__(self):
        
        # Define the correct path to the Samples directory
        self.base_path = '/Users/nickschmeiter/Desktop/KI-Projekt/Eperiment1/Data/Samples/'
        
        # Make a list containing the paths to all your .pkl files
        self.paths = [os.path.join(self.base_path, file) for file in os.listdir(self.base_path) if file.endswith('.pkl')]
        
        # make a list containing the path to all your pkl files
        #self.paths = listdir('c:/data/htw/2021_SS/AKI/Samples/')
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        
        
        #with open('c:/data/htw/2021_SS/AKI/Samples/' + self.paths[idx], 'rb') as f:
        with open(self.paths[idx], 'rb') as f:
            item = pickle.load(f)
        Y = item['Y'].to_numpy()
        y = item['y']
        X_p = item['X_p'].to_numpy()
        X_n = item['X_n'].to_numpy()
        return Y, y, X_p, X_n

def collate_fn(batch):
    Y, y, X_p, X_n = zip(*batch)
    
    # Find the maximum lengths for padding
    max_len_Y = max([len(t) for t in Y])  # Maximum length of Y
    max_len_X_p = max([len(t) for t in X_p])  # Maximum length of X_p
    max_len_X_n = max([len(t) for t in X_n])  # Maximum length of X_n
    
    # Pad the sequences to the maximum length
    Y_padded = [np.pad(t, ((0, max_len_Y - len(t)), (0, 0)), 'constant') for t in Y]
    X_p_padded = [np.pad(t, ((0, max_len_X_p - len(t)), (0, 0)), 'constant') for t in X_p]
    X_n_padded = [np.pad(t, ((0, max_len_X_n - len(t)), (0, 0)), 'constant') for t in X_n]
    
    # Convert to PyTorch tensors
    # Y has shape (batch_size, seq_len, input_size=1)
    Y_tensor = torch.tensor(Y_padded).float()  # Input size is already 1, no need to unsqueeze
    X_p_tensor = torch.tensor(X_p_padded).float()  # Input size is already 1, no need to unsqueeze
    X_n_tensor = torch.tensor(X_n_padded).float()  # Input size is already 1, no need to unsqueeze
    
    # Target labels `y`
    y_tensor = torch.tensor(y).float()  # Shape: (batch_size,)
    
    return Y_tensor, y_tensor, X_p_tensor, X_n_tensor

T = 100
batch_size = 5
q = 64

dataset = CustomDataset()
loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)

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

    print('Epoch loss: ' + str(running_loss / len(loader)))

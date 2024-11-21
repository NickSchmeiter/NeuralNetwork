import torch
import torch.nn as nn
import math

class CustomLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        # Forget gate parameters
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # Input gate parameters
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # Cell state update parameters (for the new candidate cell state)
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # Output gate parameters
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        bs, seq_sz, _ = x.shape  # (batch_size, sequence_size, input_size)
        h_t = torch.zeros(bs, self.hidden_size)  # hidden state
        c_t = torch.zeros(bs, self.hidden_size)  # cell state
        hidden_seq = []

        for t in range(seq_sz):
            x_t = x[:, t, :]  # get input at time step t

            print(f"x_t shape: {x_t.shape}")
            print(f"self.U_f shape: {self.U_f.shape}")

            # Forget gate
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.W_f + self.b_f)
            # Input gate
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.W_i + self.b_i)
            # Candidate cell state
            c_hat_t = torch.tanh(x_t @ self.U_c + h_t @ self.W_c + self.b_c)
            # Cell state update
            c_t = f_t * c_t + i_t * c_hat_t
            # Output gate
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.W_o + self.b_o)
            # Hidden state update
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(1))  # store hidden state at each time step

        hidden_seq = torch.cat(hidden_seq, dim=1)  # concatenate the sequence of hidden states
        return h_t, hidden_seq  # return final hidden state and the sequence of hidden states

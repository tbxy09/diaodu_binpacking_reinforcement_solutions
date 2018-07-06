# %load -r 37-57 /data/dotfiles_xy/nvim/plugged/pytorch-tutorial/tutorials/02-intermediate/recurrent_neural_network/main.py
# RNN Model (Many-to-One)
import torch
import torch.nn as nn
import pandas as pd
class RNN(nn.Module):
    def __init__(self,seq_len, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
#         self.bn=nn.BatchNorm1d(seq_len)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.it = iter(self.parameters())

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate RNN
#         x = self.bn(x)
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out
    def expand(self):
        [print("id:{},{}".format(id_,each.shape)) for id_,each in self.state_dict().items()]
        # pd.DataFrame.from_records(self.state_dict())
    def next(self):
        print(next(self.it))



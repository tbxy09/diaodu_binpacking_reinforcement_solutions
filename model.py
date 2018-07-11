# %load -r 37-57 /data/dotfiles_xy/nvim/plugged/pytorch-tutorial/tutorials/02-intermediate/recurrent_neural_network/main.py
# RNN Model (Many-to-One)
# %load -r 219-239 /data/dotfiles_xy/nvim/plugged/tutorials/beginner_source/nlp/word_embeddings_tutorial.py
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
class RNN(nn.Module):
    def __init__(self,seq_len, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
#         self.bn=nn.BatchNorm1d(seq_len)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.it = iter(self.parameters())
        self.dic= {}
        # self.h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        # self.c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        # c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        # Forward propagate RNN
#         x = self.bn(x)
        out, _ = self.lstm(x, (h0, c0))
        out=F.relu(out)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        # out = F.log_softmax(out,dim=1)
        return out
    def expand(self):
        [print("id:{},{}".format(id_,each.shape)) for id_,each in self.state_dict().items()]
        # pd.DataFrame.from_records(self.state_dict())
    def next(self):
        print(next(self.it))
    def init_it(self):
        self.it=iter(rnn.parameters())
    def weight_init():
        import torch.nn.init as weight_init
        for name, param in net.named_parameters():
            weight_init.normal_(param)
        #     weight_init.uniform_(param)
        #     weight_init.constant_(param,1)
            dic[name] = param

        net.load_state_dict(dic)
    def show(self):
        fig=plt.figure()
        ax=plt.gca(projection='3d')
        for id_,(key,value) in enumerate(self.dic.items()):
            xs=np.arange(dict[key].shape[0])
            ys=value
        #     xs=np.arange(ys.shape[0])
            print(xs.shape,ys.shape)
            ax.bar(xs,ys.data.numpy().T[1],zs=id_,zdir='y')

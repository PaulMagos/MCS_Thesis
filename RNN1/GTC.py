from GMM import GMM
from torch import nn
import torch

__all__ = ['GTC']

class GTC(nn.Module):
    def __init__(self, *args) -> None:
        super(GTC, self).__init__()
        input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, device, debug = args
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=(output_size+2)*mixture_dim, device=device)
        self.gmm = GMM(M = mixture_dim, device = device, debug=debug)
        # self.step = 0

    def forward(self, x):
        # self.step = self.step+1
        # if self.step == len(x)-1:
        #     self.step = 0
        return self.gmm(self.dense(self.lstm(x)[0]))
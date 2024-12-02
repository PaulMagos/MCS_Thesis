from torch.nn import Module, ModuleList, Identity, Dropout, Linear, LSTMCell, Tanh
import torch.nn.functional as F
import torch
from functools import partial
import numpy as np
from typing import Any
from torchmetrics import Metric
from GRGN.Utils.reshapes import reshape_to_nodes

__all__ = ['GMMCell']

class GMMCell(Module):
    def __init__(self, input_size: int, n_nodes: int, hidden_size: int, M: int):
        super().__init__()
        self.input_activation = Tanh() 
        self.first_stage = Linear(hidden_size, (input_size) * M + 2*M)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.M = M
        self.means = None
        self.stds = None
        self.weights = None
        
    def forward(self, x):
        model_input = self.input_activation(x)#.view(-1, self.hidden_size * self.n_nodes))
        out = self.first_stage(model_input)
        stds_index = self.input_size * self.M
        weights_index = (self.input_size + 1) * self.M
        out[..., stds_index:weights_index] = torch.exp(out[..., stds_index:weights_index])
        out[..., weights_index:] = F.softmax(out[..., weights_index:], dim=-1)
        return out
    
    @staticmethod
    def calculate_output_shape(input_shape, M):
        return (input_shape * M) + (M * 2)
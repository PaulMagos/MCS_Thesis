from torch.nn import Module, ModuleList, Identity, Dropout, Linear
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
        self.input_activation = F.tanh
        self.first_stage = Linear(hidden_size * n_nodes, (n_nodes * input_size * M) + M * 2)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.M = M
        self.means = None
        self.stds = None
        self.weights = None
        
    def forward(self, x):
        model_input = self.input_activation(x)
        out = self.first_stage(model_input.view(-1, self.hidden_size * self.n_nodes))
        D = self.n_nodes
        stds_index = D*self.M
        weghts_index = (D + 1)*self.M
        out[..., stds_index:weghts_index] = torch.exp(out[..., stds_index:weghts_index])
        out[..., weghts_index:] = F.softmax(out[..., weghts_index:], dim=-1)
        out = reshape_to_nodes(out, self.n_nodes, self.input_size, self.M)
        return out
    
    @staticmethod
    def calculate_output_shape(input_shape, M):
        return input_shape  * M + M * 2
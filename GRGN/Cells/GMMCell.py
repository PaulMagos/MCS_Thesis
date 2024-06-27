from torch.nn import Module, ModuleList, Identity, Dropout, Linear
import torch.nn.functional as F
import torch
from functools import partial
import numpy as np
from typing import Any
from torchmetrics import Metric

__all__ = ['GMMCell']

class GMMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, M: int):
        super().__init__()
        self.first_stage = Linear(hidden_size, (input_size + 2) * M)
        self.M = M
        self.means = None
        self.stds = None
        self.weights = None
        
    def forward(self, x):
        out = self.first_stage(x)
        D = out.shape[-1] // self.M - 2
        out[:, :, D*self.M:(D+1)*self.M] = torch.exp(out[:, :, D*self.M:(D+1)*self.M])
        out[:, :, (D+1)*self.M:(D+2)*self.M] = F.softmax(out[:, :, (D+1)*self.M:(D+2)*self.M], dim=-1)
        
        # self.means = out[:, :, :self.M*D]
        # self.stds  = out[:, :, self.M*D:self.M * (D+1)]
        # self.weights = out[:, :, self.M*(D+1):]
        
        # pred = self.weights * torch.normal(self.means, self.stds)
        # pred = torch.sum(pred, dim=-1)
        return out
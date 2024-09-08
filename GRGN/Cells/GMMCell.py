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
        self.first_stage = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.M = M
        self.means = None
        self.stds = None
        self.weights = None
        
    def forward(self, x):
        if self.first_stage == None:
            self.first_stage = Linear(self.hidden_size * x.size(-2), (x.size(-2) + 2) * self.M)
        out = self.first_stage(x.view(-1, x.size(-2)*self.hidden_size))
        D = out.shape[-1] // self.M - 2
        out[..., D*self.M:(D+1)*self.M] = torch.exp(out[..., D*self.M:(D+1)*self.M])
        out[..., (D+1)*self.M:(D+2)*self.M] = F.softmax(out[..., (D+1)*self.M:(D+2)*self.M], dim=1)
        return out
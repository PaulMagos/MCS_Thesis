__all__ = ['MLPLayer']
from torch import nn
import torch
import torch.optim as optim

class MLPLayer(nn.Module):
    def __init__(self, input_size: int, ): super(MLPLayer, self).__init__()
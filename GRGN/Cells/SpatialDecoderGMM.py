from tsl.ops.connectivity import asymmetric_norm, power_series, transpose
from torch_geometric.utils import (from_scipy_sparse_matrix, remove_self_loops,
                                   to_scipy_sparse_matrix)
from tsl.nn.layers.graph_convs import DiffConv
from torch_geometric.typing import OptTensor, Adj
from torch import LongTensor, Tensor
from torch_sparse import SparseTensor, remove_diag
from torch.nn import Module, Linear, PReLU
from .GMMCell import GMMCell
from typing import Optional
import torch

class SpatialDecoderGMM(Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: Optional[int] = None,
                 exog_size: int = 0,
                 order: int = 1,
                 num_components=32):
        super().__init__()
        self.num_components = num_components
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.exog_size = exog_size
        self.order = order

        # Input channels of convolution
        in_channels = input_size + (input_size + 2) * num_components + hidden_size * input_size
        
        #
        self.lin_in = Linear(in_channels, hidden_size * input_size)
        
        self.graph_conv = DiffConv(in_channels=hidden_size,
                                   out_channels=hidden_size,
                                   root_weight=False,
                                   k=1)
        
        self.gmm = GMMCell(input_size, 2 * hidden_size, num_components)
        
        # self.activation = PReLU()
        
    def __repr__(self):
        attrs = ['input_size', 'hidden_size', 'output_size', 'order', 'num_components']
        attrs = ', '.join([f'{attr}={getattr(self, attr)}' for attr in attrs])
        return f"{self.__class__.__name__}({attrs})"
        
    def compute_support(self, 
                        edge_index: LongTensor, 
                        edge_weight: OptTensor = None, 
                        num_nodes: Optional[int]=None, 
                        add_backward: bool = True
                        ):
        ei, ew = asymmetric_norm(edge_index,
                                 edge_weight,
                                 dim=1,
                                 num_nodes=num_nodes)
        ei, ew = power_series(ei, ew, self.order)
        ei, ew = remove_self_loops(ei, ew)
        if add_backward:
            ei_t, ew_t = transpose(edge_index, edge_weight)
            return (ei, ew), self.compute_support(ei_t, ew_t, num_nodes, False)
        return ei, ew
    
    def forward(self, 
                x: Tensor, 
                x_hat_1: Tensor, 
                h: Tensor, 
                edge_index: Adj, 
                edge_weight: OptTensor = None, 
                u: OptTensor = None
                ):
        # print(x.shape, x_hat_1.shape, h.shape)
        x_in = [x, x_hat_1, h.view(-1, h.size(-1)*h.size(-2))]
        if u is not None:
            x_in += [u]
        x_in = torch.cat(x_in, dim=-1)
        x_in = self.lin_in(x_in).view(-1, x.shape[-1], h.size(-1))
        if self.order > 1: 
            support = self.compute_support(edge_index, edge_weight, num_nodes=x.size(1))
            self.graph_conv._support = support
            out = self.graph_conv(x, edge_index=None)
            self.graph_conv._support = None
        else:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            elif isinstance(edge_index, SparseTensor):
                edge_index = remove_diag(edge_index)
            out = self.graph_conv(x_in, edge_index, edge_weight)
        
        out1 = torch.cat([out, h], dim=-1)
        out = self.gmm(out1)
        # out = torch.cat([out, h] , dim=-1)
        # out = self.activation(out)
        return out, out1, h
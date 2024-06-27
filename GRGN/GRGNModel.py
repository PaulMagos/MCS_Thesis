import torch
from torch import Tensor
from typing import Optional
from tsl.nn.models import BaseModel 
from tsl.nn.layers import NodeEmbedding
from torch_geometric.typing import Adj, OptTensor
from .Cells import GRGNCell
from torch.nn import Linear, Sequential, ReLU, Dropout

class GRGNModel(BaseModel):
    def __init__(self, 
                input_size: int,
                hidden_size: int = 64,
                ff_size: int = 128,
                mixture_size: int = 32,
                embedding_size: Optional[int] = None,
                n_layers: int = 1,
                n_nodes: Optional[int] = None,
                kernel_size: int = 2,
                decoder_order: int = 1,
                layer_norm: bool = True,
                dropout: float =.0,
                ff_dropout: float =.0,
                merge_mode: str = 'mlp'
                ):
        super(GRGNModel, self).__init__()
        self._name = 'GRGN'
        self.M = mixture_size
        self.input_size = input_size
        self.fwd_grgl = GRGNCell(input_size = input_size, 
                                 hidden_size = hidden_size, 
                                 mixture_size = mixture_size, 
                                 kernel_size = kernel_size, 
                                 dropout=dropout,
                                 n_nodes=n_nodes,
                                 n_layers=n_layers,
                                 layer_norm = layer_norm, 
                                 decoder_order = decoder_order)
        self.bwd_grgl = GRGNCell(input_size = input_size, 
                                 hidden_size = hidden_size, 
                                 mixture_size = mixture_size, 
                                 kernel_size = kernel_size, 
                                 dropout=dropout,
                                 n_nodes=n_nodes,
                                 n_layers=n_layers,
                                 layer_norm = layer_norm,
                                 decoder_order = decoder_order)
        
        if embedding_size is not None:
            assert n_nodes is not None
            self._embedding = NodeEmbedding(n_nodes, embedding_size)
        else:
            self.register_parameter('embedding', None)
            
        self.merge_mode = merge_mode
        if self.merge_mode == 'mlp':
            in_channels = 4 * hidden_size + input_size + embedding_size
            self.out = Sequential(Linear(in_channels, ff_size),
                                  ReLU(),
                                  Dropout(ff_dropout),
                                  Linear(ff_size, input_size)
                                  )
        elif self.merge_mode in ['mean', 'sum', 'min', 'max']:
            self.out = getattr(torch, self.merge_mode)
        else: 
            raise ValueError("Merge option %s not allowed" % self.merge_mode)
        
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u: OptTensor = None) -> list:
        """"""
        # x: [batch, steps, nodes, channels]
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_grgl(x,
                                                       edge_index,
                                                       edge_weight,
                                                       u=u)
        # Backward
        rev_x = x.flip(1)
        rev_u = u.flip(1) if u is not None else None
        *bwd, _ = self.bwd_grgl(rev_x,
                                edge_index,
                                edge_weight,
                                u=rev_u)
        bwd_out, bwd_pred, bwd_repr = [res.flip(1) for res in bwd]

        if self.merge_mode == 'mlp':
            inputs = [fwd_repr, bwd_repr]
            if self._embedding is not None:
                b, s, *_ = fwd_repr.size()  # fwd_h: [b t n f]
                inputs += [self._embedding(expand=(b, s, -1, -1))]
            generation = torch.cat(inputs, dim=-1)
            generation = self.out(generation)
        else:
            generation = torch.stack([fwd_out, bwd_out], dim=-1)
            generation = self.out(generation, dim=-1)
            
            prediction = torch.stack([fwd_pred, bwd_pred], dim=-1)
            prediction = self.out(prediction, dim=-1)

        return torch.cat([generation[..., :(self.input_size + 2) * self.M], prediction], dim=-1)
    
    def predict(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u: OptTensor = None) -> Tensor:
        """"""
        out = self.forward(x=x,
                                  u=u,
                                  edge_index=edge_index,
                                  edge_weight=edge_weight)
        
        return out
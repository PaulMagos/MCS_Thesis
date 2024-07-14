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
                mixture_size: int = 32,
                embedding_size: Optional[int] = None,
                n_layers: int = 1,
                n_nodes: Optional[int] = None,
                kernel_size: int = 2,
                decoder_order: int = 1,
                layer_norm: bool = True,
                dropout: float =.0,
                ):
        super(GRGNModel, self).__init__()
        
        # Model Name
        self._name = 'GRGN'
        # Number of gaussians (means and variances)
        self.M = mixture_size
        # Input size of the data
        self.input_size = input_size
        
        # Forward Step
        self.fwd_grgl = GRGNCell(input_size = input_size, 
                                 hidden_size = hidden_size, 
                                 mixture_size = mixture_size, 
                                 kernel_size = kernel_size, 
                                 dropout=dropout,
                                 n_nodes=n_nodes,
                                 n_layers=n_layers,
                                 layer_norm = layer_norm, 
                                 decoder_order = decoder_order)
        # Backward Step
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
            
        self.out = getattr(torch, 'mean')
        
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
        
        out = out[..., (self.input_size + 2) * self.M:]
        
        D = out.shape[-1] // self.M - 2
        
        means = out[:, :, :self.M*D]
        stds  = out[:, :, self.M*D:self.M * (D+1)]
        weights = out[:, :, self.M*(D+1):]
        
        pred = weights * torch.normal(means, stds)
        pred = torch.sum(pred, dim=-1)
        
        return pred
    
    def generation(self,
                   X: Tensor,
                   edge_index: Adj,
                   edge_weight: OptTensor = None,
                   u: OptTensor = None) -> Tensor:
    
        out = self.forward(x=X[-1],
                           u=u,
                           edge_index=edge_index,
                           edge_weight=edge_weight
                           )
        
        out = out[..., :(self.input_size + 2) * self.M]
        
        output = None
        for i in range(X.shape[0]):
            D = out.shape[-1] // self.M - 2
            
            means = out[:, :, :self.M*D]
            stds  = out[:, :, self.M*D:self.M * (D+1)]
            weights = out[:, :, self.M*(D+1):]
            
            gen = weights * torch.normal(means, stds)
            gen = torch.sum(gen, dim=-1)
            
            output = torch.cat([output, gen], -1) if output is not None else gen
            out = self.forward(x=output[-1],
                           u=u,
                           edge_index=edge_index,
                           edge_weight=edge_weight
                           )
        
            out = out[..., :(self.input_size + 2) * self.M]
        
        return output

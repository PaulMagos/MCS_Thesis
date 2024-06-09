__all__ = ['GRGNCell']
from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from tsl.nn.layers.base import NodeEmbedding
from torch_geometric.typing import Adj, OptTensor
from tsl.nn.layers import DCRNNCell
from tsl.nn.layers.recurrent.grin import SpatialDecoder

class GRGNCell(nn.Module):
    def __init__(self, 
                    input_size: int,
                    hidden_size: int,
                    n_layers: int = 1,
                    n_nodes: Optional[int] = None,
                    kernel_size: int = 2,
                    mixture_dimension: int = 20,
                    dense_convolution: bool = False,
                    decoder_order: int = 1,
                    layer_norm: bool = False,
                    dropout: float = 0.0,
                ):
        super(GRGNCell, self).__init__()
        self.__class__.__name__ = 'GRGNCell'
        
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._kernel_size = kernel_size
        self._dense_convolution  = dense_convolution
        self.M = mixture_dimension
        
        rnn_input_size = 2 * self._input_size 
                 
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(self.n_layers):
            cell = DCRNNCell(
                                input_size=rnn_input_size, 
                                hidden_size=self._hidden_size,
                                k=self._kernel_size,
                                root_weight=True
                            )   
            self.cells.append(cell)
            
            self.norms.append(nn.LayerNorm(hidden_size) if layer_norm else nn.Identity())
            
            rnn_input_size = hidden_size
            
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        self.first_stage = nn.Linear(self._hidden_size, self._hidden_size * decoder_order)
        
        self.spatial_decoder = SpatialDecoder(
                                                input_size=self._input_size, 
                                                hidden_size=self._hidden_size, 
                                                order=decoder_order
                                            )
        
        if n_nodes is not None:
            self.h0 = nn.ModuleList()
            for _ in range(self._n_layers):
                self.h0.append(NodeEmbedding(n_nodes=n_nodes, hidden_size=self._hidden_size))
        else:
            self.register_parameter('h0', None)
            
        
    def __repr__(self):
        attrs = ['input_size', 'hidden_size', 'kernel_size', 'n_layers']
        attrs = ', '.join([f'{attr}={getattr(self, attr)}' for attr in attrs])
        return f"{self.__class__.__name__}({attrs})"
    
    def get_h0(self, x):
        if self.h0 is not None:
            return [h(expand=(x.shape[0], -1, -1)) for h in self.h0]
        size = (self.n_layers, x.shape[0], x.shape[2], self.hidden_size)
        return [*torch.zeros(size, device=x.device)]
    
    def update_state(self, x, h, edge_index, edge_weight):
        # x: [batch, nodes, channels]
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            h[layer] = norm(cell(rnn_in, h[layer], edge_index, edge_weight))
            rnn_in = h[layer]
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h
    
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u: OptTensor = None,
                h: Union[List[Tensor], Tensor] = None):
        """"""
        # x: [batch, steps, nodes, channels]
        steps = x.size(1)
        
        # We don't require a mask we want to generate everything.
        # infer all valid if mask is None
        mask = torch.zeros_like(x, dtype=torch.uint8)

        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x)  # [[b n h] * n_layers]
        elif not isinstance(h, list):
            h = [*h]

        # Temporal conv
        predictions, generations, states = [], [], []
        representations = []
        for step in range(steps):
            x_s = x[:, step]
            m_s = mask[:, step]
            h_s = h[-1]
            u_s = u[:, step] if u is not None else None
            # firstly impute missing values with predictions from state
            xs_hat_1 = self.first_stage(h_s)
            # fill missing values in input with prediction
            # x_s = torch.where(m_s.bool(), x_s, xs_hat_1)
            x_s = xs_hat_1
            # prepare inputs
            # retrieve maximum information from neighbors
            xs_hat_2, repr_s = self.spatial_decoder(x_s,
                                                    m_s,
                                                    h_s,
                                                    u=u_s,
                                                    edge_index=edge_index,
                                                    edge_weight=edge_weight)
            # readout of generation state + mask to retrieve imputations
            # prepare inputs
            # x_s = torch.where(m_s.bool(), x_s, xs_hat_2)
            x_s = xs_hat_2
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=-1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using generations
            h = self.update_state(inputs, h, edge_index, edge_weight)
            # store generations and states
            generations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)

        # Aggregate outputs -> [batch, steps, nodes, channels]
        generations = torch.stack(generations, dim=1)
        predictions = torch.stack(predictions, dim=1)
        states = torch.stack(states, dim=2)
        representations = torch.stack(representations, dim=1)

        return generations, predictions, representations, states
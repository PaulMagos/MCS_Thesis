from typing import Optional, Union, List
from torch import LongTensor, Tensor
import torch
from .SpatialDecoderGMM import SpatialDecoderGMM
from tsl.nn.layers import NodeEmbedding
from .DCGRNNCell import DCGRNNNCell
from tsl.nn.layers.norm import LayerNorm
from torch.nn import Module, ModuleList, Identity, Dropout, Linear
from .GMMCell import GMMCell
from torch_geometric.typing import Adj, OptTensor

class GRGNCell(Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = 0,
                 n_layers: int = 1,
                 mixture_size: int = 32,
                 n_nodes: Optional[int] = None,
                 kernel_size: int = 2,
                 decoder_order: int = 1,
                 layer_norm: bool = False,
                 dropout: float = 0.):
        super(GRGNCell, self).__init__()
        
        # Cell parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.u_size = exog_size
        self.n_layers = n_layers
        self.mixture_size = mixture_size
        self.kernel_size = kernel_size
        
        # Dimension of the output of first stage (input of second stage) + imput dimension
        rnn_input_size = self.input_size + ((input_size + 2) * mixture_size) + hidden_size
        
        self.cells = ModuleList()
        self.norms = ModuleList()
        
        
        # First Stage - Spatio Temporal Encoder
        for i in range(self.n_layers):
            in_channels = rnn_input_size if i == 0 else hidden_size
            cell = DCGRNNNCell(input_size=in_channels,
                             hidden_size=hidden_size,
                             k=kernel_size,
                             root_weight=True)
            self.cells.append(cell)
            norm = LayerNorm(self.hidden_size) if layer_norm else Identity()
            self.norms.append(norm)
            
        # Dropout for for first stage hidden state
        self.dropout = Dropout(dropout) if dropout > 0 else None
        
        # First Stage - Mixture Density Model
        self.first_stage = GMMCell(input_size, hidden_size, mixture_size)
        
        # Second Stage - Spatial Decoder Mixture Density Model
        self.spatial_decoder = SpatialDecoderGMM(input_size=input_size,
                                              hidden_size=hidden_size,
                                              exog_size=exog_size,
                                              order=decoder_order,
                                              num_components=mixture_size
                                              )
                                            
        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = ModuleList()
            for _ in range(self.n_layers):
                self.h0.append(NodeEmbedding(n_nodes, self.hidden_size))
        else:
            self.register_parameter('h0', None)
                           
    def update_state(self, x, h, edge_index, edge_weight):
        # x: [batch, nodes, channels]
        rnn_in = x\
            
        # Update hidden state for each cell, normalizing output
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            # Pass input to cell, or pass hidden state from previous layer
            tmp_out = cell(rnn_in, h[layer], edge_index, edge_weight)
            # Normlize output of cell
            h[layer] = norm(tmp_out)
            rnn_in = h[layer]
            # Apply dropout if necessary
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h
                           
                           
    def get_h0(self, x):
        # Initialize hidden state
        if self.h0 is not None:
            return [h(expand=(x.shape[0], -1, -1)) for h in self.h0]
        size = (self.n_layers, x.shape[0], x.shape[2], self.hidden_size)
        return [*torch.zeros(size, device=x.device)]                 
                                        

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u: OptTensor = None,
                h: Union[List[Tensor], Tensor] = None):
        """"""
        # print(x.shape)
        # x: [batch, steps, nodes, channels]
        steps = x.size(1)

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
            h_s = h[-1]
            u_s = u[:, step] if u is not None else None
            # first generate gaussians means, stds from state
            xs_hat_1 = self.first_stage(h_s)
            # retrieve maximum information from neighbors
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s,
                                                    x_hat_1 = xs_hat_1,
                                                    h=h_s,
                                                    u=u_s,
                                                    edge_index=edge_index,
                                                    edge_weight=edge_weight)
            # readout of generation state 
            # prepare inputs
            inputs = xs_hat_2
            if u_s is not None:
                inputs.append(u_s)
                inputs = torch.cat(inputs, dim=-1)  # x_hat_2 + exogenous
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
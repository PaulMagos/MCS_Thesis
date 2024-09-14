import torch
from torch import Tensor
from typing import Optional
from tsl.nn.models import BaseModel 
from tsl.nn.layers import NodeEmbedding
from torch_geometric.typing import Adj, OptTensor
from .Cells import GRGNCell
from GRGN.Utils.reshapes import reshape_to_original
from tqdm import tqdm
from torch.nn import Linear, Sequential, ReLU, Dropout

class GRGNModel(BaseModel):
    def __init__(self, 
                input_size: int,
                hidden_size: int = 64,
                mixture_size: int = 32,
                mixture_weights_mode: str = 'weighted',
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
        # Nodes of the graph
        self.n_nodes = n_nodes
        
        self.mixture_weights_mode = mixture_weights_mode
        
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
                                                       u=None)
        # Backward
        rev_x = x.flip(1)
        *bwd, _ = self.bwd_grgl(rev_x,
                                edge_index,
                                edge_weight,
                                u=None)
        bwd_out, bwd_pred, bwd_repr = [res.flip(1) for res in bwd]

        out = torch.cat([fwd_pred, fwd_out, bwd_pred, bwd_out], dim=-1)
        
        return out
    
    def generate(self,
                   X: Tensor,
                   edge_index: Adj,
                   edge_weight: OptTensor = None,
                   u: OptTensor = None,
                   steps: int = 32, 
                   encoder_only = False,
                   both_mean = False,
                   **kwargs
                   ) -> Tensor:
        # check if scaler in kwargs is present
        if 'scaler' in kwargs['kwargs']:
            scaler = kwargs['kwargs']['scaler']
            X = scaler.transform(X)
        nextval = X
        output = []
        output_not_computed = []
        self.D = X.shape[-1]# * X.shape[-2]
                
        self.stds_index = self.M*self.D
        self.weights_index = self.M*(self.D+1)
        
        with tqdm(len(range(0, steps)), total=steps, desc='Generating', unit='step') as t:
            for i in range(steps):
                out = self.forward(x=nextval,
                            u=u,
                            edge_index=edge_index,
                            edge_weight=edge_weight
                            )
                
                gen =  self.get_output(out, both_mean, encoder_only)

                nextval = gen.reshape(1, 1, gen.shape[-1], 1)
                output.append(nextval)
                output_not_computed.append(out)
                t.update(1)
        output = torch.cat(output)
        if 'scaler' in kwargs['kwargs']:
            output = scaler.inverse_transform(output)
        return output

    def predict(self,
                   X: Tensor,
                   edge_index: Adj,
                   edge_weight: OptTensor = None,
                   u: OptTensor = None,
                   encoder_only = False,
                   both_mean=False,
                   **kwargs
                   ) -> Tensor:
        if 'scaler' in kwargs['kwargs']:
            scaler = kwargs['kwargs']['scaler']
            X = scaler.transform(X)
        nextval = X
        output = []
        output_not_computed = []
        steps = X.shape[0]
        
        self.D = X.shape[-1]# * X.shape[-2]
        
        self.stds_index = self.M*self.D
        self.weights_index = self.M*(self.D+1)
        
        with tqdm(len(range(0, steps)), total=steps, desc='Predicting', unit='step') as t:
            for i in range(steps):
                nextval = X[i].reshape(1, 1, X.shape[-2], 1)
                out = self.forward(x=nextval,
                            u=u,
                            edge_index=edge_index,
                            edge_weight=edge_weight
                            )
                
                gen =  self.get_output(out, both_mean, encoder_only)
                        
                output.append(gen.reshape(1, 1, gen.shape[-1], 1))
                output_not_computed.append(out)
                t.update(1)
        output = torch.cat(output)
        if 'scaler' in kwargs['kwargs']:
            output = scaler.inverse_transform(output)
        return output
    
    
    def get_output(self, out, both_mean, encoder_only):
        pred_index = (self.input_size * 2) * self.M + self.M
        if both_mean and not both_mean:
            out_fwd = (out[..., :pred_index] + out[..., pred_index:2*pred_index])/2
            out_bwd = (out[..., 2*pred_index:3*pred_index] + out[..., 3*pred_index:])/2
            out1 = (out_fwd + out_bwd)/2
        else:
            if encoder_only:
                enc_fwd = out[..., :pred_index]
                enc_bwd = out[..., 2*pred_index:3*pred_index]
                out1 = (enc_fwd + enc_bwd)/2
            else:
                dec_bwd  = out[..., pred_index:2*pred_index]
                dec_fwd = out[..., 3*pred_index:]
                out1 = (dec_bwd + dec_fwd)/2
            
        # out1 = reshape_to_original(out1, self.n_nodes, self.input_size, self.M)  
        
        means = out1[..., :self.stds_index]#.reshape(-1, self.M, self.D)
        stds  = out1[..., self.stds_index:self.weights_index]#.unsqueeze(-1)
        weights = out1[..., self.weights_index:]#.unsqueeze(-1)
        
        means = torch.where(torch.isnan(means) | torch.isinf(means), torch.zeros_like(means), means).to(means.device)
        stds = torch.where(torch.isnan(stds) | torch.isinf(stds), torch.zeros_like(stds), stds).to(stds.device)
        weights = torch.where(torch.isnan(weights) | torch.isinf(weights), torch.zeros_like(weights), weights).to(weights.device)
           
        match(self.mixture_weights_mode):
            case 'weighted':
                gen = weights * torch.normal(means, stds)
            case 'uniform':
                gen = torch.normal(means, stds)
            case 'equal_probability':
                weights = torch.ones_like(weights) * 1/self.M
                gen = weights * torch.normal(means, stds)
           
        gen = torch.mean(gen, axis=-1)
                
        return gen
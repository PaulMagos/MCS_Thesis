import torch
from torch import Tensor
from typing import Optional
from tsl.nn.models import BaseModel 
from tsl.nn.layers import NodeEmbedding
from torch_geometric.typing import Adj, OptTensor
from .Cells import GRGNCell
from tqdm import tqdm
from GRGN.Utils import reshape_to_original

class GRGNModel(BaseModel):
    def __init__(self, 
                input_size: int,
                hidden_size: int = 64,
                mixture_size: int = 32,
                exclude_bwd: bool = False,
                embedding_size: Optional[int] = 2,
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
        self.exclude_bwd = exclude_bwd
        self.D = input_size
        self.stds_index = self.M*self.D
        self.weights_index = self.M*(self.D+1)
        
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
        if not exclude_bwd:
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
            u: OptTensor = None) -> Tensor:
        """
        Forward and backward pass combining predictions and outputs.
        """
        
        # with probability 1/
        # Forward pass
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_grgl(x, edge_index, edge_weight, u=None)
        
        if not self.exclude_bwd:
            # Backward pass (reverse the input in time dimension)
            rev_x = x.flip(1)
            bwd_out, bwd_pred, bwd_repr, _ = self.bwd_grgl(rev_x, edge_index, edge_weight, u=None)
            
            # Flip backward results back to the original time direction
            bwd_out, bwd_pred, bwd_repr = [res.flip(1) for res in [bwd_out, bwd_pred, bwd_repr]]
        else:
            bwd_out, bwd_pred, bwd_repr = fwd_out, fwd_pred, fwd_repr

        # Concatenate forward and backward results
        out = torch.cat([fwd_out, bwd_out], dim=-1)
        
        return out
            
    def generate(self,
             X: Tensor,
             edge_index: Adj,
             edge_weight: OptTensor = None,
             u: OptTensor = None,
             steps: int = 32,
             disable_bar = False,
             **kwargs) -> Tensor:
    
        # Check for the presence of a scaler in kwargs
        scaler = kwargs.get('scaler')
        if scaler:
            X = scaler.transform(X)
        
        nextval = X
        output = []
        
        with tqdm(total=steps, disable=disable_bar, desc='Generating', unit='step') as t:
            for _ in range(steps):
                # Forward pass
                out = self.forward(x=nextval, u=u, edge_index=edge_index, edge_weight=edge_weight)
                
                # Get generated output
                gen = self.get_output(out)
                
                # Prepare for the next step
                nextval = gen
                output.append(gen)
                t.update(1)
        
        # Concatenate and inverse transform if scaler exists
        output = torch.cat(output)
        if scaler:
            output = scaler.inverse_transform(output)
        
        return output

    def predict(self,
            X: Tensor,
            edge_index: Adj,
            edge_weight: OptTensor = None,
            u: OptTensor = None,
            encoder_only: bool = False,
            both_mean: bool = False,
            **kwargs) -> Tensor:
        
        scaler = kwargs.get('scaler')
        if scaler:
            X = scaler.transform(X)
        
        nextval = X
        output = []
        steps = X.shape[0]

        with tqdm(total=steps, desc='Predicting', unit='step') as t:
            for i in range(steps):
                nextval = X[i].reshape(1, 1, X.shape[-2], 1)
                
                # Forward pass
                out = self.forward(x=nextval, u=None, edge_index=edge_index, edge_weight=edge_weight)
                
                # Get generated output
                gen = self.get_output(out)
                
                output.append(gen)
                t.update(1)
        
        # Concatenate and inverse transform if scaler exists
        output = torch.cat(output)
        if scaler:
            output = scaler.inverse_transform(output)
        
        return output
    
    
    def get_output(self, out):
        pred_index = (self.input_size + 2) * self.M 
        
        # Extract encoder and decoder components
        dec_fwd, dec_bwd = out[..., :pred_index], out[..., pred_index:]
        
        # Helper function to compute and combine forward and backward components
        def compute_mean(fwd, bwd):
            out_fwd, out_bwd = self.compute_for(fwd), self.compute_for(bwd)
            return torch.mean(torch.cat([out_fwd, out_bwd], axis=-1), axis=-1, keepdim=True)

        output = compute_mean(dec_fwd, dec_bwd)
        
        return output
    
    def compute_for(self, out):
        # Extract means, stds, and weights
        means, stds, weights = out[..., :self.stds_index], out[..., self.stds_index:self.weights_index], out[..., self.weights_index:]
        
        
        # Clean up invalid values (NaN or Inf) in means, stds, and weights
        def clean_tensor(tensor):
            return torch.where(torch.isnan(tensor) | torch.isinf(tensor), torch.zeros_like(tensor), tensor).to(tensor.device)

        means, stds, weights = clean_tensor(means), clean_tensor(stds), clean_tensor(weights)

        normal = torch.normal(means, stds)
        # Generate output based on mixture weights mode
        gen = weights * normal
        
        # Compute mean over the last axis and reshape the result
        gen = torch.mean(gen, axis=-1)
        gen = gen.reshape(1, 1, gen.shape[-1], 1)
        
        return gen
import torch
from torch import Tensor
from typing import Optional
from tsl.nn.models import BaseModel 
from tsl.nn.layers import NodeEmbedding
from torch_geometric.typing import Adj, OptTensor
from .Cells import GRGNCell
import random
from tqdm import tqdm
import numpy as np
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
                steps_ahead: int = 5,
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
        self.steps_ahead = steps_ahead
        
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
        out = torch.cat([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=-1)
        
        return out
    
    def autoregression(self,
             X: Tensor,
             edge_index: Adj,
             edge_weight: OptTensor = None,
             u: OptTensor = None,
             steps: int = 32,
             add_noise = True,
             noise_mean: float = 0.,
             noise_stddev: float = 1.,
             disable_bar = False,
             enc_dec_mean = False,
             exclude_bwd = False,
             **kwargs) -> Tensor:
    
        # Check for the presence of a scaler in kwargs
        scaler = kwargs.get('scaler')
        if scaler:
            X = scaler.transform(X)
        if exclude_bwd: self.exclude_bwd = True
        # nextval = X
        output = None
        nextval = X
        noiser = torch.distributions.Normal(noise_mean, noise_stddev)
        
        with tqdm(total=steps, disable=disable_bar, desc='Generating', unit='step') as t:
            for _ in range(steps):
                # Forward pass
                steps_ahead = self.steps_ahead if nextval.shape[1] > self.steps_ahead else nextval.shape[1]
                out = self.forward(x=nextval[:, -steps_ahead:, :, :], u=u, edge_index=edge_index, edge_weight=edge_weight)
                
                gen = self.get_output(out, enc_dec_mean)
                noise = noiser.sample(gen.size()).to(X.device)
                
                nextval = gen + noise if add_noise else gen
                output = torch.cat([output, gen[:, -1:, :, :]], dim=1) if output is not None else gen[:, -1:, :, :]
                t.update(1)

        # Concatenate and inverse transform if scaler exists
        if scaler:
            output = scaler.inverse_transform(output)
        if exclude_bwd: self.exclude_bwd = False
        
        return output
            
    def generate(self,
             edge_index: Adj,
             edge_weight: OptTensor = None,
             u: OptTensor = None,
             steps: int = 32,
             noise_mean: float = 0.,
             noise_stddev: float = 5.,
             disable_bar = False,
             enc_dec_mean = False,
             exclude_bwd = False,
             **kwargs) -> Tensor:
    
        # Check for the presence of a scaler in kwargs
        scaler = kwargs.get('scaler')
        if exclude_bwd: self.exclude_bwd = True

        # nextval = X
        output = None
        means = torch.ones((1, self.steps_ahead, self.n_nodes, self.input_size)) * noise_mean
        stds = torch.ones((1, self.steps_ahead, self.n_nodes, self.input_size)) * noise_stddev
        noiser = torch.distributions.Normal(means, stds)
        nextval = noiser.sample()
        
        with tqdm(total=steps, disable=disable_bar, desc='Generating', unit='step') as t:
            for _ in range(steps):
                # Forward pass
                out = self.forward(x=nextval[:, -self.steps_ahead:, :, :], u=u, edge_index=edge_index, edge_weight=edge_weight)
                
                gen = self.get_output(out, enc_dec_mean)
                
                nextval = gen
                output = torch.cat([output, gen[:, -1:, :, :]], dim=1) if output is not None else gen[:, -1:, :, :]
                t.update(1)

        # Concatenate and inverse transform if scaler exists
        if scaler:
            output = scaler.inverse_transform(output)
        if exclude_bwd: self.exclude_bwd = False
        
        return output
    
    def predict(self,
            X: Tensor,
            edge_index: Adj,
            edge_weight: OptTensor = None,
            u: OptTensor = None,
            enc_dec_mean: bool = False,
            steps = 32,
            exclude_bwd = False,
            **kwargs) -> Tensor:
        
        scaler = kwargs.get('scaler')
        if scaler:
            X = scaler.transform(X)
        
        if exclude_bwd: self.exclude_bwd = True
        nextval = X.permute(1, 0, 2, 3)
        output = None
        steps = steps//nextval.shape[0]
        
        with tqdm(total=steps, desc='Predicting', unit='step') as t:
            for i in range(steps):
                # Forward pass
                out = self.forward(x=nextval, u=None, edge_index=edge_index, edge_weight=edge_weight)
                
                # Get generated output
                gen = self.get_output(out, enc_dec_mean)
                
                nextval = gen
                
                output = torch.cat([output, gen], dim=0) if output is not None else gen
                t.update(1)
        
        # Concatenate and inverse transform if scaler exists
        if scaler:
            output = scaler.inverse_transform(output)
        if exclude_bwd: self.exclude_bwd = False
        
        return output
    
    
    def imputation(self,
                   X: Tensor,
                   mask: Tensor,
                   edge_index: Adj,
                   edge_weight: OptTensor = None,
                   u: OptTensor = None,
                   enc_dec_mean: bool = False,
                    exclude_bwd = False,
                   **kwargs) -> Tensor:
        
        scaler = kwargs.get('scaler')
        if scaler:
            X = scaler.transform(X)
        if exclude_bwd: self.exclude_bwd = True
        
        # Forward pass to get predictions
        out = self.forward(x=X, u=None, edge_index=edge_index, edge_weight=edge_weight)
        
        # Get generated output
        gen = self.get_output(out, enc_dec_mean)
        
        gen = torch.cat([X[0:1], gen[:-1]], dim=0).to(X.device)
        output = torch.where(mask, gen, X)

        # Inverse transform the output if a scaler is provided
        if scaler:
            output = scaler.inverse_transform(output)
        if exclude_bwd: self.exclude_bwd = False

        return output
    
    def get_output(self, out, enc_dec_mean):
        pred_index = (self.input_size + 2) * self.M 
        
        # Extract encoder and decoder components
        dec_fwd, dec_bwd = out[..., :pred_index], out[..., pred_index:pred_index*2]
        enc_fwd, enc_bwd = out[..., pred_index*2:pred_index*3], out[..., pred_index*3:]
        
        # Helper function to compute and combine forward and backward components
        def compute_mean(fwd, bwd):
            out_fwd = self.compute_for(fwd[..., :self.stds_index], fwd[..., self.stds_index:self.weights_index], fwd[..., self.weights_index:])
            out_bwd = self.compute_for(bwd[..., :self.stds_index], bwd[..., self.stds_index:self.weights_index], bwd[..., self.weights_index:])
            if self.exclude_bwd: return out_fwd
            return torch.mean(torch.cat([out_fwd, out_bwd], axis=-1), axis=-1, keepdim=True)

        output = compute_mean(dec_fwd, dec_bwd)
        if enc_dec_mean:
            output2 = compute_mean(enc_fwd, enc_bwd)
            output = torch.cat([output2, output], axis=-1)
            output = torch.mean(output, axis=-1, keepdim=True)
        
        return output
    
    def compute_for(self, means, stds, weights):
        """
        Args:
            means: Tensor of shape (b, s, n, m), representing the means.
            stds: Tensor of shape (b, s, n, m), representing the standard deviations.
            weights: Tensor of shape (b, s, n, m), representing the mixture weights.
        
        Returns:
            A tensor `gen` with sampled values from the distribution.
        """
        # Clean up invalid values (NaN or Inf) in means, stds, and weights
        def clean_tensor(tensor):
            return torch.where(torch.isnan(tensor) | torch.isinf(tensor), torch.zeros_like(tensor), tensor).to(tensor.device)

        means, stds, weights = clean_tensor(means), clean_tensor(stds), clean_tensor(weights)

        # Initialize the generated tensor
        gen = torch.zeros(size=means.shape[:-1] + (1,), device=means.device)
        
        # Sample uniform random values for each node and batch
        u = torch.rand(size=weights.shape[:-1], device=weights.device)  # Shape: (b, s, n)
        
        # Get the minimum index based on cumulative weights (vectorized)
        m = self.get_min_index(u, weights)
        
        # Gather the means and standard deviations for the chosen indices (m)
        batch_indices = torch.arange(means.shape[0], device=means.device).view(-1, 1, 1)
        seq_indices = torch.arange(means.shape[1], device=means.device).view(1, -1, 1)
        node_indices = torch.arange(means.shape[2], device=means.device).view(1, 1, -1)
        
        selected_means = means[batch_indices, seq_indices, node_indices, m]
        selected_stds = stds[batch_indices, seq_indices, node_indices, m]

        # Generate normally distributed random values using the selected means and stds
        gen = torch.distributions.Normal(selected_means, torch.sqrt(selected_stds)).rsample().unsqueeze(-1)
        
        return gen
    
    def get_min_index(self, val, pis):
        """
        Get the smallest index in the m dimension for each (b, s) where
        the cumulative sum of the pis along the m dimension is greater than or equal to val.
        
        Args:
            val: A float tensor of shape (b, s, n) with random values between 0 and 1.
            pis: Tensor of shape (b, s, n, m), representing the probabilities.
            
        Returns:
            A tensor of shape (b, s, n) with the smallest index in the m dimension for each (b, s, n).
        """
        # Compute the cumulative sum along the m dimension
        cumsum_pis = torch.cumsum(pis, dim=-1)
        
        # Expand val to have the same shape as cumsum_pis for broadcasting
        val_expanded = val.unsqueeze(-1)  # Shape becomes (b, s, n, 1)
        
        # Create a mask where the cumulative sum is greater than or equal to val
        mask = cumsum_pis >= val_expanded
        
        # Create an index tensor with values ranging from 0 to m-1
        indices = torch.arange(pis.shape[-1], device=pis.device)
        
        # Use torch.where to replace False with a large value (mask.shape[-1] ensures it's beyond valid index range)
        valid_indices = torch.where(mask, indices, pis.shape[-1])
        
        # Find the minimum valid index (first occurrence of mask >= val) along the m dimension
        min_index, _ = torch.min(valid_indices, dim=-1)
        
        return min_index
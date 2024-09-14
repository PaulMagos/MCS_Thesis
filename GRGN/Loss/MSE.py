import torch
from functools import partial
import numpy as np
from typing import Any
from torchmetrics import Metric
from GRGN.Utils.reshapes import reshape_to_original

class MSE_Custom(Metric):
    def __init__(self, mixture_weights_mode: str = 'weighted', **kwargs: Any):
        # set 'full_state_update' before Metric instantiation
        super(MSE_Custom, self).__init__(**kwargs)
        
        self.mixture_weights_mode = mixture_weights_mode
        
        self.add_state('value',
                       dist_reduce_fx='sum',
                       default=torch.tensor(0., dtype=torch.float))
        self.add_state('numel',
                       dist_reduce_fx='sum',
                       default=torch.tensor(0., dtype=torch.float))
        
    def loss_function(self, y_pred, y_true, **kwargs):
        """
        GMM loss function.
        Assumes that y_pred has (D+2)*M dimensions and y_true has D dimensions. The first 
        M*D features are treated as means, the next M features as standard devs and the last 
        M features as mixture components of the GMM. 
        """
        firts_pred = y_pred.shape[-1]//4

        y_pred_fwd = y_pred[..., firts_pred:firts_pred*2]
        y_pred_bwd = y_pred[..., 3*firts_pred:]
        
        def loss(y_true, y_pred, M):
            means = y_pred[..., :M]
            stds  = y_pred[..., M:M*2]
            weights = y_pred[..., M*2:]
            
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
                    
            # MSE between gen and y_pred 
            diff = torch.mean(torch.square(gen - y_true), axis=-1)
            return diff

        D = y_true.shape[-1]
        M = y_pred_fwd.shape[-1] // ((D*2) + 1)
        
        fwd_loss = loss(y_true, y_pred_fwd, M)
        bwd_loss = loss(y_true, y_pred_bwd, M)
            
        result = (fwd_loss.mean() + bwd_loss.mean())/2
            
        return result

    def update(self, y_pred, y_true, **kwargs):
        self.value = self.loss_function(y_pred, y_true, **kwargs)
    
    def compute(self):
        return self.value
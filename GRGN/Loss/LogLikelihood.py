import torch
from functools import partial
import numpy as np
from typing import Any
from torchmetrics import Metric

class LogLikelihood(Metric):
    def __init__(self, first: bool = True, **kwargs: Any):
        # set 'full_state_update' before Metric instantiation
        super(LogLikelihood, self).__init__(**kwargs)
        
        self.first = first
        
        self.metric_fn = self.loss_function
        
        self.sqrt = torch.sqrt(2. * torch.tensor(np.pi))
        
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
        firts_pred = y_pred.shape[-1]//2
        if self.first:
            y_pred = y_pred[..., :firts_pred]
        else:
            y_pred = y_pred[..., firts_pred:]
        
        def loss(m, M, D, y_true, y_pred):
            mu = y_pred[..., D*m:(m+1)*D]
            sigma = y_pred[..., D*M+m]
            alpha = y_pred[..., (D+1)*M+m]
            
            # sqrt = torch.sqrt(2. * torch.tensor(np.pi))
            # Calculate exponent term
            exponent = -torch.sum((mu - y_true)**2, -1) / (2*sigma**2)
            # Handle potential NaN or Inf values in exponent
            exponent = torch.where(torch.isnan(exponent) | torch.isinf(exponent), torch.zeros_like(exponent), exponent)
            # Calculate left term
            left = alpha / (sigma * self.sqrt)
            # Handle potential NaN or Inf values in left term
            left = torch.where(torch.isnan(left) | torch.isinf(left), torch.zeros_like(left), left)
            # Calculate loss
            loss = left * torch.exp(exponent)
            # Handle potential NaN or Inf values in loss
            loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
            
            return loss.unsqueeze(-1)

        D = y_true.shape[-1]
        M = y_pred.shape[-1] // (D+2)
        
        Y_Pred_shape = y_true.shape
        new_shape = (M, *Y_Pred_shape)
        
        result = torch.zeros(new_shape)
        for m in range(M):
            result[m] = loss(m, M, D, y_true, y_pred)
            
        # Handling NaN and Inf values
        result = torch.where(torch.isnan(result) | torch.isinf(result), torch.zeros_like(result), result)
        
        # Avoiding division by zero
        result = torch.where(result == 0, torch.ones_like(result) * 1e-8, result)
        return torch.log(result.sum(0).mean(0).mean(-2))

    def update(self, y_pred, y_true, **kwargs):
        self.value = self.loss_function(y_pred, y_true, **kwargs)
    
    def compute(self):
        return self.value
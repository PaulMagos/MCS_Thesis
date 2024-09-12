import torch
from functools import partial
import numpy as np
from typing import Any
from torchmetrics import Metric
from GRGN.Utils.reshapes import reshape_to_original

class LogLikelihood(Metric):
    def __init__(self, encoder_only: bool = False, both = False, **kwargs: Any):
        # set 'full_state_update' before Metric instantiation
        super(LogLikelihood, self).__init__(**kwargs)
        
        self.encoder_only = encoder_only
        
        self.both = both
        
        self.sqrt = torch.sqrt(2. * torch.tensor(np.pi))
        
        self.add_state('value',
                       dist_reduce_fx='sum',
                       default=torch.tensor(0., dtype=torch.float))
        self.add_state('numel',
                       dist_reduce_fx='sum',
                       default=torch.tensor(0., dtype=torch.float))
        
    def loss_function (self, y_pred, y_true, **kwargs):
        return self.loss(y_pred, y_true, **kwargs) if not self.both else self.double_loss(y_pred, y_true, **kwargs)
        
    def double_loss(self, y_pred, y_true, **kwargs):
        current_state = self.encoder_only
        first = self.loss(y_pred, y_true)
        self.encoder_only = not current_state
        second = self.loss(y_pred, y_true)
        self.encoder_only = current_state
        return (first + second)/2
        
    def loss(self, y_pred, y_true, **kwargs):
        """
        GMM loss function.
        Assumes that y_pred has (D+2)*M dimensions and y_true has D dimensions. The first 
        M*D features are treated as means, the next M features as standard devs and the last 
        M features as mixture components of the GMM. 
        """
        self.sqrt = self.sqrt.to(y_pred.device)
        firts_pred = y_pred.shape[-1]//2
        if self.encoder_only:
            y_pred = y_pred[..., :firts_pred]
        else:
            y_pred = y_pred[..., firts_pred:]
        
        def loss(m, M, D, y_true, y_pred):
            y_true_local = y_true.view(-1, D * y_true.shape[-1])
            mu = y_pred[..., D*m:(m+1)*D]
            sigma = y_pred[..., D*M+(D*m): D*M+D*(m+1)]
            alpha = y_pred[..., (D*2)*M+(D*m) : (D*2)*M+(D*(m+1))]
            
            # sqrt = torch.sqrt(2. * torch.tensor(np.pi))
            # Calculate exponent term
            exponent = -torch.sum((mu - y_true_local)**2, -1) / (2*sigma**2)
            # Handle potential NaN or Inf values in exponent
            exponent = torch.where(torch.isnan(exponent) | torch.isinf(exponent), torch.zeros_like(exponent), exponent).to(y_pred.device)
            # Calculate left term
            left = alpha / (sigma * self.sqrt)
            # Handle potential NaN or Inf values in left term
            left = torch.where(torch.isnan(left) | torch.isinf(left), torch.zeros_like(left), left).to(y_pred.device)
            # Calculate loss
            loss = left * torch.exp(exponent)
            # Handle potential NaN or Inf values in loss
            loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss).to(y_pred.device)
            return loss.sum(-1)

        D = y_true.shape[-1]
        M = y_pred.shape[-1] // (D * 3)
        y_pred = reshape_to_original(y_pred, y_true.shape[-2], D, M)
        D = y_true.shape[-1] * y_true.shape[-2]
        
        new_shape = (M, 1)
        
        result = torch.zeros(new_shape).to(y_pred.device)
        for m in range(M):
            result[m] = loss(m, M, D, y_true, y_pred)
            
        # Handling NaN and Inf values
        result = torch.where(torch.isnan(result) | torch.isinf(result), torch.zeros_like(result), result).to(y_pred.device)
        
        # Avoiding division by zero
        result = torch.where(result == 0, torch.ones_like(result) * 1e-8, result).to(y_pred.device)
        result = result.sum(0)
        result = -torch.log(result).to(y_pred.device)
        return result

    def update(self, y_pred, y_true, **kwargs):
        self.value = self.loss_function(y_pred, y_true, **kwargs)
    
    def compute(self):
        return self.value
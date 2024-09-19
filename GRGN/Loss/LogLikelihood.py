import torch
from functools import partial
import numpy as np
from typing import Any
from torchmetrics import Metric
from GRGN.Utils.reshapes import reshape_to_original

class LogLikelihood(Metric):
    def __init__(self, enc=False, dec=False, **kwargs: Any):
        super(LogLikelihood, self).__init__(**kwargs)  # Initialize Metric before add_state
        
        self.enc = enc
        self.dec = dec
        
        if self.dec == self.enc == False:
            # Error can't have both
            self.dec = True
            self.enc  = True
        self.sqrt = torch.sqrt(2. * torch.tensor(torch.pi))

        # Add states after the Metric is initialized
        self.add_state('value', default=torch.tensor([0.], dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('numel', default=torch.tensor(0., dtype=torch.float), dist_reduce_fx='sum')
        
    def loss_function(self, y_pred, y_true, **kwargs):
        """
        GMM loss function.
        Assumes that y_pred has (D+2)*M dimensions and y_true has D dimensions.
        """
        self.sqrt = self.sqrt.to(y_pred.device)
        firts_pred = y_pred.shape[-1] // 4
        if self.enc:
            y_pred_fwd = y_pred[..., firts_pred*2:firts_pred*3]
            y_pred_bwd = y_pred[..., firts_pred*3:]
        if self.dec:
            y_pred_fwd  = y_pred[..., :firts_pred]
            y_pred_bwd   = y_pred[..., firts_pred:firts_pred*2]
        
        def loss_inner(m, M, D, y_true, y_pred):
            y_true_local = y_true[..., :]
            
            mu = y_pred[..., D * m:D * (m + 1)]
            sigma = y_pred[..., D * M + m: D * M + (m + 1)]
            alpha = y_pred[..., (D + 1) * M + m:(D + 1) * M + m + 1]
            
            # Calculate exponent term
            sum = -torch.sum((mu - y_true_local)**2, -1, keepdim=True)
            exponent = sum / (2 * sigma**2)
            
            # Calculate left term
            left = alpha / (sigma * self.sqrt)
            
            # Calculate loss
            loss = left * torch.exp(exponent)
            loss = loss.mean(dim=(0, 1))
            return loss

        D = y_true.shape[-1]
        M = y_pred_fwd.shape[-1] // (D + 2)
        
        new_shape = (M, y_true.shape[-2], D)
        result_fwd = torch.zeros(new_shape).to(y_pred.device)
        result_bwd = torch.zeros(new_shape).to(y_pred.device)
        
        for m in range(M):
            result_fwd[m] = loss_inner(m, M, D, y_true, y_pred_fwd)
            result_bwd[m] = loss_inner(m, M, D, y_true, y_pred_bwd)
        result_fwd = -torch.log(result_fwd.sum(0)).mean(0)
        result_bwd = -torch.log(result_bwd.sum(0)).mean(0)
        
        result1 = result_fwd.to(y_pred.device)
        result2 = result_bwd.to(y_pred.device)
        result = (result1 + result2)/2
        
        return result

    def update(self, y_pred, y_true, **kwargs):
        if not self.enc and not self.dec:
            self.enc = True
            loss_enc = self.loss_function(y_pred, y_true, **kwargs)
            self.enc = False
            self.dec = True
            loss_dec = self.loss_function(y_pred, y_true, **kwargs)
            self.dec = False
            loss = (loss_enc + loss_dec)/2
        else:
            # Accumulate the loss and track the number of samples processed
            loss = self.loss_function(y_pred, y_true, **kwargs)
        
        if loss.dim() == 0:  # If loss is a scalar (shape [])
            loss = loss.unsqueeze(0)

        self.value += loss
        self.numel += 1

    def compute(self):
        # Return the accumulated value
        if self.numel > 0:
            return self.value / self.numel
        return self.value
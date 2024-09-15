import torch
from functools import partial
import numpy as np
from typing import Any
from torchmetrics import Metric
from GRGN.Utils.reshapes import reshape_to_original

class LogLikelihood(Metric):
    def __init__(self, encoder_only: bool = False, both: bool = False, weights_mode: str = 'weighted', **kwargs: Any):
        super(LogLikelihood, self).__init__(**kwargs)  # Initialize Metric before add_state
        
        self.encoder_only = encoder_only
        self.both = both
        self.weights_mode = weights_mode
        self.sqrt = torch.sqrt(2. * torch.tensor(torch.pi))

        # Add states after the Metric is initialized
        self.add_state('value', default=torch.tensor([0.], dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('numel', default=torch.tensor(0., dtype=torch.float), dist_reduce_fx='sum')
        
    def loss_function(self, y_pred, y_true, **kwargs):
        return self.loss(y_pred, y_true, **kwargs) if not self.both else self.double_loss(y_pred, y_true, **kwargs)
        
    def double_loss(self, y_pred, y_true, **kwargs):
        current_state = self.encoder_only
        first = self.loss(y_pred, y_true)
        self.encoder_only = not current_state
        second = self.loss(y_pred, y_true)
        self.encoder_only = current_state
        return (first + second) / 2
        
    def loss(self, y_pred, y_true, **kwargs):
        """
        GMM loss function.
        Assumes that y_pred has (D+2)*M dimensions and y_true has D dimensions.
        """
        self.sqrt = self.sqrt.to(y_pred.device)
        firts_pred = y_pred.shape[-1] // 4
        if self.encoder_only:
            y_pred_fwd = y_pred[..., :firts_pred]
            y_pred_bwd = y_pred[..., 2*firts_pred:3*firts_pred]
        else:
            y_pred_fwd = y_pred[..., firts_pred:firts_pred*2]
            y_pred_bwd = y_pred[..., 3*firts_pred:]
        
        def loss_inner(m, M, D, y_true, y_pred):
            y_true_local = y_true.view(-1, D * y_true.shape[-1])
            
            start_index_gaussian = D * m
            end_index_gaussian = D * (m + 1)
            
            sigma_index = D * M
            alpha_index = (D + 1) * M
            
            mu = y_pred[..., start_index_gaussian:end_index_gaussian]
            sigma = y_pred[..., sigma_index + m]
            alpha = y_pred[..., alpha_index + m]
            if self.weights_mode == 'uniform':
                alpha = torch.ones_like(alpha)
            elif self.weights_mode == 'equal_probability':
                alpha = torch.ones_like(alpha) / M
            
            # Calculate exponent term
            exponent = -torch.sum((mu - y_true_local)**2, -1) / (2 * sigma**2)
            exponent = torch.where(torch.isnan(exponent) | torch.isinf(exponent), torch.zeros_like(exponent), exponent).to(y_pred.device)
            
            # Calculate left term
            left = alpha / (sigma * self.sqrt)
            left = torch.where(torch.isnan(left) | torch.isinf(left), torch.zeros_like(left), left).to(y_pred.device)
            
            # Calculate loss
            loss = left * torch.exp(exponent)
            loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss).to(y_pred.device)
            
            dimensions = [i for i in range(len(loss.shape))]
            return loss.sum(dim=dimensions)

        D = y_true.shape[-1]
        M = y_pred_fwd.shape[-1] // (D + 2)
        y_pred_fwd = reshape_to_original(y_pred_fwd, y_true.shape[-2], D, M)
        y_pred_bwd = reshape_to_original(y_pred_bwd, y_true.shape[-2], D, M)
        D = y_true.shape[-1] * y_true.shape[-2]
        
        new_shape = (M * 2, 1)
        result = torch.zeros(new_shape).to(y_pred.device)
        
        for m in range(M):
            result[m] = loss_inner(m, M, D, y_true, y_pred_fwd)
            result[m + M] = loss_inner(m, M, D, y_true, y_pred_bwd)
        
        result = (result[:M] + result[M:]) / 2
        
        result = torch.where(torch.isnan(result) | torch.isinf(result), torch.zeros_like(result), result).to(y_pred.device)
        result = torch.where(result == 0, torch.ones_like(result) * 1e-8, result).to(y_pred.device)
        result = result.sum(0)
        result = -torch.log(result).to(y_pred.device)
        
        return result

    def update(self, y_pred, y_true, **kwargs):
        # Accumulate the loss and track the number of samples processed
        loss = self.loss_function(y_pred, y_true, **kwargs)
        
        if loss.dim() == 0:  # If loss is a scalar (shape [])
            loss = loss.unsqueeze(0)

        self.value += loss
        self.numel += y_true.numel()

    def compute(self):
        # Return the accumulated value
        return self.value / self.numel
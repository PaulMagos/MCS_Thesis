from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

__all__= ['GMM', 'gmm_loss']

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'
torch.set_default_device(device)
 
class GMM(nn.Module):
    def __init__(self, M=10, device = 'cpu', debug=False, **kwargs) -> None:
        super(GMM, self).__init__()
        self.M = M
        self.device = device
        self.debug = debug
    
    def __call__(self, X):
        D = X.shape[-1] // self.M - 2
        # Leave mu values as they are since they're unconstrained
        # Scale sigmas with exp, so all values are non-negative
        X[:, :, D*self.M:(D+1)*self.M] = torch.exp(X[:, :, D*self.M:(D+1)*self.M]).to(self.device)
        # Scale alphas with softmax, so all values are between [0,1] and sum up to 1
        X[:, :, (D+1)*self.M:(D+2)*self.M] = F.softmax(X[:, :, (D+1)*self.M:(D+2)*self.M], dim=1).to(self.device)
        if self.debug:
            print(X[0].shape)
        return X[0].to(self.device)

def gmm_loss(y_true, y_pred):
    """
    GMM loss function.
    Assumes that y_pred has (D+2)*M dimensions and y_true has D dimensions. The first 
    M*D features are treated as means, the next M features as standard devs and the last 
    M features as mixture components of the GMM. 
    """

    def loss(m, M, D, y_true, y_pred):
        mu = y_pred[:, D*m:(m+1)*D]
        sigma = y_pred[:, D*M+m]
        alpha = y_pred[:, (D+1)*M+m]
        
        sqrt = torch.sqrt(2. * torch.tensor(np.pi))
        # Calculate exponent term
        exponent = -torch.sum((mu - y_true)**2, -1) / (2*sigma**2)
        # Handle potential NaN or Inf values in exponent
        exponent = torch.where(torch.isnan(exponent) | torch.isinf(exponent), torch.zeros_like(exponent), exponent)
        # Calculate left term
        left = alpha / (sigma * sqrt)
        # Handle potential NaN or Inf values in left term
        left = torch.where(torch.isnan(left) | torch.isinf(left), torch.zeros_like(left), left)
        # Calculate loss
        loss = left * torch.exp(exponent)
        # Handle potential NaN or Inf values in loss
        loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
        
        return loss

    D = y_true.shape[-1]
    M = y_pred.shape[-1] // (D+2)
    result = torch.zeros(M, 1)
    for m in range(M):
        result[m] = loss(m, M, D, y_true, y_pred)
        
    # Handling NaN and Inf values
    result = torch.where(torch.isnan(result) | torch.isinf(result), torch.zeros_like(result), result)
    
    # Avoiding division by zero
    result = torch.where(result == 0, torch.ones_like(result) * 1e-8, result)
    
    return -torch.log(result.sum(0))

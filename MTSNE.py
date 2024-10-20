import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class MTSNE:
    def __init__(self, n_components=2, perplexity=30, temporal_weight=1.0, max_iter=1000, random_state=None):
        """
        Parameters:
        - n_components: Number of dimensions for the output embedding (usually 2 or 3 for visualization).
        - perplexity: Perplexity parameter for t-SNE.
        - temporal_weight: Weight for the temporal smoothness constraint.
        - n_iter: Number of iterations for optimization.
        - random_state: Seed for reproducibility.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.temporal_weight = temporal_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.tsne = None  # To store the t-SNE model
        self.Y_train = None  # To store the fitted embedding for training data
        
    def fit_transform(self, X):
        """
        Fit m-TSNE to the multivariate time series data.
        
        Parameters:
        - X: Input multivariate time series data. Shape: (n_samples, time_steps, n_features)
        
        Returns:
        - Y: Embedded space representation with shape (n_samples, time_steps, n_components)
        """
        n_samples, time_steps, n_features = X.shape
        
        # Step 1: Flatten the data for t-SNE by treating each time step as a separate data point
        X_flat = X.reshape(n_samples * time_steps, n_features)
        
        # Apply standard t-SNE on the flattened data
        self.tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, max_iter=self.max_iter, random_state=self.random_state)
        Y_flat = self.tsne.fit_transform(X_flat)  # Shape: (n_samples * time_steps, n_components)
        
        # Reshape back into (n_samples, time_steps, n_components)
        Y = Y_flat.reshape(n_samples, time_steps, self.n_components)
        
        # Store the fitted embedding
        self.Y_train = Y
        
        # Step 2: Apply temporal smoothing (regularization)
        Y_smoothed = self.temporal_smoothness(Y, time_steps)
        
        return Y_smoothed
    
    def temporal_smoothness(self, Y, time_steps):
        """
        Apply temporal smoothness regularization to the t-SNE embeddings.
        
        Parameters:
        - Y: t-SNE embedding. Shape: (n_samples, time_steps, n_components)
        - time_steps: Number of time steps.
        
        Returns:
        - Y_smoothed: Smoothed embedding with temporal continuity.
        """
        n_samples, _, n_components = Y.shape
        
        # Step 3: Compute the temporal distances (difference between consecutive time steps)
        for i in range(n_samples):
            for t in range(1, time_steps):
                # Encourage temporal continuity by penalizing large differences between consecutive time steps
                diff = Y[i, t] - Y[i, t-1]
                Y[i, t] -= self.temporal_weight * diff  # Adjust by the temporal weight
        
        return Y
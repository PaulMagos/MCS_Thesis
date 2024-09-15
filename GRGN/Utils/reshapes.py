import torch

__all__ = ['reshape_to_nodes', 'reshape_to_original']

def reshape_to_nodes(input_tensor: torch.Tensor, num_nodes: int, input_shape: int, mixture_size: int) -> torch.Tensor:
    """
    Reshapes the input tensor of shape (steps, mixture_size * (num_nodes * input_shape + 2)) to (steps, num_nodes, mixture_size * (input_shape + 2)).
    The first part of the tensor contains the means, followed by stds and weights repeated for each node.
    
    Args:
    - input_tensor: Input tensor of shape (steps, mixture_size * (num_nodes * input_shape + 2))
    - num_nodes: Number of nodes (e.g., 325)
    - input_shape: The size of the input per node (e.g., 32)
    - mixture_size: The size of the mixture (e.g., 32)
    
    Returns:
    - Reshaped tensor of shape (steps, num_nodes, mixture_size * (input_shape + 2))
    """
    steps = input_tensor.shape[0]
    means_size = mixture_size * num_nodes * input_shape  # Size of means (mixture_size * num_nodes * input_shape)
    stds_size = mixture_size  # Std size (32) 
    
    # Extract means, stds, and weights from the input tensor
    means = input_tensor[..., :means_size]  # First part are the means (1, mixture_size * num_nodes * input_shape)
    stds = input_tensor[..., means_size:means_size + stds_size]  # Next part are the stds (1, mixture_size)
    weights = input_tensor[..., means_size + stds_size:]  # Weights (1, mixture_size)

    # Reshape means to (1, num_nodes, mixture_size * input_shape)
    means_reshaped = means.reshape(steps, num_nodes, mixture_size * input_shape)
    
    # Repeat stds and weights across all nodes 
    # stds_repeated = stds.reshape(steps, num_nodes, mixture_size * input_shape)  # Repeat stds for each node (1, num_nodes, mixture_size)
    # weights_repeated = weights.reshape(steps, num_nodes, mixture_size * input_shape)  # Repeat stds for each node (1, num_nodes, mixture_size)
    stds_repeated = stds.unsqueeze(1).repeat(1, num_nodes, 1)  # Repeat stds for each node (1, num_nodes, mixture_size)
    weights_repeated = weights.unsqueeze(1).repeat(1, num_nodes, 1)  # Repeat weights for each node (1, num_nodes, mixture_size)

    # Concatenate means, stds, and weights along the last dimension
    final_tensor = torch.cat((means_reshaped, stds_repeated, weights_repeated), dim=-1)

    return final_tensor  # Final shape is (1, num_nodes, mixture_size * (input_shape + 2))


def reshape_to_original(final_tensor: torch.Tensor, num_nodes: int, input_shape: int, mixture_size: int) -> torch.Tensor:
    """
    Reverts a tensor from shape (steps, num_nodes, mixture_size * (input_shape + 2)) back to the original shape (steps, mixture_size * (num_nodes * input_shape + 2)).
    
    Args:
    - final_tensor: Input tensor of shape (steps, num_nodes, mixture_size * (input_shape + 2))
    - num_nodes: Number of nodes (e.g., 325)
    - input_shape: The size of the input per node (e.g., 32)
    - mixture_size: The size of the mixture (e.g., 32)
    
    Returns:
    - Reverted tensor of shape (steps, mixture_size * (num_nodes * input_shape + 2))
    """
    
    steps = final_tensor.shape[0]
    means_size = mixture_size * input_shape  # Size of means for one node
    stds_size = mixture_size # Size of stds

    # Extract means, stds, and weights from the input tensor
    means = final_tensor[..., :means_size]  # First part are the means (1, num_nodes, mixture_size * input_shape)
    stds = final_tensor[..., means_size:means_size + stds_size]  # Stds (1, mixture_size)
    weights = final_tensor[..., means_size + stds_size:]  # Weights (1, mixture_size)

    # Flatten the means across nodes
    means_flat = means.reshape(steps, -1)  # Flatten means to (1, mixture_size * num_nodes * input_shape)

    # stds_flat = stds.reshape(steps, -1)   # Flatten stds to (mixture_size * num_nodes,)
    
    # weights_flat = weights.reshape(steps, -1)   # Flatten stds to (mixture_size * num_nodes,)

    # means_stds = torch.cat([means_flat, stds_flat], dim=-1)  # Concatenate stds and means
    # original_tensor = torch.cat([means_flat, stds_flat, weights_flat], dim=-1)  # Concatenate stds and means

    # original_tensor = torch.cat([means_flat, weights.reshape(-1, weights.shape[-1] * weights.shape[-2])[..., -mixture_size:]], dim=-1)
    original_tensor = torch.cat([means_flat, stds.reshape(-1, stds.shape[-1] * stds.shape[-2])[..., -mixture_size:], weights.reshape(-1, weights.shape[-1] * weights.shape[-2])[..., -mixture_size:]], dim=-1)

    return original_tensor  # Final shape is (1, mixture_size * (num_nodes * input_shape + 2))
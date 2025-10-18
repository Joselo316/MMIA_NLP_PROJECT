from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    
    # Create position indices
    position = torch.arange(seqlen, device=device).float()
    
    # Create frequency indices for each dimension pair
    dim_indices = torch.arange(0, head_dim // 2, device=device).float()
    
    # Compute frequencies: theta^(-2i/d) for i in [0, d/2-1]
    freqs = 1.0 / (theta ** (2 * dim_indices / head_dim))
    
    # Create position-frequency matrix: pos * freqs
    # Shape: (seqlen, head_dim//2)
    freqs_cis = torch.outer(position, freqs)
    
    # Compute cos and sin values
    cos_freqs = torch.cos(freqs_cis)
    sin_freqs = torch.sin(freqs_cis)
    
    # Reshape to match the complex representation
    # freqs_cis should have shape (seqlen, head_dim//2)
    # We need to reshape it to match query shape for broadcasting
    cos_freqs = cos_freqs.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, head_dim//2)
    sin_freqs = sin_freqs.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, head_dim//2)
    
    # Expand to match the batch and head dimensions
    cos_freqs = cos_freqs.expand(query.shape[0], -1, query.shape[2], -1)
    sin_freqs = sin_freqs.expand(query.shape[0], -1, query.shape[2], -1)
    
    # Reshape query and key to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    
    # Apply rotary embeddings using complex multiplication
    # For complex number z = a + bi and rotation matrix R = [cos -sin; sin cos]
    # R * z = (a*cos - b*sin) + i(a*sin + b*cos)
    query_out_real = query_real * cos_freqs - query_imag * sin_freqs
    query_out_imag = query_real * sin_freqs + query_imag * cos_freqs
    
    key_out_real = key_real * cos_freqs - key_imag * sin_freqs
    key_out_imag = key_real * sin_freqs + key_imag * cos_freqs
    
    # Combine real and imaginary parts back to original shape
    query_out = torch.stack([query_out_real, query_out_imag], dim=-1).reshape(query.shape).type_as(query)
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1).reshape(key.shape).type_as(key)
    
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def rf_filter(
    img: torch.Tensor,
    sigma_s: float,
    sigma_r: float,
    num_iterations: int = 3,
    joint_image: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Domain transform recursive edge-preserving filter.

    Args:
        img (torch.Tensor): Input image to be filtered [B, C, H, W] or [C, H, W].
        sigma_s (float): Filter spatial standard deviation.
        sigma_r (float): Filter range standard deviation.
        num_iterations (int, optional): Number of iterations to perform. Default: 3.
        joint_image (torch.Tensor, optional): Optional image for joint filtering [B, C_j, H, W] or [C_j, H, W].

    Returns:
        torch.Tensor: Filtered image with same dimensions and type as input.

    Reference:
        "Domain Transform for Edge-Aware Image and Video Processing"
        Eduardo S. L. Gastal and Manuel M. Oliveira
        ACM Transactions on Graphics. Volume 30 (2011), Number 4.
        Proceedings of SIGGRAPH 2011, Article 69.
    """
    # Add batch dimension if not present
    img_is_3d = img.dim() == 3
    if img_is_3d:
        img = img.unsqueeze(0)
    
    # Ensure input is a float tensor
    input_dtype = img.dtype
    I = img.float()
    
    # Handle joint filtering
    if joint_image is not None:
        if joint_image.dim() == 3:
            joint_image = joint_image.unsqueeze(0)
        J = joint_image.float()
        if I.shape[2:] != J.shape[2:]:
            raise ValueError("Input and joint images must have equal height and width.")
    else:
        J = I
    
    # Extract image dimensions
    # batch_size, num_channels, height, width = I.shape
    
    # Compute the domain transform
    dHdx, dVdy = _compute_domain_transform(J, sigma_s, sigma_r)
    
    # Perform the filtering
    F_out = I
    sigma_H = sigma_s

    # Compute sigma for each iteration once
    sigma_H_i_values = [
        sigma_H * torch.sqrt(torch.tensor(3.0, device=I.device)) * 
        (2**(num_iterations - (i + 1))) / 
        torch.sqrt(torch.tensor(4.0**num_iterations - 1.0, device=I.device))
        for i in range(num_iterations)
    ]
    
    for i in range(num_iterations):
        # Apply horizontal filter
        F_out = domain_recursive_filter_horizontal(F_out, dHdx, sigma_H_i_values[i])
        
        # Transpose for vertical filtering
        F_out = F_out.transpose(2, 3)
        dVdy_t = dVdy.transpose(2, 3)
        
        # Apply filter to transposed image (vertical filtering)
        F_out = domain_recursive_filter_horizontal(F_out, dVdy_t, sigma_H_i_values[i])
        
        # Transpose back
        F_out = F_out.transpose(2, 3)
    
    # Remove batch dimension if input didn't have one
    if img_is_3d:
        F_out = F_out.squeeze(0)
        
    # Return output with same dtype as input
    return F_out.to(input_dtype)


def _compute_domain_transform(
    J: torch.Tensor, 
    sigma_s: float, 
    sigma_r: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the domain transform derivatives for horizontal and vertical filtering.
    
    Args:
        J (torch.Tensor): Input image tensor [B, C, H, W].
        sigma_s (float): Filter spatial standard deviation.
        sigma_r (float): Filter range standard deviation.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Domain transform derivatives (dHdx, dVdy).
    """
    # Calculate horizontal gradients (along width)
    dIcdx = J[:, :, :, 1:] - J[:, :, :, :-1]
    # For vertical gradients (along height)
    dIcdy = J[:, :, 1:, :] - J[:, :, :-1, :]

    # Initialize gradient tensors with zeros
    dIdx = torch.zeros(J.shape[0], 1, J.shape[2], J.shape[3], 
                       device=J.device, dtype=J.dtype)
    dIdy = torch.zeros(J.shape[0], 1, J.shape[2], J.shape[3], 
                       device=J.device, dtype=J.dtype)

    # Compute L1-norm distance of neighboring pixels
    for c in range(J.shape[1]):
        # Add channel contribution to horizontal gradient
        padded_dIdx = F.pad(torch.abs(dIcdx[:, c:c+1]), (0, 1, 0, 0))
        dIdx = dIdx + padded_dIdx

        # Add channel contribution to vertical gradient
        padded_dIdy = F.pad(torch.abs(dIcdy[:, c:c+1]), (0, 0, 0, 1))
        dIdy = dIdy + padded_dIdy

    # Compute domain transform derivatives
    dHdx = 1 + sigma_s / sigma_r * dIdx
    dVdy = 1 + sigma_s / sigma_r * dIdy
    
    return dHdx, dVdy


def domain_recursive_filter_horizontal(
    I: torch.Tensor, 
    D: torch.Tensor, 
    sigma: torch.Tensor
) -> torch.Tensor:
    """
    Apply the recursive filter along the horizontal dimension.
    
    Args:
        I (torch.Tensor): Input tensor [B, C, H, W]
        D (torch.Tensor): Domain transform derivatives [B, 1, H, W]
        sigma (torch.Tensor): Sigma parameter for the filter
        
    Returns:
        torch.Tensor: Filtered tensor
    """
    # Feedback coefficient (Appendix of the paper)
    a = torch.exp(-torch.sqrt(torch.tensor(2.0, device=I.device)) / sigma)
    
    # Create a copy of the input for output
    F = I.clone()
    
    # Compute feedback coefficients
    V = a ** D
    
    batch_size, num_channels, height, width = I.shape
    
    # Vectorized implementation if possible (for small tensors)
    if width <= 1000 and height * batch_size * num_channels <= 10000:
        # Left -> Right filter - vectorized implementation
        for i in range(1, width):
            F[:, :, :, i] = F[:, :, :, i] + V[:, :, :, i] * (F[:, :, :, i-1] - F[:, :, :, i])
        
        # Right -> Left filter - vectorized implementation
        for i in range(width-2, -1, -1):
            F[:, :, :, i] = F[:, :, :, i] + V[:, :, :, i+1] * (F[:, :, :, i+1] - F[:, :, :, i])
    else:
        # Fallback to chunked processing for large tensors to avoid OOM errors
        chunk_size = 10
        for h_chunk in range(0, height, chunk_size):
            h_end = min(h_chunk + chunk_size, height)
            
            # Left -> Right filter
            for i in range(1, width):
                update = V[:, :, h_chunk:h_end, i] * (
                    F[:, :, h_chunk:h_end, i-1] - F[:, :, h_chunk:h_end, i]
                )
                F[:, :, h_chunk:h_end, i] = F[:, :, h_chunk:h_end, i] + update
            
            # Right -> Left filter
            for i in range(width-2, -1, -1):
                update = V[:, :, h_chunk:h_end, i+1] * (
                    F[:, :, h_chunk:h_end, i+1] - F[:, :, h_chunk:h_end, i]
                )
                F[:, :, h_chunk:h_end, i] = F[:, :, h_chunk:h_end, i] + update
    
    return F
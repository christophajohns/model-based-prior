import math
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
    
    # Calculate horizontal gradients (along width) for all channels
    dIcdx = J[:, :, :, 1:] - J[:, :, :, :-1] # [B, C, H, W-1]
    # Calculate vertical gradients (along height) for all channels
    dIcdy = J[:, :, 1:, :] - J[:, :, :-1, :] # [B, C, H-1, W]

    # Compute L1 norm over channels and pad
    dIdx = torch.sum(torch.abs(dIcdx), dim=1, keepdim=True) # [B, 1, H, W-1]
    dIdx = F.pad(dIdx, (0, 1, 0, 0)) # [B, 1, H, W]

    dIdy = torch.sum(torch.abs(dIcdy), dim=1, keepdim=True) # [B, 1, H-1, W]
    dIdy = F.pad(dIdy, (0, 0, 0, 1)) # [B, 1, H, W]

    # Compute domain transform derivatives
    dHdx = 1 + sigma_s / sigma_r * dIdx
    dVdy = 1 + sigma_s / sigma_r * dIdy
    
    return dHdx, dVdy

def _combine(state_L: Tuple[torch.Tensor, torch.Tensor], 
             state_R: Tuple[torch.Tensor, torch.Tensor],
             backward: bool = False,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Associative combine operation for forward scan (Left->Right).
    Represents the composition of affine transformations for y_i = B_i + A_i * y_{i-1}:
    If y_{i-1} = state_L.a * x + state_L.b
    and y_i     = state_R.a * y_{i-1} + state_R.b
    Then y_i    = state_R.a * (state_L.a * x + state_L.b) + state_R.b
                = (state_R.a * state_L.a) * x + (state_R.a * state_L.b + state_R.b)
    
    Args:
        state_L: Tuple (a_L, b_L) - transform applied first. 
                 a_L [B, 1, H, K], b_L [B, C, H, K].
        state_R: Tuple (a_R, b_R) - transform applied second.
                 a_R [B, 1, H, K], b_R [B, C, H, K].
        backward: bool - direction of the combination.

    Returns:
        Tuple (new_a, new_b) representing the combined transformation.
    """
    a_L, b_L = state_L
    a_R, b_R = state_R
    
    if not backward:
        new_a = a_R * a_L
        # PyTorch handles broadcasting: a_R [B, 1, H, K] * b_L [B, C, H, K] -> [B, C, H, K]
        new_b = a_R * b_L + b_R
    else: # backward
        new_a = a_L * a_R
        # PyTorch handles broadcasting: a_L [B, 1, H, K] * b_R [B, C, H, K] -> [B, C, H, K]
        new_b = a_L * b_R + b_L 
    
    return new_a, new_b

def _combine_forward(state_L: Tuple[torch.Tensor, torch.Tensor], 
                     state_R: Tuple[torch.Tensor, torch.Tensor]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Associative combine operation for forward scan (Left->Right).
    Represents the composition of affine transformations for y_i = B_i + A_i * y_{i-1}:
    If y_{i-1} = state_L.a * x + state_L.b
    and y_i     = state_R.a * y_{i-1} + state_R.b
    Then y_i    = state_R.a * (state_L.a * x + state_L.b) + state_R.b
                = (state_R.a * state_L.a) * x + (state_R.a * state_L.b + state_R.b)
    
    Args:
        state_L: Tuple (a_L, b_L) - transform applied first. 
                 a_L [B, 1, H, K], b_L [B, C, H, K].
        state_R: Tuple (a_R, b_R) - transform applied second.
                 a_R [B, 1, H, K], b_R [B, C, H, K].

    Returns:
        Tuple (new_a, new_b) representing the combined transformation.
    """
    return _combine(state_L=state_L, state_R=state_R, backward=False)

def _combine_backward(state_L: Tuple[torch.Tensor, torch.Tensor], 
                      state_R: Tuple[torch.Tensor, torch.Tensor]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Associative combine operation for composing backward steps (Right->Left).
    Represents the composition T_L * T_R where T_i maps z_{i+1} -> z_i via z_i = N_i + M_i*z_{i+1}.
    Let T_L = (M_L, N_L) and T_R = (M_R, N_R).
    Applying T_R then T_L: z_i = N_L + M_L * z_{i+1} = N_L + M_L * (N_R + M_R * z_{i+2})
                            = (N_L + M_L * N_R) + (M_L * M_R) * z_{i+2}
    Combined transform is (M_L * M_R, N_L + M_L * N_R)
    
    Args:
        state_L: Tuple (a_L, b_L) = (M_L, N_L) - transform applied second (e.g., T_i).
                 a_L [B, 1, H, K], b_L [B, C, H, K].
        state_R: Tuple (a_R, b_R) = (M_R, N_R) - transform applied first (e.g., T_{i+1}).
                 a_R [B, 1, H, K], b_R [B, C, H, K].

    Returns:
        Tuple (new_a, new_b) representing the combined transformation T_L * T_R.
    """
    return _combine(state_L=state_L, state_R=state_R, backward=True)

def _parallel_scan(initial_state: Tuple[torch.Tensor, torch.Tensor], 
                   combine_op,
                   suffix: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a parallel prefix scan using the Hillis-Steele algorithm.
    Operates on sequences of affine transformations (a, b).

    Args:
        initial_state: Tuple (A_initial, B_initial) representing the sequence of
                       transformations M_1, M_2, ..., M_{W_scan}. 
                       A_initial is [B, 1, H, W_scan], 
                       B_initial is [B, C, H, W_scan].
        combine_op: The associative combination function (e.g., _combine_forward).
        suffix: bool Whether to use the suffix scan instead.

    Returns:
        Tuple (A_final, B_final) containing the accumulated transformations at each position.
        Shape is the same as input. A_final[..., k] contains the composition 
        M_1 M_2 ... M_{k+1} (using 0-based index k).
    """
    # We modify the state in-place. The input tensors are created specifically for this.
    A_current, B_current = initial_state
    
    # Get dimensions (use B_current as it has the channel dim C)
    # A_current is [B, 1, H, W_scan]
    # B_current is [B, C, H, W_scan]
    B, C, H, W_scan = B_current.shape 

    # Handle sequence length 0 or 1
    if W_scan <= 0:
        return A_current, B_current # Return the (empty) input directly
    if W_scan == 1:
        return A_current, B_current # Scan of length 1 is just the element itself

    # Calculate number of steps needed for Hillis-Steele
    num_steps = math.ceil(math.log2(W_scan))

    # Hillis-Steele Scan for Prefix Computation
    for k in range(num_steps):
        dist = 2**k
        
        # Elements at index j >= dist combine info from index j - dist
        
        # Prepare the left state (elements from j - dist)
        # Indices: [0, 1, ..., W_scan - dist - 1]
        state_L_a = A_current[..., :-dist]
        state_L_b = B_current[..., :-dist]
        
        # Prepare the right state (elements from j)
        # Indices: [dist, dist+1, ..., W_scan - 1]
        state_R_a = A_current[..., dist:]
        state_R_b = B_current[..., dist:]

        # Combine states using the provided associative operator
        # For forward scan: compose(state_L, state_R) -> M_R * M_L
        combined_state = combine_op((state_L_a, state_L_b), (state_R_a, state_R_b))

        # Update the right part (indices [dist, W_scan - 1]) with the combined state
        if suffix:
            A_current[..., :-dist] = combined_state[0]
            B_current[..., :-dist] = combined_state[1]
        else:
            A_current[..., dist:] = combined_state[0]
            B_current[..., dist:] = combined_state[1]

    return A_current, B_current


def _parallel_suffix_scan(initial_state: Tuple[torch.Tensor, torch.Tensor], 
                          combine_op) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a parallel suffix scan using the Hillis-Steele algorithm.
    Combines elements from right to left.

    Args:
        initial_state: Tuple (A_initial, B_initial) representing the sequence
                       T_0, T_1, ..., T_{W_scan-1}.
                       Shapes [B, 1, H, W_scan], [B, C, H, W_scan].
        combine_op: The associative combination function (e.g., _combine_backward).
                    It should implement the composition T_L * T_R.

    Returns:
        Tuple (A_final, B_final) containing the suffix accumulated transformations.
        A_final[..., k] contains the composition T_k T_{k+1} ... T_{W_scan-1}.
    """
    return _parallel_scan(initial_state=initial_state, combine_op=combine_op, suffix=True)


def domain_recursive_filter_horizontal(
    I: torch.Tensor, 
    D: torch.Tensor, 
    sigma: torch.Tensor
) -> torch.Tensor:
    """
    Apply the recursive filter along the horizontal dimension using parallel scan.
    This is a drop-in replacement for the original sequential version.
    
    Args:
        I (torch.Tensor): Input tensor [B, C, H, W].
        D (torch.Tensor): Domain transform derivatives [B, 1, H, W].
        sigma (torch.Tensor): Sigma parameter for the filter (scalar tensor).
        
    Returns:
        torch.Tensor: Filtered tensor [B, C, H, W].
    """
    # ----- Setup -----
    
    # Ensure sigma is a scalar tensor on the correct device
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=I.device, dtype=torch.float32)
    elif sigma.numel() != 1:
         raise ValueError("sigma must be a scalar value or a tensor with one element.")
    else:
        sigma = sigma.to(I.device).squeeze() # Ensure scalar

    # Base feedback coefficient 'a'
    a = torch.exp(-torch.sqrt(torch.tensor(2.0, device=I.device)) / sigma) 
    
    # Spatially varying feedback coefficients 'v_i'
    V = a ** D  # Shape [B, 1, H, W]
    
    B, C, H, W = I.shape
    
    # Handle width = 1 edge case: filter has no effect
    if W <= 1:
        return I.clone() 

    # ----- Forward Pass (Left -> Right) -----
    # Recurrence: y_i = (1 - v_i) * x_i + v_i * y_{i-1} 
    #             y_i = B_i + A_i * y_{i-1}
    # where A_i = v_i, B_i = (1 - v_i) * x_i
    # We need prefix products P_k = M_1 M_2 ... M_{k+1} using _combine_forward op.
    # Then y_{k+1} = P_k.b + P_k.a * y_0

    x = I
    v = V # Shape [B, 1, H, W]

    # Define the sequence of forward transformations M_1, ..., M_{W-1}
    A_fwd = v[..., 1:]             # Shape [B, 1, H, W-1] (Coeffs v_1 to v_{W-1})
    B_fwd = (1 - A_fwd) * x[..., 1:] # Shape [B, C, H, W-1] (Terms (1-v_i)x_i for i=1 to W-1)

    # Perform parallel prefix scan on the forward transformations
    initial_state_fwd = (A_fwd.clone(), B_fwd.clone())
    # S_A_fwd[k], S_B_fwd[k] represents combined transform M_1...M_{k+1}
    S_A_fwd, S_B_fwd = _parallel_scan(initial_state_fwd, _combine_forward) 

    # Compute the forward result y_1, ..., y_{W-1}
    y_0 = x[..., 0:1] # Shape [B, C, H, 1]
    
    # Apply the accumulated transforms P_k to y_0
    # y_{k+1} = S_A_fwd[k] * y_0 + S_B_fwd[k] (Note: P_k.a -> S_A_fwd[k], P_k.b -> S_B_fwd[k])
    F_L_scan_part = S_A_fwd * y_0 + S_B_fwd # Shape [B, C, H, W-1] (Contains y_1 to y_{W-1})

    # Assemble the full forward pass result [y_0, y_1, ..., y_{W-1}]
    F_L = torch.cat((y_0, F_L_scan_part), dim=3) # Shape [B, C, H, W]


    # ----- Backward Pass (Right -> Left) -----
    # Recurrence: z_i = (1 - v_{i+1}) * y_i + v_{i+1} * z_{i+1}
    #             z_i = N_i + M_i * z_{i+1}
    # where M_i = v_{i+1}, N_i = (1 - v_{i+1}) * y_i
    # Let T_i = (M_i, N_i) map z_{i+1} -> z_i. Combine op T_L*T_R is _combine_backward.
    # We need suffix products Suf_k = T_k T_{k+1} ... T_{W-2}.
    # Then z_k = Suf_k.b + Suf_k.a * z_{W-1}.

    y = F_L # Input to backward pass is the result of the forward pass

    # Define the sequence of backward transformations T_0, ..., T_{W-2}
    M_bwd = v[..., 1:]             # M_i = v_{i+1} for i=0..W-2. Shape [B, 1, H, W-1]
    N_bwd = (1 - M_bwd) * y[..., :-1] # N_i = (1-v_{i+1})y_i for i=0..W-2. Shape [B, C, H, W-1]
    
    # Perform parallel suffix scan on sequence T=(M,N) using _combine_backward
    initial_state_bwd = (M_bwd.clone(), N_bwd.clone())
    # S_M[k], S_N[k] = Suf_k = T_k ... T_{W-2}
    S_M_bwd, S_N_bwd = _parallel_suffix_scan(initial_state_bwd, _combine_backward) 

    # Compute the backward result z_0, ..., z_{W-2}
    z_last = y[..., -1:] # Shape [B, C, H, 1] (This is z_{W-1})

    # Apply the suffix transforms Suf_k to z_{W-1}
    # z_k = S_N_bwd[k] + S_M_bwd[k] * z_last
    F_R_scan_part = S_M_bwd * z_last + S_N_bwd # Shape [B, C, H, W-1] (Contains z_0 to z_{W-2})

    # Assemble the full backward pass result [z_0, ..., z_{W-2}, z_{W-1}]
    F_R = torch.cat((F_R_scan_part, z_last), dim=3) # Shape [B, C, H, W]

    return F_R

if __name__ == "__main__":
    import cProfile
    import pstats
    import io
    import time
    import torch.profiler as profiler

    # --- Configuration ---
    # --- Choose device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if str(device) == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # --- Input tensor parameters ---
    batch_size = 1        # Test with batching
    channels = 3          # Standard RGB
    height = 64          # Moderate height
    width = 64           # Moderate width

    # --- Filter parameters ---
    sigma_s = 60.0
    sigma_r = 0.4         
    num_iterations = 3    # Default value

    # --- Data Type ---
    dtype = torch.float32 # Common dtype

    # --- Profiling options ---
    profile_joint = True # Set to False to profile without joint filtering
    joint_channels = 1     # E.g., use a grayscale image for joint filtering
    
    # --- Create Dummy Input Data ---
    print(f"\n--- Setting up Input Data ---")
    print(f"Image shape: ({batch_size}, {channels}, {height}, {width})")
    img = torch.rand(batch_size, channels, height, width, 
                     device=device, dtype=dtype)
    
    joint_image = None
    if profile_joint:
        print(f"Joint Image shape: ({batch_size}, {joint_channels}, {height}, {width})")
        joint_image = torch.rand(batch_size, joint_channels, height, width, 
                                 device=device, dtype=dtype) * 0.5 + 0.25 # Different data range
    else:
         print("Profiling without joint image.")

    print(f"Filter params: sigma_s={sigma_s}, sigma_r={sigma_r}, num_iterations={num_iterations}")
    print("-" * 30)
    
    # --- Warm-up Run (especially important for GPU) ---
    print("Performing warm-up run...")
    _ = rf_filter(img.clone(), sigma_s, sigma_r, num_iterations, 
                  joint_image.clone() if joint_image is not None else None)
    if str(device) == "cuda":
        torch.cuda.synchronize() # Ensure GPU work is finished
    print("Warm-up complete.")

    # --- Profiling with cProfile (High-level Python function calls) ---
    print("\n--- Profiling with cProfile ---")
    pr = cProfile.Profile()
    pr.enable()
    
    start_time_cprofile = time.time()
    filtered_img_cprofile = rf_filter(img.clone(), sigma_s, sigma_r, num_iterations, 
                                      joint_image.clone() if joint_image is not None else None)
    if str(device) == "cuda":
        torch.cuda.synchronize()
    end_time_cprofile = time.time()
    
    pr.disable()
    
    print(f"cProfile run wall time: {end_time_cprofile - start_time_cprofile:.4f} seconds")
    
    s = io.StringIO()
    # Sort stats by cumulative time ('cumtime') and print top 30 functions
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative') 
    ps.print_stats(30) 
    print(s.getvalue())
    print("-" * 30)

    # --- Profiling with torch.profiler (Detailed PyTorch Ops, CPU/GPU, Memory) ---
    print("\n--- Profiling with torch.profiler ---")
    # Note: 'with_stack=True' can add overhead but gives source locations.
    #       'profile_memory=True' tracks memory usage.
    #       'record_shapes=True' records tensor shapes.
    activities = [profiler.ProfilerActivity.CPU]
    if str(device) == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)
        
    with profiler.profile(activities=activities, 
                          record_shapes=True, 
                          profile_memory=True, 
                          with_stack=True) as prof:
        with profiler.record_function("rf_filter_total"): # Add a top-level marker
            start_time_torchprof = time.time()
            filtered_img_torchprof = rf_filter(img.clone(), sigma_s, sigma_r, num_iterations, 
                                            joint_image.clone() if joint_image is not None else None)
            if str(device) == "cuda":
                torch.cuda.synchronize()
            end_time_torchprof = time.time()

    print(f"torch.profiler run wall time: {end_time_torchprof - start_time_torchprof:.4f} seconds")

    # --- Print torch.profiler results ---
    # Print sorted by self CPU time total
    print("\n--- torch.profiler results (sorted by self CPU time) ---")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=20))

    # Print sorted by total CPU time total (includes time in called functions)
    print("\n--- torch.profiler results (sorted by total CPU time) ---")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))

    if str(device) == "cuda":
        # Print sorted by self CUDA time total
        print("\n--- torch.profiler results (sorted by self CUDA time) ---")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        
        # Print sorted by total CUDA time total
        print("\n--- torch.profiler results (sorted by total CUDA time) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print memory usage information
    print("\n--- torch.profiler results (sorted by self CPU memory usage) ---")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))

    if str(device) == "cuda":
        print("\n--- torch.profiler results (sorted by self CUDA memory usage) ---")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    # --- Optional: Export for Chrome Trace Viewer ---
    trace_file = "rf_filter_profile_trace.json"
    print(f"\nExporting torch.profiler trace to '{trace_file}'...")
    try:
        prof.export_chrome_trace(trace_file)
        print(f"Trace file saved. Open in Chrome via chrome://tracing")
    except Exception as e:
        print(f"Could not export trace: {e}")
        
    print("-" * 30)
    print("Profiling complete.")
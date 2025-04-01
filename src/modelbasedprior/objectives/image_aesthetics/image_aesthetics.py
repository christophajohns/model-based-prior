import torch
from typing import List, Optional, Tuple

from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.transforms import unnormalize
from torchvision.transforms.functional import rgb_to_grayscale

from modelbasedprior.objectives.image_similarity import generate_image
from modelbasedprior.objectives.image_aesthetics.domain_transform_filter import rf_filter

# Constants from the paper or reasonable defaults
DEFAULT_PYRAMID_LEVELS = 8
DOMAIN_TRANSFORM_ITERATIONS = 3
DEFAULT_TONE_U_PARAM = 0.05
DEFAULT_TONE_O_PARAM = 0.05
# Small epsilon to avoid division by zero or log(0)
EPS = 1e-6


def _compute_pyramid_and_details(
    img_norm: torch.Tensor, 
    k_levels: int = 8,
    domain_transform_iterations: int = 3,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Computes a Domain Transform pyramid and corresponding detail layers.

    Args:
        img_norm: Normalized input image tensor (B, C, H, W), range [0, 1].
        k_levels: The number of pyramid levels (K in the paper).

    Returns:
        Tuple containing:
        - List[LP]: Domain transform pyramid levels [LP^1, ..., LP^K].
        - List[D]: Detail layers [D^1, ..., D^K]. D^k = |LP^{k-1} - LP^k|
                   (with LP^0 = img_norm). Note K levels produce K details.
    """
    # Ensure the input is (B, H, W, C)
    if len(img_norm.shape) == 4 and img_norm.shape[1] == 3:  # (B, C, H, W)
        img_norm = img_norm.permute(0, 2, 3, 1)
    elif len(img_norm.shape) != 4:
        raise ValueError("Input img_norm must be a 4D tensor (B, C, H, W) or (B, H, W, C)")


    pyramid_lp = []
    detail_d = []
    last_lp = img_norm

    for k in range(1, k_levels + 1):
        sigma = 2 ** (k + 1)

        current_lp = rf_filter(img=last_lp, sigma_s=sigma, sigma_r=1.0, num_iterations=domain_transform_iterations, joint_image=last_lp)

        pyramid_lp.append(current_lp)
        detail = torch.abs(last_lp - current_lp)
        detail_d.append(detail)
        last_lp = current_lp

    # Convert back to (B, C, H, W) for the pyramid and detail layers
    pyramid_lp = [lp.permute(0, 3, 1, 2) for lp in pyramid_lp]
    detail_d = [d.permute(0, 3, 1, 2) for d in detail_d]


    return pyramid_lp, detail_d


def _compute_focus_map(
    detail_d: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Computes the focus map levels and in/out-of-focus masks.

    Simplified: Focus map F^k is just the detail layer D^k.
    In-focus F^if is where F^1 (D^1) is large.

    Args:
        detail_d: List of detail layers [D^1, ..., D^K].

    Returns:
        Tuple containing:
        - List[F]: Focus map levels [F^1, ..., F^K] (same as detail_d here).
        - Fif: Binary mask for in-focus region (B, 1, H, W).
        - Foof: Binary mask for out-of-focus region (B, 1, H, W).
    """
    focus_map_f = detail_d # Simplified: F^k = D^k
    f1 = focus_map_f[0] # Detail layer D1 (B, C, H, W)

    # Determine in-focus based on magnitude of D1 (e.g., mean over channels)
    # Use grayscale version of f1 for thresholding to make it channel-independent?
    # Or simply use max over channel dim? Let's use mean.
    f1_magnitude = f1.mean(dim=1, keepdim=True) # (B, 1, H, W)

    # Define in-focus as pixels with high magnitude in F1/D1
    # Use a quantile, e.g., top 20% brightest pixels in F1 magnitude map
    # Flatten spatial dims for quantile calculation
    b, _, h, w = f1_magnitude.shape
    f1_flat = f1_magnitude.view(b, -1)
    # Ensure quantile value is computed per image in batch
    # Use try-except for quantile in case some images are constant
    try:
        # quantile requires float, ensure f1_flat is float
        thresholds = torch.quantile(f1_flat.float(), 0.8, dim=1, keepdim=True) # (B, 1)
    except RuntimeError as e:
        # Likely "quantile() input tensor is too large" or similar, use mean instead
        print(f"Warning: Quantile failed ({e}), using mean as threshold.")
        thresholds = f1_flat.mean(dim=1, keepdim=True) # (B, 1)

    thresholds = thresholds.view(b, 1, 1, 1) # Reshape for broadcasting (B, 1, 1, 1)
    fif = (f1_magnitude >= thresholds).bool() # (B, 1, H, W)
    foof = ~fif # (B, 1, H, W)

    return focus_map_f, fif, foof

def _compute_sharpness(f1: torch.Tensor) -> torch.Tensor:
    """Eq: Ψ_sh = μ(|F¹|)"""
    # F¹ is D¹ = |I - LP¹| (B, C, H, W)
    # Take mean over spatial dims (H, W) and channels (C)
    return f1.mean(dim=(-1, -2, -3)) # (B,)

def _compute_depth(focus_map_f: List[torch.Tensor]) -> torch.Tensor:
    """Eq: Ψ_de = argmax_k [Σ(F^k > 0)] for k = [2, K]
       Simplified: argmax_k [ mean(F^k) ] for k = [2, K] """
    if len(focus_map_f) < 2:
        return torch.zeros(focus_map_f[0].shape[0], device=focus_map_f[0].device) # Return 0 if no levels k>=2

    means = []
    # Iterate from k=2 up to K (index 1 to K-1 in the list)
    for fk in focus_map_f[1:]:
        # Mean magnitude over C, H, W
        mean_mag = fk.mean(dim=(-1, -2, -3)) # (B,)
        means.append(mean_mag)

    # Stack means for levels k=2..K: shape (K-1, B)
    stacked_means = torch.stack(means, dim=0)
    # Find the index (0 to K-2) of the max mean for each image in batch
    # Add 2 to the index to get the level k (from 2 to K)
    # argmax returns first max index in case of ties
    depth_index = torch.argmax(stacked_means, dim=0) + 2
    return depth_index.float() # Return as float tensor (B,)

def _compute_clarity(
    contrast_c: torch.Tensor,
    fif: torch.Tensor,
    foof: torch.Tensor,
    ) -> torch.Tensor:
    """Eq: Ψ_cl = A^oof * (|μ(C · F^if) - μ(C · F^oof)|)"""
    b, _, h, w = fif.shape
    num_pixels = h * w

    # Calculate area of out-of-focus region A^oof
    # Sum boolean mask spatially, divide by total pixels
    aoof = foof.sum(dim=(-1, -2), keepdim=True) / num_pixels # (B, 1, 1, 1)

    # Compute mean contrast in in-focus (fif) and out-of-focus (foof) regions
    # Mask C and compute mean only over non-zero mask elements
    c_masked_if = contrast_c * fif
    c_masked_oof = contrast_c * foof

    # Sum over spatial dims (H, W) and channels (C)
    sum_c_if = c_masked_if.sum(dim=(-1, -2, -3)) # (B,)
    sum_c_oof = c_masked_oof.sum(dim=(-1, -2, -3)) # (B,)

    # Count non-zero elements in masks (sum over H, W)
    count_if = fif.sum(dim=(-1, -2)).squeeze(-1) # (B, 1) -> (B,)
    count_oof = foof.sum(dim=(-1, -2)).squeeze(-1) # (B,)

    # Mean = Sum / Count, handle potential division by zero if mask is all False
    mean_c_if = torch.nan_to_num(sum_c_if / (count_if + EPS))
    mean_c_oof = torch.nan_to_num(sum_c_oof / (count_oof + EPS))

    clarity = aoof.squeeze() * torch.abs(mean_c_if - mean_c_oof) # (B,)
    return clarity


def _compute_tone(
    luminance_l: torch.Tensor,
    u_param: float = 0.05,
    o_param: float = 0.05
    ) -> torch.Tensor:
    """Eq: Ψ_to = c^u * c^o * |p⁹⁵(L) - p⁵(L)|
       c^u = min(u, p³⁰(L) - p⁵(L)) / u
       c^o = min(o, p⁹⁵(L) - p⁷⁰(L)) / o
    """
    b, _, h, w = luminance_l.shape
    l_flat = luminance_l.view(b, -1) # (B, H*W)

    # Compute required percentiles (quantiles)
    # Need to handle potential errors if image is constant
    try:
        p5 = torch.quantile(l_flat, 0.05, dim=1)
        p30 = torch.quantile(l_flat, 0.30, dim=1)
        p70 = torch.quantile(l_flat, 0.70, dim=1)
        p95 = torch.quantile(l_flat, 0.95, dim=1)
    except RuntimeError as e:
         print(f"Warning: Quantile failed for Tone ({e}), returning 0.")
         return torch.zeros(b, device=luminance_l.device)


    # Compute penalty terms c^u, c^o (ensure non-negative differences)
    diff_30_5 = torch.relu(p30 - p5)
    diff_95_70 = torch.relu(p95 - p70)

    cu = torch.min(torch.tensor(u_param, device=l_flat.device), diff_30_5) / u_param
    co = torch.min(torch.tensor(o_param, device=l_flat.device), diff_95_70) / o_param

    # Compute tone
    tone = cu * co * torch.abs(p95 - p5) # (B,)
    return tone


def _compute_colorfulness(img_norm: torch.Tensor) -> torch.Tensor:
    """
    Computes colorfulness based on Hasler & Süsstrunk (2003).
    Eq: Ψ_co = sqrt(σ_rg² + σ_yb²) + 0.3 * sqrt(μ_rg² + μ_yb²)
    where rg = R - G, yb = 0.5*(R+G) - B
    """
    R = img_norm[:, 0, :, :]
    G = img_norm[:, 1, :, :]
    B = img_norm[:, 2, :, :]

    rg = R - G
    yb = 0.5 * (R + G) - B

    # Calculate mean and std dev for rg and yb across spatial dimensions (H, W)
    # Keep batch dim B
    mu_rg = rg.mean(dim=(-1, -2)) # (B,)
    mu_yb = yb.mean(dim=(-1, -2)) # (B,)
    std_rg = rg.std(dim=(-1, -2)) # (B,)
    std_yb = yb.std(dim=(-1, -2)) # (B,)

    # Combine means and std devs
    mean_c = torch.sqrt(mu_rg**2 + mu_yb**2 + EPS)
    std_c = torch.sqrt(std_rg**2 + std_yb**2 + EPS)

    # Compute colorfulness score
    colorfulness = std_c + 0.3 * mean_c # (B,)
    return colorfulness


class ImageAestheticsLoss(SyntheticTestFunction):
    """
    Automated Aesthetic Analysis Loss Function based on Aydın et al. (2015).

    Computes an aesthetic loss for an image based on modifications
    (brightness, contrast, saturation, hue). A lower loss corresponds to
    a potentially higher aesthetic score. The score is derived from
    metrics defined in the paper: Sharpness, Depth, Clarity, Tone,
    and Colorfulness.

    The score is computed similarly to Eq. 4 in the paper:
    Score = Sharpness * mean(Depth, Clarity, Tone, Colorfulness)
    The returned loss is the negative of this score. Minimizing the loss
    maximizes the underlying aesthetic score.
    """
    dim = 4
    _check_grad_at_opt: bool = False
    _brightness_bounds = (0.8, 1.25)
    _contrast_bounds = (0.8, 1.25)
    _saturation_bounds = (0.8, 1.25)
    _hue_bounds = (-0.25, 0.25)

    def __init__(
        self,
        original_image: torch.Tensor,
        k_levels: int = DEFAULT_PYRAMID_LEVELS,
        tone_u: float = DEFAULT_TONE_U_PARAM,
        tone_o: float = DEFAULT_TONE_O_PARAM,
        domain_transform_iterations: int = DOMAIN_TRANSFORM_ITERATIONS,
        noise_std: Optional[float] = None,
        negate: bool = False, # Default False: Output is loss (lower is better)
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            original_image: The base image (C, H, W) as a uint8 tensor [0, 255].
            k_levels: Number of pyramid levels (K) for metric calculation.
            tone_u: Underexposure parameter for the Tone metric.
            tone_o: Overexposure parameter for the Tone metric.
            noise_std: Standard deviation of Gaussian noise added to the output.
            negate: If True, negate the loss output. Standard BoTorch use cases
                    for minimization typically leave this False.
            bounds: Custom bounds for the input parameters. If None, uses defaults.
        """
        if bounds is None:
            bounds = [
                self._brightness_bounds,
                self._contrast_bounds,
                self._saturation_bounds,
                self._hue_bounds,
            ]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self._original_image = original_image
        self._k_levels = k_levels
        self._tone_u = tone_u
        self._tone_o = tone_o
        self._domain_transform_iterations = domain_transform_iterations
        # Note: _optimizers and _optimal_value are typically omitted for non-analytic functions

    def _compute_all_metrics(self, image_batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute all raw aesthetic metrics for a batch of images."""
        # 1. Normalize image batch to [0, 1]
        img_norm = image_batch / 255.0

        # 2. Pre-computation: Pyramid, Details, Contrast, Focus Map, Masks
        _pyramid_lp, detail_d = _compute_pyramid_and_details(img_norm, self._k_levels, self._domain_transform_iterations)
        # Multi-scale contrast C = sum(|D^k|) over k=1..K
        contrast_c = torch.stack(detail_d, dim=0).sum(dim=0) # (B, C, H, W)
        focus_map_f, fif, foof = _compute_focus_map(detail_d)
        luminance_l = rgb_to_grayscale(img_norm, num_output_channels=1) # (B, 1, H, W)

        # 3. Compute individual metrics
        sharpness = _compute_sharpness(focus_map_f[0])
        depth = _compute_depth(focus_map_f)
        clarity = _compute_clarity(contrast_c, fif, foof)
        tone = _compute_tone(luminance_l, self._tone_u, self._tone_o)
        colorfulness = _compute_colorfulness(img_norm)

        # TODO: Consider normalizing raw metrics before combining, as their
        # scales might be very different. E.g., using sigmoid or min-max scaling
        # based on expected ranges. Without calibration data (Sec 5.3), this
        # is ad-hoc.

        return sharpness, depth, clarity, tone, colorfulness
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the aesthetic loss for a batch of parameter settings X.

        Args:
            X: Tensor of shape (n, q, d) or (b, d) with d=4 parameters
               (brightness, contrast, saturation, hue). Values expected in
               original (unnormalized) space defined by bounds.

        Returns:
            Tensor of shape (b,) if X was (b, d).
            Tensor of shape (n, q) if X was (n, q, d).
            Lower values indicate better aesthetic scores.
            If negate=True was passed to __init__, the loss is negated.
        """
        # Check input dimension and store if it was 2D
        was_2d = X.dim() == 2
        # Ensure X has shape (n, q, d) internally
        if was_2d:
            X = X.unsqueeze(1)  # Convert (b, d) -> (b, 1, d)

        n, q, _d = X.shape

        # generate_image expects (n, q, d) and returns (n, q, C, H, W)
        candidate_images_nq = generate_image(X, self._original_image) # Pass the 3D X

        # Flatten to (n*q, C, H, W) for batch metric computation
        candidate_images_flat = candidate_images_nq.flatten(0, 1)

        # Compute all metrics for the batch
        sh, de, cl, to, co = self._compute_all_metrics(candidate_images_flat)
        # sh, de, cl, to, co are all tensors of shape (b,) where b = n*q

        # Combine metrics using Eq. 4 structure to get aesthetic score
        other_metrics = torch.stack([de, cl, to, co], dim=1) # (n*q, 4)
        mean_others = other_metrics.mean(dim=1) # (n*q,)
        score = sh * mean_others # (n*q,)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert score to loss
        loss = -score # (n*q,)

        # Reshape loss back to (n, q)
        loss_nq = loss.view(n, q)

        # Squeeze the q-dimension if the original input X was 2D
        # Also ensure output tensor properties match input X
        return loss_nq.squeeze(1).to(X) if was_2d else loss_nq.to(X)
    

# Example Usage
if __name__ == "__main__":
    import cProfile
    import pstats
    import io
    # import os
    # from dotenv import load_dotenv
    # from torchvision.io import read_image
    # from torchvision.transforms.functional import resize

    # load_dotenv()

    # Use a small random image for testing
    # test_image_uint8 = (torch.rand(3, 64, 64) * 255).floor().to(torch.uint8)
    test_image_uint8 = (torch.rand(3, 16, 16) * 255).floor().to(torch.uint8)

    # --- Example with real image (requires image file and torchvision.io) ---
    # ava_flowers_dir = os.getenv("AVA_FLOWERS_DIR") # Make sure this env var is set
    # if ava_flowers_dir and os.path.exists(ava_flowers_dir):
    #     try:
    #         img_path = os.path.join(ava_flowers_dir, '43405.jpg') # Example image
    #         test_image_uint8 = read_image(img_path)
    #         # Resize for faster testing
    #         test_image_uint8 = resize(test_image_uint8, [128, 128], antialias=True)
    #         print(f"Loaded and resized image: {img_path}, shape: {test_image_uint8.shape}")
    #     except Exception as e:
    #         print(f"Could not load real image: {e}. Using random image.")
    #         test_image_uint8 = (torch.rand(3, 64, 64) * 255).floor().to(torch.uint8)
    # else:
    #     print("AVA_FLOWERS_DIR not set or found. Using random image.")
    #     test_image_uint8 = (torch.rand(3, 64, 64) * 255).floor().to(torch.uint8)
    # -----------------------------------------------------------------------

    print(f"Using image of shape: {test_image_uint8.shape}")

    # Instantiate the objective function (Loss version)
    image_aesthetics_loss = ImageAestheticsLoss(
        original_image=test_image_uint8,
        negate=False # Standard setup: minimize the loss
    )

    # Generate some random parameter settings (unnormalized)
    X_rand = unnormalize(torch.rand(2, 1, image_aesthetics_loss.dim), bounds=image_aesthetics_loss.bounds)
    print("\nRandom Parameters (X_rand shape", X_rand.shape, "):")
    print(X_rand)
    # Output is now loss (lower is better)
    loss_rand = image_aesthetics_loss(X_rand)
    print("Aesthetic Loss:", loss_rand)

    # Generate parameters assumed to be good (neutral adjustments)
    X_neutral = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]], dtype=X_rand.dtype, device=X_rand.device)
    print("\nNeutral Parameters (X_neutral shape", X_neutral.shape, "):")
    print(X_neutral)
    # Output is loss
    loss_neutral = image_aesthetics_loss(X_neutral)
    print("Aesthetic Loss:", loss_neutral)


    # Test batch evaluation (n=3, q=2)
    print("\nBatch Evaluation (n=3, q=2):")
    qX = unnormalize(torch.rand(3, 2, image_aesthetics_loss.dim), bounds=image_aesthetics_loss.bounds)
    print("Batch Parameters (qX shape", qX.shape, "):")
    # Output is loss
    q_loss = image_aesthetics_loss(qX)
    print("Batch Aesthetic Loss (shape", q_loss.shape, "):")
    print(q_loss)

    # Test evaluation with 2D input (b, d)
    print("\nBatch Evaluation with 2D input (b=5, d=4):")
    X_flat_test = unnormalize(torch.rand(5, image_aesthetics_loss.dim), bounds=image_aesthetics_loss.bounds)
    print("Flat Parameters (X_flat_test shape", X_flat_test.shape, "):")
    # Output is loss
    flat_loss = image_aesthetics_loss(X_flat_test)
    print("Flat Aesthetic Loss (shape", flat_loss.shape, "):")
    print(flat_loss)

    # Test metric computation directly on an image
    print("\nDirect Metric Computation Test:")
    test_batch_img = test_image_uint8.unsqueeze(0).float()
    # Need instance to call _compute_all_metrics
    metrics = image_aesthetics_loss._compute_all_metrics(test_batch_img)
    metric_names = ["Sharpness", "Depth", "Clarity", "Tone", "Colorfulness"]
    print("Raw metrics for original image:")
    for name, val in zip(metric_names, metrics):
        print(f"  {name}: {val.item():.4f}")

    # --- Profiling Section ---
    print("\n--- Profiling Batch Evaluation (n=3, q=2) ---")
    # Prepare the input data *before* starting the profiler
    qX = unnormalize(torch.rand(3, 2, image_aesthetics_loss.dim), bounds=image_aesthetics_loss.bounds)

    # Create a profiler object
    pr = cProfile.Profile()

    # Enable profiling and run the function call
    pr.enable()
    q_loss = image_aesthetics_loss(qX) # The call to profile
    pr.disable()

    print("Profiling finished.")
    print("Batch Aesthetic Loss (shape", q_loss.shape, "):")
    # print(q_loss) # Optionally print the result

    # Create a stream to write the stats to
    s = io.StringIO()
    # Sort stats by cumulative time spent in function and its callees
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

    # Print the top 25 most time-consuming parts
    ps.print_stats(25)
    print("\nProfiler Output (Top 25 by Cumulative Time):")
    print(s.getvalue())
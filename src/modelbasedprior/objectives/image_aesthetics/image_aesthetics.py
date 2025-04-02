import torch
from typing import List, Optional, Tuple

from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from torchvision.transforms.functional import rgb_to_grayscale

from modelbasedprior.objectives.image_similarity import generate_image
from modelbasedprior.objectives.image_aesthetics.domain_transform_filter import rf_filter

# Constants from the paper or reasonable defaults
DEFAULT_PYRAMID_LEVELS = 8
DOMAIN_TRANSFORM_ITERATIONS = 5
DEFAULT_TONE_U_PARAM = 0.05
DEFAULT_TONE_O_PARAM = 0.05
DEFAULT_NUM_INIT_SAMPLES = 32 # Number of Sobol/random samples for range estimation
# Small epsilon to avoid division by zero or log(0)
EPS = 1e-6


def _compute_pyramid_and_details(
    img_norm: torch.Tensor, 
    k_levels: int = 8,
    domain_transform_iterations: int = 5,
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

        current_lp = rf_filter(img=last_lp, sigma_s=sigma, sigma_r=1.0 if k < k_levels else float('inf'), num_iterations=domain_transform_iterations, joint_image=last_lp)

        pyramid_lp.append(current_lp)
        detail = torch.abs(last_lp - current_lp)
        detail_d.append(detail)
        last_lp = current_lp

    # Convert back to (B, C, H, W) for the pyramid and detail layers
    pyramid_lp = [lp.permute(0, 3, 1, 2) for lp in pyramid_lp]
    detail_d = [d.permute(0, 3, 1, 2) for d in detail_d]


    return pyramid_lp, detail_d


def _compute_focus_map(
    detail_d: List[torch.Tensor],
    original_image: torch.Tensor,
    detail_filter_sigma_s: float = 60.0,
    detail_filter_sigma_r: float = 0.4,
    detail_filter_iterations: int = 5,
    epsilon: float = EPS
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    r"""
    Computes focus map levels F^k and in/out-of-focus masks (Fif, Foof).

    This version implements the precise method described in Aydin et al. 2015,
    Sec 5.2 & Eq. (3), intended as a higher-fidelity replacement for simpler
    approximations. It requires filtering detail layers using the original image
    as guidance and performs sequential level assignment based on magnitude dominance.

    Args:
        detail_d: List of K detail layers [D^1, ..., D^K] (B, C, H, W each),
                  output from a pyramid decomposition function (e.g., using rf_filter).
        original_image: The input RGB image (B, C, H, W), normalized [0, 1]. Crucial
                        for guiding the filtering of detail layers (joint_image).
        detail_filter_sigma_s: Spatial sigma for the rf_filter step transforming D^k to \hat{D}^k.
        detail_filter_sigma_r: Range sigma for the rf_filter step transforming D^k to \hat{D}^k.
        detail_filter_iterations: Number of iterations for the rf_filter step.
        epsilon: Small value for comparing magnitudes against zero.

    Returns:
        Tuple containing:
        - focus_map_f: List[F^1, ..., F^K] (B, C, H, W each). Contains filtered detail
                       values (\hat{D}^k) assigned based on dominance.
        - fif_mask: Boolean mask for in-focus region (B, 1, H, W), where |F^1| > epsilon.
        - foof_mask: Boolean mask for out-of-focus region (B, 1, H, W).

    Raises:
        ValueError: If detail_d list is empty.
    """
    if not detail_d:
        raise ValueError("Input detail list 'detail_d' cannot be empty.")

    num_detail_layers = len(detail_d) # K
    B, C, H, W = detail_d[0].shape
    device = detail_d[0].device

    # 1. Filter D^k -> \hat{D}^k using original image as guidance in rf_filter
    detail_d_hat = []
    for d_k in detail_d:
        d_hat_k = rf_filter(
            img=d_k,
            sigma_s=detail_filter_sigma_s,
            sigma_r=detail_filter_sigma_r,
            num_iterations=detail_filter_iterations,
            joint_image=original_image
        )
        detail_d_hat.append(d_hat_k)

    # 2. Compute Focus Map Levels F^k via sequential dominance assignment
    focus_map_f = [torch.zeros_like(d) for d in detail_d_hat]
    # Mask tracking pixels not yet assigned to a focus level
    active_pixel_mask = torch.ones(B, 1, H, W, dtype=torch.bool, device=device)

    # Compare levels k and k+1 (indices 0 to K-2)
    for k_idx in range(num_detail_layers - 1):
        d_hat_k = detail_d_hat[k_idx]
        d_hat_k_plus_1 = detail_d_hat[k_idx + 1]

        # Compare mean absolute magnitude across channels
        mag_k = torch.abs(d_hat_k).mean(dim=1, keepdim=True)
        mag_k_plus_1 = torch.abs(d_hat_k_plus_1).mean(dim=1, keepdim=True)
        k_dominates_mask = (mag_k > mag_k_plus_1 + epsilon) # Add epsilon for robustness

        # Pixels assigned to F^k are those active AND where k dominates k+1
        assign_to_fk_mask = active_pixel_mask & k_dominates_mask

        # Assign values from \hat{D}^k to F^k
        focus_map_f[k_idx] = torch.where(
            assign_to_fk_mask.expand(-1, C, -1, -1), d_hat_k, focus_map_f[k_idx]
        )

        # Deactivate pixels that were just assigned
        active_pixel_mask = active_pixel_mask & (~assign_to_fk_mask)

    # Assign remaining active pixels to the last level F^K
    if num_detail_layers > 0:
        k_idx_last = num_detail_layers - 1
        focus_map_f[k_idx_last] = torch.where(
            active_pixel_mask.expand(-1, C, -1, -1),
            detail_d_hat[k_idx_last],
            focus_map_f[k_idx_last]
        )

    # 3. Compute In-focus mask Fif based on |F^1| > epsilon
    f1 = focus_map_f[0]
    fif_mask = (torch.abs(f1).mean(dim=1, keepdim=True) > epsilon)

    # 4. Compute Out-of-focus mask Foof
    foof_mask = ~fif_mask

    # Ensure boolean return type matches original simplified function's intent
    return focus_map_f, fif_mask, foof_mask

def _compute_sharpness(f1: torch.Tensor) -> torch.Tensor:
    """Eq: Ψ_sh = μ(|F¹|)"""
    # F¹ is D¹ = |I - LP¹| (B, C, H, W)
    # Take mean over spatial dims (H, W) and channels (C)
    return f1.mean(dim=(-1, -2, -3)) # (B,)

def _compute_depth(
    focus_map_f: List[torch.Tensor],
    epsilon: float = EPS
) -> torch.Tensor:
    """
    Computes the Depth aesthetic attribute Ψ_de based on Aydin et al. 2015, Table 1.

    Ψ_de = argmax_k [Area(F^k)] for k = [2, K]
    Calculates the area for each focus map level F^k (k>=2) and returns the
    level k corresponding to the largest area.

    Args:
        focus_map_f: List of K focus map levels [F^1, ..., F^K] computed by
                     the precise `_compute_focus_map` function.
        epsilon: Small value for comparing magnitudes against zero when calculating area.

    Returns:
        depth_level: Tensor (B,) containing the level index k (in range [2, K])
                     that maximizes the non-zero area per image. Returns 0.0 if
                     fewer than 2 focus map levels exist. Float tensor.

    Raises:
        ValueError: If focus_map_f list is empty.
    """
    num_focus_levels = len(focus_map_f) # K

    # Need at least F^1 and F^2 to compute depth for k>=2
    if num_focus_levels < 2:
        if not focus_map_f:
             raise ValueError("Focus map list 'focus_map_f' is empty.")
        # Return default depth 0 if only F^1 exists
        return torch.zeros(focus_map_f[0].shape[0], dtype=torch.float, device=focus_map_f[0].device)

    # Calculate area (pixel count > epsilon) for levels k=2 to K
    areas_k2_to_K = []
    # Iterate over focus maps F^2 to F^K (indices 1 to K-1)
    for k_idx in range(1, num_focus_levels):
        fk = focus_map_f[k_idx] # This is F^{k_idx+1}

        # Area is the count of pixels where mean abs magnitude > epsilon
        non_zero_mask = (torch.abs(fk).mean(dim=1, keepdim=True) > epsilon)
        # Sum over H, W, and channel dims to get count per batch element
        area = torch.sum(non_zero_mask, dim=(-1, -2, -3)) # Shape (B,)
        areas_k2_to_K.append(area)

    # Stack areas for levels k=2..K -> shape (K-1, B)
    stacked_areas = torch.stack(areas_k2_to_K, dim=0)

    # Find index (0 to K-2) corresponding to max area within levels k=2..K
    max_area_relative_idx = torch.argmax(stacked_areas, dim=0) # Shape (B,)

    # Convert relative index back to actual level k (index 0 -> k=2, etc.)
    # Resulting levels are in the range [2, K]
    depth_level = max_area_relative_idx + 2

    return depth_level.float()

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
        num_init_samples: int = DEFAULT_NUM_INIT_SAMPLES,
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
            domain_transform_iterations: Number of iterations for the domain transform filter.
            num_init_samples: Number of quasi-random (Sobol) or random samples
                used to estimate metric output ranges during initialization.
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
        self._num_init_samples = num_init_samples
        # Note: _optimizers and _optimal_value are typically omitted for non-analytic functions
        self._metric_stats = self._estimate_metric_stats()

    @torch.no_grad() # Disable gradients for estimation
    def _estimate_metric_stats(self) -> dict:
        """
        Estimates statistics (median, IQR) for each metric by sampling
        the parameter space. These stats are used for sigmoid normalization
        for all metrics except 'depth'.
        """
        device = self._original_image.device
        dtype = torch.float32 # Use float for parameters

        # 1. Define parameter points for estimation (same as before)
        bounds_t = self.bounds.t().to(device=device, dtype=dtype)
        b_min, b_max = bounds_t[0]
        c_min, c_max = bounds_t[1]
        s_min, s_max = bounds_t[2]
        h_min, h_max = bounds_t[3]

        X_extreme_list = [
            [1.0, 1.0, 1.0, 0.0], [b_min, 1.0, 1.0, 0.0], [b_max, 1.0, 1.0, 0.0],
            [1.0, c_min, 1.0, 0.0], [1.0, c_max, 1.0, 0.0], [1.0, 1.0, s_min, 0.0],
            [1.0, 1.0, s_max, 0.0], [1.0, 1.0, 1.0, h_min], [1.0, 1.0, 1.0, h_max],
            [b_min, c_min, s_min, h_min], [b_max, c_max, s_max, h_max]
        ]
        X_extreme = torch.tensor(X_extreme_list, device=device, dtype=dtype)

        # 2. Generate Sobol or random samples (same as before)
        if self._num_init_samples > 0:
            X_norm_samples = draw_sobol_samples(
                bounds=torch.tensor([[0.0] * self.dim, [1.0] * self.dim], device=device, dtype=dtype),
                n=self._num_init_samples, q=1,
                seed=torch.randint(10000, (1,)).item()
            ).squeeze(1)
            X_samples = unnormalize(X_norm_samples, bounds=self.bounds)
            X_est = torch.cat([X_extreme, X_samples], dim=0)
        else:
            X_est = X_extreme

        X_est = X_est.unsqueeze(1) # Shape (num_points, 1, d)

        # 3. Generate images (same as before)
        est_images_nq = generate_image(X_est, self._original_image.to(device))
        est_images_flat = est_images_nq.flatten(0, 1) # (num_points, C, H, W)

        # 4. Compute metrics (same as before)
        sh_vals, de_vals, cl_vals, to_vals, co_vals = self._compute_all_metrics(est_images_flat)

        # 5. Helper to get median/IQR stats, handling NaNs/Infs and edge cases
        def get_stats(vals: torch.Tensor) -> dict:
            if vals.numel() == 0:
                print(f"Warning: Empty tensor passed to get_stats. Using default stats (median=0.5, iqr=1.0).")
                return {'median': 0.5, 'iqr': 1.0}

            valid_vals = vals[~torch.isnan(vals) & ~torch.isinf(vals)].flatten()

            if valid_vals.numel() < 2: # Need at least 2 points for IQR
                print(f"Warning: < 2 valid values for metric during range estimation. Using default stats (median={valid_vals.mean().item() if valid_vals.numel() > 0 else 0.5}, iqr=1.0).")
                median = valid_vals.mean().item() if valid_vals.numel() > 0 else 0.5
                iqr = 1.0
            else:
                try:
                    # Calculate quartiles
                    q25, median_val, q75 = torch.quantile(
                        valid_vals,
                        torch.tensor([0.25, 0.5, 0.75], device=valid_vals.device, dtype=valid_vals.dtype)
                    )
                    median = median_val.item()
                    iqr = (q75 - q25).item()
                    # Ensure IQR is not too small to avoid division by zero in sigmoid scale
                    if iqr < EPS:
                        # print(f"Warning: Estimated IQR is near zero ({iqr}). Setting to {EPS}.")
                        iqr = EPS
                except RuntimeError as e:
                     print(f"Warning: Quantile calculation failed during stats estimation ({e}). Using default stats (median=0.5, iqr=1.0).")
                     median = 0.5
                     iqr = 1.0

            return {'median': median, 'iqr': iqr}

        stats = {}
        metrics_data = {
            # Compute stats for all, even if depth's aren't used for sigmoid
            "sharpness": sh_vals, "depth": de_vals, "clarity": cl_vals,
            "tone": to_vals, "colorfulness": co_vals
        }
        for key, vals in metrics_data.items():
            stats[key] = get_stats(vals)
            # print(f"Estimated stats for {key}: {stats[key]}") # Debug print

        return stats

    def _compute_all_metrics(self, image_batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute all raw aesthetic metrics for a batch of images."""
        # 1. Normalize image batch to [0, 1]
        img_norm = image_batch / 255.0

        # 2. Pre-computation: Pyramid, Details, Contrast, Focus Map, Masks
        _pyramid_lp, detail_d = _compute_pyramid_and_details(img_norm, self._k_levels, self._domain_transform_iterations)
        # Multi-scale contrast C = sum(|D^k|) over k=1..K
        contrast_c = torch.stack(detail_d, dim=0).sum(dim=0) # (B, C, H, W)
        focus_map_f, fif, foof = _compute_focus_map(detail_d, img_norm, detail_filter_iterations=self._domain_transform_iterations)
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

        # return torch.ones_like(sharpness), torch.zeros_like(depth), torch.zeros_like(clarity), torch.zeros_like(tone), torch.zeros_like(colorfulness)
        return sharpness, depth, clarity, tone, colorfulness
    
    def _normalize_metrics(
        self,
        sharpness_raw: torch.Tensor,
        depth_raw: torch.Tensor,
        clarity_raw: torch.Tensor,
        tone_raw: torch.Tensor,
        colorfulness_raw: torch.Tensor,
    ):
        """
        Normalizes the image aesthetics metrics.
        - 'depth' is normalized using theoretical bounds [0, K].
        - Other metrics are normalized using a sigmoid function based on
          median and IQR estimated during initialization.

        Args:
            sharpness_raw: Raw sharpness values.
            depth_raw: Raw depth values.
            clarity_raw: Raw clarity values.
            tone_raw: Raw tone values.
            colorfulness_raw: Raw colorfulness values.

        Returns:
            Tuple of normalized tensors (sh_norm, de_norm, cl_norm, to_norm, co_norm),
            each mapped approximately to the [0, 1] range.
        """
        # Sigmoid scale factor (adjusts steepness, larger means steeper)
        # A value of ~6 means the output goes from ~0.1 to ~0.9 over the IQR.
        sigmoid_scale_factor = 6.0

        normalized_metrics = {}
        raw_metrics = {
            "sharpness": sharpness_raw, "depth": depth_raw, "clarity": clarity_raw,
            "tone": tone_raw, "colorfulness": colorfulness_raw
        }

        for key, val_raw in raw_metrics.items():
            if key == "depth":
                # Use theoretical bounds [0, K] for depth normalization
                # Note: _compute_depth returns 0 if K < 2, and [2, K] otherwise.
                # Using [0, K] covers all cases.
                min_val = 0.0
                max_val = float(self._k_levels) # K
                range_val = max_val - min_val

                # Handle NaN before normalization (replace with min_val)
                val_nan_handled = torch.nan_to_num(val_raw, nan=min_val)

                if range_val < EPS:
                    # If K=0 (or 1, making range=0 or 1), normalize to 0.5 or 0
                    # Let's normalize to 0 if range is effectively zero
                    normalized_val = torch.zeros_like(val_nan_handled)
                else:
                    # Apply min-max scaling
                    normalized_val = (val_nan_handled - min_val) / range_val

                # Clamp result explicitly to [0, 1] as a safety measure
                normalized_metrics[key] = torch.clamp(normalized_val, 0.0, 1.0)

            else:
                # Use sigmoid normalization for other metrics
                stats = self._metric_stats[key]
                median = stats['median']
                iqr = stats['iqr'] # Already ensured >= EPS during estimation

                # Handle NaN before normalization (replace with median)
                val_nan_handled = torch.nan_to_num(val_raw, nan=median)

                # Calculate sigmoid scale `k` based on IQR
                # k = sigmoid_scale_factor / iqr maps ~IQR range to steep part of sigmoid
                k = sigmoid_scale_factor / iqr

                # Apply sigmoid: sigmoid(k * (value - center))
                normalized_val = torch.sigmoid(k * (val_nan_handled - median))
                normalized_metrics[key] = normalized_val # Sigmoid naturally outputs [0, 1]

        return (
            normalized_metrics["sharpness"],
            normalized_metrics["depth"],
            normalized_metrics["clarity"],
            normalized_metrics["tone"],
            normalized_metrics["colorfulness"],
        )
    
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
        sh_raw, de_raw, cl_raw, to_raw, co_raw = self._compute_all_metrics(candidate_images_flat)
        # sh, de, cl, to, co are all tensors of shape (b,) where b = n*q

        # Normalize using estimated output ranges
        sh_norm, de_norm, cl_norm, to_norm, co_norm = self._normalize_metrics(
            sh_raw, de_raw, cl_raw, to_raw, co_raw
        )

        # Combine *normalized* metrics using Eq. 4 structure to get aesthetic score
        other_metrics_norm = torch.stack([de_norm, cl_norm, to_norm, co_norm], dim=1) # (n*q, 4)
        mean_others_norm = other_metrics_norm.mean(dim=1) # (n*q,)
        # Use normalized sharpness
        score = sh_norm * mean_others_norm # (n*q,)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0) # Handled by normalization/clamping now

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
    metrics_raw = image_aesthetics_loss._compute_all_metrics(test_batch_img)
    metrics = image_aesthetics_loss._normalize_metrics(*metrics_raw)
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
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Optional, Tuple
from botorch.test_functions.synthetic import SyntheticTestFunction
from pyiqa.archs.ssim_arch import ssim
import cv2
from botorch.utils.transforms import normalize

class ScatterPlotQualityLoss(SyntheticTestFunction):
    r"""Scatter plot quality test function.

    3-dimensional function (usually evaluated on `[3, 53]` for the first
    dimension (marker size), `[5, 255]` for the second dimension (marker opacity), and
    `[0.5, 1.5]` for the third dimension (aspect ratio):

        f(x) = w_\alpha E_\alpha(x) + w_r E_r(x) + w_\mu I_\mu(x) + w_\sigma I_\sigma(x) + w_{\bar{\mu}} I_{\bar{\mu}}(x) + w_{\bar{\sigma}} I_{\bar{\sigma}}(x) + w_l I_l(x) + w_p I_p(x) + w_c S_c(x) + w_o S_o(x)

    where each `w` is a weight in the range `[-1, 1]`.

    The default weight settings represent the settings for the correlation task.

    For full details, see Micallef et al. (2017; DOI: 10.1109/TVCG.2017.2674978).

    It is unclear whether f has a global optimum.
    This depends on the given dataset and the weights.
    """

    dim = 3
    _check_grad_at_opt: bool = False
    _optimal_value: float = 0.0  # Every objective component is in the range [0, 1] with 0 being the best -> the optimal value is thus 0; however, more realistically around 11 for w_overplotting = 12
    _marker_size_bounds = (3., 53.)
    _marker_opacity_bounds = (5., 255.)
    _aspect_ratio_bounds = (0.5, 1.5)

    def __init__(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        outliers: Optional[torch.Tensor] = None,
        desired_opacity: float = 0.5,
        desired_contrast: float = 0.1,
        weight_angle_difference: float = 0.5,
        weight_axis_ratio_difference: float = 1.0,
        weight_opacity: float = -0.5,
        weight_contrast: float = 0.0,
        weight_opacity_difference: float = 0.5,
        weight_contrast_difference: float = 0.5,
        weight_marker_overlap: float = -0.5,
        weight_overplotting: float = 12.0,
        weight_class_perception: float = 0.0,
        weight_outlier_perception: float = 0.0,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        use_approximate_model: bool = False,
    ) -> None:
        r"""
        Args:
            x_data: The data to plot on the x-axis of the scatter plot.
            y_data: The data to plot on the y-axis of the scatter plot.
            class_labels: Optional tensor of class labels for class perception
            outliers: Optional tensor indicating outlier points
            desired_opacity: Target opacity for the plot (default: 0.5)
            desired_contrast: Target contrast for the plot (default: 0.1)
            weight_angle_difference: The weight for the angle difference component.
            weight_axis_ratio_difference: The weight for the axis ratio difference component.
            weight_opacity: The weight for the opacity component.
            weight_contrast: The weight for the contrast component.
            weight_opacity_difference: The weight for the opacity difference component.
            weight_contrast_difference: The weight for the contrast difference component.
            weight_marker_overlap: The weight for the marker overlap component.
            weight_overplotting: The weight for the overplotting component.
            weight_class_perception: The weight for the class perception component.
            weight_outlier_perception: The weight for the outlier perception component.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if bounds is None:
            bounds = [self._marker_size_bounds, self._marker_opacity_bounds, self._aspect_ratio_bounds]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.x_data = x_data
        self.y_data = y_data
        self.class_labels = class_labels
        self.outliers = outliers
        self.desired_opacity = desired_opacity
        self.desired_contrast = desired_contrast
        self._weight_angle_difference = weight_angle_difference
        self._weight_axis_ratio_difference = weight_axis_ratio_difference
        self._weight_opacity = weight_opacity
        self._weight_contrast = weight_contrast
        self._weight_opacity_difference = weight_opacity_difference
        self._weight_contrast_difference = weight_contrast_difference
        self._weight_marker_overlap = weight_marker_overlap
        self._weight_overplotting = weight_overplotting
        self._weight_class_perception = weight_class_perception
        self._weight_outlier_perception = weight_outlier_perception
        if use_approximate_model:
            self._approximate_model = ScatterPlotQualityLossRegressor()
            self._approximate_model.load_state_dict(torch.load('best_model.pth'))
            self._approximate_model.eval()
        self._use_approximate_model = use_approximate_model

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        if self._use_approximate_model:
            with torch.no_grad():
                X_normalized = normalize(X, self.bounds)
                X_normalized_double = X_normalized.double()  # Ensure dtype=torch.float64
                if X_normalized_double.dim() == 1:  # [3]
                    X_normalized_double = X_normalized_double.unsqueeze(0)
                elif X_normalized_double.dim() == 3:  # [N, 1, 3]
                    X_normalized_double = X_normalized_double.squeeze(1)
                return self._approximate_model(X_normalized_double)
        batch_shape = X.shape[:-1]
        results = torch.zeros(batch_shape, device=X.device)

        for idx in range(X.shape[0]):
            result = self.evaluate_single(X[idx].squeeze()) # X may have shape (N, 4) or (N, 1, 4)
            results[idx] = result
    
        return results.unsqueeze(-1).double()
    
    def evaluate_single(self, x: torch.Tensor) -> torch.Tensor:
        marker_size, marker_opacity, aspect_ratio = x.tolist()

        # Generate base plot image
        image = self._generate_plot_image(self.x_data, self.y_data, marker_size, marker_opacity, aspect_ratio)
        
        # Compute all components
        angle_diff, axis_ratio_diff = self._compute_angle_difference_and_axis_ratio(self.x_data, self.y_data, marker_size, marker_opacity, aspect_ratio) if (self._weight_angle_difference != 0) or (self._weight_axis_ratio_difference != 0) else (torch.tensor(0.0), torch.tensor(0.0))
        
        opacity, contrast = self._compute_opacity_contrast(image) if (self._weight_opacity != 0) or (self._weight_contrast != 0) else (torch.tensor(0.0), torch.tensor(0.0))
        opacity_diff = torch.abs(opacity - self.desired_opacity)
        contrast_diff = torch.abs(contrast - self.desired_contrast)
        
        marker_pixels = ScatterPlotQualityLoss._compute_marker_pixels(marker_size, marker_opacity, aspect_ratio)
        marker_overlap = self._compute_marker_overlap(image, marker_pixels, self.x_data, marker_size, marker_opacity, aspect_ratio) if self._weight_marker_overlap != 0 else torch.tensor(0)
        overplotting = self._compute_overplotting(image, marker_pixels, self.x_data, marker_size, marker_opacity, aspect_ratio) if self._weight_overplotting != 0 else torch.tensor(0)
        
        class_perception = self._compute_class_perception(image, marker_size, marker_opacity, aspect_ratio) if self._weight_class_perception != 0 else torch.tensor(0)
        outlier_perception = self._compute_outlier_perception(image, marker_size, marker_opacity, aspect_ratio) if self._weight_outlier_perception != 0 else torch.tensor(0)

        linear_correlation = (
            self._weight_angle_difference * angle_diff +
            self._weight_axis_ratio_difference * axis_ratio_diff
        )
        image_quality = (
            self._weight_opacity * opacity +
            self._weight_contrast * contrast +
            self._weight_opacity_difference * opacity_diff +
            self._weight_contrast_difference * contrast_diff +
            self._weight_marker_overlap * marker_overlap +
            self._weight_overplotting * overplotting
        )
        structural_similarity = (
            self._weight_class_perception * class_perception +
            self._weight_outlier_perception * outlier_perception
        )
        return linear_correlation + image_quality + structural_similarity

    @staticmethod
    def _generate_plot_image(x: torch.Tensor, y: torch.Tensor, marker_size: float, marker_opacity: float,
                            aspect_ratio: float, color: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate a scatter plot and return it as a grayscale image tensor."""
        fig, ax = plt.subplots(figsize=(2 * aspect_ratio, 2))
        canvas = FigureCanvasAgg(fig)

        ax.axis("off")
        ax.scatter(x.numpy(), y.numpy(), s=marker_size, 
                  alpha=marker_opacity / 255.0,
                  c=color.numpy() if color is not None else "black")
        canvas.draw()
        
        width, height = fig.get_size_inches() * fig.get_dpi()
        rgba_image_flat = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        image_rgba = rgba_image_flat.reshape(int(height), int(width), 4)
        plt.close(fig)

        # Get the Alpha channel and construct an RGB image by repeating it for all RGB channels
        image_gray = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2GRAY)  # shape: (H, W)
        # Convert to RGB by repeating the grayscale value for all RGB channels
        image_rgb = image_gray.reshape(image_gray.shape[0], image_gray.shape[1], 1).repeat(3, axis=2)
        return torch.tensor(image_rgb).permute(2, 0, 1) / 255.0  # Permute from (H, W, C) to (C, H, W) and normalize to [0, 1]
    
    @staticmethod
    def _calculate_ellipse_params(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate ellipse parameters: major axis, minor axis, and angle."""
        cov_matrix = torch.cov(torch.stack([x, y]))
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
        major_axis = 2 * torch.sqrt(eigvals[1])
        minor_axis = 2 * torch.sqrt(eigvals[0])
        angle = torch.atan2(eigvecs[1, 0], eigvecs[0, 0])
        return major_axis, minor_axis, angle
    
    @staticmethod
    def _compute_perceived_ellipse_params(edge_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Calculate perceived ellipse parameters: major axis, minor axis, and angle."""
        # Convert Torch tensor to numpy for OpenCV edge detection (if needed, depending on func._generate_plot_image output)
        edge_image_np = edge_image.squeeze(0).cpu().numpy().astype('uint8')
        
        # Find contours in the edge image
        contours, _ = cv2.findContours(edge_image_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Fit ellipse to the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(largest_contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            angle_in_degree = ellipse[2]
            angle = torch.deg2rad(angle_in_degree)
            return torch.tensor(major_axis), torch.tensor(minor_axis), angle
        else:
            return None
        
    @staticmethod
    def _get_edge_image(gray_image: torch.Tensor) -> torch.Tensor:
        """Get the edge image from the grayscale plot image with Gaussian smoothing (σ = 4px)."""
        # Convert the grayscale image to NumPy array for OpenCV
        gray_image_np = gray_image.squeeze(0).cpu().numpy().astype('uint8')
        
        # Apply Gaussian blur with sigma=4 and appropriate kernel size
        blurred_image = cv2.GaussianBlur(gray_image_np, (25, 25), sigmaX=4)
        
        # Apply Canny edge detection after the Gaussian blur
        edges = torch.tensor(cv2.Canny(blurred_image, 
                                    threshold1=100, threshold2=200, 
                                    apertureSize=3, L2gradient=True)).unsqueeze(0).to(gray_image.device)
        
        return edges
    
    @staticmethod
    def _compute_angle_difference_and_axis_ratio(x: torch.Tensor, y: torch.Tensor, marker_size: float, marker_opacity: float, aspect_ratio: float) -> torch.Tensor:
        """Compute the angle difference and axis ratio between computed and perceived ellipses."""
        x_scaled = (x - x.min()) / (x.max() - x.min())
        y_scaled = (y - y.min()) / (y.max() - y.min())

        major_c, minor_c, angle_c = ScatterPlotQualityLoss._calculate_ellipse_params(x_scaled, y_scaled)

        image = ScatterPlotQualityLoss._generate_plot_image(x_scaled, y_scaled, marker_size, marker_opacity, aspect_ratio)
        ellipse_params = ScatterPlotQualityLoss._compute_perceived_ellipse_params(ScatterPlotQualityLoss._get_edge_image(image))
        if ellipse_params is None:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        major_p, minor_p, angle_p = ellipse_params
        return torch.abs(angle_c - angle_p) / torch.pi, torch.abs((minor_c / major_c) - (minor_p / major_p))

    @staticmethod
    def _compute_angle_difference(x_scaled: torch.Tensor, y_scaled: torch.Tensor, marker_size: float, marker_opacity: float, aspect_ratio: float) -> torch.Tensor:
        """Compute the angle difference between computed and perceived ellipses."""
        _, _, angle_c = ScatterPlotQualityLoss._calculate_ellipse_params(x_scaled, y_scaled)

        image = ScatterPlotQualityLoss._generate_plot_image(x_scaled, y_scaled, marker_size, marker_opacity, aspect_ratio)
        ellipse_params = ScatterPlotQualityLoss._compute_perceived_ellipse_params(ScatterPlotQualityLoss._get_edge_image(image))
        if ellipse_params is None:
            return torch.tensor(0.0)
        
        _, _, angle_p = ellipse_params
        return torch.abs(angle_c - angle_p) / torch.pi
    
    @staticmethod
    def _compute_axis_ratio_difference(x_scaled: torch.Tensor, y_scaled: torch.Tensor, marker_size: float, marker_opacity: float, aspect_ratio: float) -> torch.Tensor:
        """Compute the axis ratio difference between computed and perceived ellipses."""
        major_c, minor_c, _ = ScatterPlotQualityLoss._calculate_ellipse_params(x_scaled, y_scaled)

        image = ScatterPlotQualityLoss._generate_plot_image(x_scaled, y_scaled, marker_size, marker_opacity, aspect_ratio)
        ellipse_params = ScatterPlotQualityLoss._compute_perceived_ellipse_params(ScatterPlotQualityLoss._get_edge_image(image))
        if ellipse_params is None:
            return torch.tensor(0.0)
        
        major_p, minor_p, _ = ellipse_params
        return torch.abs((minor_c / major_c) - (minor_p / major_p))
    
    @staticmethod
    def _compute_opacity_contrast(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the opacity and contrast of the plot."""
        # Find pixels with non-zero opacity (grayscale value > 0)
        # image is a 3D tensor with shape (C, H, W)
        non_zero_pixels = image[image > 0]
        
        if len(non_zero_pixels) == 0:
            # Handle case where no non-zero pixels are found (this could happen with a completely transparent image)
            return torch.tensor(0.0).to(image.device), torch.tensor(0.0).to(image.device)
        
        # Calculate I_mu as the average value of non-zero pixels
        opacity = torch.mean(non_zero_pixels)

        # Calculate I_sigma as the standard deviation of the non-zero pixels
        contrast = torch.std(non_zero_pixels)

        return opacity, contrast
    
    @staticmethod
    def _compute_marker_pixels(marker_size: float, marker_opacity: float, aspect_ratio: float) -> torch.Tensor:
        """
        Compute the number of pixels covered by a single marker (|M|).
        """
        # Generate a scatter plot with only one marker centered in the image
        x_single = torch.tensor([0.0])  # Single marker at center
        y_single = torch.tensor([0.0])
        
        # Generate a plot image with one marker using the _generate_plot_image function
        one_marker_image = ScatterPlotQualityLoss._generate_plot_image(x_single, y_single, marker_size, marker_opacity, aspect_ratio)
        
        # Count the non-zero pixels (|M|) in the alpha channel (this is the set of pixels covered by the marker)
        marker_pixels = (one_marker_image > 0).sum()
        
        return marker_pixels
    
    @staticmethod
    def _compute_marker_overlap(image: torch.Tensor, marker_pixels: torch.Tensor, x: torch.Tensor, marker_size: float, marker_opacity: float, aspect_ratio: float) -> torch.Tensor:
        """Compute the marker overlap in the plot."""
        # Calculate |P|, the number of non-zero pixels in the full scatter plot (all markers)
        n_non_zero_pixels = (image > 0).sum()  # This is |P|, the set of pixels with non-zero opacity
        
        if marker_pixels == 0:
            # Handle case where no marker pixels are covered
            return torch.tensor(0.0).to(image.device)
        
        # Compute I_l = 1 - |P| / (n * |M|)
        number_of_markers = x.shape[0]
        overlap = 1 - (n_non_zero_pixels / (number_of_markers * marker_pixels))
        return overlap
    
    @staticmethod
    def _compute_overplotting(image: torch.Tensor, marker_pixels: torch.Tensor, x: torch.Tensor, marker_size: float, marker_opacity: float, aspect_ratio: float) -> torch.Tensor:
        """Compute the overplotting in the plot."""
        # Calculate sum(p in P), the sum of all actual opacities in the full scatter plot
        actual_opacity_sum = image.sum()  # Sum of all opacity values in the full image (|P|)
        
        # Calculate sum(p in M), the sum of opacities for a single marker
        marker_opacity_sum = marker_pixels.sum()
        
        if marker_opacity_sum == 0:
            # Handle case where no marker opacity is found (e.g., fully transparent or very small marker)
            return torch.tensor(0.0).to(image.device)
        
        # Compute I_p = 1 - (sum(p in P) / (n * sum(p in M)))
        number_of_markers = x.shape[0]
        overplotting = 1 - (actual_opacity_sum / (number_of_markers * marker_opacity_sum))
        return overplotting
    
    def _compute_class_perception(self, image_without_class: torch.Tensor, marker_size: float, marker_opacity: float, aspect_ratio: float) -> torch.Tensor:
        """Compute the class perception score."""
        if self.class_labels is None:
            return torch.tensor(0.0)
        
        image_with_class = self._generate_plot_image(self.x_data, self.y_data, marker_size, marker_opacity, aspect_ratio, self.class_labels)
        
        return 1 - ssim(image_without_class.unsqueeze(0), image_with_class.unsqueeze(0))
    
    def _compute_outlier_perception(self, image_all: torch.Tensor, marker_size: float, marker_opacity: float, aspect_ratio: float) -> torch.Tensor:
        """Compute the outlier perception score."""
        if self.outliers is None:
            return torch.tensor(0.0)
        
        outlier_mask = self.outliers.bool()
        image_outliers = self._generate_plot_image(
            self.x_data[outlier_mask], self.y_data[outlier_mask], marker_size, marker_opacity, aspect_ratio)
        
        return 1 - ssim(image_all.unsqueeze(0), image_outliers.unsqueeze(0))
    
class ScatterPlotQualityLossRegressor(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dims: list[int] = [64, 64, 64], output_dim: int = 1, dropout_rate: float = 0.1):
        """
        Neural network model for regression with BatchNorm, residual connections, and dropout.

        Args:
            input_dim (int): Dimensionality of input features (e.g., 3).
            hidden_dims (list[int]): List of integers specifying the number of neurons in each hidden layer.
            output_dim (int): Dimensionality of the output (default: 1 for regression).
            dropout_rate (float): Dropout rate for regularization.
        """
        super(ScatterPlotQualityLossRegressor, self).__init__()
        
        self.layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim, dtype=torch.double),
                nn.BatchNorm1d(hidden_dim, dtype=torch.double),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            if in_dim != hidden_dim:
                self.skip_connections.append(nn.Linear(in_dim, hidden_dim, dtype=torch.double))
            else:
                self.skip_connections.append(nn.Identity())
            in_dim = hidden_dim
        
        self.output_layer = nn.Linear(in_dim, output_dim, dtype=torch.double)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Normalized input tensor of shape [batch_size, input_dim] with dtype torch.float64.
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim] with dtype torch.float64.
        """
        for layer, skip in zip(self.layers, self.skip_connections):
            x = layer(x) + skip(x)
        return self.output_layer(x)
    
if __name__ == "__main__":
    x_data = torch.randn(100)
    y_data = 0.7 * x_data + 0.3 * torch.randn(100)
    class_labels = torch.randint(0, 3, (100,))
    outliers = torch.zeros(100, dtype=torch.bool)
    outliers[torch.randint(0, 100, (5,))] = True
    
    func = ScatterPlotQualityLoss(
        x_data=x_data,
        y_data=y_data,
        class_labels=class_labels,
        outliers=outliers,
        use_approximate_model=True,
    )
    exact_func = ScatterPlotQualityLoss(
        x_data=x_data,
        y_data=y_data,
        class_labels=class_labels,
        outliers=outliers,
        use_approximate_model=False,
    )
    
    # Test with a batch of 5 different parameter sets
    X = torch.tensor([
        [30.0, 128.0, 1.0],
        [10.0, 200.0, 0.8],
        [50.0, 50.0, 1.2],
        [15.0, 150.0, 1.1],
        [40.0, 100.0, 0.9]
    ]).double()

    # Show the generated images
    # for i in range(X.shape[0]):
    #     image = func._generate_plot_image(x_data, y_data, X[i, 0].item(), X[i, 1].item(), X[i, 2].item())
    #     plt.imshow(image.permute(1, 2, 0))
    #     plt.show()
    
    with torch.no_grad():
        results = func(X)
    print(f"Test results: {results}")
    print(f"Test results shape: {results.shape}")

    # Run grid search over the parameter bounds to
    # estimate min and max
    import itertools
    from tqdm import tqdm

    def grid_search(obj_func, num_points=10):
        """Perform a grid search over the search space."""
        marker_sizes = torch.linspace(*obj_func._marker_size_bounds, num_points)
        marker_opacities = torch.linspace(*obj_func._marker_opacity_bounds, num_points)
        aspect_ratios = torch.linspace(*obj_func._aspect_ratio_bounds, num_points)

        best_value = float("inf")
        worst_value = float("-inf")
        best_x = None
        worst_x = None

        for ms, mo, ar in tqdm(
            itertools.product(marker_sizes, marker_opacities, aspect_ratios),
            total=marker_sizes.size(0)*marker_opacities.size(0)*aspect_ratios.size(0)
        ):
            x = torch.tensor([ms, mo, ar]).double()
            with torch.no_grad():
                value = obj_func.evaluate_true(x.unsqueeze(0)).item()
            
            if value < best_value:
                best_value = value
                best_x = x.clone()
                print("New best value:", best_x, value)
            
            if value > worst_value:
                worst_value = value
                worst_x = x.clone()
                print("New worst value:", worst_x, value)

        return best_x, best_value, worst_x, worst_value

    # Example usage (assuming obj_func is an instance of ScatterPlotQualityLoss)
    best_x, best_val, worst_x, worst_val = grid_search(exact_func, num_points=20)
    print(f"Best solution: {best_x.numpy()}, Value: {best_val}")
    print(f"Worst solution: {worst_x.numpy()}, Value: {worst_val}")
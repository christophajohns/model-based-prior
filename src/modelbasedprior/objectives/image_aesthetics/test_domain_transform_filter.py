import torch
import torch.nn.functional as F
import numpy as np
import pytest
from PIL import Image
import os
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from modelbasedprior.objectives.image_aesthetics.domain_transform_filter import rf_filter, domain_recursive_filter_horizontal, _compute_domain_transform

class TestRFFilter:
    """
    Unit tests for the Domain Transform Recursive Edge-Preserving Filter.
    """

    @staticmethod
    def _create_sample_image() -> torch.Tensor:
        """Generate a sample image tensor (Helper static method)."""
        # Create a simple gradient image
        h, w = 64, 64
        x = torch.linspace(0, 1, w).reshape(1, 1, 1, w).repeat(1, 1, h, 1)
        y = torch.linspace(0, 1, h).reshape(1, 1, h, 1).repeat(1, 1, 1, w)
        # Create a 3-channel image with different patterns
        img = torch.cat([
            x,
            y,
            (x + y) / 2
        ], dim=1)

        # Add some edges
        img[:, :, h//3:h//3+2, :] = 1.0
        img[:, :, :, w//3:w//3+2] = 0.0

        return img

    @pytest.fixture
    def sample_image(self) -> torch.Tensor:
        """Pytest fixture that provides the sample image."""
        # Call the static helper method to get the image
        return TestRFFilter._create_sample_image()
    
    @pytest.fixture
    def noisy_image(self, sample_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a noisy version of the sample image."""
        noise = torch.randn_like(sample_image) * 0.1
        noisy = torch.clamp(sample_image + noise, 0, 1)
        return noisy, sample_image  # return both noisy and original
    
    def test_output_shape_and_type(self, sample_image: torch.Tensor):
        """Test that output shape and dtype match input."""
        # Test with different dtypes
        for dtype in [torch.float32, torch.float16, torch.uint8]:
            img = sample_image.to(dtype)
            if dtype == torch.uint8:
                img = (img * 255).to(dtype)
            
            # Run filter
            output = rf_filter(img, sigma_s=10.0, sigma_r=0.1)
            
            # Check shape and dtype
            assert output.shape == img.shape
            assert output.dtype == img.dtype
            
            # Check output range for uint8
            if dtype == torch.uint8:
                assert output.min() >= 0
                assert output.max() <= 255
    
    def test_different_dimensions(self, sample_image: torch.Tensor):
        """Test the filter with and without batch dimension."""
        # With batch dimension (4D)
        output_4d = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1)
        assert output_4d.dim() == 4
        
        # Without batch dimension (3D)
        img_3d = sample_image.squeeze(0)
        output_3d = rf_filter(img_3d, sigma_s=10.0, sigma_r=0.1)
        assert output_3d.dim() == 3
        
        # Compare results (should be the same)
        assert torch.allclose(output_4d.squeeze(0), output_3d)
    
    def test_joint_filtering(self, sample_image: torch.Tensor):
        """Test that joint filtering works correctly."""
        # Create a guidance image (e.g., a blurred version)
        guidance = rf_filter(sample_image, sigma_s=2.0, sigma_r=0.1)
        
        # Test joint filtering
        output = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1, joint_image=guidance)
        
        # Result should be different from both input and regular filtered
        regular_output = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1)
        
        # They should be different but not completely different
        assert not torch.allclose(output, regular_output)
        assert not torch.allclose(output, sample_image)
        
        # Test with 3D joint image
        guidance_3d = guidance.squeeze(0)
        output_with_3d = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1, joint_image=guidance_3d)
        
        # Should match output with 4D joint image
        assert torch.allclose(output, output_with_3d)
        
        # Test error when dimensions don't match
        with pytest.raises(ValueError):
            wrong_size = torch.rand(1, 3, 32, 32)  # Different size
            rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1, joint_image=wrong_size)
    
    def test_domain_transform_computation(self, sample_image: torch.Tensor):
        """Test the domain transform computation function."""
        dHdx, dVdy = _compute_domain_transform(sample_image, sigma_s=10.0, sigma_r=0.1)
        
        # Check output shapes
        assert dHdx.shape == (1, 1, sample_image.shape[2], sample_image.shape[3])
        assert dVdy.shape == (1, 1, sample_image.shape[2], sample_image.shape[3])
        
        # Check values are reasonable (min value should be 1.0)
        assert dHdx.min() >= 1.0
        assert dVdy.min() >= 1.0
    
    def test_horizontal_filter(self, sample_image: torch.Tensor):
        """Test the horizontal filter function."""
        # Compute domain transform
        dHdx, _ = _compute_domain_transform(sample_image, sigma_s=10.0, sigma_r=0.1)
        
        # Apply horizontal filter
        sigma = torch.tensor(10.0)
        output = domain_recursive_filter_horizontal(sample_image, dHdx, sigma)
        
        # Output should be different but have same shape
        assert output.shape == sample_image.shape
        assert not torch.allclose(output, sample_image)
    
    def test_iteration_effect(self, sample_image: torch.Tensor):
        """Test the effect of different numbers of iterations."""
        # Run with different iteration counts
        output_1 = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1, num_iterations=1)
        output_3 = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1, num_iterations=3)
        output_5 = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1, num_iterations=5)
        
        # More iterations should lead to more smoothing
        # Calculate gradient magnitude as a measure of smoothness
        def calc_smoothness(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            grad_mag = torch.sqrt(dx.pow(2).mean() + dy.pow(2).mean())
            return grad_mag
            
        grad_1 = calc_smoothness(output_1)
        grad_3 = calc_smoothness(output_3)
        grad_5 = calc_smoothness(output_5)
        
        # More iterations should reduce gradient magnitude (more smoothing)
        assert grad_1 > grad_3
        assert grad_3 > grad_5
    
    def test_sigma_effect(self, sample_image: torch.Tensor):
        """Test the effect of different sigma values."""
        # Run with different sigma values
        output_small_s = rf_filter(sample_image, sigma_s=5.0, sigma_r=0.1)
        output_large_s = rf_filter(sample_image, sigma_s=20.0, sigma_r=0.1)
        
        output_small_r = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.05)
        output_large_r = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.2)
        
        # Calculate variance as a measure of smoothing
        def calc_variance(img):
            return torch.var(img)
            
        # Larger spatial sigma should lead to more smoothing
        assert calc_variance(output_small_s) > calc_variance(output_large_s)
        
        # Smaller range sigma should preserve more edges (less variance reduction)
        assert calc_variance(output_small_r) > calc_variance(output_large_r)
    
    def test_device_compatibility(self, sample_image: torch.Tensor):
        """Test that the filter works on different devices."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")
            
        # Run on CPU
        output_cpu = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1)
        
        # Run on GPU
        img_gpu = sample_image.cuda()
        output_gpu = rf_filter(img_gpu, sigma_s=10.0, sigma_r=0.1)
        
        # Results should be similar
        assert torch.allclose(output_cpu, output_gpu.cpu(), rtol=1e-5, atol=1e-5)
    
    def test_noise_reduction(self, noisy_image: Tuple[torch.Tensor, torch.Tensor]):
        """Test the filter's ability to reduce noise while preserving edges."""
        noisy, original = noisy_image
        
        # Filter the noisy image
        filtered = rf_filter(noisy, sigma_s=10.0, sigma_r=0.1, num_iterations=3)
        
        # Calculate MSE between filtered and original
        mse_noisy = torch.mean((noisy - original) ** 2)
        mse_filtered = torch.mean((filtered - original) ** 2)
        
        # Filtered image should be closer to original than noisy
        assert mse_filtered < mse_noisy
        
        # Calculate edge preservation
        def calc_edge_preservation(img1, img2):
            # Compute gradients
            dx1 = img1[:, :, :, 1:] - img1[:, :, :, :-1]  # [B, C, H, W-1]
            dy1 = img1[:, :, 1:, :] - img1[:, :, :-1, :]  # [B, C, H-1, W]
            
            dx2 = img2[:, :, :, 1:] - img2[:, :, :, :-1]  # [B, C, H, W-1]
            dy2 = img2[:, :, 1:, :] - img2[:, :, :-1, :]  # [B, C, H-1, W]
            
            # Make sure gradients have the same size by restricting to common regions
            common_width = min(dx1.shape[3], dx2.shape[3])
            common_height = min(dy1.shape[2], dy2.shape[2])
            
            # Crop to common dimensions
            dx1 = dx1[:, :, :common_height, :common_width]
            dx2 = dx2[:, :, :common_height, :common_width]
            dy1 = dy1[:, :, :common_height, :common_width]
            dy2 = dy2[:, :, :common_height, :common_width]
            
            # Calculate gradient magnitudes (properly sized now)
            g1 = torch.sqrt(dx1.pow(2) + dy1.pow(2))
            g2 = torch.sqrt(dx2.pow(2) + dy2.pow(2))
            
            # Return correlation
            return torch.corrcoef(torch.stack([g1.flatten(), g2.flatten()]))[0, 1]
        
        # Edge preservation should be high between filtered and original
        edge_preservation = calc_edge_preservation(filtered, original)
        assert edge_preservation > 0.7  # Reasonable threshold


class TestRFFilterIntegration:
    """
    Integration tests for the Domain Transform Recursive Edge-Preserving Filter.
    """
    
    @pytest.fixture
    def real_images(self) -> Tuple[torch.Tensor, ...]:
        """Load real test images or create synthetic test patterns."""
        # Create test pattern: sharp edges with noise
        h, w = 256, 256
        
        # Create checkerboard pattern
        x = torch.linspace(-1, 1, w).view(1, 1, 1, w).repeat(1, 3, h, 1)
        y = torch.linspace(-1, 1, h).view(1, 1, h, 1).repeat(1, 3, 1, w)
        checkerboard = ((x.sign() * y.sign() + 1) / 2)
        
        # Create gradient pattern
        gradient = torch.cat([
            torch.linspace(0, 1, w).reshape(1, 1, 1, w).repeat(1, 1, h, 1),
            torch.linspace(0, 1, h).reshape(1, 1, h, 1).repeat(1, 1, 1, w),
            torch.ones(1, 1, h, w) * 0.5
        ], dim=1)
        
        # Create a more complex pattern
        complex_pattern = torch.zeros(1, 3, h, w)
        # Add circles
        for cx, cy, r in [(h//4, w//4, h//8), (h//2, w//2, h//6), (3*h//4, 3*w//4, h//7)]:
            for i in range(h):
                for j in range(w):
                    dist = torch.sqrt(torch.tensor(((i-cy)**2 + (j-cx)**2), dtype=torch.float32))
                    if dist < r:
                        complex_pattern[0, 0, i, j] = 1.0
                        complex_pattern[0, 1, i, j] = 0.5
                        
        # Add some lines
        complex_pattern[0, 2, h//3:h//3+2, :] = 1.0
        complex_pattern[0, 1, :, w//3:w//3+2] = 1.0
        
        # Add noise to all patterns
        noise_level = 0.1
        patterns = []
        for pattern in [checkerboard, gradient, complex_pattern]:
            noisy = pattern + torch.randn_like(pattern) * noise_level
            noisy = torch.clamp(noisy, 0, 1)
            patterns.append(noisy)
            
        return tuple(patterns)
    
    def test_edge_preserving_smoothing(self, real_images: Tuple[torch.Tensor, ...]):
        """Test that the filter preserves edges while smoothing noise."""
        for img in real_images:
            # Apply filter with different parameters
            filtered_strong = rf_filter(img, sigma_s=20.0, sigma_r=0.1, num_iterations=3)
            filtered_edge_preserving = rf_filter(img, sigma_s=20.0, sigma_r=0.05, num_iterations=3)
            
            # Calculate gradients to detect edges
            def detect_edges(tensor):
                # Simple Sobel-like edge detection - ensuring dimensions match
                if tensor.shape[3] < 3 or tensor.shape[2] < 3:
                    # Handle very small images
                    return torch.zeros((tensor.shape[0], tensor.shape[1], 
                                    max(1, tensor.shape[2]-2), 
                                    max(1, tensor.shape[3]-2)), 
                                    device=tensor.device)
                
                dx = tensor[:, :, :, 2:] - tensor[:, :, :, :-2]
                dy = tensor[:, :, 2:, :] - tensor[:, :, :-2, :]
                
                # Resize dy to match dx dimensions
                if dx.shape[2] > dy.shape[2]:
                    dy = F.pad(dy, (0, 0, 0, dx.shape[2] - dy.shape[2]), "constant", 0)
                
                # Resize dx to match dy dimensions
                if dy.shape[3] > dx.shape[3]:
                    dx = F.pad(dx, (0, dy.shape[3] - dx.shape[3], 0, 0), "constant", 0)
                
                # Ensure we're operating on matching dimensions
                common_height = min(dx.shape[2], dy.shape[2])
                common_width = min(dx.shape[3], dy.shape[3])
                
                dx = dx[:, :, :common_height, :common_width]
                dy = dy[:, :, :common_height, :common_width]
                
                return torch.sqrt(dx.pow(2) + dy.pow(2))
            
            edges_original = detect_edges(img)
            edges_strong = detect_edges(filtered_strong)
            edges_preserving = detect_edges(filtered_edge_preserving)
            
            # The edge-preserving filter should maintain stronger edges than the strong smoothing
            assert edges_preserving.mean() > edges_strong.mean()
            
            # Both filtered results should reduce the overall gradient (noise reduction)
            assert edges_original.mean() > edges_preserving.mean()
            assert edges_original.mean() > edges_strong.mean()
    
    def test_computational_performance(self, real_images: Tuple[torch.Tensor, ...]):
        """Test the computational performance of the filter."""
        # Skip if time tracking isn't available
        try:
            import time
        except ImportError:
            pytest.skip("time module not available")
        
        img = real_images[0]  # Use first test image
        
        # Warm-up
        for _ in range(3):
            _ = rf_filter(img, sigma_s=10.0, sigma_r=0.1)
        
        # Time several runs
        times = []
        for _ in range(5):
            start_time = time.time()
            _ = rf_filter(img, sigma_s=10.0, sigma_r=0.1)
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Performance assertion - should be reasonably fast for a 256x256 image
        # Adjust the threshold based on your hardware expectations
        assert avg_time < 0.5, f"Filter too slow: {avg_time:.4f} seconds (threshold: 0.5s)"
    
    def test_joint_filtering_edge_preservation(self, real_images: Tuple[torch.Tensor, ...]):
        """Test joint filtering for targeted smoothing while preserving edges."""
        img = real_images[2]  # Use the complex pattern
        
        # Create a modified joint image (e.g., one with enhanced edges)
        joint_img = img.clone()

        def enhance_edges(joint_img):
            # Calculate gradients
            dx = joint_img[:, :, :, 1:] - joint_img[:, :, :, :-1]  # [B, C, H, W-1]
            dy = joint_img[:, :, 1:, :] - joint_img[:, :, :-1, :]  # [B, C, H-1, W]
            
            # Get common dimensions
            h, w = joint_img.shape[2], joint_img.shape[3]
            
            # Sum along channel dimension
            dx_sum = dx.pow(2).sum(dim=1, keepdim=True)  # [B, 1, H, W-1]
            dy_sum = dy.pow(2).sum(dim=1, keepdim=True)  # [B, 1, H-1, W]
            
            # Create padded versions to match original image size
            dx_padded = F.pad(dx_sum, (0, 1, 0, 0), "constant", 0)  # [B, 1, H, W]
            dy_padded = F.pad(dy_sum, (0, 0, 0, 1), "constant", 0)  # [B, 1, H, W]
            
            # Calculate edge magnitude on properly padded tensors
            edge_magnitude = torch.sqrt(dx_padded + dy_padded)  # [B, 1, H, W]
            
            # Create enhanced joint image
            enhanced_joint = joint_img.clone()
            enhanced_joint = enhanced_joint + edge_magnitude  # Enhance edges
            enhanced_joint = torch.clamp(enhanced_joint, 0, 1)  # Keep values in valid range
            
            return enhanced_joint
        
        # Enhance edges in the joint image
        joint_img = enhance_edges(joint_img)
        
        # Apply filtering with joint guidance
        filtered_regular = rf_filter(img, sigma_s=15.0, sigma_r=0.1)
        filtered_joint = rf_filter(img, sigma_s=15.0, sigma_r=0.1, joint_image=joint_img)
        
        # Calculate total variation as a measure of edge preservation
        def total_variation(tensor):
            dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
            dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
            return torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))
        
        tv_original = total_variation(img)
        tv_regular = total_variation(filtered_regular)
        tv_joint = total_variation(filtered_joint)
        
        # Joint filtering should preserve more variation than regular filtering
        assert tv_joint > tv_regular
        # Both should be lower than the original (noise reduction)
        assert tv_original > tv_joint
        assert tv_original > tv_regular
    
    def test_sigma_parameters_effect(self, real_images: Tuple[torch.Tensor, ...]):
        """Test the effect of sigma parameters on filtering results."""
        img = real_images[1]  # Use gradient image
        
        # Apply filtering with different sigma combinations
        results = {}
        for sigma_s in [5.0, 15.0, 30.0]:
            for sigma_r in [0.05, 0.1, 0.2]:
                key = f"s{sigma_s}_r{sigma_r}"
                results[key] = rf_filter(img, sigma_s=sigma_s, sigma_r=sigma_r)
        
        # Higher spatial sigma (sigma_s) should lead to more spatial smoothing
        assert torch.var(results["s5.0_r0.1"]) > torch.var(results["s30.0_r0.1"])
        
        # Lower range sigma (sigma_r) should better preserve edges
        edge_preservation_low_r = torch.mean(torch.abs(results["s15.0_r0.05"] - img))
        edge_preservation_high_r = torch.mean(torch.abs(results["s15.0_r0.2"] - img))
        assert edge_preservation_low_r < edge_preservation_high_r
    
    def visualize_results(self, real_images: Tuple[torch.Tensor, ...], output_dir: Optional[str] = None):
        """
        Visualize filtering results with different parameters.
        
        This is not an automatic test, but a helper for manual visual inspection.
        """
        if output_dir is None:
            output_dir = "filter_results"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each test image
        for idx, img in enumerate(real_images):
            # Apply different filters
            results = {
                "original": img,
                "smooth_weak": rf_filter(img, sigma_s=5.0, sigma_r=0.2),
                "smooth_strong": rf_filter(img, sigma_s=30.0, sigma_r=0.2),
                "edge_preserving": rf_filter(img, sigma_s=15.0, sigma_r=0.05),
                "iterations_1": rf_filter(img, sigma_s=15.0, sigma_r=0.1, num_iterations=1),
                "iterations_5": rf_filter(img, sigma_s=15.0, sigma_r=0.1, num_iterations=5)
            }
            
            # Create joint filtered version
            joint_img = img.clone()
            # Enhance edges for joint filtering
            dx = joint_img[:, :, :, 1:] - joint_img[:, :, :, :-1]
            dy = joint_img[:, :, 1:, :] - joint_img[:, :, :-1, :]
            edge_magnitude = torch.sqrt(dx.pow(2).sum(dim=1, keepdim=True) + 
                                       dy.pow(2).sum(dim=1, keepdim=True))
            edge_magnitude = F.pad(edge_magnitude, (0, 1, 0, 1))
            joint_img = torch.clamp(joint_img * (1 + 5 * edge_magnitude), 0, 1)
            
            # Add joint filtering result
            results["joint_filtered"] = rf_filter(img, sigma_s=15.0, sigma_r=0.1, joint_image=joint_img)
            
            # Plot results
            plt.figure(figsize=(20, 10))
            for i, (name, result) in enumerate(results.items()):
                plt.subplot(2, 4, i+1)
                plt.imshow(result.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.title(name)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"filter_comparison_{idx}.png"))
            plt.close()
            
            # Save individual images
            for name, result in results.items():
                img_np = (result.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_pil.save(os.path.join(output_dir, f"img{idx}_{name}.png"))


def test_end_to_end():
    """
    End-to-end test with a real-world image if available.
    """
    # Try to load a real image if PIL is available
    try:
        from PIL import Image
        import urllib.request
        import tempfile
        
        # Download a test image (or use a local one if available)
        image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        
        with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
            try:
                urllib.request.urlretrieve(image_url, f.name)
                img = Image.open(f.name)
            except:
                # Fallback to generating a test image
                img = Image.new('RGB', (256, 256))
                for x in range(256):
                    for y in range(256):
                        r = int(x / 256 * 255)
                        g = int(y / 256 * 255)
                        b = int(((x+y) / 2) / 256 * 255)
                        img.putpixel((x, y), (r, g, b))
            
            # Convert to tensor
            img_np = np.array(img) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)
            
            # Apply filtering with different parameters
            filtered_default = rf_filter(img_tensor, sigma_s=10.0, sigma_r=0.1)
            filtered_strong = rf_filter(img_tensor, sigma_s=30.0, sigma_r=0.2)
            filtered_edge = rf_filter(img_tensor, sigma_s=20.0, sigma_r=0.05)

            def create_edge_magnitude(gray):
                # Create edge magnitude map handling dimension issues
                dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]  # [B, 1, H, W-1]
                dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]  # [B, 1, H-1, W]
                
                # Pad tensors to match original dimensions
                dx_padded = F.pad(dx, (0, 1, 0, 0), "constant", 0)  # [B, 1, H, W]
                dy_padded = F.pad(dy, (0, 0, 0, 1), "constant", 0)  # [B, 1, H, W]
                
                # Calculate edge magnitude with properly padded tensors
                edge_magnitude = torch.sqrt(dx_padded**2 + dy_padded**2)
                
                return edge_magnitude
            
            # Joint filtering
            # Create an edge-enhanced guidance image
            gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
            edge_magnitude = create_edge_magnitude(gray)
            
            # Use the edge information for joint filtering
            filtered_joint = rf_filter(img_tensor, sigma_s=20.0, sigma_r=0.1, joint_image=edge_magnitude)
            
            # Check basic properties
            assert filtered_default.shape == img_tensor.shape
            assert filtered_strong.shape == img_tensor.shape
            assert filtered_edge.shape == img_tensor.shape
            assert filtered_joint.shape == img_tensor.shape
            
            # Save results for visual inspection if output directory exists
            output_dir = "test_output"
            if os.path.exists(output_dir) or os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
                # Function to convert tensor to PIL Image
                def tensor_to_image(t):
                    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    return Image.fromarray(arr)
                
                # Save original and filtered images
                tensor_to_image(img_tensor).save(os.path.join(output_dir, "original.png"))
                tensor_to_image(filtered_default).save(os.path.join(output_dir, "filtered_default.png"))
                tensor_to_image(filtered_strong).save(os.path.join(output_dir, "filtered_strong.png"))
                tensor_to_image(filtered_edge).save(os.path.join(output_dir, "filtered_edge.png"))
                tensor_to_image(filtered_joint).save(os.path.join(output_dir, "filtered_joint.png"))
                
                # Create comparison figure
                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 3, 1)
                plt.imshow(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.title("Original")
                plt.axis('off')
                
                plt.subplot(2, 3, 2)
                plt.imshow(filtered_default.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.title("Default Filtering")
                plt.axis('off')
                
                plt.subplot(2, 3, 3)
                plt.imshow(filtered_strong.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.title("Strong Smoothing")
                plt.axis('off')
                
                plt.subplot(2, 3, 4)
                plt.imshow(filtered_edge.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.title("Edge Preserving")
                plt.axis('off')
                
                plt.subplot(2, 3, 5)
                plt.imshow(filtered_joint.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.title("Joint Filtering")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "comparison.png"))
                plt.close()
                
    except ImportError:
        pytest.skip("PIL or matplotlib not available, skipping end-to-end test")


if __name__ == "__main__":
    # Run some basic tests and visualizations when executed as a script
    test_rf = TestRFFilter()
    sample_image = test_rf._create_sample_image()
    
    # Apply different filter settings
    filtered_default = rf_filter(sample_image, sigma_s=10.0, sigma_r=0.1)
    filtered_strong = rf_filter(sample_image, sigma_s=30.0, sigma_r=0.2)
    filtered_edge = rf_filter(sample_image, sigma_s=20.0, sigma_r=0.05)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Save results
    for i, (name, img) in enumerate([
        ("original", sample_image),
        ("filtered_default", filtered_default),
        ("filtered_strong", filtered_strong),
        ("filtered_edge", filtered_edge)
    ]):
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(img_np)
        plt.title(name)
        plt.axis('off')
        plt.savefig(f"output/{name}.png")
        plt.close()
    
    # Run end-to-end test
    test_end_to_end()
    
    print("Tests and visualizations completed. Check the 'output' directory for results.")
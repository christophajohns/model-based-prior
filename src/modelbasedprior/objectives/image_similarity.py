import torch
from typing import List, Optional, Tuple
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.prior import unnormalize
from pyiqa.archs.psnr_arch import psnr
from pyiqa.archs.ssim_arch import ssim
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue

def generate_image(X: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
        # Ensure X has shape (n, q, d), then flatten to (n*q, d) for processing
        n, q, d = X.shape
        X_flat = X.view(n * q, d)

        images = []
        for x in X_flat:
            image = original_image.clone().detach()
            brightness, contrast, saturation, hue = x
            image = adjust_brightness(image, brightness.item())
            image = adjust_contrast(image, contrast.item())
            image = adjust_saturation(image, saturation.item())
            image = adjust_hue(image, hue.item())
            images.append(image)

        # Stack images and reshape back to (n, q, C, H, W)
        return torch.stack(images).view(n, q, *original_image.shape)
    
class ImageSimilarityLoss(SyntheticTestFunction):
    dim = 4
    _check_grad_at_opt: bool = False
    _brightness_bounds = (0.3, 3.)
    _contrast_bounds = (0.3, 3.)
    _saturation_bounds = (0.3, 3.)
    _hue_bounds = (-0.5, 0.5)

    def __init__(
        self,
        original_image: torch.Tensor,
        optimizer: Tuple[float, ...] = (1.0, 1.0, 1.0, 0.0),
        weight_psnr: float = 0.5,
        weight_ssim: float = 0.5,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        if bounds is None:
            bounds = [self._brightness_bounds, self._contrast_bounds, self._saturation_bounds, self._hue_bounds]
        self._optimizers = [optimizer]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self._original_image = original_image
        self._weight_psnr = weight_psnr
        self._weight_ssim = weight_ssim
        self._original_target_image = generate_image(torch.tensor(optimizer).reshape(1, 1, -1), self._original_image)  # Ensure shape (1, 1, d)
        self._target_image = self._normalize_image(self._original_target_image)
        self._optimal_value = self.evaluate_true(torch.tensor(optimizer).reshape(1, 1, -1)).item()

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # Ensure X has shape (n, q, d)
        if X.dim() == 2:  
            X = X.unsqueeze(1)  # Convert (n, d) -> (n, 1, d)

        n, q, d = X.shape

        # Generate images for all input parameter settings
        original_candidate_image = generate_image(X, self._original_image)  # (n, q, C, H, W)
        candidate_image = self._normalize_image(original_candidate_image)

        # Expand target image to match (n, q, C, H, W)
        target_image = self._target_image.expand(n, q, -1, -1, -1)  # Efficient broadcast without extra memory

        # Compute similarity metrics
        psnr_score = psnr(candidate_image.flatten(0, 1), target_image.flatten(0, 1)) / 80.0  # Normalize PSNR to [0, 1]
        ssim_score = ssim(candidate_image.flatten(0, 1), target_image.flatten(0, 1))  # Flatten (n*q, C, H, W)

        # Reshape results back to (n, q)
        psnr_score = psnr_score.view(n, q)
        ssim_score = ssim_score.view(n, q)

        # Compute final loss
        score = -self._weight_psnr * psnr_score + self._weight_ssim * (1 - ssim_score)
        return score.squeeze(1).to(X) if X.shape[1] == 1 else score.to(X)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return image / 255.0
    
if __name__ == "__main__":
    # import os
    # from dotenv import load_dotenv
    # from torchvision.io import read_image
    # from torchvision.transforms.functional import resize

    # load_dotenv()

    original_image = (torch.rand(3, 16, 16) * 255).floor().to(torch.uint8)
    # ava_flowers_dir = os.getenv("AVA_FLOWERS_DIR")
    # original_image = resize(read_image(os.path.join(ava_flowers_dir, '43405.jpg')), 64)  # Downsample
    image_similarity = ImageSimilarityLoss(original_image=original_image)
    X_rand = unnormalize(torch.rand(2, 4), image_similarity.bounds)
    X_best = image_similarity.optimizers
    X = torch.cat([X_rand, X_best], dim=0)
    print(image_similarity(X))

    qX = unnormalize(torch.rand(3, 2, 4), image_similarity.bounds)
    print(image_similarity(qX))
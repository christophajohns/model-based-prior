import torch
from typing import List, Optional, Tuple
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.prior import normalize, unnormalize
from pyiqa.archs.psnr_arch import psnr
from pyiqa.archs.ssim_arch import ssim
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
    
class ImageSimilarityLoss(SyntheticTestFunction):
    r"""Image similarity test function.

    4-dimensional function (usually evaluated on `[0.3, 3]` for the first
    three dimensions and `[-0.5, 0.5]` for the last dimension):

        f(x) = -w_{PSNR} * \text{PSNR}(\text{Image}(x), \text{Image}(z)) + w_{SSIM} * (1 - \text{SSIM}(\text{Image}(x), \text{Image}(z)))

    where `w_{PSNR} + w_{SSIM} = 1` and `z` is the set of target parameter settings.

    f has one minimizer for its global minimum `z_1` as specified during initialization with
    `f(z_1) = -w_{PSNR} * 80` (80 is the maximum PSNR value for the default epsilon value of 1e-8).

    PSNR ranges from 0 to infinity with higher values indicating better similarity.
    SSIM ranges from 0 to 1 with 1 indicating perfect similarity.

    The input to the function represents the parameter settings for brightness,
    contrast, saturation, and hue to manipulate an original image.
    """

    dim = 4
    _check_grad_at_opt: bool = False
    _brightness_bounds = (0.3, 3.)
    _contrast_bounds = (0.3, 3.)
    _saturation_bounds = (0.3, 3.)
    _hue_bounds = (-0.5, 0.5)

    def __init__(
        self,
        original_image: torch.Tensor,
        optimizer: Tuple[float, ...] = (1.0, 1.0, 1.0, 0.0),  # keep the original image
        weight_psnr: float = 0.5,
        weight_ssim: float = 0.5,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            original_image: The original image RGB tensor (3, H, W).
            optimizer: The optimizer location.
            weight_psnr: The weight for the PSNR component.
            weight_ssim: The weight for the SSIM component.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if bounds is None:
            bounds = [self._brightness_bounds, self._contrast_bounds, self._saturation_bounds, self._hue_bounds]
        self._optimizers = [optimizer]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self._original_image = original_image
        self._weight_psnr = weight_psnr
        self._weight_ssim = weight_ssim
        self._original_target_image = self._generate_image(torch.tensor(optimizer).reshape(1, -1))
        self._target_image = self._normalize_image(self._original_target_image)
        self._optimal_value = self.evaluate_true(torch.tensor(optimizer).reshape(1, -1)).item()

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        original_candidate_image = self._generate_image(X)
        candidate_image = self._normalize_image(original_candidate_image)
        target_image = self._target_image.repeat(X.size()[0], 1, 1, 1) # Repeat the target image to match the number of rows in X
        psnr_score = psnr(candidate_image, target_image) / 80.0  # Normalize PSNR to [0, 1]
        ssim_score = ssim(candidate_image, target_image)
        score = -self._weight_psnr * psnr_score + self._weight_ssim * (1 - ssim_score)  # 1 - SSIM because SSIM is range [0, 1] with 1 being the best; -PSNR because PSNR is range [0, inf] with higher values being better
        return score.to(X)
    
    def _generate_image(self, X: torch.Tensor) -> torch.Tensor:
        X_unit_normalized = normalize(X.squeeze(1), self.bounds)  # X may have shape (N, 4) or (N, 1, 4)
        X_normalized = unnormalize(X_unit_normalized, torch.tensor([self._brightness_bounds, self._contrast_bounds, self._saturation_bounds, self._hue_bounds]).T)
        images = []
        for x in X_normalized:
            image = self._original_image.clone().detach()
            brightness, contrast, saturation, hue = x
            image = adjust_brightness(image, brightness.item())
            image = adjust_contrast(image, contrast.item())
            image = adjust_saturation(image, saturation.item())
            image = adjust_hue(image, hue.item())
            images.append(image)
        return torch.stack(images)
    
    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return image / 255.0
    
if __name__ == "__main__":
    original_image = (torch.rand(3, 16, 16) * 255).floor().to(torch.uint8)
    image_similarity = ImageSimilarityLoss(original_image=original_image)
    X_rand = unnormalize(torch.rand(2, 4), image_similarity.bounds)
    X_best = image_similarity.optimizers
    X = torch.cat([X_rand, X_best], dim=0)
    print(image_similarity(X))
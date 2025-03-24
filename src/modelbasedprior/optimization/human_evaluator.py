import torch
import logging
from typing import Protocol
import matplotlib.pyplot as plt
from modelbasedprior.optimization.bo import maximize
from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss

class HumanEvaluatorRenderer(Protocol):
    def render(self, X: torch.Tensor) -> torch.Tensor:
        """Render the point in a specific way (CLI, image, etc.) and collect a human rating."""
        ...

class CLIHumanEvaluatorRenderer:
    def render(self, X: torch.Tensor) -> torch.Tensor:
        """Render the point in a CLI and get user rating."""
        point = X.numpy()
        print(f"\nEvaluate this point: {point}")
        return torch.tensor(float(input("Enter rating: ")), dtype=X.dtype)

class ImageHumanEvaluatorRenderer:
    def __init__(self, original_image: torch.Tensor):
        self.image_similarity = ImageSimilarityLoss(original_image)

    def render(self, X: torch.Tensor) -> torch.Tensor:
        """Render the image and collect a human rating."""
        transformed_images = self.image_similarity._generate_image(X.view(1, 1, -1))  # shape: n x q x C x H x W
        n, q, C, H, W = transformed_images.shape  # Extract shape

        ratings = torch.zeros(n, q, dtype=X.dtype)  # Store ratings

        for i in range(n):
            for j in range(q):
                img = transformed_images[i, j].cpu().permute(1, 2, 0).numpy()  # Convert (C, H, W) → (H, W, C)
                
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"Batch {i+1}, Image {j+1}")
                plt.draw()  # Draw the figure without blocking
                plt.pause(0.001)  # Allow UI to update
                
                rating = float(input(f"Enter rating for batch {i+1}, image {j+1}: "))
                ratings[i, j] = rating
                
                plt.clf()  # Clear the figure for the next image

        plt.close()  # Close figure after all images are rated
        return ratings

class HumanEvaluatorObjective:
    def __init__(self, renderer: HumanEvaluatorRenderer = CLIHumanEvaluatorRenderer(), dim=2, bounds=None):
        self.renderer = renderer  # Inject dependency
        self.dim = dim
        if bounds is None:
            bounds = [(-5.12, 5.12) for _ in range(self.dim)]
        self.bounds = torch.tensor(bounds).t()
    
    def __call__(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        orig_shape = X.shape
        if X.dim() == 2:
            X = X.unsqueeze(1)  # Add q dimension
        
        batch_shape, q, dim = X.shape
        ratings = torch.zeros(batch_shape, q, dtype=X.dtype, device=X.device)
        
        for i in range(batch_shape):
            for j in range(q):
                ratings[i, j] = self.renderer.render(X[i, j].cpu())  # Delegation
        
        return ratings.squeeze(1) if len(orig_shape) == 2 else ratings

if __name__ == "__main__":
    from botorch.utils.prior import DefaultPrior
    from modelbasedprior.objectives.sphere import Sphere
    from modelbasedprior.logger import setup_logger
    from modelbasedprior.prior import ModelBasedPrior

    # CLI-based evaluation
    objective = Sphere(dim=2, negate=True)
    global_optima = objective.optimizers  # here: objective.bounds.T.mean(dim=-1) => [0, 0]
    offset = 0.1 * (objective.bounds[1,:] - objective.bounds[0,:])
    parameter_default = global_optima[0] + offset
    user_prior = DefaultPrior(bounds=objective.bounds, parameter_defaults=parameter_default, confidence=0.25)
    
    logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

    cli_renderer = CLIHumanEvaluatorRenderer()
    human_evaluator_objective = HumanEvaluatorObjective(renderer=cli_renderer, dim=objective.dim, bounds=objective._bounds)

    # result_X, result_y, model = maximize(human_evaluator_objective, user_prior=user_prior, num_trials=5, logger=logger)

    # Image-based evaluation
    original_image = (torch.rand(3, 16, 16) * 255).floor().to(torch.uint8)
    image_similarity = ImageSimilarityLoss(original_image=original_image, negate=True)
    user_prior = ModelBasedPrior(
        bounds=image_similarity.bounds,
        predict_func=image_similarity,
        minimize=False,
    )
    image_renderer = ImageHumanEvaluatorRenderer(original_image)
    human_evaluator = HumanEvaluatorObjective(renderer=image_renderer, dim=image_similarity.dim, bounds=image_similarity._bounds)

    result_X, result_y, model = maximize(human_evaluator, user_prior=user_prior, num_trials=5, logger=logger)
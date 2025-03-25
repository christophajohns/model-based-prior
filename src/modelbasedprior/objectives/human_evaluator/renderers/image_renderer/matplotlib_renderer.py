import torch
import matplotlib.pyplot as plt
from modelbasedprior.objectives.image_similarity import generate_image

class MatplotlibImageHumanEvaluatorRenderer:
    def __init__(self, original_image: torch.Tensor):
        self._original_image = original_image

    def render(self, X: torch.Tensor) -> torch.Tensor:
        """Render the image and collect a human rating."""
        transformed_images = generate_image(X.view(1, 1, -1), self._original_image)  # shape: n x q x C x H x W
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

if __name__ == "__main__":
    import logging
    from modelbasedprior.logger import setup_logger
    from modelbasedprior.prior import ModelBasedPrior
    from modelbasedprior.objectives.human_evaluator.human_evaluator_objective import HumanEvaluatorObjective
    from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss
    from modelbasedprior.optimization.bo import maximize

    logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

    # Image-based evaluation
    original_image = (torch.rand(3, 16, 16) * 255).floor().to(torch.uint8)
    image_similarity = ImageSimilarityLoss(original_image=original_image, negate=True)
    user_prior = ModelBasedPrior(
        bounds=image_similarity.bounds,
        predict_func=image_similarity,
        minimize=False,
    )
    image_renderer = MatplotlibImageHumanEvaluatorRenderer(original_image)
    human_evaluator = HumanEvaluatorObjective(renderer=image_renderer, dim=image_similarity.dim, bounds=image_similarity._bounds)

    result_X, result_y, model = maximize(human_evaluator, user_prior=user_prior, num_trials=5, logger=logger)
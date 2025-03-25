import time
import webbrowser
import torch
from threading import Thread
from typing import Tuple
from modelbasedprior.objectives.human_evaluator.renderers.image_renderer.web_renderer.web_server import run_server, set_latest_image, reset_latest_image, get_latest_rating, reset_latest_rating, set_target_image
from modelbasedprior.objectives.image_similarity import generate_image

class WebImageHumanEvaluatorRenderer:
    def __init__(self, original_image: torch.Tensor, optimal_transformation: Tuple[float, ...] | None = None):
        self._original_image = original_image
        self._optimal_transformation = optimal_transformation
        self._target_image = generate_image(torch.tensor(self._optimal_transformation).view(1, 1, -1), self._original_image).squeeze() if self._optimal_transformation is not None else None
        self.server_thread = None
        self.browser_opened = False  # Track whether the browser was opened


    def start_server(self):
        """Run FastAPI server in a separate thread."""
        self.server_thread = Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(1)  # Give the server time to start

    def render(self, X: torch.Tensor) -> torch.Tensor:
        """Render an image, launch the web UI, and collect a human rating."""
        if self.server_thread is None:
            self.start_server()

        if self._target_image is not None:
            set_target_image(self._target_image.cpu().permute(1, 2, 0))

        transformed_images = generate_image(X.view(1, 1, -1), self._original_image)  # shape: n x q x C x H x W
        n, q, C, H, W = transformed_images.shape  # Extract shape

        ratings = torch.zeros(n, q, dtype=X.dtype)  # Store ratings

        for i in range(n):
            for j in range(q):
                img = transformed_images[i, j].cpu().permute(1, 2, 0)  # Convert (C, H, W) → (H, W, C)
                
                # Set the latest image (in memory, no file writing!)
                set_latest_image(img)

                # Reset the rating before asking for new input
                reset_latest_rating()

                # Open the UI
                if not self.browser_opened:  # Open browser only once
                    webbrowser.open("http://127.0.0.1:8000")
                    self.browser_opened = True

                # Wait for user input
                while get_latest_rating() is None:
                    time.sleep(0.5)  # Wait for rating to be submitted

                # Retrieve the rating
                rating = torch.tensor(get_latest_rating(), dtype=X.dtype)
                ratings[i, j] = rating

                reset_latest_rating()  # Clear rating for the next round
                reset_latest_image()

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
    image_renderer = WebImageHumanEvaluatorRenderer(original_image) # , optimal_transformation=(1.3, 1.3, 1.3, 0.1))
    human_evaluator = HumanEvaluatorObjective(renderer=image_renderer, dim=image_similarity.dim, bounds=image_similarity._bounds)

    result_X, result_y, model = maximize(human_evaluator, user_prior=user_prior, num_trials=5, logger=logger)
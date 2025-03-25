import os
import logging
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
from dotenv import load_dotenv
from torchvision.transforms.functional import resize
from modelbasedprior.prior import ModelBasedPrior
from modelbasedprior.objectives import HumanEvaluatorObjective
from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss, generate_image
from modelbasedprior.logger import setup_logger
from modelbasedprior.optimization.bo import maximize
from modelbasedprior.objectives.human_evaluator.renderers import WebImageHumanEvaluatorRenderer
from modelbasedprior.visualization.visualization import (
    make_grid,
    show_images,
)

load_dotenv()

def compute_best_X_y(X: torch.Tensor, y: torch.Tensor):
    # Compute cumulative maximum of y
    best_y, best_indices = torch.cummax(y, dim=0)
    
    # Select corresponding best X values
    best_X = X[best_indices.squeeze()]
    
    return best_X, best_y

def plot_image_similarity(X: torch.Tensor, original_image: torch.Tensor, target_image: torch.Tensor):
    candidate_images = generate_image(X.unsqueeze(0), original_image).squeeze()
    show_images(make_grid([original_image, target_image, *candidate_images]))

def plot_optimization_trace(
        ax: plt.Axes,
        y: torch.Tensor,
        num_initial_samples: int = 4,
        xlabel: str = r"$\text{Iteration } i$",
        ylabel: str = r"$\text{Observed } f_i(x)$",
    ) -> None:
    """
    Plot the observed function values over iterations.

    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        y (torch.Tensor): A 1D tensor of shape (n,), containing the observed y values per iteration.
        num_initial_samples (int): Number of initial random samples.
    """
    iterations = torch.arange(len(y))

    ax.plot(iterations, y, marker='o', linestyle='-', alpha=0.9)

    # Indicate initial samples
    ax.axvline(x=num_initial_samples - 0.5, color="darkgrey", linestyle="--", alpha=0.5)

    ax.set_xticks(iterations)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, linestyle="--", alpha=0.5)

ORIGINAL_IMAGE_PATH = os.getenv("ORIGINAL_IMAGE_PATH")
OPTIMAL_CONFIGURATION = (0.8, 1.2, 1.2, 0.1)  # brightness, contrast, saturation, hue
SEED = 23489
NUM_TRIALS = 2
NUM_INITIAL_SAMPLES = 2

original_image = read_image(ORIGINAL_IMAGE_PATH)
downsampled_original_image = resize(original_image, 64)

prior_predict_func = ImageSimilarityLoss(original_image=downsampled_original_image, optimizer=OPTIMAL_CONFIGURATION, weight_psnr=0.5, weight_ssim=0.5, negate=True)

user_prior = ModelBasedPrior(bounds=prior_predict_func.bounds, predict_func=prior_predict_func, minimize=False, seed=SEED)

logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

image_renderer = WebImageHumanEvaluatorRenderer(original_image, optimal_transformation=OPTIMAL_CONFIGURATION)
human_evaluator = HumanEvaluatorObjective(renderer=image_renderer, dim=prior_predict_func.dim, bounds=prior_predict_func._bounds)
# human_evaluator = prior_predict_func  # Mock

result_X, result_y, model = maximize(human_evaluator, user_prior=user_prior, num_trials=NUM_TRIALS, num_initial_samples=NUM_INITIAL_SAMPLES, logger=logger)

result_best_X, result_best_y = compute_best_X_y(result_X, result_y)

plot_image_similarity(result_X, original_image, image_renderer._target_image)
plot_image_similarity(result_best_X, original_image, image_renderer._target_image)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)
plot_optimization_trace(axes[0], num_initial_samples=NUM_INITIAL_SAMPLES, y=result_y, xlabel='', ylabel=r"$\text{Observed } f_i(x)$")
plot_optimization_trace(axes[1], num_initial_samples=NUM_INITIAL_SAMPLES, y=result_best_y, ylabel=r"$\text{Best Observed } f_i^*(x)$")
plt.show()
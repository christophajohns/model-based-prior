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
from modelbasedprior.objectives.image_aesthetics import ImageAestheticsLoss
from modelbasedprior.logger import setup_logger
from modelbasedprior.optimization.bo import maximize
from modelbasedprior.objectives.human_evaluator.renderers import WebImageHumanEvaluatorRenderer
from modelbasedprior.visualization.visualization import (
    make_grid,
    show_images,
)

load_dotenv()


# Configuration
SMOKE_TEST = True  # Set to True to use a synthetic function instead of an actual human evaluator
ORIGINAL_IMAGE_PATH = os.getenv("ORIGINAL_IMAGE_PATH")
OPTIMAL_CONFIGURATION = None # (0.8, 1.2, 1.2, 0.1)  # brightness, contrast, saturation, hue OR None
SEED = 23489
NUM_TRIALS = 5
NUM_INITIAL_SAMPLES = 4


# Helper functions
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
        **kwargs,
    ) -> None:
    """
    Plot the observed function values over iterations.

    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        y (torch.Tensor): A 1D tensor of shape (n,), containing the observed y values per iteration.
        num_initial_samples (int): Number of initial random samples.
    """
    iterations = torch.arange(len(y))

    ax.plot(iterations, y, marker='o', linestyle='-', alpha=0.9, **kwargs)

    # Indicate initial samples
    ax.axvline(x=num_initial_samples - 0.5, color="darkgrey", linestyle="--", alpha=0.5)

    ax.set_xticks(iterations)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, linestyle="--", alpha=0.5)


# Main script
original_image = read_image(ORIGINAL_IMAGE_PATH)
downsampled_original_image = resize(original_image, 64)

if OPTIMAL_CONFIGURATION is None:
    prior_predict_func = ImageAestheticsLoss(original_image=downsampled_original_image, negate=True)
else:
    prior_predict_func = ImageSimilarityLoss(original_image=downsampled_original_image, optimizer=OPTIMAL_CONFIGURATION, weight_psnr=0.5, weight_ssim=0.5, negate=True)

user_prior = ModelBasedPrior(bounds=prior_predict_func.bounds, predict_func=prior_predict_func, minimize=False, seed=SEED)

logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

if OPTIMAL_CONFIGURATION is None:
    image_renderer = WebImageHumanEvaluatorRenderer(original_image)
else:
    image_renderer = WebImageHumanEvaluatorRenderer(original_image, optimal_transformation=OPTIMAL_CONFIGURATION)

if SMOKE_TEST:
    human_evaluator = prior_predict_func  # Mock
else:
    human_evaluator = HumanEvaluatorObjective(renderer=image_renderer, dim=prior_predict_func.dim, bounds=prior_predict_func._bounds)

result_X, result_y, model = maximize(human_evaluator, user_prior=user_prior, num_trials=NUM_TRIALS, num_initial_samples=NUM_INITIAL_SAMPLES, logger=logger)

result_best_X, result_best_y = compute_best_X_y(result_X, result_y)

if OPTIMAL_CONFIGURATION is None:
    # Without target image
    plot_image_similarity(result_X, original_image, original_image)
    plot_image_similarity(result_best_X, original_image, original_image)
else:
    # For target image
    plot_image_similarity(result_X, original_image, image_renderer._target_image)
    plot_image_similarity(result_best_X, original_image, image_renderer._target_image)

prior_predict_y = prior_predict_func(result_X)
prior_predict_best_y = prior_predict_func(result_best_X)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True)
plot_optimization_trace(axes[0][0], num_initial_samples=NUM_INITIAL_SAMPLES, y=result_y, xlabel='', ylabel=r"$\text{Observed } f(x_i)$")
plot_optimization_trace(axes[1][0], num_initial_samples=NUM_INITIAL_SAMPLES, y=result_best_y, ylabel=r"$\text{Best Observed } f(x_i^*)$")
plot_optimization_trace(axes[0][1], num_initial_samples=NUM_INITIAL_SAMPLES, y=prior_predict_y, xlabel='', ylabel=r"$\text{Prior Observed } \hat{f}(x_i)$", color='darkorange')
plot_optimization_trace(axes[1][1], num_initial_samples=NUM_INITIAL_SAMPLES, y=prior_predict_best_y, ylabel=r"$\text{Prior Best Observed } \hat{f}(x_i^*)$", color='darkorange')
fig.tight_layout()
plt.show()

print("\nSample Parameters (result_X shape", result_X.shape, "):")
print(result_X)

print("\nSample Scores (result_y shape", result_y.shape, "):")
print(result_y)

print("\nBest Observed Parameters (result_best_X shape", result_best_X.shape, "):")
print(result_best_X)

print("\nBest Observed Scores (result_best_y shape", result_best_y.shape, "):")
print(result_best_y)
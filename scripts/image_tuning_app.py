import os
import logging
import torch
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from botorch.utils.transforms import unnormalize
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
from torch.quasirandom import SobolEngine

load_dotenv()


# Configuration
SMOKE_TEST = True  # Set to True to use a synthetic function instead of an actual human evaluator
ORIGINAL_IMAGE_PATH = os.getenv("ORIGINAL_IMAGE_PATH")
ORIGINAL_MODIFICATION = (1., 1., 1., -0.)
OPTIMAL_CONFIGURATION = None  # (0.8, 1.2, 1.2, 0.1)  # brightness, contrast, saturation, hue OR None
SEED = 23489
NUM_ANCHORING_SAMPLES = 6  # Number of random samples for anchoring mitigation
NUM_TRIALS = 5
NUM_INITIAL_SAMPLES = 4
IAL_K_LEVELS = 6  # Image Aesthetics Loss K-levels (original: 8; smaller is faster)
IAL_ITERATIONS = 3  # Image Aesthetics Loss Domain Transform Filter Iterations (original: 5; smaller is faster)
DOWNSAMPLING_SIZE = 64


# Helper functions
def compute_best_X_y(X: torch.Tensor, y: torch.Tensor):
    best_y, best_indices = torch.cummax(y, dim=0)
    best_X = X[best_indices.squeeze()]
    return best_X, best_y


def plot_image_similarity(X: torch.Tensor, original_image: torch.Tensor, modified_image: torch.Tensor, target_image: torch.Tensor):
    candidate_images = generate_image(X.unsqueeze(0), modified_image).squeeze()
    show_images(make_grid([original_image, modified_image, target_image, *candidate_images]))


def plot_optimization_trace(
        ax: plt.Axes,
        y: torch.Tensor,
        num_initial_samples: int = 4,
        xlabel: str = r"$\text{Iteration } i$",
        ylabel: str = r"$\text{Observed } f_i(x)$",
        **kwargs,
    ) -> None:
    iterations = torch.arange(len(y))
    ax.plot(iterations, y, marker='o', linestyle='-', alpha=0.9, **kwargs)
    ax.axvline(x=num_initial_samples - 0.5, color="darkgrey", linestyle="--", alpha=0.5)
    ax.set_xticks(iterations)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5)


# --- Main script ---
torch.manual_seed(SEED) # Set seed for reproducibility

logger = setup_logger(level=logging.INFO)

original_image = read_image(ORIGINAL_IMAGE_PATH)
downsampled_original_image = resize(original_image, DOWNSAMPLING_SIZE)

# Modify the original image before optimization
modified_image = generate_image(torch.tensor([[ORIGINAL_MODIFICATION]]), original_image).squeeze()
downsampled_modified_image = resize(modified_image, DOWNSAMPLING_SIZE)

# Determine prior function
if OPTIMAL_CONFIGURATION is None:
    prior_predict_func = ImageAestheticsLoss(original_image=downsampled_modified_image, negate=True, k_levels=IAL_K_LEVELS, domain_transform_iterations=IAL_ITERATIONS)
else:
    # Assuming OPTIMAL_CONFIGURATION is a tuple/list, convert to tensor
    optimal_config_tensor = torch.tensor([[OPTIMAL_CONFIGURATION]], dtype=torch.float32)
    target_image = generate_image(optimal_config_tensor, modified_image).squeeze()
    downsampled_target_image = resize(target_image, DOWNSAMPLING_SIZE)
    prior_predict_func = ImageSimilarityLoss(reference_image=downsampled_target_image, original_image=downsampled_modified_image, weight_psnr=0.5, weight_ssim=0.5, negate=True)


bounds = prior_predict_func.bounds
dim = prior_predict_func.dim

user_prior = ModelBasedPrior(bounds=bounds, predict_func=prior_predict_func, minimize=False, seed=SEED)


# Create renderer and human evaluator objective
if OPTIMAL_CONFIGURATION is None:
    image_renderer = WebImageHumanEvaluatorRenderer(modified_image)
    target_image_for_plotting = modified_image # For plotting later, show modified as "target"
else:
    # We already generated the target_image above
    image_renderer = WebImageHumanEvaluatorRenderer(modified_image, optimal_transformation=OPTIMAL_CONFIGURATION)
    target_image_for_plotting = target_image # For plotting later

if SMOKE_TEST:
    human_evaluator = prior_predict_func  # Mock
else:
    # Pass bounds directly if HumanEvaluatorObjective expects them separately
    human_evaluator = HumanEvaluatorObjective(renderer=image_renderer, dim=dim, bounds=bounds.t())


# Anchoring Mitigation
if not SMOKE_TEST and NUM_ANCHORING_SAMPLES > 0:
    logger.info(f"--- Starting Anchoring Mitigation Phase ({NUM_ANCHORING_SAMPLES} samples) ---")
    logger.info("You will now rate a few random image modifications to get familiar with the range.")
    logger.info("These ratings WILL NOT be used for the optimization process.")

    # Generate random samples within bounds
    sobol_engine = SobolEngine(dimension=dim, scramble=True, seed=SEED)
    # Draw enough points, considering potential burn-in if needed, though not strictly necessary here
    anchoring_samples_unit = sobol_engine.draw(NUM_ANCHORING_SAMPLES)

    anchoring_samples = unnormalize(anchoring_samples_unit, bounds)
    anchoring_ratings = []

    for i in range(NUM_ANCHORING_SAMPLES):
        sample_to_eval = anchoring_samples[i].unsqueeze(0) # Add batch dimension
        logger.info(f"Presenting anchoring sample {i+1}/{NUM_ANCHORING_SAMPLES}...")
        # Call the human evaluator - this triggers the UI and waits for user input
        # We store the rating just for potential logging, but don't use it further.
        # Assuming human_evaluator returns a tensor score
        rating = human_evaluator(sample_to_eval)
        anchoring_ratings.append(rating.item()) # Store scalar rating
        logger.info(f"Received anchoring rating {i+1}/{NUM_ANCHORING_SAMPLES}: {rating.item():.4f}")

    logger.info("--- Anchoring Mitigation Phase Complete ---")
    # Optional: Print the collected anchoring ratings
    # logger.info(f"Anchoring ratings (for info only): {anchoring_ratings}")


# Proceed with the actual Bayesian Optimization
logger.info(f"--- Starting Bayesian Optimization ({NUM_TRIALS} trials, {NUM_INITIAL_SAMPLES} initial) ---")
result_X, result_y, model = maximize(human_evaluator, user_prior=user_prior, num_trials=NUM_TRIALS, num_initial_samples=NUM_INITIAL_SAMPLES, logger=logger)

logger.info("--- Bayesian Optimization Complete ---")

# --- Post-processing and Plotting ---
result_best_X, result_best_y = compute_best_X_y(result_X, result_y)

logger.info("Plotting results...")
# Ensure result_X and result_best_X are 2D for plotting
if result_X.ndim == 1: result_X = result_X.unsqueeze(0)
if result_best_X.ndim == 1: result_best_X = result_best_X.unsqueeze(0)

plot_image_similarity(result_X, original_image, modified_image, target_image_for_plotting)
plot_image_similarity(result_best_X, original_image, modified_image, target_image_for_plotting)


# Ensure result_X and result_best_X are 2D for prior evaluation
prior_predict_y = prior_predict_func(result_X)
prior_predict_best_y = prior_predict_func(result_best_X)

# Make sure prior predictions are 1D tensors for plotting
if prior_predict_y.ndim == 0: prior_predict_y = prior_predict_y.unsqueeze(0)
if prior_predict_best_y.ndim == 0: prior_predict_best_y = prior_predict_y.unsqueeze(0)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True)
plot_optimization_trace(axes[0][0], num_initial_samples=NUM_INITIAL_SAMPLES, y=result_y, xlabel='', ylabel=r"$\text{Observed } f(x_i)$")
axes[0][0].set_title("Human Ratings")
plot_optimization_trace(axes[1][0], num_initial_samples=NUM_INITIAL_SAMPLES, y=result_best_y, ylabel=r"$\text{Best Observed } f(x_i^*)$")
plot_optimization_trace(axes[0][1], num_initial_samples=NUM_INITIAL_SAMPLES, y=prior_predict_y, xlabel='', ylabel=r"$\text{Prior Predicted } \hat{f}(x_i)$", color='darkorange')
axes[0][1].set_title("Prior Model Predictions")
plot_optimization_trace(axes[1][1], num_initial_samples=NUM_INITIAL_SAMPLES, y=prior_predict_best_y, ylabel=r"$\text{Prior Best Predicted } \hat{f}(x_i^*)$", color='darkorange')
fig.suptitle("Optimization Traces", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
plt.show()

print("\n--- Final Results ---")
print(f"\nSample Parameters (result_X shape {result_X.shape}):")
print(result_X.cpu().numpy()) # Use .cpu().numpy() for printing

print(f"\nSample Scores (result_y shape {result_y.shape}):")
print(result_y.cpu().numpy()) # Use .cpu().numpy() for printing

print(f"\nBest Observed Parameters (result_best_X shape {result_best_X.shape}):")
# Handle potential multiple best points (if cummax returns multiple indices for the same max value)
# Print only the last one if multiple exist for simplicity, or handle as needed
print(result_best_X[-1].cpu().numpy() if result_best_X.shape[0] > 0 else "N/A")

print(f"\nBest Observed Scores (result_best_y shape {result_best_y.shape}):")
print(result_best_y[-1].cpu().numpy() if result_best_y.shape[0] > 0 else "N/A")

print("\n--- End of Script ---")
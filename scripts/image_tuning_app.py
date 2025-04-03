#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Bayesian Image Tuning Script

This script provides an interactive interface for tuning multiple images using
Bayesian optimization with human feedback. It processes images from a specified
directory, applies random initial transformations, and allows users to evaluate
and optimize the transformations through an interactive UI.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

# Import project-specific modules
from modelbasedprior.prior import ModelBasedPrior
from modelbasedprior.objectives import HumanEvaluatorObjective
from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss, generate_image
from modelbasedprior.objectives.image_aesthetics import ImageAestheticsLoss
from modelbasedprior.logger import setup_logger
from modelbasedprior.optimization.bo import maximize
from modelbasedprior.objectives.human_evaluator.renderers import WebImageHumanEvaluatorRenderer
from modelbasedprior.visualization.visualization import make_grid, show_images

# Default values
SMOKE_TEST = False  # Set to True to use a synthetic function instead of an actual human evaluator
AVA_FLOWERS_DIR = os.getenv("AVA_FLOWERS_DIR")  # image directory
IMAGE_IDS = ["43405", "117679", "189197"]  # Example IDs
OPTIMAL_CONFIGURATION = None  # (0.8, 1.2, 1.2, 0.1)  # brightness, contrast, saturation, hue OR None
SEED = 23489
NUM_ANCHORING_SAMPLES = 2  # Number of random samples for anchoring mitigation
NUM_TRIALS = 2
NUM_INITIAL_SAMPLES = 2
IAL_K_LEVELS = 6  # Image Aesthetics Loss K-levels (original: 8; smaller is faster)
IAL_ITERATIONS = 3  # Image Aesthetics Loss Domain Transform Filter Iterations (original: 5; smaller is faster)
DOWNSAMPLING_SIZE = 64

# Configure logging
logger = logging.getLogger(__name__)


class ImageTuningConfig:
    """Configuration for image tuning parameters."""
    
    def __init__(self, **kwargs):
        """Initialize with default parameters, overridden by any provided kwargs."""
        # Default configuration
        self.smoke_test = SMOKE_TEST
        self.seed = SEED
        self.num_anchoring_samples = NUM_ANCHORING_SAMPLES
        self.num_trials = NUM_TRIALS
        self.num_initial_samples = NUM_INITIAL_SAMPLES
        self.ial_k_levels = IAL_K_LEVELS
        self.ial_iterations = IAL_ITERATIONS
        self.downsampling_size = DOWNSAMPLING_SIZE
        self.optimal_configuration = OPTIMAL_CONFIGURATION  # Optional target configuration
        
        # Override defaults with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
                
    @classmethod
    def from_args(cls, args):
        """Create configuration from command line arguments."""
        return cls(
            smoke_test=args.smoke_test,
            seed=args.seed,
            num_anchoring_samples=args.num_anchoring_samples,
            num_trials=args.num_trials,
            num_initial_samples=args.num_initial_samples,
            ial_k_levels=args.ial_k_levels,
            ial_iterations=args.ial_iterations,
            downsampling_size=args.downsampling_size
        )


class ImageTuner:
    """Main class for tuning images using Bayesian optimization with human feedback."""
    
    def __init__(self, config: ImageTuningConfig):
        """Initialize the image tuner with the given configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.config.seed)
        
    @staticmethod
    def get_image_paths(directory: str, image_ids: List[str]) -> List[str]:
        """Get valid image paths from directory based on image IDs."""
        image_paths = []
        for img_id in image_ids:
            path = os.path.join(directory, f"{img_id}.jpg")
            if os.path.exists(path):
                image_paths.append(path)
            else:
                logger.warning(f"Image {img_id}.jpg not found in {directory}")
        return image_paths
    
    @staticmethod
    def get_random_modification(bounds_dict: Optional[Dict] = None) -> Tuple[float, float, float, float]:
        """Generate random modification parameters within bounds using torch."""
        if bounds_dict is None:
            bounds_dict = {
                'brightness': ImageAestheticsLoss._brightness_bounds,
                'contrast': ImageAestheticsLoss._contrast_bounds,
                'saturation': ImageAestheticsLoss._saturation_bounds,
                'hue': ImageAestheticsLoss._hue_bounds
            }
        
        brightness = torch.FloatTensor(1).uniform_(*bounds_dict['brightness']).item()
        contrast = torch.FloatTensor(1).uniform_(*bounds_dict['contrast']).item()
        saturation = torch.FloatTensor(1).uniform_(*bounds_dict['saturation']).item()
        hue = torch.FloatTensor(1).uniform_(*bounds_dict['hue']).item()
        
        return (brightness, contrast, saturation, hue)
    
    @staticmethod
    def compute_best_X_y(X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the best parameters and scores seen so far at each iteration."""
        best_y, best_indices = torch.cummax(y, dim=0)
        best_X = X[best_indices.squeeze()]
        return best_X, best_y
    
    @staticmethod
    def plot_image_similarity(X: torch.Tensor, original_image: torch.Tensor, 
                             modified_image: torch.Tensor, target_image: torch.Tensor):
        """Plot image similarity visualization."""
        candidate_images = generate_image(X.unsqueeze(0).unsqueeze(0) if X.dim() == 1 else X.unsqueeze(0), modified_image).squeeze()
        show_images(make_grid([original_image, modified_image, target_image, *candidate_images]))
    
    @staticmethod
    def plot_optimization_trace(
            ax: plt.Axes,
            y: torch.Tensor,
            num_initial_samples: int = 4,
            xlabel: str = r"$\text{Iteration } i$",
            ylabel: str = r"$\text{Observed } f_i(x)$",
            **kwargs,
        ) -> None:
        """Plot optimization trace on the given axes."""
        iterations = torch.arange(len(y))
        ax.plot(iterations, y, marker='o', linestyle='-', alpha=0.9, **kwargs)
        ax.axvline(x=num_initial_samples - 0.5, color="darkgrey", linestyle="--", alpha=0.5)
        ax.set_xticks(iterations)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.5)
    
    def process_single_image(self, image_path: str, random_modification: Optional[Tuple] = None) -> Optional[torch.Tensor]:
        """Process a single image with optimization."""
        logger.info(f"=== Processing image: {image_path} ===")
        
        try:
            # Load and preprocess image
            original_image = read_image(image_path, mode=ImageReadMode.RGB).to(self.device)
            downsampled_original_image = resize(original_image, self.config.downsampling_size)
            
            # Use random modification if provided, otherwise generate one
            if random_modification is None:
                random_modification = self.get_random_modification()
            
            logger.info(f"Using initial modification: {random_modification}")
            
            # Modify the original image before optimization
            random_mod_tensor = torch.tensor([[random_modification]], dtype=torch.float32, device=self.device)
            modified_image = generate_image(random_mod_tensor, original_image).squeeze()
            downsampled_modified_image = resize(modified_image, self.config.downsampling_size)
            
            # Determine prior function
            if self.config.optimal_configuration is None:
                prior_predict_func = ImageAestheticsLoss(
                    original_image=downsampled_modified_image, 
                    negate=True, 
                    k_levels=self.config.ial_k_levels, 
                    domain_transform_iterations=self.config.ial_iterations
                ).to(self.device)
                target_image_for_plotting = modified_image
            else:
                # Using a target configuration if provided
                optimal_config_tensor = torch.tensor(
                    [[self.config.optimal_configuration]], 
                    dtype=torch.float32, 
                    device=self.device
                )
                target_image = generate_image(optimal_config_tensor, modified_image).squeeze()
                downsampled_target_image = resize(target_image, self.config.downsampling_size)
                prior_predict_func = ImageSimilarityLoss(
                    reference_image=downsampled_target_image, 
                    original_image=downsampled_modified_image, 
                    weight_psnr=0.5, 
                    weight_ssim=0.5, 
                    negate=True
                ).to(self.device)
                target_image_for_plotting = target_image
            
            bounds = prior_predict_func.bounds
            dim = prior_predict_func.dim
            
            user_prior = ModelBasedPrior(
                bounds=bounds, 
                predict_func=prior_predict_func, 
                minimize=False, 
                seed=self.config.seed
            )
            
            # Create renderer and human evaluator objective
            image_renderer = WebImageHumanEvaluatorRenderer(
                modified_image, 
                optimal_transformation=self.config.optimal_configuration
            )
            
            if self.config.smoke_test:
                human_evaluator = prior_predict_func  # Use mock evaluator for testing
            else:
                human_evaluator = HumanEvaluatorObjective(
                    renderer=image_renderer, 
                    dim=dim, 
                    bounds=bounds.t()
                )
            
            # Anchoring Mitigation
            if not self.config.smoke_test and self.config.num_anchoring_samples > 0:
                self._run_anchoring_mitigation(human_evaluator, bounds, dim)
            
            # Run Bayesian Optimization
            logger.info(f"--- Starting Bayesian Optimization ({self.config.num_trials} trials, "
                       f"{self.config.num_initial_samples} initial) ---")
            result_X, result_y, model = maximize(
                human_evaluator, 
                user_prior=user_prior, 
                num_trials=self.config.num_trials, 
                num_initial_samples=self.config.num_initial_samples, 
                logger=logger
            )
            
            logger.info("--- Bayesian Optimization Complete ---")
            
            # Post-processing and visualization
            self._visualize_results(
                result_X, result_y, prior_predict_func, 
                original_image, modified_image, target_image_for_plotting, 
                image_path
            )
            
            # Return the best configuration
            result_best_X, result_best_y = self.compute_best_X_y(result_X, result_y)
            return result_best_X[-1] if result_best_X.shape[0] > 0 else None
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _run_anchoring_mitigation(self, human_evaluator, bounds, dim):
        """Run anchoring mitigation phase with random samples."""
        logger.info(f"--- Starting Anchoring Mitigation Phase ({self.config.num_anchoring_samples} samples) ---")
        logger.info("You will now rate a few random image modifications to get familiar with the range.")
        logger.info("These ratings WILL NOT be used for the optimization process.")
        
        # Generate random samples within bounds
        sobol_engine = SobolEngine(dimension=dim, scramble=True, seed=self.config.seed)
        anchoring_samples_unit = sobol_engine.draw(self.config.num_anchoring_samples)
        anchoring_samples = unnormalize(anchoring_samples_unit, bounds)
        
        for i in range(self.config.num_anchoring_samples):
            sample_to_eval = anchoring_samples[i].unsqueeze(0)  # Add batch dimension
            logger.info(f"Presenting anchoring sample {i+1}/{self.config.num_anchoring_samples}...")
            try:
                rating = human_evaluator(sample_to_eval)
                logger.info(f"Received anchoring rating {i+1}/{self.config.num_anchoring_samples}: {rating.item():.4f}")
            except Exception as e:
                logger.error(f"Error during anchoring sample {i+1}: {str(e)}")
        
        logger.info("--- Anchoring Mitigation Phase Complete ---")
    
    def _visualize_results(self, result_X, result_y, prior_predict_func, 
                          original_image, modified_image, target_image, image_path):
        """Create visualizations of optimization results."""
        logger.info("Generating result visualizations...")
        
        # Get best results
        result_best_X, result_best_y = self.compute_best_X_y(result_X, result_y)
        
        # Ensure tensors have proper dimensions for plotting
        if result_X.ndim == 1: 
            result_X = result_X.unsqueeze(0)
        if result_best_X.ndim == 1: 
            result_best_X = result_best_X.unsqueeze(0)
        
        # Plot image comparisons
        self.plot_image_similarity(result_X, original_image, modified_image, target_image)
        self.plot_image_similarity(result_best_X, original_image, modified_image, target_image)
        
        # Get prior predictions
        prior_predict_y = prior_predict_func(result_X)
        prior_predict_best_y = prior_predict_func(result_best_X)
        
        # Ensure proper dimensions for plotting
        if prior_predict_y.ndim == 0: 
            prior_predict_y = prior_predict_y.unsqueeze(0)
        if prior_predict_best_y.ndim == 0: 
            prior_predict_best_y = prior_predict_y.unsqueeze(0)
        
        # Create optimization trace plots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True)
        self.plot_optimization_trace(
            axes[0][0], 
            num_initial_samples=self.config.num_initial_samples, 
            y=result_y, 
            xlabel='', 
            ylabel=r"$\text{Observed } f(x_i)$"
        )
        axes[0][0].set_title("Human Ratings")
        
        self.plot_optimization_trace(
            axes[1][0], 
            num_initial_samples=self.config.num_initial_samples, 
            y=result_best_y, 
            ylabel=r"$\text{Best Observed } f(x_i^*)$"
        )
        
        self.plot_optimization_trace(
            axes[0][1], 
            num_initial_samples=self.config.num_initial_samples, 
            y=prior_predict_y, 
            xlabel='', 
            ylabel=r"$\text{Prior Predicted } \hat{f}(x_i)$", 
            color='darkorange'
        )
        axes[0][1].set_title("Prior Model Predictions")
        
        self.plot_optimization_trace(
            axes[1][1], 
            num_initial_samples=self.config.num_initial_samples, 
            y=prior_predict_best_y, 
            ylabel=r"$\text{Prior Best Predicted } \hat{f}(x_i^*)$", 
            color='darkorange'
        )
        
        fig.suptitle(f"Optimization Traces - {os.path.basename(image_path)}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        
        # Print summary
        image_basename = os.path.basename(image_path)
        print(f"\n--- Final Results for {image_basename} ---")
        print(f"\nInitial Modification Parameters:")
        for i, param in enumerate(['brightness', 'contrast', 'saturation', 'hue']):
            try:
                print(f"  {param}: {result_X[0][i].item():.4f}")
            except:
                pass
        
        print(f"\nBest Parameters Found:")
        best_idx = result_best_y.argmax().item()
        for i, param in enumerate(['brightness', 'contrast', 'saturation', 'hue']):
            try:
                print(f"  {param}: {result_best_X[best_idx][i].item():.4f}")
            except:
                pass
        
        print(f"\nBest Score: {result_best_y.max().item():.4f}")
    
    def process_images(self, image_dir: str, image_ids: List[str]) -> None:
        """Process multiple images in sequence."""
        image_paths = self.get_image_paths(image_dir, image_ids)
        
        if not image_paths:
            logger.error("No valid image paths found. Please check the directory and image IDs.")
            return
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"\n\n=== Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ===")
            
            # Generate random modification parameters
            random_mod = self.get_random_modification()
            
            # Process the image
            try:
                best_config = self.process_single_image(image_path, random_mod)
                
                # Add a separator between images
                if i < len(image_paths) - 1:
                    input("\nPress Enter to continue to the next image...")
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user. Exiting...")
                break
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                if i < len(image_paths) - 1:
                    proceed = input("\nError occurred. Press Enter to continue to the next image or 'q' to quit: ")
                    if proceed.lower() == 'q':
                        break
        
        logger.info("\n--- All Images Processed ---")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Bayesian Image Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--image-dir", 
        type=str,
        default=AVA_FLOWERS_DIR,
        help="Directory containing images to process"
    )
    
    parser.add_argument(
        "--image-ids", 
        type=str, 
        nargs="+",
        default=IMAGE_IDS,
        help="List of image IDs to process (without file extension)"
    )
    
    parser.add_argument(
        "--smoke-test", 
        action="store_true",
        default=SMOKE_TEST,
        help="Run in smoke test mode (use synthetic evaluator)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=SEED,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--num-anchoring-samples", 
        type=int, 
        default=NUM_ANCHORING_SAMPLES,
        help="Number of random samples for anchoring mitigation"
    )
    
    parser.add_argument(
        "--num-trials", 
        type=int, 
        default=NUM_TRIALS,
        help="Number of optimization trials to run"
    )
    
    parser.add_argument(
        "--num-initial-samples", 
        type=int, 
        default=NUM_INITIAL_SAMPLES,
        help="Number of initial samples for Bayesian optimization"
    )
    
    parser.add_argument(
        "--ial-k-levels", 
        type=int, 
        default=IAL_K_LEVELS,
        help="Image Aesthetics Loss K-levels"
    )
    
    parser.add_argument(
        "--ial-iterations", 
        type=int, 
        default=IAL_ITERATIONS,
        help="Image Aesthetics Loss Domain Transform Filter Iterations"
    )
    
    parser.add_argument(
        "--downsampling-size", 
        type=int, 
        default=DOWNSAMPLING_SIZE,
        help="Size to downsample images to for internal processing"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logger(level=log_level)
    
    try:
        # Create configuration from args
        config = ImageTuningConfig.from_args(args)
        
        # Create tuner and process images
        tuner = ImageTuner(config)
        tuner.process_images(args.image_dir, args.image_ids)
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
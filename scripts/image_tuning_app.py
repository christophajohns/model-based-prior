#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Bayesian Image Tuning Script

This script provides an interactive interface for tuning multiple images using
Bayesian optimization with human feedback. It processes images from a specified
directory, applies random initial transformations, and allows users to evaluate
and optimize the transformations through an interactive UI. Saves the data for
each participant.
"""

import os
import sys
import logging
import argparse
import json
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
DEFAULT_IMAGE_DIR = os.getenv("AVA_FLOWERS_DIR")  # image directory
IMAGE_IDS = ["43405", "117679", "189197", "953980", "735492"]  # Example IDs
DEFAULT_SAVE_DIR = os.getenv("IMAGE_TUNING_SAVE_DIR", "./image_tuning_results") # Resolve env var or use default "./image_tuning_results"
DEFAULT_PARTICIPANT_ID = 99999
OPTIMAL_CONFIGURATION = None  # (0.8, 1.2, 1.2, 0.1)  # brightness, contrast, saturation, hue OR None
SEED = 23489
NUM_ANCHORING_SAMPLES = 3  # Number of random samples for anchoring mitigation
NUM_TRIALS = 5
NUM_INITIAL_SAMPLES = 4
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
        self.participant_id = DEFAULT_PARTICIPANT_ID
        self.save_path = None # Will be set externally based on args/defaults
        
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
            downsampling_size=args.downsampling_size,
            participant_id=args.participant_id,
            save_path=args.save_path # Pass the final determined path from main/args
        )
    
    def to_dict(self):
        """Convert configuration to a dictionary for saving."""
        # Simple conversion, assumes all attributes are serializable or okay to represent as is
        # Convert tensors or other complex types if necessary here
        config_dict = self.__dict__.copy()
        # Example: convert optimal_configuration tuple if it exists
        if isinstance(config_dict.get('optimal_configuration'), tuple):
             config_dict['optimal_configuration'] = list(config_dict['optimal_configuration'])
        return config_dict


class ImageTuner:
    """Main class for tuning images using Bayesian optimization with human feedback."""
    
    def __init__(self, config: ImageTuningConfig):
        """Initialize the image tuner with the given configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.config.seed)

        if self.config.save_path:
            try:
                os.makedirs(self.config.save_path, exist_ok=True)
                logger.info(f"Ensured save directory exists: {self.config.save_path}")

                # Save the configuration file for this run
                config_filename = f"config_p{self.config.participant_id}_run.json"
                config_filepath = os.path.join(self.config.save_path, config_filename)
                try:
                    with open(config_filepath, 'w') as f:
                        json.dump(self.config.to_dict(), f, indent=4)
                    logger.info(f"Run configuration saved to {config_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save configuration file: {e}")

            except OSError as e:
                logger.error(f"Failed to create save directory {self.config.save_path}: {e}. Saving will be disabled.")
                self.config.save_path = None # Disable saving if directory fails
        else:
            logger.warning("No save path provided in configuration. Results will not be saved.")
        
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
            result_best_X, result_best_y = self.compute_best_X_y(result_X, result_y)
            best_final_config = result_best_X[-1] if result_best_X.shape[0] > 0 else None

            self._save_run_data(
                image_path=image_path,
                initial_modification=random_modification,
                result_X=result_X,
                result_y=result_y,
                result_best_X_trace=result_best_X,
                result_best_y_trace=result_best_y,
                best_final_config=best_final_config
            )

            self._visualize_results(
                result_X, result_y, prior_predict_func, 
                original_image, modified_image, target_image_for_plotting, 
                image_path
            )

            return best_final_config # Return the best config found
            
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

        # Set the renderer to training
        def set_renderer_training_mode(new_value: bool):
            if hasattr(human_evaluator.renderer, "is_training"):
                human_evaluator.renderer.is_training = new_value
        set_renderer_training_mode(True)
        
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
                set_renderer_training_mode(False)

        # Set the renderer back
        set_renderer_training_mode(False)
        
        logger.info("--- Anchoring Mitigation Phase Complete ---")
    
    def _visualize_results(self, result_X, result_y, prior_predict_func, 
                          original_image, modified_image, target_image, image_path):
        """Create visualizations of optimization results."""
        logger.info("Generating result visualizations...")
        image_basename = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_basename)[0]
        save_plots = self.config.save_path is not None # Check if saving is enabled

        image_specific_save_dir = None
        if save_plots:
            # self.config.save_path is the participant directory
            image_specific_save_dir = os.path.join(self.config.save_path, image_name_no_ext)
            # Ensure directory exists (might be redundant if _save_run_data ran, but safe)
            try:
                os.makedirs(image_specific_save_dir, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create image-specific save directory {image_specific_save_dir} for plots: {e}. Plot saving disabled.")
                save_plots = False # Disable saving if dir creation fails here
        
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

        if save_plots:
            plot_filename = f"plot_traces_{image_name_no_ext}_p{self.config.participant_id}.png"
            save_plot_path = os.path.join(image_specific_save_dir, plot_filename)
            try:
                fig.savefig(save_plot_path)
                logger.info(f"Trace plot saved to {save_plot_path}")
            except Exception as e:
                logger.error(f"Failed to save trace plot: {e}")

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

    def _save_run_data(self, image_path: str, initial_modification: Tuple,
                       result_X: Optional[torch.Tensor], result_y: Optional[torch.Tensor],
                       result_best_X_trace: Optional[torch.Tensor], result_best_y_trace: Optional[torch.Tensor],
                       best_final_config: Optional[torch.Tensor]) -> None:
        """Saves the results of a single image optimization run."""
        if not self.config.save_path:
            logger.debug("Save path not configured, skipping saving run data.")
            return

        image_basename = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_basename)[0]

        image_specific_save_dir = os.path.join(self.config.save_path, image_name_no_ext)
        try:
            os.makedirs(image_specific_save_dir, exist_ok=True) # Create dir if not exists
        except OSError as e:
            logger.error(f"Failed to create image-specific save directory {image_specific_save_dir}: {e}. Skipping save.")
            return

        save_filename = f"results_{image_name_no_ext}_p{self.config.participant_id}.pt"
        save_filepath = os.path.join(image_specific_save_dir, save_filename)

        data_to_save = {
            'image_path': image_path,
            'initial_random_modification': initial_modification,
             # Save tensors on CPU to avoid device issues when loading later
            'result_X': result_X.cpu() if result_X is not None else None,
            'result_y': result_y.cpu() if result_y is not None else None,
            'result_best_X_trace': result_best_X_trace.cpu() if result_best_X_trace is not None else None,
            'result_best_y_trace': result_best_y_trace.cpu() if result_best_y_trace is not None else None,
            'best_final_config': best_final_config.cpu() if best_final_config is not None else None,
            # Optionally include config again, though it's saved separately
            # 'config_snapshot': self.config.to_dict()
        }

        try:
            torch.save(data_to_save, save_filepath)
            logger.info(f"Run data saved to {save_filepath}")
        except Exception as e:
            logger.error(f"Failed to save run data for {image_path} to {save_filepath}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
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
        default=DEFAULT_IMAGE_DIR,
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
        "--participant-id",
        type=int,
        default=DEFAULT_PARTICIPANT_ID,
        help="Identifier for the current participant."
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None, # Default to None, logic in main will handle env var and final default
        help=f"Base directory to save results. If not set, uses IMAGE_TUNING_SAVE_DIR env var, then defaults to '{DEFAULT_SAVE_DIR}'. A subdirectory 'participant_<ID>' will be created."
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
    
    args = parser.parse_args()

    if not args.image_dir:
         parser.error("--image-dir is required if AVA_FLOWERS_DIR environment variable is not set.")

    return args


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logger(level=log_level)
    
    try:
        # Determine Final Save Path
        base_save_dir = args.save_dir if args.save_dir is not None else DEFAULT_SAVE_DIR
        # Construct the full path including the participant ID subdir
        participant_save_path = os.path.join(base_save_dir, f"participant_{args.participant_id}")
        args.save_path = participant_save_path # Add the determined path to args temporarily for from_args

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
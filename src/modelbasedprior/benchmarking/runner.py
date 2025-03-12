"""Benchmark runner for the synthetic evaluation."""

import os
import numpy as np
import torch
import logging
from typing import Literal, List
from dataclasses import dataclass
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP
from botorch.test_functions.synthetic import Hartmann, Shekel
from botorch.acquisition.prior_monte_carlo import qPriorExpectedImprovement, qPriorLogExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.user_prior import PiBO
from dotenv import load_dotenv
from torchvision.io import read_image
from torchvision.transforms.functional import resize
import seaborn as sns

from modelbasedprior.benchmarking.database import (
    create_database,
    get_database,
)
from modelbasedprior.optimization.pbo import maximize as pbo_maximize
from modelbasedprior.optimization.bo import maximize as bo_maximize
from modelbasedprior.optimization.prior_sampling import maximize as prior_sampling_maximize
from modelbasedprior.objectives.sphere import Sphere
from modelbasedprior.objectives.shekel import ShekelNoGlobal, Shekel2D, Shekel2DNoGlobal
from modelbasedprior.objectives.image_similarity import ImageSimilarityLoss
from modelbasedprior.objectives.scatterplot_quality import ScatterPlotQualityLoss
from modelbasedprior.objectives.mr_layout_quality import MRLayoutQualityLoss
from modelbasedprior.prior import get_default_prior, get_model_based_prior

load_dotenv()

def save_model(gp: SingleTaskGP | PairwiseGP, path: str):
    torch.save(gp.state_dict(), path)

def pibo_factory(**kwargs):
    """
    Creates an instance of the PiBO class with specified parameters.
    Parameters:
    **kwargs: Arbitrary keyword arguments.
        - user_prior (optional): User-defined prior.
        - decay_beta (optional): Decay rate for the beta parameter.
        - prior_floor (optional): Minimum value for the prior.
        - log_acq_floor (optional): Minimum value for the log acquisition function.
        - nonneg_acq (optional): Boolean indicating if the acquisition function should be non-negative.
        - Additional keyword arguments are passed to the raw acquisition function.
    Returns:
    PiBO: An instance of the PiBO class initialized with the provided parameters.
    """
    pibo_kwargs = {
        key: kwargs.pop(key)
        for key in ["user_prior", "decay_beta", "prior_floor", "log_acq_floor", "nonneg_acq"]  # "custom_decay" is not working because it was not implemented by Hvarfner et al.
        if key in kwargs
    }
    # cache_root = kwargs.pop("cache_root", False)
    return PiBO(raw_acqf_kwargs={**kwargs}, **pibo_kwargs, acqf_factory=qPriorLogExpectedImprovement)

@dataclass
class ExperimentConfig:
    seed: int | None = None
    optimization_method: Literal["BO", "PBO"] = "BO"
    objective: Literal["Sphere", "Sphere1D", "SphereNoisy", "Shekel", "Hartmann", "ImageSimilarity", "ScatterPlotQuality", "MRLayoutQuality"] = "Sphere"
    num_trials: int = 40
    num_paths: int = 64
    num_initial_samples: int = 4
    prior_type: Literal["Biased", "Unbiased", "BiasedCertain", "UnbiasedUncertain", "None"] = "None"
    prior_injection_method: Literal["None", "ColaBO", "piBO"] = "ColaBO"

class BenchmarkRunner:
    def __init__(self,
                 db_path: str = "experiments.db",
                 gp_dir_path: str = "data/",
                 seed: int = 0,
                 logger: logging.Logger = logging.getLogger(__name__),
                ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.db_path = db_path
        create_database(db_path)
        self.db = get_database(db_path)
        self.logger = logger
        self.gp_dir_path = gp_dir_path
        if not os.path.exists(gp_dir_path):
            os.makedirs(gp_dir_path)

    def run(self, configs: List[ExperimentConfig], n_runs_per_config: int = 1, **kwargs):
        n_total_experiments = len(configs) * n_runs_per_config
        self.logger.info(f"Running {len(configs)} configurations with {n_runs_per_config} runs per config ({n_total_experiments} total experiments)")

        for _ in range(n_runs_per_config):
            self.logger.info(f"Starting run {_ + 1}")

            seed = int(self.rng.integers(0, 2**32))

            for config_idx, config in enumerate(configs):
                experiment_idx = _ * len(configs) + config_idx
                self.logger.info(f"Running experiment {experiment_idx + 1}/{n_total_experiments}")

                config.seed = seed

                if self.experiment_exists(config):
                    self.logger.info(f"Skipping experiment {experiment_idx + 1}/{n_total_experiments}")
                    continue
                self.run_experiment(config, **kwargs)

        self.logger.info("Benchmark completed.")

    def experiment_exists(self, config: ExperimentConfig):
        return self.db.experiment_exists(
            optimization_type=config.optimization_method,
            num_trials=config.num_trials,
            num_paths=config.num_paths,
            num_initial_samples=config.num_initial_samples,
            objective_type=config.objective,
            prior_type=config.prior_type,
            seed=config.seed,
            prior_injection_method=config.prior_injection_method,
        )

    def run_experiment(self, config: ExperimentConfig, **kwargs):
        self.logger.info(f"Starting experiment {config}")

        objective = self.get_objective(config.objective, **kwargs)
        user_prior = self.get_user_prior(prior_type=config.prior_type, objective=objective, seed=config.seed)

        temperature = None
        if hasattr(user_prior, "temperature"):
            temperature = user_prior.temperature

        if config.optimization_method in ["BO", "PriorSampling"]:
            if config.optimization_method == "BO":
                result_X, result_y, gp = bo_maximize(
                    objective,
                    user_prior=user_prior,
                    num_trials=config.num_trials,
                    logger=self.logger,
                    num_paths=config.num_paths,
                    seed=config.seed,
                    num_initial_samples=config.num_initial_samples,
                    acq_func_factory=pibo_factory if config.prior_injection_method == "piBO" else qPriorLogExpectedImprovement,
                    acqf_func_kwargs=dict(
                        resampling_fraction=0.5 if config.num_paths < 1024 else 256/config.num_paths,
                        custom_decay=1.0,
                    ),
                )
            else:  # config.optimization_method == "PriorSampling"
                result_X, result_y, gp = prior_sampling_maximize(
                    objective,
                    user_prior=user_prior,
                    num_trials=config.num_trials,
                    logger=self.logger,
                    num_paths=config.num_paths,
                    seed=config.seed,
                    num_initial_samples=config.num_initial_samples,
                )
            result_best_y, result_best_y_indices = torch.cummax(result_y, dim=0)
            result_best_y = result_best_y.squeeze(-1)
            result_best_y_indices = result_best_y_indices.squeeze(-1)
            result_best_X = result_X[result_best_y_indices]

            experiment_config = self.db.add_bo_results(
                result_best_X=result_best_X,
                result_best_y=result_best_y,
                result_X=result_X,
                result_y=result_y,
                objective=objective,
                seed=config.seed,
                n_trials=config.num_trials,
                n_paths=config.num_paths,
                n_initial_samples=config.num_initial_samples,
                temperature=temperature,
                prior=user_prior,
                prior_type=config.prior_type,
                objective_type=config.objective,
                prior_injection_method=config.prior_injection_method,
                optimization_type=config.optimization_method,
            )

        elif config.optimization_method == "PBO":
            result_X, result_comparisons, result_best_X, gp = pbo_maximize(
                objective,
                user_prior=user_prior,
                num_trials=config.num_trials,
                logger=self.logger,
                num_paths=config.num_paths,
                num_samples_per_iteration=2,
                include_current_best=True,
                seed=config.seed,
                num_initial_samples=config.num_initial_samples,
                acq_func_factory=pibo_factory if config.prior_injection_method == "piBO" else qPriorLogExpectedImprovement,
                acqf_func_kwargs=dict(
                    resampling_fraction=0.5 if config.num_paths < 1024 else 256/config.num_paths,
                    custom_decay=1.0,
                ),
            )
            result_y = -objective.evaluate_true(result_X) if objective.negate else objective.evaluate_true(result_X)
            result_best_y = -objective.evaluate_true(result_best_X) if objective.negate else objective.evaluate_true(result_best_X)

            experiment_config = self.db.add_pbo_results(
                result_best_X=result_best_X,
                result_best_y=result_best_y,
                result_X=result_X,
                result_comparisons=result_comparisons,
                result_y=result_y,
                objective=objective,
                seed=config.seed,
                n_trials=config.num_trials,
                n_paths=config.num_paths,
                n_initial_samples=config.num_initial_samples,
                temperature=temperature,
                prior=user_prior,
                prior_type=config.prior_type,
                objective_type=config.objective,
                prior_injection_method=config.prior_injection_method,
            )

        save_model(gp, os.path.join(self.gp_dir_path, f"gp_{experiment_config.id}.pt"))

        self.logger.info(f"Finished experiment {experiment_config}")

    @staticmethod
    def get_user_prior(
        prior_type: Literal["Biased", "Unbiased", "BiasedCertain", "UnbiasedUncertain", "Default", "None"] | None = None,
        objective: Sphere | Shekel | Shekel2D | Hartmann | ImageSimilarityLoss | ScatterPlotQualityLoss | MRLayoutQualityLoss | None = None,
        seed: int = 42,
    ):
        if (prior_type is None) or (prior_type == "None"):
            return None
        
        if prior_type == "Default":
            return get_default_prior(objective=objective, offset_factor=0.1, confidence=0.25)
        
        temperature = 1.0
        if "Certain" in prior_type:
            temperature = 0.1
        elif "MoreCertain" in prior_type:
            temperature = 0.01
        elif "Uncertain" in prior_type:
            temperature = 10.0
        elif "MoreUncertain" in prior_type:
            temperature = 100.0
        
        if "Unbiased" in prior_type:
            return get_model_based_prior(objective=objective, objective_model=objective, minimize=False, temperature=temperature)

        if isinstance(objective, Sphere):
            objective_model = lambda x: objective(x - 0.5)

        elif isinstance(objective, Shekel):
            objective_model = ShekelNoGlobal(m=objective.m, negate=True)

        elif isinstance(objective, Shekel2D):
            objective_model = Shekel2DNoGlobal(m=objective.m, negate=True)

        elif isinstance(objective, ImageSimilarityLoss):
            objective_model = ImageSimilarityLoss(original_image=objective._original_image, weight_psnr=0.2, weight_ssim=0.8, negate=True)

        elif isinstance(objective, ScatterPlotQualityLoss):
            objective_model = ScatterPlotQualityLoss(
                x_data=objective.x_data,
                y_data=objective.y_data,
                weight_angle_difference = objective._weight_angle_difference,
                weight_axis_ratio_difference = objective._weight_axis_ratio_difference,
                weight_opacity = objective._weight_opacity,
                weight_contrast = objective._weight_contrast,
                weight_opacity_difference = objective._weight_opacity_difference,
                weight_contrast_difference = objective._weight_contrast_difference,
                weight_marker_overlap = objective._weight_marker_overlap,
                weight_overplotting = 0,  # is 12 in original
                # weight_overplotting = objective._weight_overplotting,
                weight_class_perception = objective._weight_class_perception,
                weight_outlier_perception = objective._weight_outlier_perception,
                use_approximate_model=True,  # trained on weight_overplotting=0
                negate=True,
            )

        elif isinstance(objective, MRLayoutQualityLoss):
            objective_model = MRLayoutQualityLoss(
                weight_arm=0,
                weight_euclidean=objective.weight_euclidean,
                weight_neck=0,
                weight_semantic=0,
                negate=True,
            )

        return get_model_based_prior(objective=objective, objective_model=objective_model, minimize=False, temperature=temperature, seed=seed)
    
    @staticmethod
    def get_objective(objective_name: str, noise_std: float = 0.1, minimal_noise_std: float = 0.001):
        if objective_name == "Sphere":
            return Sphere(dim=2, negate=True, noise_std=minimal_noise_std)
        if objective_name == "Sphere1D":
            return Sphere(dim=1, negate=True, noise_std=minimal_noise_std)
        if objective_name == "SphereNoisy":
            return Sphere(dim=2, negate=True, noise_std=noise_std)
        if objective_name == "Shekel":
            return Shekel(negate=True, noise_std=minimal_noise_std)  # dim=4, num_maximizers=10
        if objective_name == "Shekel2D":
            return Shekel2D(negate=True, m=5, noise_std=minimal_noise_std)
        if objective_name == "Hartmann":
            return Hartmann(dim=6, negate=True, noise_std=minimal_noise_std)
        if objective_name == "ImageSimilarity":
            # original_image = (torch.rand(3, 16, 16) * 255).floor().to(torch.uint8)
            ava_flowers_dir = os.getenv("AVA_FLOWERS_DIR")
            original_image = resize(read_image(os.path.join(ava_flowers_dir, '43405.jpg')), 64)  # Downsample
            return ImageSimilarityLoss(original_image=original_image, optimizer=(0.8, 1.2, 1.2, 0.1), weight_psnr=0.5, weight_ssim=0.5, negate=True, noise_std=minimal_noise_std)
        if objective_name == "ScatterPlotQuality":
            df = sns.load_dataset('mpg')  # Load Cars dataset from seaborn
            x_data = torch.tensor(df['horsepower'].values, dtype=torch.float32)
            y_data = torch.tensor(df['mpg'].values, dtype=torch.float32)
            return ScatterPlotQualityLoss(x_data=x_data, y_data=y_data, negate=True, noise_std=minimal_noise_std)
        if objective_name == "MRLayoutQuality":
            return MRLayoutQualityLoss(negate=True, noise_std=minimal_noise_std)
        raise ValueError(f"Objective {objective_name} not recognized")


if __name__ == "__main__":
    import tempfile
    from modelbasedprior.logger import setup_logger
    temporary_gp_dir = tempfile.TemporaryDirectory()
    gp_dir_path = temporary_gp_dir.name

    logger = setup_logger(level=logging.DEBUG)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

    configs = [
        ExperimentConfig(
            seed=2195314465,
            optimization_method='BO',
            objective='Shekel',
            num_trials=40,
            num_paths=65376,
            num_initial_samples=4,
            prior_type='Biased',
            prior_injection_method='ColaBO',
        )
    ]

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_path = tmp.name
        runner = BenchmarkRunner(seed=0, db_path=db_path, gp_dir_path=gp_dir_path)
        runner.run(configs)

    temporary_gp_dir.cleanup()
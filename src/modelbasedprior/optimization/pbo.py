import torch
import logging
import numpy as np

from typing import Tuple, Callable
from itertools import combinations
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.sampling.pathwise_sampler import PathwiseSampler

from botorch.acquisition.prior_monte_carlo import qPriorExpectedImprovement, PriorMCAcquisitionFunction
from botorch.utils.prior import UserPriorLocation

from modelbasedprior.optimization.bo import generate_data, generate_data_from_prior, Objective

def generate_comparisons(y: torch.Tensor) -> torch.Tensor:
    """Create noisy pairwise comparisons from utility values."""
    y = y.squeeze()

    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    
    c0 = y[all_pairs[:, 0]]
    c1 = y[all_pairs[:, 1]]
    
    reverse_comp = (c0 < c1).numpy()  # Corresponds to maximizing the objective function, use the reverse comparison for minimization
    all_pairs[reverse_comp, :] = np.flip(all_pairs[reverse_comp, :], axis=1)
    
    return torch.tensor(all_pairs).long()

def init_and_fit_model(X: torch.Tensor, comparisons: torch.Tensor, bounds: torch.Tensor) -> PairwiseGP:
    """Initialize and fit a pairwise Gaussian process model."""
    model = PairwiseGP(datapoints=X, comparisons=comparisons, input_transform=Normalize(d=X.shape[-1], bounds=bounds))
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

def map_to_existing(X: torch.Tensor, points: torch.Tensor, tolerance: float = 1e-6, 
                    match_strategy: str = "first") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map points to their indices in X if they exist, or append them to X and return new indices.
    
    Args:
        X: Tensor of shape (n, d) containing previously evaluated points
        points: Tensor of shape (m, d) containing points to map
        tolerance: Floating point tolerance for equality comparison
        match_strategy: Strategy for handling multiple matches:
            - "first": Use the first matching index
            - "closest": Use the index of the closest point
            - "error": Raise an error if multiple matches are found
        
    Returns:
        Tuple containing:
            - Updated X tensor with new points appended (if any)
            - Tensor of indices mapping each point in 'points' to its position in the updated X
    
    Raises:
        ValueError: If match_strategy is "error" and multiple matches are found
    """
    # Initialize indices list to store results
    indices = []
    new_X = X.clone()  # Create a copy to avoid modifying the input tensor
    
    for point in points:
        # Compute L1 distance between point and all points in new_X
        diff = (new_X - point).abs().sum(dim=1)
        
        # Find indices where the difference is less than tolerance
        existing_idx = torch.where(diff < tolerance)[0]
        
        if len(existing_idx) > 0:
            # A point very close to this one already exists in X
            if len(existing_idx) == 1 or match_strategy == "first":
                # Only one match or using first match strategy
                indices.append(existing_idx[0].item())
            elif match_strategy == "closest":
                # Find the closest match
                closest_idx = torch.argmin(diff[existing_idx]).item()
                indices.append(existing_idx[closest_idx].item())
            elif match_strategy == "error":
                # Raise error for multiple matches
                raise ValueError(f"Multiple matches found for point {point}. "
                                 f"Matching indices: {existing_idx.tolist()}")
            else:
                raise ValueError(f"Unknown match strategy: {match_strategy}")
        else:
            # Point is new, append it to new_X
            new_X = torch.cat([new_X, point.unsqueeze(0)], dim=0)
            indices.append(len(new_X) - 1)  # Index is the last position
    
    # Convert indices list to tensor
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    
    return new_X, indices_tensor

def make_new_data(X: torch.Tensor, next_X: torch.Tensor, comparisons: torch.Tensor, objective: Objective):
    """Generate new pairwise comparisons and update the data."""
    # Map next_X points back to their indices in X (or append if new)
    X, next_indices = map_to_existing(X, next_X)

    # Generate new comparisons between the points in next_X (now represented by next_indices)
    new_comps = torch.tensor(list(combinations(next_indices.tolist(), 2)))

    # Generate noisy comparisons for these new pairs (every pair is compared once)
    X_comp = X[next_indices]
    y_comp = objective(X_comp)  # Get utility values for the corresponding points
    noisy_comps = generate_comparisons(y_comp)

    # Adjust the comparisons based on noisy_comps, ensuring the correct ordering
    adjusted_comps = new_comps.clone()  # Create a copy of new_comps to modify

    # Flip pairs in new_comps if needed, based on noisy_comps
    for i in range(noisy_comps.shape[0]):
        if noisy_comps[i, 0] > noisy_comps[i, 1]:
            # Flip the pair in adjusted_comps to match the preference in noisy_comps
            adjusted_comps[i] = torch.flip(new_comps[i], dims=[0])

    # Append the new comparisons to the existing comparisons
    comparisons = torch.cat([comparisons, adjusted_comps])

    return X, comparisons

def maximize(
    objective: Objective,
    user_prior: UserPriorLocation | None = None,
    include_current_best: bool = True,
    num_trials: int = 20,
    num_initial_samples: int = 2,  # if set to 1, Standardize will fail
    num_samples_per_iteration: int = 2,  # if include_current_best is True, this needs to be at least 2
    num_paths: int = 512,
    acq_func_factory: Callable[..., PriorMCAcquisitionFunction] = qPriorExpectedImprovement,
    acqf_func_kwargs: dict = dict(
        resampling_fraction=0.5,
        custom_decay=1.0,
    ),
    acqf_func_optimization_kwargs: dict = dict(
        num_restarts=8,
        raw_samples=256,
    ),
    seed: int = 123,
    logger: logging.Logger = logging.getLogger(__name__),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the preferential Bayesian optimization (maximize objective)."""
    assert num_samples_per_iteration > 1 or not include_current_best, "At least 2 samples are needed per iteration if include_current_best is True"

    logger.info("Starting Bayesian optimization")
    logger.debug(f"Parameters: trials={num_trials}, initial_samples={num_initial_samples}, "
                f"paths={num_paths}, seed={seed}")

    # Seed
    torch.manual_seed(seed)
    logger.debug(f"Set random seed to {seed}")

    # Store for the current best point at any iteration
    all_best_X = torch.Tensor([])

    # Initial data generation
    if user_prior is None:
        logger.info("Generating initial data without prior")
        init_X, init_y = generate_data(objective, num_initial_samples)
    else:
        logger.info("Generating initial data from user prior")
        init_X, init_y = generate_data_from_prior(objective, user_prior, num_initial_samples)
    
    logger.debug(f"Initial X shape: {init_X.shape}, Initial y shape: {init_y.shape}")

    # Generate initial comparisons
    logger.info("Generating initial comparisons")
    init_comparisons = generate_comparisons(init_y)
    logger.debug(f"Initial comparisons shape: {init_comparisons.shape}")

    # Initialize and fit the model
    logger.info("Initializing and fitting model")
    model = init_and_fit_model(init_X, init_comparisons, objective.bounds)

    # Store the point with the highest approximate utility at each iteration
    best_X = init_X[torch.argmax(model.utility)].unsqueeze(0)  # NOTE: best_X is only one point even though there may be more than 1 comparison
    all_best_X = torch.cat([all_best_X, best_X])  # NOTE: all_best_X will thus be of length num_trials + 1

    # Run the optimization loop for the specified number of batches
    for iteration in range(1, num_trials + 1):
        logger.info(f"Starting iteration {iteration}/{num_trials}")

        # Get the next parameters to evaluate
        sample_shape = torch.Size([num_paths])
        logger.debug(f"Drawing Matheron paths with shape {sample_shape}")
        paths = draw_matheron_paths(model=model, sample_shape=sample_shape)
        
        sampler = PathwiseSampler(sample_shape=sample_shape)
        acq_func = acq_func_factory(
            model=model,
            paths=paths,
            sampler=sampler,
            X_baseline=init_X,
            user_prior=user_prior,
            **acqf_func_kwargs,
        )
        logger.debug("Created acquisition function")

        # Optimize the acquisition function to get the next candidate points
        logger.debug("Optimizing acquisition function")
        next_X, _ = optimize_acqf(
            acq_func,
            bounds=objective.bounds,
            q=num_samples_per_iteration - 1 if include_current_best else num_samples_per_iteration,
            **acqf_func_optimization_kwargs,
        )

        # Add the current best point to next_X to ensure it is compared (will be mapped back to existing points)
        logger.debug("Determining the new comparison")
        best_X = init_X[torch.argmax(model.utility)].unsqueeze(0)  # note that the point with the highest estimated utility is not necessarily the point with the highest actual utility in the candidate set
        if include_current_best:
            next_X = torch.cat([next_X, best_X])

        # Update the data with the new observations
        logger.debug("Updating data with new observations")
        init_X, init_comparisons = make_new_data(init_X, next_X, init_comparisons, objective)
        
        # Refit the model with updated data
        logger.debug("Refitting model with updated data")
        model = init_and_fit_model(init_X, init_comparisons, objective.bounds)

        # Store the point with the highest approximate utility at each iteration
        all_best_X = torch.cat([all_best_X, best_X])

    logger.info("Optimization completed")
    logger.debug(f"Final X shape: {init_X.shape}, Final comparisons shape: {init_comparisons.shape}, Final best X shape: {all_best_X.shape}")
    
    # Return the evaluated data
    return init_X, init_comparisons, all_best_X, model

def load_model(path: str, datapoints: torch.Tensor, comparisons: torch.Tensor, bounds: torch.Tensor) -> PairwiseGP:
    """Load the model from a file."""
    gp = PairwiseGP(  # PairwiseGP for PBO
        datapoints=datapoints,
        comparisons=comparisons,
        input_transform=Normalize(d=datapoints.shape[-1], bounds=bounds),
    )
    mll = PairwiseLaplaceMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.load_state_dict(torch.load(path, weights_only=True))
    return gp

if __name__ == "__main__":
    from botorch.utils.prior import DefaultPrior
    from modelbasedprior.prior import ModelBasedPrior
    from modelbasedprior.objectives.sphere import Sphere
    from modelbasedprior.logger import setup_logger
    from modelbasedprior.benchmarking.runner import pibo_factory

    objective = Sphere(dim=2, negate=True)
    # global_optima = objective.optimizers  # here: objective.bounds.T.mean(dim=-1) => [0, 0]
    # offset = 0.1 * (objective.bounds[1,:] - objective.bounds[0,:])
    # parameter_default = global_optima[0] + offset
    # user_prior = DefaultPrior(bounds=objective.bounds, parameter_defaults=parameter_default, confidence=0.25)
    user_prior = ModelBasedPrior(bounds=objective.bounds, predict_func=lambda x: objective(x - 0.5), temperature=1.0, minimize=False)
    
    logger = setup_logger(level=logging.DEBUG)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

    result_X, result_comparisons, result_best_X, model = maximize(objective, user_prior=user_prior, num_initial_samples=4, num_trials=5, logger=logger, acq_func_factory=pibo_factory, include_current_best=False)
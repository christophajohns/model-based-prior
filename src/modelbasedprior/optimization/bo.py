import torch
import logging

from typing import Tuple, Callable, Protocol
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.sampling.pathwise_sampler import PathwiseSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.prior_monte_carlo import qPriorExpectedImprovement, PriorMCAcquisitionFunction
from botorch.utils.prior import UserPriorLocation, unnormalize

class Objective(Protocol):
    bounds: torch.Tensor  # Example: torch.tensor([[-5.12, -5.12], [5.12, 5.12]])
    dim: int

    def __call__(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ...

def init_and_fit_model(X: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor) -> SingleTaskGP:
    """Initialize and fit a Gaussian process model."""
    model = SingleTaskGP(train_X=X, train_Y=y, input_transform=Normalize(d=X.shape[-1], bounds=bounds), outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

def generate_data(objective: Objective, n: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random input data X and corresponding utility values."""
    X = unnormalize(torch.rand(n, objective.dim, dtype=torch.float64), objective.bounds)
    y = objective(X).reshape(-1, 1)
    return X, y

def generate_data_from_prior(
        objective: Objective,
        user_prior: UserPriorLocation,
        n: int = 1,
        max_retries: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate input data X from a prior and corresponding utility values."""
    default_X = user_prior.default.unsqueeze(0).to(dtype=torch.float64)
    default_y = objective(default_X).reshape(-1, 1)

    if n == 1:
        return default_X, default_y

    # Sample n-1 unique points from the prior that are not the default point
    sampled_X = torch.empty(0, objective.dim, dtype=torch.float64)
    while sampled_X.size(0) < n-1:
        i = 0
        while (i < max_retries):
            attempt_sampled_X = user_prior.sample(1).to(dtype=torch.float64)
            sample_close_to_default = torch.allclose(attempt_sampled_X, default_X)
            sample_in_sampled_X = torch.any(torch.all(torch.isclose(attempt_sampled_X, sampled_X), dim=-1))
            if not sample_close_to_default and not sample_in_sampled_X:
                sampled_X = torch.cat([sampled_X, attempt_sampled_X])
                break
            i += 1
        if i == max_retries:
            raise RuntimeError(f"Could not sample a new point after {max_retries} attempts")      

    sampled_y = objective(sampled_X).reshape(-1, 1)
    
    X = torch.cat([default_X, sampled_X])
    y = torch.cat([default_y, sampled_y])
    return X, y

def make_new_data(X: torch.Tensor, y: torch.Tensor, next_X: torch.Tensor, objective: Objective) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate new ratings and update the data."""
    # Generate potentially noisy utility values for the new points
    y_next = objective(next_X).reshape(-1, 1)  # Get utility values for the corresponding points

    # Append the new points and noisy utility values to the existing data
    X = torch.cat([X, next_X])
    y = torch.cat([y, y_next])

    return X, y

def maximize(
    objective: Objective,
    user_prior: UserPriorLocation | None = None,
    num_trials: int = 20,
    num_initial_samples: int = 2,  # if set to 1, Standardize will fail
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the Bayesian optimization with absolute ratings (maximize objective)."""
    logger.info("Starting Bayesian optimization")
    logger.debug(f"Parameters: trials={num_trials}, initial_samples={num_initial_samples}, "
                f"paths={num_paths}, seed={seed}")

    # Seed
    torch.manual_seed(seed)
    logger.debug(f"Set random seed to {seed}")

    # Initial data generation
    if user_prior is None:
        logger.info("Generating initial data without prior")
        init_X, init_y = generate_data(objective, num_initial_samples)
    else:
        logger.info("Generating initial data from user prior")
        init_X, init_y = generate_data_from_prior(objective, user_prior, num_initial_samples)
    
    logger.debug(f"Initial X shape: {init_X.shape}, Initial y shape: {init_y.shape}")

    # Initialize and fit the model
    logger.info("Initializing and fitting model")
    model = init_and_fit_model(init_X, init_y, objective.bounds)

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
            q=1,
            **acqf_func_optimization_kwargs,
        )

        # Update the data with the new observations
        logger.debug("Updating data with new observations")
        init_X, init_y = make_new_data(init_X, init_y, next_X, objective)
        
        # Refit the model with updated data
        logger.debug("Refitting model with updated data")
        model = init_and_fit_model(init_X, init_y, objective.bounds)

    logger.info("Optimization completed")
    logger.debug(f"Final X shape: {init_X.shape}, Final y shape: {init_y.shape}")
    
    # Return the evaluated data
    return init_X, init_y, model

def load_model(path: str, train_X: torch.Tensor, train_Y: torch.Tensor, bounds: torch.Tensor) -> SingleTaskGP:
    """Load the model from a file."""
    gp = SingleTaskGP(  # SingleTaskGP for BO
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1)
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.load_state_dict(torch.load(path, weights_only=True))
    return gp

if __name__ == "__main__":
    from botorch.utils.prior import DefaultPrior
    from modelbasedprior.objectives.sphere import Sphere
    from modelbasedprior.logger import setup_logger

    objective = Sphere(dim=2, negate=True)
    global_optima = objective.optimizers  # here: objective.bounds.T.mean(dim=-1) => [0, 0]
    offset = 0.1 * (objective.bounds[1,:] - objective.bounds[0,:])
    parameter_default = global_optima[0] + offset
    user_prior = DefaultPrior(bounds=objective.bounds, parameter_defaults=parameter_default, confidence=0.25)
    
    logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

    result_X, result_y, model = maximize(objective, user_prior=user_prior, num_trials=5, logger=logger)
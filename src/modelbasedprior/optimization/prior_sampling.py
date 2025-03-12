import torch
import logging
from typing import Tuple

from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.prior import UserPriorLocation

from modelbasedprior.optimization.bo import init_and_fit_model, generate_data, generate_data_from_prior, make_new_data

def maximize(
    objective: SyntheticTestFunction,
    user_prior: UserPriorLocation | None = None,
    num_trials: int = 20,
    num_initial_samples: int = 2,  # if set to 1, Standardize will fail
    num_paths: int = 512,
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

        # Draw the next sample from the sampler
        logger.debug("Drawing the next sample from the sampler")
        next_X, _ = generate_data_from_prior(objective, user_prior, 1)

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
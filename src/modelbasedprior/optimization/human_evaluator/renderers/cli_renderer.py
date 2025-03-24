import torch
import logging
from modelbasedprior.optimization.bo import maximize

class CLIHumanEvaluatorRenderer:
    def render(self, X: torch.Tensor) -> torch.Tensor:
        """Render the point in a CLI and get user rating."""
        point = X.numpy()
        print(f"\nEvaluate this point: {point}")
        return torch.tensor(float(input("Enter rating: ")), dtype=X.dtype)

if __name__ == "__main__":
    from botorch.utils.prior import DefaultPrior
    from modelbasedprior.objectives.sphere import Sphere
    from modelbasedprior.logger import setup_logger
    from modelbasedprior.optimization.human_evaluator.human_evaluator_objective import HumanEvaluatorObjective

    # CLI-based evaluation
    objective = Sphere(dim=2, negate=True)
    global_optima = objective.optimizers  # here: objective.bounds.T.mean(dim=-1) => [0, 0]
    offset = 0.1 * (objective.bounds[1,:] - objective.bounds[0,:])
    parameter_default = global_optima[0] + offset
    user_prior = DefaultPrior(bounds=objective.bounds, parameter_defaults=parameter_default, confidence=0.25)
    
    logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

    cli_renderer = CLIHumanEvaluatorRenderer()
    human_evaluator_objective = HumanEvaluatorObjective(renderer=cli_renderer, dim=objective.dim, bounds=objective._bounds)

    result_X, result_y, model = maximize(human_evaluator_objective, user_prior=user_prior, num_trials=5, logger=logger)
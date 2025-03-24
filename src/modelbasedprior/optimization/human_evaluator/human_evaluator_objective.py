import torch
import logging
from modelbasedprior.optimization.bo import maximize
from modelbasedprior.optimization.human_evaluator.renderers import HumanEvaluatorRenderer, CLIHumanEvaluatorRenderer

class HumanEvaluatorObjective:
    def __init__(self, renderer: HumanEvaluatorRenderer = CLIHumanEvaluatorRenderer(), dim=2, bounds=None):
        self.renderer = renderer  # Inject dependency
        self.dim = dim
        if bounds is None:
            bounds = [(-5.12, 5.12) for _ in range(self.dim)]
        self.bounds = torch.tensor(bounds).t()
    
    def __call__(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        orig_shape = X.shape
        if X.dim() == 2:
            X = X.unsqueeze(1)  # Add q dimension
        
        batch_shape, q, dim = X.shape
        ratings = torch.zeros(batch_shape, q, dtype=X.dtype, device=X.device)
        
        for i in range(batch_shape):
            for j in range(q):
                ratings[i, j] = self.renderer.render(X[i, j].cpu())  # Delegation
        
        return ratings.squeeze(1) if len(orig_shape) == 2 else ratings

if __name__ == "__main__":
    from botorch.utils.prior import DefaultPrior
    from modelbasedprior.objectives.sphere import Sphere
    from modelbasedprior.logger import setup_logger

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
import logging
from botorch.utils.prior import DefaultPrior
from modelbasedprior.objectives.sphere import Sphere
from modelbasedprior.logger import setup_logger
from modelbasedprior.optimization.bo import maximize

objective = Sphere(dim=2, negate=True)
global_optima = objective.optimizers  # here: objective.bounds.T.mean(dim=-1) => [0, 0]
offset = 0.1 * (objective.bounds[1,:] - objective.bounds[0,:])
parameter_default = global_optima[0] + offset
user_prior = DefaultPrior(bounds=objective.bounds, parameter_defaults=parameter_default, confidence=0.25)

logger = setup_logger(level=logging.INFO)  # or logging.DEBUG for more detailed output or logging.WARNING for less output

result_X, result_y, model = maximize(objective, user_prior=user_prior, num_trials=5, logger=logger)
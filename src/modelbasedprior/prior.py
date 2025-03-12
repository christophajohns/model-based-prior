import torch
from botorch.utils.prior import UserPriorLocation, DefaultPrior, normalize, unnormalize
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Callable

class ModelBasedPrior(UserPriorLocation):
    def __init__(self, *args, minimize = True, predict_func = lambda x: torch.tensor([0.0] * x.size()[0]), n_samples = 500, temperature = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_using_model = predict_func
        self.n_samples = n_samples
        self.temperature = temperature
        self.minimize = minimize

        # Generate Sobol samples from the parameter space
        sobol = torch.quasirandom.SobolEngine(dimension=self.dim)
        x_sobol = sobol.draw(n_samples).double()
        x_samples = unnormalize(x_sobol, self.bounds)

        # Compute the function values for the prediction model
        y_pred_samples = predict_func(x_samples)

        # Normalize the function values into the range [0, 1] (1 = max, 0 = min)
        # to avoid overflow (and underflow) when computing the exponential
        self.min_y_pred_samples = torch.min(y_pred_samples)
        self.max_y_pred_samples = torch.max(y_pred_samples)
        y_pred_samples_normalized = (y_pred_samples - self.min_y_pred_samples) / (self.max_y_pred_samples - self.min_y_pred_samples)

        # Compute the Boltzmann distribution
        if self.minimize:
            y_pred_samples_normalized = 1 - y_pred_samples_normalized

        # Normalize the function values to the range [-5, 5] (5 = max, -5 = min)
        # to cover the entire transformation range
        y_pred_samples_normalized = 10 * y_pred_samples_normalized - 5
        self.logsumexp = torch.logsumexp(y_pred_samples_normalized / temperature, dim=0)

        self.rng = torch.Generator(device='cpu').manual_seed(self.seed)

        self.samples = x_samples
        self.sample_log_probs = y_pred_samples_normalized / temperature - self.logsumexp
        self.sample_probs = torch.exp(self.sample_log_probs)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Acts as the joint pdf function in the normalized space. As such, the input 
        enters in the normalized space defined by self.bounds.

        Args: 
            X (torch.Tensor): The normalized input tensor

        Returns:
            torch.Tensor: The log probability density of each sample in the original space.
        """
        X_unnormalized = unnormalize(X, self.bounds)
        y_pred = self.predict_using_model(X_unnormalized)
        y_pred_normalized = (y_pred - self.min_y_pred_samples) / (self.max_y_pred_samples - self.min_y_pred_samples)
        if self.minimize:
            y_pred_normalized = 1 - y_pred_normalized
        y_pred_normalized = 10 * y_pred_normalized - 5
        return (y_pred_normalized / self.temperature - self.logsumexp).reshape(-1, 1)

    def sample(self, num_samples: int = 1, *args, **kwargs):
        """Retrieves samples in the actual, true search space

        Args:
            num_samples ([type]): Number of samples
        """
        # Sample from the model based prior by generating samples from the Boltzmann distribution
        number_of_generated_samples = max(kwargs.get("number_of_generated_samples", self.samples.size(0)), num_samples)
        samples = unnormalize(torch.rand(number_of_generated_samples, self.dim), self.bounds)
        sample_log_probs = self.evaluate(normalize(samples, self.bounds)).squeeze()
        sample_log_probs_normalized = sample_log_probs - torch.logsumexp(sample_log_probs, dim=0)
        sample_probs_normalized = torch.exp(sample_log_probs_normalized)
        return samples[torch.multinomial(sample_probs_normalized, num_samples, generator=self.rng)]
    
    @property
    def _default(self):
        """Return the normalized default prior location.
        
        Here, we return the normalized sample with the highest probability density.
        """
        max_idx = torch.argmax(self.sample_probs)
        return normalize(self.samples[max_idx], self.bounds)
    
    @property
    def default(self):
        """Return the unnormalized default prior location."""
        normalized_default = self._default
        return unnormalize(normalized_default, self.bounds)
    
def get_default_prior(objective: SyntheticTestFunction, offset_factor: float = 0.1, confidence: float = 0.25) -> DefaultPrior:
    global_optima = objective.optimizers  # here: objective.bounds.T.mean(dim=-1) => [0, 0]
    offset = offset_factor * (objective.bounds[1,:] - objective.bounds[0,:])
    parameter_default = global_optima[0] + offset
    return DefaultPrior(bounds=objective.bounds, parameter_defaults=parameter_default, confidence=confidence)

def get_model_based_prior(
        objective: SyntheticTestFunction,
        objective_model: Callable[[torch.Tensor], torch.Tensor],
        *prior_args,
        **prior_kwargs,
    ) -> ModelBasedPrior:
    return ModelBasedPrior(
        bounds=objective.bounds,
        predict_func=objective_model,
        *prior_args,
        **prior_kwargs,
    )
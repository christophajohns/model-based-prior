from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List, Optional, Tuple
import torch

class Sphere(SyntheticTestFunction):
    r"""Sphere test function.

    d-dimensional function (usually evaluated on `[-5.12, 5.12]^d`):

        f(x) = sum_{i=1}^d x_i^2

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        if bounds is None:
            bounds = [(-5.12, 5.12) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sum(X ** 2, dim=-1)
    
if __name__ == "__main__":
    sphere = Sphere(dim=3)
    X = torch.rand(2, 3)
    print(sphere(X))
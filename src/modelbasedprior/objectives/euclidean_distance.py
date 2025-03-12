from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List, Optional, Tuple
import torch

class EuclideanDistance(SyntheticTestFunction):
    r"""EuclideanDistance test function.

    d-dimensional function (usually evaluated on `[-5.12, 5.12]^d`):

        f(x) = sqrt(sum_{i=1}^d (x_i - z_i)^2)

    f has one minimizer for its global minimum at `z_1 = (z_1^0, z_1^1, ..., z_1^d)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 3,
        optimizer: Tuple[float, ...] = (0.0, 0.0, 0.0),
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
        self._optimizers = [optimizer]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum((X - torch.tensor(self._optimizers[0])) ** 2, dim=-1))
    
if __name__ == "__main__":
    distance = EuclideanDistance(dim=3, optimizer=(0.0, 0.0, 0.0))
    X = torch.rand(2, 3)
    print(distance(X))
from botorch.test_functions.synthetic import SyntheticTestFunction, Shekel
from typing import List, Optional, Tuple
import torch

class ShekelNoGlobal(SyntheticTestFunction):
    r"""Shekel test function without the global optimum.

    4-dimensional function (usually evaluated on `[0, 10]^4`):

        f(x) = -sum_{i=1}^10 (sum_{j=1}^4 (x_j - A_{ji})^2 + C_i)^{-1}

    f has one minimizer for its global minimum at `z_1 = (8, 8, 8, 8)` with
    `f(z_1) = -5.1600`.
    """

    dim = 4
    _bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    _optimizers = [(8, 8, 8, 8)]

    def __init__(
        self,
        m: int = 9,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            m: Defaults to 9.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.m = m
        optvals = {5: -5.0852, 7: -5.1132, 9: -5.1358, 10: -5.1600}
        self._optimal_value = optvals[self.m]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

        self.register_buffer(
            "beta", torch.tensor([2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=torch.float)
        )
        C_t = torch.tensor(
            [
                [1, 8, 6, 3, 2, 5, 8, 6, 7],
                [1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                [1, 8, 6, 3, 2, 5, 8, 6, 7],
                [1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            ],
            dtype=torch.float,
        )
        self.register_buffer("C", C_t.transpose(-1, -2))

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        self.to(device=X.device, dtype=X.dtype)
        beta = self.beta / 10.0  # should this be 9 instead?
        result = -sum(
            1 / (torch.sum((X - self.C[i]) ** 2, dim=-1) + beta[i])
            for i in range(self.m - 1)
        )
        return result
    
class Shekel2D(SyntheticTestFunction):
    r"""2D variant of the Shekel test function.

    2-dimensional function (usually evaluated on `[0, 10]^2`):

        f(x) = -sum_{i=1}^10 (sum_{j=1}^4 (x_j - A_{ji})^2 + C_i)^{-1}

    f has one minimizer for its global minimum at `z_1 = (4, 4)`.
    """

    dim = 2
    _bounds = [(0.0, 10.0), (0.0, 10.0)]
    _optimizers = [(4, 4)]

    def __init__(
        self,
        m: int = 10,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            m: Defaults to 10.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.m = m
        optvals = {5: -10.3012, 7: -10.7698, 10: -11.0298}
        self._optimal_value = optvals[self.m]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

        self.register_buffer(
            "beta", torch.tensor([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=torch.float)
        )
        C_t = torch.tensor(
            [
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            ],
            dtype=torch.float,
        )
        self.register_buffer("C", C_t.transpose(-1, -2))

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        self.to(device=X.device, dtype=X.dtype)
        beta = self.beta / 10.0
        result = -sum(
            1 / (torch.sum((X - self.C[i]) ** 2, dim=-1) + beta[i])
            for i in range(self.m)
        )
        return result
    
class Shekel2DNoGlobal(SyntheticTestFunction):
    r"""2D variant of the Shekel test function without its global optimum.

    2-dimensional function (usually evaluated on `[0, 10]^4`):

        f(x) = -sum_{i=1}^10 (sum_{j=1}^4 (x_j - A_{ji})^2 + C_i)^{-1}

    f has one minimizer for its global minimum at `z_1 = (4, 4)`.
    """

    dim = 2
    _bounds = [(0.0, 10.0), (0.0, 10.0)]
    _optimizers = [(8, 8)]

    def __init__(
        self,
        m: int = 9,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            m: Defaults to 9.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.m = m
        optvals = {5: -5.1671, 7: -5.2229, 9: -5.2677, 10: -5.3156}
        self._optimal_value = optvals[self.m]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

        self.register_buffer(
            "beta", torch.tensor([2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=torch.float)
        )
        C_t = torch.tensor(
            [
                [1, 8, 6, 3, 2, 5, 8, 6, 7],
                [1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            ],
            dtype=torch.float,
        )
        self.register_buffer("C", C_t.transpose(-1, -2))

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        self.to(device=X.device, dtype=X.dtype)
        beta = self.beta / 10.0
        result = -sum(
            1 / (torch.sum((X - self.C[i]) ** 2, dim=-1) + beta[i])
            for i in range(self.m - 1)
        )
        return result
    
if __name__ == "__main__":
    shekel2D_5 = Shekel2D(m=5)
    shekel2D_7 = Shekel2D(m=7)
    shekel2D_10 = Shekel2D(m=10)
    X_sample = torch.rand(2, 2)
    X_best = shekel2D_5.optimizers
    X = torch.cat([X_sample, X_best], dim=0)
    print("X", X)
    print("Shekel2D (m=5)", shekel2D_5(X))
    print("Shekel2D (m=7)", shekel2D_7(X))
    print("Shekel2D (m=10)", shekel2D_10(X))

    shekel2D_noglobal_5 = Shekel2DNoGlobal(m=5)
    shekel2D_noglobal_7 = Shekel2DNoGlobal(m=7)
    shekel2D_noglobal_10 = Shekel2DNoGlobal(m=10)
    shekel2D_noglobal_9 = Shekel2DNoGlobal(m=9)
    X_sample = torch.rand(2, 2)
    X_best = shekel2D_noglobal_5.optimizers
    X = torch.cat([X_sample, X_best], dim=0)
    print("X", X)
    print("Shekel2DNoGlobal (m=5)", shekel2D_noglobal_5(X))
    print("Shekel2DNoGlobal (m=7)", shekel2D_noglobal_7(X))
    print("Shekel2DNoGlobal (m=10)", shekel2D_noglobal_10(X))
    print("Shekel2DNoGlobal (m=9)", shekel2D_noglobal_9(X))

    shekel_noglobal_5 = ShekelNoGlobal(m=5)
    shekel_noglobal_7 = ShekelNoGlobal(m=7)
    shekel_noglobal_10 = ShekelNoGlobal(m=10)
    shekel_noglobal_9 = ShekelNoGlobal(m=9)
    X_sample = torch.rand(2, 4)
    X_best = shekel_noglobal_5.optimizers
    X = torch.cat([X_sample, X_best], dim=0)
    print("X", X)
    print("ShekelNoGlobal (m=5)", shekel_noglobal_5(X))
    print("ShekelNoGlobal (m=7)", shekel_noglobal_7(X))
    print("ShekelNoGlobal (m=10)", shekel_noglobal_10(X))
    print("ShekelNoGlobal (m=9)", shekel_noglobal_9(X))

    shekel_5 = Shekel(m=5)
    shekel_7 = Shekel(m=7)
    shekel_10 = Shekel(m=10)
    X_sample = torch.rand(2, 4)
    X_best = shekel_5.optimizers
    X = torch.cat([X_sample, X_best], dim=0)
    print("X", X)
    print("Shekel (m=5)", shekel_5(X))
    print("Shekel (m=7)", shekel_7(X))
    print("Shekel (m=10)", shekel_10(X))

    import plotly.graph_objects as go

    # Plot the Shekel2D function
    x = torch.linspace(0, 10, 100)
    y = torch.linspace(0, 10, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    Z = shekel2D_5(Z).reshape(100, 100)
    fig = go.Figure(layout=dict(title="Shekel2D (m=5)"), data=[go.Surface(x=x, y=y, z=Z)])
    fig.show()

    # Plot the Shekel2DNoGlobal function
    shekel2D_no_global = Shekel2DNoGlobal(m=5)
    Z = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    Z = shekel2D_no_global(Z).reshape(100, 100)
    fig = go.Figure(layout=dict(title="Shekel2DNoGlobal (m=5)"), data=[go.Surface(x=x, y=y, z=Z)])
    fig.show()
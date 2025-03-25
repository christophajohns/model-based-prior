import torch
from typing import Protocol

class HumanEvaluatorRenderer(Protocol):
    def render(self, X: torch.Tensor) -> torch.Tensor:
        """Render the point in a specific way (CLI, image, etc.) and collect a human rating."""
        ...

if __name__ == "__main__":
    class TestRenderer:
        def render(self, X: torch.Tensor) -> torch.Tensor:
            print(X)

    renderer = TestRenderer()
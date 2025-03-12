import numpy as np
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List, Optional, Tuple
import torch

def get_semantic_agreement(positive_associations: torch.Tensor,
                           negative_associations: torch.Tensor,
                           distances: torch.Tensor,
                           positive_weight: float = 0.375,
                           negative_weight: float = 0.675,
    ):
    """Return the semantic agreement of an element (i.e., a score that is based on the distance
    between the closest semantically related object in the user's environment
    and the element as weighted by the association score)."""
    # Replace zeros with a tiny value to avoid division by zero
    small_float = 0.001 # If distance is zero, add 1mm to avoid division by zero
    squared_distances_without_zero = torch.where(
        distances==0,
        small_float,
        distances**2
    )
    max_positive_association_distance = (positive_associations / squared_distances_without_zero).max()
    max_negative_association_distance = (negative_associations / squared_distances_without_zero).max()
    return positive_weight * max_positive_association_distance - negative_weight * max_negative_association_distance

def get_semantic_mismatch(positive_associations: torch.Tensor,
                           negative_associations: torch.Tensor,
                           distances: torch.Tensor,
                           positive_weight: float = 0.375,
                           negative_weight: float = 0.675):
    """Return the semantic mismatch of an element (i.e., a cost that is based on the distance
    between the closest semantically related object in the user's environment
    and the element as weighted by the association score)."""
    semantic_agreement = get_semantic_agreement(
        positive_associations,
        negative_associations,
        distances,
        positive_weight,
        negative_weight,
    )
    return 1 / (1 + torch.exp(torch.clamp(semantic_agreement, -1e2, 1e2)))

def get_semantic_cost(
    element_position: torch.Tensor,
    object_positions: torch.Tensor,
    positive_association_scores: torch.Tensor,
    negative_association_scores: torch.Tensor,
    positive_association_weight: float = 0.375,
    negative_association_weight: float = 0.625,
):
    """
    Calculate the semantic cost of an element based on its position 
    relative to semantically related objects.
    
    Args:
        element_position (torch.Tensor): Position of the element
        object_positions (torch.Tensor): Positions of semantic objects
        positive_association_scores (torch.Tensor): Positive association scores
        negative_association_scores (torch.Tensor): Negative association scores
        positive_association_weight (float): Weight for positive associations
        negative_association_weight (float): Weight for negative associations
    
    Returns:
        torch.Tensor: Semantic cost score
    """
    # Handle empty object positions case
    if object_positions.numel() == 0:
        return torch.tensor(1.0, device=element_position.device)
    
    # Compute distances with automatic gradient tracking
    distances = torch.norm(element_position - object_positions, dim=1)
    
    # Calculate semantic mismatch
    semantic_mismatch = get_semantic_mismatch(
        positive_association_scores,
        negative_association_scores,
        distances,
        positive_association_weight,
        negative_association_weight
    )
    
    return semantic_mismatch


class SemanticAgreementCost(SyntheticTestFunction):
    r"""SemanticAgreementCost test function.

    3-dimensional function (usually evaluated on `[-2.0, 2.0]^3`):

        f(x) = (1 + (w_+ * max(positive_associations / distances^2) - w_- * max(negative_associations / distances^2)))^(-1)    
    """

    dim = 3
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        object_positions: Optional[torch.Tensor] = None,
        positive_association_scores: Optional[torch.Tensor] = None,
        negative_association_scores: Optional[torch.Tensor] = None,
        positive_association_weight: float = 0.375,
        negative_association_weight: float = 0.625,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            object_positions: The positions of the semantic objects.
            positive_association_scores: The positive association scores.
            negative_association_scores: The negative association scores.
            positive_association_weight: The weight for the positive association component.
            negative_association_weight: The weight for the negative association component.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 3
        if bounds is None:
            bounds = [(-2., 2.) for _ in range(self.dim)]

        if object_positions is None:
            # Default objects: One positively associated object to the front right, one negatively associated object to the front left
            object_positions = torch.tensor([[1., 1., 1.], [-1., 1., 1.]])
            positive_association_scores = torch.tensor([1., -1.])
            negative_association_scores = torch.tensor([-1., 1.])
            # Optimizer is the first object position as a list of lists
            self._optimizers = [object_positions[0].tolist()]

        # Check that the object positions are in the correct format
        assert object_positions.shape[1] == 3, "The object positions must be a 3D array."

        # Check that the association scores are in the correct format
        assert positive_association_scores.shape == negative_association_scores.shape, "The association scores must have the same shape."
        assert positive_association_scores.shape[0] == object_positions.shape[0], "The association scores must have the same number of elements as the object positions."

        self.object_positions = object_positions
        self.positive_association_scores = positive_association_scores
        self.negative_association_scores = negative_association_scores
        self.positive_association_weight = positive_association_weight
        self.negative_association_weight = negative_association_weight

        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        batch_shape = X.shape[:-1]
        results = torch.zeros(batch_shape, device=X.device)

        # Ensure X is at least 2D (for single sample cases)
        if X.dim() == 1:
            X = X.unsqueeze(0)

        for idx in range(X.shape[0]):
            result = get_semantic_cost(
                X[idx].squeeze(),  # X may have shape (N, 3) or (N, 1, 3) 
                self.object_positions,
                self.positive_association_scores,
                self.negative_association_scores,
                self.positive_association_weight,
                self.negative_association_weight,
            )
            results[idx] = result
    
        return results.double()
    
if __name__ == "__main__":
    torch.manual_seed(0)
    semantic_agreement_cost = SemanticAgreementCost()
    X = torch.rand(2, 3)
    X_batch = torch.rand(128, 1, 3)
    X_single = torch.rand(3)
    print(X)
    print(semantic_agreement_cost(X))
    print(semantic_agreement_cost(X).shape)
    print(semantic_agreement_cost(X_batch).shape)
    print(semantic_agreement_cost(X_single).shape)
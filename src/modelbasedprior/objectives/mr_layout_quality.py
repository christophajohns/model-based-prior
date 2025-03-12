from modelbasedprior.objectives.ergonomics import NeckErgonomicsCost, ExponentialArmErgonomicsCost
from modelbasedprior.objectives.semantic_agreement import SemanticAgreementCost
from modelbasedprior.objectives.euclidean_distance import EuclideanDistance
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List, Optional, Tuple
import torch
import numpy as np

class MRLayoutQualityLoss(SyntheticTestFunction):
    r"""MRLayoutQualityLoss test function as weighted combination
    of the NeckErgonomicsCost, ExponentialArmErgonomicsCost, and
    SemanticAgreementCost functions as well as the EuclideanDistance function.

    3-dimensional function (usually evaluated on `[-2.0, 2.0]^3`):

        f(x) = w_neck * NeckErgonomicsCost(x) + w_arm * ExponentialArmErgonomicsCost(x) + w_semantic * SemanticAgreementCost(x) + w_euclidean * EuclideanDistance(x)

    where w_neck, w_arm, w_semantic, and w_euclidean are the weights for the NeckErgonomicsCost, ExponentialArmErgonomicsCost, SemanticAgreementCost and EuclideanDistance functions, respectively.
    """

    dim = 3
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        weight_neck: float = 0.25,
        weight_arm: float = 0.25,
        weight_semantic: float = 0.25,
        weight_euclidean: float = 0.25,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            weight_neck: The weight for the NeckErgonomicsCost function.
            weight_arm: The weight for the ExponentialArmErgonomicsCost function.
            weight_semantic: The weight for the SemanticAgreementCost function.
            weight_euclidean: The weight for the EuclideanDistance function.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 3
        if bounds is None:
            bounds = [(-2., 2.) for _ in range(self.dim)]

        self.weight_neck = weight_neck
        self.weight_arm = weight_arm
        self.weight_semantic = weight_semantic
        self.weight_euclidean = weight_euclidean

        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

        self.neck_ergonomics_cost = NeckErgonomicsCost(eye_position=torch.tensor([0.0, 0.0, 0.0]))
        self.exp_arm_ergonomics_cost = ExponentialArmErgonomicsCost(shoulder_joint_position=torch.tensor([0.0, -0.2, 0.0]))
        self.semantic_agreement_cost = SemanticAgreementCost(
            object_positions=torch.tensor([[1., -0.5, 1.], [-1., -0.5, 1.]]),
            positive_association_scores=torch.tensor([1., -1.]),
            negative_association_scores=torch.tensor([-1., 1.]),
        )
        self.euclidean_distance = EuclideanDistance(optimizer=(0.3, -0.1, 0.4))

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return self.weight_neck * self.neck_ergonomics_cost(X) + self.weight_arm * self.exp_arm_ergonomics_cost(X) + self.weight_semantic * self.semantic_agreement_cost(X) + self.weight_euclidean * self.euclidean_distance(X)

    
if __name__ == "__main__":
    mr_layout_quality = MRLayoutQualityLoss()
    X = torch.rand(2, 3)
    print(mr_layout_quality(X))
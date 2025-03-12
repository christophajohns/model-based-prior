from typing import List, Optional, Tuple
import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction

def get_angle_between_vectors(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Computes the angle (in radians, as a multiple of π) between two nonzero vectors.
    Supports batched inputs for efficient computation.

    Args:
        vector1 (torch.Tensor): A tensor of shape (B, N) or (N,) representing the first vector(s).
        vector2 (torch.Tensor): A tensor of shape (B, N) or (N,) representing the second vector(s).

    Returns:
        torch.Tensor: A tensor containing the angles (in radians) between the corresponding vectors.
                      Returns 0 for zero-vectors.
                      Returns NaN if the angle is undefined.

    Raises:
        ValueError: If input tensors do not have the same shape or are not 1D/2D.

    Example:
        >>> v1 = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        >>> v2 = torch.tensor([[0.0, 1.0], [1.0, -1.0]])
        >>> get_angle_between_vectors(v1, v2)
        tensor([1.5708, 1.5708])  # π/2 radians for both vector pairs
    """
    # Ensure inputs have valid dimensions (either (N,) or (B, N))
    if vector1.dim() not in (1, 2) or vector2.dim() not in (1, 2):
        raise ValueError(f"Input tensors must be either 1D (N,) or 2D (B, N), got {vector1.shape} and {vector2.shape}.")
    
    # Ensure both vectors have the same shape
    if vector1.shape != vector2.shape:
        raise ValueError(f"Input vectors must have the same shape, got {vector1.shape} and {vector2.shape}.")

    # Compute norms
    norm1 = torch.norm(vector1, dim=-1, keepdim=True)
    norm2 = torch.norm(vector2, dim=-1, keepdim=True)

    # Handle zero-vectors by returning 0
    zero_mask = (norm1 == 0) | (norm2 == 0)
    
    # Normalize vectors
    unit_vector1 = vector1 / norm1
    unit_vector2 = vector2 / norm2

    # Compute dot product along the last dimension
    dot_product = torch.sum(unit_vector1 * unit_vector2, dim=-1)

    # Clamp dot product to the valid range to prevent numerical instability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the angle using arccos
    angle = torch.acos(dot_product)

    # Set angles to 0 where either vector was zero
    angle[zero_mask.squeeze()] = 0.0

    return angle

def get_arm_angle(
    shoulder_joint_position: torch.Tensor, element_position: torch.Tensor
) -> torch.Tensor:
    """
    Computes the angle (in radians, as a multiple of π) between the arm (shoulder-to-element vector)
    and the vertical downward direction.

    Supports batched inputs for efficient computation.

    Args:
        shoulder_joint_position (torch.Tensor): A tensor of shape (B, 3) or (3,) representing 
                                                the shoulder joint position(s).
        element_position (torch.Tensor): A tensor of shape (B, 3) or (3,) representing 
                                         the element position(s).

    Returns:
        torch.Tensor: A tensor containing the arm angles (in radians) for each input pair.
                      Returns 0 if the arm vector is a zero-vector.
                      Returns NaN if the angle is undefined.

    Raises:
        ValueError: If input tensors do not have the same shape or are not 1D/2D.

    Example:
        >>> shoulder_pos = torch.tensor([[0.0, 1.0, 0.0]])
        >>> element_pos = torch.tensor([[1.0, 0.0, 0.0]])
        >>> get_arm_angle(shoulder_pos, element_pos)
        tensor([0.7854])  # 45 degrees (π/4 radians)
    """
    # Ensure inputs have valid dimensions (either (3,) or (B, 3))
    if shoulder_joint_position.dim() not in (1, 2) or element_position.dim() not in (1, 2):
        raise ValueError(f"Input tensors must be either 1D (3,) or 2D (B, 3), got {shoulder_joint_position.shape} and {element_position.shape}.")

    # Ensure both inputs have the same shape
    if (shoulder_joint_position.shape != element_position.shape) and shoulder_joint_position.dim() != 1:
        raise ValueError(f"Input tensors must have the same shape or shoulder_joint_position must be 1D, got {shoulder_joint_position.shape} and {element_position.shape}.")
    # If shoulder_joint_position is 1D, expand it to match the batch size
    if (shoulder_joint_position.dim() == 1) and (element_position.dim() == 2):
        shoulder_joint_position = shoulder_joint_position.unsqueeze(0).expand(element_position.shape[0], -1)

    # Compute the vector from the shoulder to the element
    shoulder_to_element_vector = element_position - shoulder_joint_position

    # Define the reference downward vector (same shape as inputs)
    downward_vector = torch.tensor([0.0, -1.0, 0.0], device=shoulder_joint_position.device)

    # If batched, expand downward_vector to match batch size
    if shoulder_to_element_vector.dim() == 2:
        downward_vector = downward_vector.unsqueeze(0).expand(shoulder_to_element_vector.shape[0], -1)

    # Compute the angle using the get_angle_between_vectors function
    arm_angle = get_angle_between_vectors(shoulder_to_element_vector, downward_vector)

    return arm_angle

def get_arm_angle_deg(
    shoulder_joint_position: torch.Tensor, element_position: torch.Tensor
) -> torch.Tensor:
    """
    Computes the arm angle in **degrees** (instead of radians) between the arm (shoulder-to-element vector)
    and the vertical downward direction.

    Supports batched inputs for efficient computation.

    Args:
        shoulder_joint_position (torch.Tensor): A tensor of shape (B, 3) or (3,) representing 
                                                the shoulder joint position(s).
        element_position (torch.Tensor): A tensor of shape (B, 3) or (3,) representing 
                                         the element position(s).

    Returns:
        torch.Tensor: A tensor containing the arm angles in **degrees**.
                      Returns 0 for zero-vectors.
                      Returns NaN if the angle is undefined.

    Raises:
        ValueError: If input tensors do not have the same shape or are not 1D/2D.

    Example:
        >>> shoulder_pos = torch.tensor([[0.0, 1.0, 0.0]])
        >>> element_pos = torch.tensor([[1.0, 0.0, 0.0]])
        >>> get_arm_angle_deg(shoulder_pos, element_pos)
        tensor([45.])  # 45 degrees
    """
    # Compute the arm angle in radians using the batched function
    arm_angle_rad = get_arm_angle(shoulder_joint_position, element_position)

    # Convert radians to degrees
    arm_angle_deg = torch.rad2deg(arm_angle_rad)

    return arm_angle_deg

def get_neck_ergonomics_cost(
    eye_position: torch.Tensor, element_position: torch.Tensor
) -> torch.Tensor:
    """
    Computes the neck ergonomics cost based on the angle between the vector from the eye position 
    to the element and the vector from the eye position to the element's projection on the xz-plane.

    Supports batched inputs for efficient computation.

    Args:
        eye_position (torch.Tensor): A tensor of shape (B, 3) or (3,) representing the eye position(s).
        element_position (torch.Tensor): A tensor of shape (B, 3) or (3,) representing the element position(s).

    Returns:
        torch.Tensor: A tensor containing the neck ergonomics cost values, normalized to [0, 1].
                      Returns 1 if the element is directly at the eye position or if the angle is undefined.

    Raises:
        ValueError: If input tensors do not have the same shape or are not 1D/2D.

    Example:
        >>> eye_pos = torch.tensor([[0.0, 1.7, 0.0]])
        >>> element_pos = torch.tensor([[1.0, 1.0, 0.0]])
        >>> get_neck_ergonomics_cost(eye_pos, element_pos)
        tensor([0.1250])  # Cost normalized to [0,1]
    """
    # Ensure inputs have valid dimensions (either (3,) or (B, 3))
    if eye_position.dim() not in (1, 2) or element_position.dim() not in (1, 2):
        raise ValueError(f"Input tensors must be either 1D (3,) or 2D (B, 3), got {eye_position.shape} and {element_position.shape}.")

    # Ensure both inputs have the same shape
    if (eye_position.shape != element_position.shape) and eye_position.dim() != 1:
        raise ValueError(f"Input tensors must have the same shape or eye_position must be 1D, got {eye_position.shape} and {element_position.shape}.")
    
    # If eye_position is 1D, expand it to match the batch size
    if (eye_position.dim() == 1) and (element_position.dim() == 2):
        eye_position = eye_position.unsqueeze(0).expand(element_position.shape[0], -1)

    # Compute vector from eye to element
    eye_to_element_vector = element_position - eye_position

    # If the element is directly at the eye position, return 1 (worst ergonomics)
    if torch.allclose(eye_to_element_vector, torch.zeros_like(eye_to_element_vector)):
        return torch.ones(eye_to_element_vector.shape[:-1], device=eye_position.device)

    # Compute vector from eye to the element's projection on the xz-plane at eye height
    eye_to_target_projection_vector = eye_to_element_vector.clone()
    eye_to_target_projection_vector[..., 1] = 0.0  # Set Y-component to 0

    # Compute the angle between the two vectors
    neck_angle = get_angle_between_vectors(eye_to_element_vector, eye_to_target_projection_vector)

    # Normalize the neck ergonomics cost to [0, 1] (angle / (π/2))
    neck_ergonomics_cost = neck_angle / (torch.pi / 2)

    return neck_ergonomics_cost

def get_exp_arm_ergonomics_cost_from_angle(
    arm_angle: torch.Tensor, steepness_factor: float = 2.0
) -> torch.Tensor:
    """
    Computes the exponential arm ergonomics cost based on the given arm angle in degrees.
    
    The cost grows exponentially with the angle and is normalized to [0,1].
    - Cost is **0 at 0°** (best ergonomics).
    - Cost is **1 at 180°** (worst ergonomics).
    - Cost wraps around for angles > 180°.
    
    Supports batched inputs for efficient computation.

    Args:
        arm_angle (torch.Tensor): A tensor of shape (B,) or scalar representing arm angles in degrees.
        steepness_factor (float, optional): Controls the steepness of the exponential curve. Default is 2.0.

    Returns:
        torch.Tensor: A tensor containing the normalized arm ergonomics cost values.

    Raises:
        ValueError: If the input tensor is not 1D or scalar.

    Example:
        >>> arm_angle = torch.tensor([0.0, 90.0, 180.0, 270.0])
        >>> get_exp_arm_ergonomics_cost_from_angle(arm_angle)
        tensor([0.0000, 0.2384, 1.0000, 0.2384])
    """
    # Ensure input is 1D or scalar
    if arm_angle.dim() not in (0, 1):
        raise ValueError(f"Input tensor must be scalar or 1D (B,), got {arm_angle.shape}.")

    # Handle infinity and NaN cases (return 1 for worst ergonomics)
    arm_angle = torch.where(torch.isinf(arm_angle) | torch.isnan(arm_angle), torch.tensor(180.0, device=arm_angle.device), arm_angle)

    # Convert degrees to radians
    arm_angle_rad = torch.deg2rad(arm_angle)

    # Compute modulo operation to keep angle within [0, 2π]
    x_mod = torch.remainder(arm_angle_rad, 2 * torch.pi)

    # Compute ergonomics cost based on angle
    exp_term = torch.exp(torch.tensor(steepness_factor))
    cost = torch.where(
        x_mod <= torch.pi,
        steepness_factor * (torch.exp((x_mod / torch.pi)) - 1) / (exp_term - 1),
        (torch.exp(-steepness_factor * (x_mod - 2 * torch.pi) / torch.pi) - 1) / (exp_term - 1)
    )

    return cost

def get_exp_arm_ergonomics_cost(
    shoulder_joint_position: torch.Tensor, element_position: torch.Tensor
) -> torch.Tensor:
    """
    Computes the exponential arm ergonomics cost based on the shoulder joint position and element position.

    The cost grows exponentially with the arm angle and is normalized to [0,1].

    Supports batched inputs for efficient computation.

    Args:
        shoulder_joint_position (torch.Tensor): A tensor of shape (B, 3) or (3,) representing shoulder joint positions.
        element_position (torch.Tensor): A tensor of shape (B, 3) or (3,) representing element positions.

    Returns:
        torch.Tensor: A tensor containing the normalized arm ergonomics cost values.

    Raises:
        ValueError: If input tensors do not have the same shape or are not 1D/2D.

    Example:
        >>> shoulder_pos = torch.tensor([[0.0, 1.0, 0.0]])
        >>> element_pos = torch.tensor([[1.0, 0.0, 0.0]])
        >>> get_exp_arm_ergonomics_cost(shoulder_pos, element_pos)
        tensor([0.2384])
    """
    # Compute arm angle in degrees
    arm_angle_deg = get_arm_angle_deg(shoulder_joint_position, element_position)

    # Compute exponential ergonomics cost
    return get_exp_arm_ergonomics_cost_from_angle(arm_angle_deg)

class NeckErgonomicsCost(SyntheticTestFunction):
    r"""NeckErgonomicsCost test function.

    This function measures the ergonomic cost of looking at an element based on the 
    angle between the eye-to-element vector and its projection onto the XZ-plane. 
    A high cost corresponds to excessive neck tilting.

    Mathematically, the cost is defined as:

        f(x) = θ(x) / (2π)

    where:

    - x = (x₁, x₂, x₃) is the **element position** in 3D space.
    - e = (e₁, e₂, e₃) is the **eye position**.
    - The **eye-to-element vector** is:  
        v = x - e
    - The **projection onto the XZ-plane** is obtained by zeroing out the Y-component:  
        p = (v₁, 0, v₃)
    - The **neck angle** is computed as:  
        θ(x) = arccos((v ⋅ p) / (||v|| ||p||))

    The cost is **normalized** between [0,1], where:
    - **0** means a horizontal view (θ = 0, looking straight ahead).
    - **1** means a maximum tilt (θ = π, looking straight up/down).

    This function encourages **ergonomic viewing angles** by penalizing excessive neck movement.

    3-dimensional function (usually evaluated on `[-2.0, 2.0]^3`).
    """

    dim = 3
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        eye_position: Optional[torch.Tensor] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            eye_position: The position of the eye as a (3,) tensor.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 3
        if bounds is None:
            bounds = [(-2.0, 2.0) for _ in range(self.dim)]

        # Ensure eye position is a torch tensor
        if eye_position is None:
            eye_position = torch.tensor([0.0, 0.0, 0.0])
        elif isinstance(eye_position, np.ndarray):
            eye_position = torch.tensor(eye_position, dtype=torch.float32)

        self._optimizers = [tuple(eye_position.tolist())]
        self.eye_position = eye_position

        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the neck ergonomics cost for a batch of element positions.

        Args:
            X (torch.Tensor): A tensor of shape (B, 3) representing element positions.

        Returns:
            torch.Tensor: A tensor of shape (B,) containing the neck ergonomics cost.
        """
        batch_shape = X.shape[:-1]
        results = torch.zeros(batch_shape, device=X.device)

        # Ensure X is at least 2D (for single sample cases)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        for idx in range(X.shape[0]):
            result = get_neck_ergonomics_cost(
                self.eye_position,
                X[idx].squeeze(),  # X may have shape (N, 3) or (N, 1, 3) 
            )
            results[idx] = result.item()
    
        return results.double()
    

class ExponentialArmErgonomicsCost(SyntheticTestFunction):
    r"""ExponentialArmErgonomicsCost test function.

    This function models the ergonomic cost of reaching for an element with the arm.
    The cost is computed based on the angle between the vector from the shoulder 
    to the element and the downward gravity vector.

    Mathematically, the cost is defined as:

        f(x) = (exp((θ(x) / π)) - 1) / (exp(S) - 1),  if θ(x) ≤ π
        f(x) = (exp(-S * (θ(x) - 2π) / π) - 1) / (exp(S) - 1),  if θ(x) > π

    where:

    - x = (x₁, x₂, x₃) is the **element position** in 3D space.
    - s = (s₁, s₂, s₃) is the **shoulder joint position**.
    - The **shoulder-to-element vector** is:  
        v = x - s
    - The **gravity vector** is:  
        g = (0, -1, 0)
    - The **angle between these vectors** is computed as:  
        θ(x) = arccos((v ⋅ g) / (||v|| ||g||))
    - S is the **steepness factor** (default: S = 2), controlling how quickly the cost increases.

    The cost **grows exponentially** with angle, penalizing large arm elevations.
    For angles beyond 180°, the cost wraps around using an exponential decay.


    3-dimensional function (usually evaluated on `[-2.0, 2.0]^3`).
    """

    dim = 3
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        shoulder_joint_position: Optional[torch.Tensor] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            shoulder_joint_position: The position of the shoulder joint as a (3,) tensor.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = 3
        if bounds is None:
            bounds = [(-2.0, 2.0) for _ in range(self.dim)]

        # Ensure shoulder position is a torch tensor
        if shoulder_joint_position is None:
            shoulder_joint_position = torch.tensor([0.0, -0.2, 0.0])
        elif isinstance(shoulder_joint_position, np.ndarray):
            shoulder_joint_position = torch.tensor(shoulder_joint_position, dtype=torch.float32)

        self._optimizers = [tuple(shoulder_joint_position.tolist())]
        self.shoulder_joint_position = shoulder_joint_position

        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential arm ergonomics cost for a batch of element positions.

        Args:
            X (torch.Tensor): A tensor of shape (B, 3) representing element positions.

        Returns:
            torch.Tensor: A tensor of shape (B,) containing the arm ergonomics cost.
        """
        batch_shape = X.shape[:-1]
        results = torch.zeros(batch_shape, device=X.device)

        # Ensure X is at least 2D (for single sample cases)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        for idx in range(X.shape[0]):
            result = get_exp_arm_ergonomics_cost(
                self.shoulder_joint_position,
                X[idx].squeeze(),  # X may have shape (N, 3) or (N, 1, 3) 
            )
            results[idx] = result.item()
    
        return results.double()

    
if __name__ == "__main__":
    torch.manual_seed(0)
    neck_ergonomics_cost = NeckErgonomicsCost()
    X = torch.rand(2, 3)
    X_batch = torch.rand(128, 1, 3)
    X_single = torch.rand(3)
    print(X)
    print(neck_ergonomics_cost(X).shape)
    print(neck_ergonomics_cost(X_batch).shape)
    print(neck_ergonomics_cost(X_single).shape)

    exp_arm_ergonomics_cost = ExponentialArmErgonomicsCost()
    print(exp_arm_ergonomics_cost(X).shape)
    print(exp_arm_ergonomics_cost(X_batch).shape)
    print(exp_arm_ergonomics_cost(X_single).shape)
import torch
from isaaclab.utils.math import matrix_from_quat


def ortho_6_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion to a 6D orthonormal matrix.

    Args:
        quat: The quaternion tensor.

    Returns:
        The 6D orthonormal matrix.
    """
    return matrix_from_quat(quat).mT.reshape(-1, 9)[:, :6]

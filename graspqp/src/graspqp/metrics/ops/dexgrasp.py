import torch


@torch.jit.script
def calc_e_fc(
    contact_pts: torch.Tensor,
    contact_normals: torch.Tensor,
    torque_weight: float = 1.0
) -> torch.Tensor:
    """
    Calculate the force closure metric (E_fc) for a batch of contact points and normals.
    Args:
        contact_pts (torch.Tensor): Tensor of shape (batch_size, n_contact, 3) representing contact points.
        contact_normals (torch.Tensor): Tensor of shape (batch_size, n_contact, 3) representing contact normals.
        torque_weight (float): Weight for the torque component.
    Returns:
        torch.Tensor: Scalar metric for each batch element.
    """
    batch_size, n_contact, _ = contact_pts.shape
    # Reshape normals for matrix multiplication
    contact_normals = contact_normals.reshape(batch_size, 1, 3 * n_contact)
    # Transformation matrix for torque computation
    transformation_matrix = torch.tensor(
        [
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0]
        ],
        dtype=torch.float, device=contact_pts.device
    ) * torque_weight
    # Build grasp matrix g
    eye = torch.eye(3, dtype=torch.float, device=contact_pts.device)
    eye_expanded = eye.expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3)
    torque = (contact_pts @ transformation_matrix).view(batch_size, 3 * n_contact, 3)
    g = torch.cat([eye_expanded, torque], dim=2).float().to(contact_pts.device)
    # Compute norm
    norm = torch.norm(contact_normals @ g, dim=[1, 2])
    return norm * norm


class DexgraspSpanMetric(torch.nn.Module):
    """
    Metric for evaluating the span of dexterous grasps.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        contact_pts: torch.Tensor,
        contact_normals: torch.Tensor,
        cog: torch.Tensor,
        torque_weight: float = 0.0,
        with_solution: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            contact_pts (torch.Tensor): (batch_size, n_contact, 3) contact points
            contact_normals (torch.Tensor): (batch_size, n_contact, 3) contact normals
            cog (torch.Tensor): (batch_size, 3) center of gravity
            torque_weight (float): weight for torque term
            with_solution (bool): whether to return dummy solution tensor
        Returns:
            torch.Tensor: metric value (and dummy solution if requested)
        """
        # Center contact points by subtracting center of gravity
        contact_pts = contact_pts - cog.unsqueeze(1)
        if with_solution:
            # Return metric and dummy solution tensor
            return calc_e_fc(contact_pts, contact_normals, torque_weight), torch.ones_like(contact_pts[..., 0])
        return calc_e_fc(contact_pts, contact_normals, torque_weight)

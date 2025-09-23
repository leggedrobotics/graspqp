import pytest
import torch

from graspqp.metrics.ops.span import OverallFrictionConeSpanMetric


def test_cpu():
    metric = OverallFrictionConeSpanMetric.from_dim(num_wrenches=12, wrench_dim=6, device="cpu")

    batch_size = 16
    contact_pts = torch.rand(batch_size, 12, 3)
    contact_normals = torch.rand(batch_size, 12, 3)
    contact_normals = contact_normals / torch.norm(contact_normals, dim=-1, keepdim=True)
    cog = torch.zeros(batch_size, 3)

    metric(contact_pts, contact_normals, cog=cog)


def test_gpu():
    metric = OverallFrictionConeSpanMetric.from_dim(num_wrenches=12, wrench_dim=6)

    batch_size = 16
    contact_pts = torch.rand(batch_size, 12, 3)
    contact_normals = torch.rand(batch_size, 12, 3)
    contact_normals = contact_normals / torch.norm(contact_normals, dim=-1, keepdim=True)
    cog = torch.zeros(batch_size, 3)

    metric = metric.to("cuda")
    metric(contact_pts.cuda(), contact_normals.cuda(), cog=cog.cuda())


if __name__ == "__main__":
    pytest.main([__file__, "-s"])

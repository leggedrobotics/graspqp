import torch
import pytest

from graspqp.metrics import DexgraspSpanMetric, TDGSpanMetric, GraspQPSpanMetric


def test_cpu():
    for metric in [
        DexgraspSpanMetric(),
        TDGSpanMetric(device="cpu"),
        GraspQPSpanMetric(),
    ]:
        print("Testing ", metric.__class__.__name__)

        batch_size = 16
        contact_pts = torch.rand(batch_size, 12, 3)
        contact_normals = torch.rand(batch_size, 12, 3)
        contact_normals = contact_normals / torch.norm(
            contact_normals, dim=-1, keepdim=True
        )
        cog = torch.zeros(batch_size, 3)
        metric = metric.to("cpu")

        metric(contact_pts, contact_normals, cog=cog, torque_weight=1.0)


def test_gpu():
    for metric in [
        DexgraspSpanMetric(),
        TDGSpanMetric(device="cuda"),
        GraspQPSpanMetric(),
    ]:
        print("Testing ", metric.__class__.__name__)

        batch_size = 16
        contact_pts = torch.rand(batch_size, 12, 3).to("cuda")
        contact_normals = torch.rand(batch_size, 12, 3).to("cuda")
        contact_normals = contact_normals / torch.norm(
            contact_normals, dim=-1, keepdim=True
        )
        cog = torch.zeros(batch_size, 3).to("cuda")
        metric = metric.to("cuda")

        metric(contact_pts, contact_normals, cog=cog, torque_weight=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])

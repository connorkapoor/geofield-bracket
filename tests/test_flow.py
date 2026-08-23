"""Flow model tests: shapes, CFM loss finiteness, OT pairing, and
SE(3)-equivariance of the velocity field w.r.t. condition tokens."""
import torch

from geofield.fields.programs.common import random_rotation
from geofield.model.encoder import N_VEC
from geofield.model.flow import LatentFlow, LatentNormalizer, cfm_loss, ot_pair, sample
from geofield.tokens.schema import Token

torch.manual_seed(0)


def _flow():
    return LatentFlow(latent_dim=32, m_fine=16, m_coarse=4, dim=48,
                      n_layers=2, k=8).eval()


def _z(B=2, flow=None):
    M = flow.m_fine + flow.m_coarse
    D = 3 + flow.latent_dim + N_VEC * 3
    return torch.randn(B, M, D)


def _cond(B=2):
    return [[Token("fixed_point", [0.2, 0.1, 0.0, 0, 0, 0, 1],
                   {"axis": [0.0, 0, 1], "diameter": 0.1, "fixed": True}),
             Token("load", [-0.2, 0.0, 0.1, 0, 0, 0, 1],
                   {"direction": [1.0, 0, 0], "magnitude": 300.0, "kind": "force"})]
            for _ in range(B)]


def test_velocity_shapes_and_loss():
    flow = _flow()
    z1 = _z(flow=flow)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        v = flow(z1, torch.tensor([0.3, 0.7]), _cond())
    assert v.shape == z1.shape
    loss = cfm_loss(flow, z1, _cond(), gen)
    assert torch.isfinite(loss)


def test_ot_pairing_reduces_cost():
    flow = _flow()
    z0, z1 = _z(flow=flow), _z(flow=flow)
    z0p = ot_pair(z0, z1, flow.m_fine)
    before = (z0[..., :3] - z1[..., :3]).pow(2).sum(-1).mean()
    after = (z0p[..., :3] - z1[..., :3]).pow(2).sum(-1).mean()
    assert after <= before + 1e-6
    # permutation preserves the multiset of rows
    assert torch.allclose(z0.sort(dim=1).values, z0p.sort(dim=1).values)


def test_velocity_equivariance():
    """Rotating (z poses+vectors, cond tokens) rotates the velocity."""
    flow = _flow()
    z = _z(B=1, flow=flow)
    cond = _cond(B=1)
    g = torch.Generator().manual_seed(3)
    R = random_rotation(g)
    ld = flow.latent_dim

    def rot_z(z, R):
        out = z.clone()
        out[..., :3] = z[..., :3] @ R.T
        v = z[..., 3 + ld:].reshape(*z.shape[:2], N_VEC, 3)
        out[..., 3 + ld:] = (v @ R.T).reshape(*z.shape[:2], N_VEC * 3)
        return out

    t = torch.tensor([0.5])
    with torch.no_grad():
        v1 = flow(z, t, cond)
        cond_r = [[tok.transform(R, torch.zeros(3)) for tok in c] for c in cond]
        v2 = flow(rot_z(z, R), t, cond_r)
    expected = rot_z(v1, R)
    err = (v2 - expected).abs().max().item()
    scale = v1.abs().max().item() + 1e-9
    assert err / scale < 1e-4, f"flow equivariance error {err / scale:.2e}"


def test_sampling_runs_and_normalizer():
    flow = _flow()
    z = sample(flow, _cond(2), n=2, steps=4)
    assert z.shape == _z(flow=flow).shape
    assert torch.isfinite(z).all()
    norm = LatentNormalizer.fit(_z(B=8, flow=flow))
    z1 = _z(flow=flow)
    assert torch.allclose(norm.denorm(norm.norm(z1)), z1, atol=1e-5)

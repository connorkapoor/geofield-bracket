"""Model tests: shapes, equivariance of the full encode/decode path,
latent serialization roundtrip, and head registry extension."""
import pytest
import torch

from geofield.fields.programs.common import random_rotation
from geofield.model.decoder import GeoFieldModel
from geofield.model.encoder import farthest_point_indices
from geofield.model.heads import ScalarHead, field_specs, register_field
from geofield.model.invariants import pairwise_invariants
from geofield.model.latent import LatentSet
from geofield.tokens.schema import Token

torch.manual_seed(0)


def _small_model():
    return GeoFieldModel(dim=64, n_layers=2, m_fine=32, m_coarse=8, k_input=16)


def _batch(B=2, N=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, 3, generator=g) * 0.5
    f = torch.randn(B, N, generator=g) * 0.1
    grad = torch.randn(B, N, 3, generator=g)
    grad = grad / torch.linalg.vector_norm(grad, dim=-1, keepdim=True)
    return x, f, grad


def _tokens(B=2):
    return [[Token("load", [0.1, 0.2, 0.3, 0, 0, 0, 1],
                   {"direction": [0.0, 0, -1.0], "magnitude": 500.0, "kind": "force"}),
             Token("build_dir", None, {"direction": [0.0, 0, 1.0]})]
            for _ in range(B)]


def test_invariants_are_invariant():
    g = torch.Generator().manual_seed(1)
    R = random_rotation(g)
    t = torch.randn(3, generator=g)
    xi, xj = torch.randn(64, 3), torch.randn(64, 3)
    gi = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    gj = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
    fi, fj = torch.randn(64), torch.randn(64)
    a = pairwise_invariants(xi, gi, fi, xj, gj, fj)
    b = pairwise_invariants(xi @ R.T + t, gi @ R.T, fi, xj @ R.T + t, gj @ R.T, fj)
    assert torch.allclose(a, b, atol=1e-5)


def test_fps_deterministic_and_spread():
    x = torch.randn(2, 256, 3)
    i1 = farthest_point_indices(x, 16)
    i2 = farthest_point_indices(x, 16)
    assert torch.equal(i1, i2)
    assert i1.unique(dim=-1).shape[-1] == 16  # no duplicates


def test_forward_shapes():
    model = _small_model().eval()
    x, f, grad = _batch()
    toks = _tokens()
    with torch.no_grad():
        lat = model.encode(x, f, grad, toks)
        out = model.decode(lat, x[:, :32], "sdf")
    assert lat.fine.h.shape == (2, 32, 64)
    assert lat.coarse.h.shape == (2, 8, 64)
    assert out.shape == (2, 32, 1)
    assert lat.pooled().shape == (2, 64)


def test_sdf_grad_autograd():
    model = _small_model().eval()
    x, f, grad = _batch()
    lat = model.encode(x, f, grad)
    fq, gq = model.sdf_and_grad(lat, x[:, :16])
    assert fq.shape == (2, 16)
    assert gq.shape == (2, 16, 3)
    assert torch.isfinite(gq).all()


@pytest.mark.parametrize("field_id,shape", [("sdf", 1), ("displacement", 3),
                                            ("support_need", 1)])
def test_equivariance_encode_decode(field_id, shape):
    """Rotating inputs+tokens+queries: invariant fields unchanged, vector
    fields rotate. Untrained model, double precision tolerance ~1e-4."""
    model = _small_model().eval()
    x, f, grad = _batch(B=1, N=128)
    toks = _tokens(B=1)
    q = x[:, :48]
    g = torch.Generator().manual_seed(5)
    R = random_rotation(g)
    t = torch.randn(3, generator=g) * 0.2

    with torch.no_grad():
        lat1 = model.encode(x, f, grad, toks)
        out1 = model.decode(lat1, q, field_id, toks)
        toks_r = [[tok.transform(R, t) for tok in rec] for rec in toks]
        lat2 = model.encode(x @ R.T + t, f, grad @ R.T, toks_r)
        out2 = model.decode(lat2, q @ R.T + t, field_id, toks_r)

    if field_id == "displacement":
        expected = out1 @ R.T
    else:
        expected = out1
    err = (out2 - expected).abs().max().item()
    scale = out1.abs().max().item() + 1e-6
    assert err / scale < 5e-3, f"equivariance error {err / scale:.2e}"


def test_latent_flatten_roundtrip():
    model = _small_model().eval()
    x, f, grad = _batch()
    with torch.no_grad():
        lat = model.encode(x, f, grad)
    z = lat.flatten()
    assert z.shape == (2, 32 + 8, 3 + 64 + 3 * 8)
    lat2 = LatentSet.unflatten(z, m_fine=32, dim=64)
    assert torch.allclose(lat2.fine.pos, lat.fine.pos)
    assert torch.allclose(lat2.coarse.h, lat.coarse.h)
    assert torch.allclose(lat2.fine.v, lat.fine.v)


def test_register_field_extends_decoder():
    register_field("test_thermal", 1, "l1", ScalarHead, equivariant=False,
                   cond_types=("load",))
    assert "test_thermal" in field_specs()
    model = _small_model().eval()  # heads built AFTER registration include it
    x, f, grad = _batch()
    with torch.no_grad():
        lat = model.encode(x, f, grad)
        out = model.decode(lat, x[:, :8], "test_thermal", _tokens())
    assert out.shape == (2, 8, 1)


def test_masked_completion_inputs():
    """Encoder must accept a reduced input set (masking drops tokens)."""
    model = _small_model().eval()
    x, f, grad = _batch()
    with torch.no_grad():
        lat = model.encode(x[:, :64], f[:, :64], grad[:, :64])
        out = model.decode(lat, x[:, 64:96], "sdf")
    assert out.shape == (2, 32, 1)

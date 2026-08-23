"""Tests for the token schema: registry, transforms, serialization."""
import warnings

import pytest
import torch

from geofield.fields.programs.common import random_rotation
from geofield.tokens.schema import (
    Token, filter_known, quat_to_rotmat, registered_token_types,
    register_token_type, rotmat_to_quat, tokens_from_json, tokens_to_json)


def test_phase1_types_registered():
    for t in ["fixed_point", "load", "material", "build_dir", "spindle_dir", "envelope"]:
        assert t in registered_token_types()


def test_quat_roundtrip():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        R = random_rotation(g)
        q = rotmat_to_quat(R)
        assert torch.allclose(quat_to_rotmat(q), R, atol=1e-5)


def test_direction_params_normalized():
    t = Token("load", pose=[0, 0, 0, 0, 0, 0, 1],
              params={"direction": [3.0, 0, 0], "magnitude": 100.0, "kind": "force"})
    assert torch.allclose(t.params["direction"], torch.tensor([1.0, 0, 0]))


def test_token_transform_rotates_pose_and_directions():
    g = torch.Generator().manual_seed(1)
    R = random_rotation(g)
    t = torch.randn(3, generator=g)
    tok = Token("fixed_point", pose=[0.1, 0.2, 0.3, 0, 0, 0, 1],
                params={"axis": [0.0, 0, 1], "diameter": 0.1, "fixed": True})
    moved = tok.transform(R, t)
    assert torch.allclose(moved.position, R @ tok.position + t, atol=1e-5)
    assert torch.allclose(moved.params["axis"], R @ tok.params["axis"], atol=1e-5)
    assert moved.params["diameter"] == pytest.approx(0.1)
    # orientation part composes: R_new == R @ R_old
    R_old = quat_to_rotmat(tok.pose[3:])
    R_new = quat_to_rotmat(moved.pose[3:])
    assert torch.allclose(R_new, R @ R_old, atol=1e-4)


def test_global_token_transform():
    b = Token("build_dir", pose=None, params={"direction": [0.0, 0, 1]})
    g = torch.Generator().manual_seed(2)
    R = random_rotation(g)
    moved = b.transform(R, torch.zeros(3))
    assert moved.pose is None
    assert torch.allclose(moved.params["direction"], R @ torch.tensor([0.0, 0, 1]), atol=1e-5)


def test_scale():
    tok = Token("fixed_point", pose=[0.2, 0, 0, 0, 0, 0, 1],
                params={"axis": [0.0, 0, 1], "diameter": 0.1, "fixed": False})
    s = tok.scale(2.0)
    assert torch.allclose(s.position, torch.tensor([0.4, 0, 0]))
    assert s.params["diameter"] == pytest.approx(0.2)


def test_serialization_roundtrip():
    toks = [
        Token("fixed_point", [0.1, 0.2, 0.3, 0, 0, 0, 1],
              {"axis": [0.0, 0, 1], "diameter": 0.08, "fixed": True}),
        Token("material", None, {"E": 68.9e9, "nu": 0.33, "yield": 276e6,
                                 "density": 2700.0, "name": "Al6061"}),
    ]
    back = tokens_from_json(tokens_to_json(toks))
    assert back[0].type == "fixed_point"
    assert torch.allclose(back[0].pose, toks[0].pose)
    assert torch.allclose(back[0].params["axis"], toks[0].params["axis"])
    assert back[1].params["name"] == "Al6061"
    assert back[1].pose is None


def test_unknown_type_warns_and_filtered():
    with pytest.warns(UserWarning, match="Unknown token type"):
        tok = Token("warp_drive", None, {"power": 9000.0})
    assert filter_known([tok]) == []


def test_register_new_type_extends_without_backbone_edits():
    register_token_type("thermal_source", {"direction": "direction", "watts": "scalar"})
    tok = Token("thermal_source", pose=[0, 0, 0, 0, 0, 0, 1],
                params={"direction": [1.0, 0, 0], "watts": 5.0})
    assert filter_known([tok]) == [tok]

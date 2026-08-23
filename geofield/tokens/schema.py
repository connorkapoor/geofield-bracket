"""Token types, registry, serialization, and SE(3) transforms.

A Token is a typed condition attached to a shape: an interface point, a load,
a material, a process direction, an envelope. Tokens transform WITH the field:
`token.transform(R, t)` moves posed tokens rigidly and rotates every parameter
registered as a direction, so a rotated record stays self-consistent.

Poses are 7-vectors [px, py, pz, qx, qy, qz, qw] (position + unit quaternion,
scalar-last). Global tokens (build_dir, spindle_dir, material) have pose=None.

Unknown token types encountered at inference are ignored with a warning —
this is the extension contract: new domains add token types via
`register_token_type` without touching consumers.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field as dc_field

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# param kinds: "scalar", "bool", "str", "direction" (unit 3-vector, rotates
# with the field), "vec3" (3-vector in local units, rotates AND scales with
# the field), "size3" (3-vector of body-frame dimensions: scales with the
# field but does NOT rotate — orientation lives in the pose quaternion).
_REGISTRY: dict[str, dict[str, str]] = {}


def register_token_type(name: str, param_schema: dict[str, str]) -> None:
    """Register a token type with its parameter kinds. Idempotent per name."""
    for k, kind in param_schema.items():
        assert kind in ("scalar", "bool", "str", "direction", "vec3", "size3"), \
            (name, k, kind)
    _REGISTRY[name] = dict(param_schema)


def token_type_schema(name: str) -> dict[str, str] | None:
    return _REGISTRY.get(name)


def registered_token_types() -> list[str]:
    return sorted(_REGISTRY)


# Phase-1 token types.
register_token_type("fixed_point", {"axis": "direction", "diameter": "scalar", "fixed": "bool"})
register_token_type("load", {"direction": "direction", "magnitude": "scalar", "kind": "str"})
register_token_type("material", {"E": "scalar", "nu": "scalar", "yield": "scalar",
                                 "density": "scalar", "name": "str"})
register_token_type("build_dir", {"direction": "direction"})
register_token_type("spindle_dir", {"direction": "direction"})
register_token_type("envelope", {"half_extents": "size3"})
# meters (or source units) per normalized unit; lets physics run in real units
register_token_type("scale", {"unit_mm": "scalar"})


# ---------------------------------------------------------------------------
# Quaternion helpers (scalar-last [qx, qy, qz, qw])
# ---------------------------------------------------------------------------

def quat_identity() -> Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 1.0])


def rotmat_to_quat(R: Tensor) -> Tensor:
    """3x3 rotation matrix -> unit quaternion, scalar-last."""
    R = R.reshape(3, 3)
    t = R.diagonal().sum()
    if t > 0:
        s = torch.sqrt(t + 1.0) * 2
        return torch.stack([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                            (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(R.diagonal().argmax())
    j, k = (i + 1) % 3, (i + 2) % 3
    s = torch.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = torch.empty(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q / torch.linalg.vector_norm(q)


def quat_to_rotmat(q: Tensor) -> Tensor:
    x, y, z, w = q
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type: str
    pose: Tensor | None = None                  # [7] or None for global tokens
    params: dict = dc_field(default_factory=dict)

    def __post_init__(self):
        if self.type not in _REGISTRY:
            warnings.warn(f"Unknown token type '{self.type}': token will be "
                          f"ignored by consumers.", stacklevel=2)
        if self.pose is not None:
            self.pose = torch.as_tensor(self.pose, dtype=torch.float32).reshape(7)
        schema = _REGISTRY.get(self.type, {})
        for k, kind in schema.items():
            if k in self.params and kind in ("direction", "vec3", "size3"):
                v = torch.as_tensor(self.params[k], dtype=torch.float32).reshape(3)
                if kind == "direction":
                    v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)
                self.params[k] = v

    @property
    def position(self) -> Tensor | None:
        return None if self.pose is None else self.pose[:3]

    def transform(self, R: Tensor, t: Tensor) -> "Token":
        R = torch.as_tensor(R, dtype=torch.float32)
        t = torch.as_tensor(t, dtype=torch.float32)
        pose = None
        if self.pose is not None:
            p = R @ self.pose[:3] + t
            q = quat_mul(rotmat_to_quat(R), self.pose[3:])
            pose = torch.cat([p, q])
        params = {}
        schema = _REGISTRY.get(self.type, {})
        for k, v in self.params.items():
            if schema.get(k) in ("direction", "vec3"):
                params[k] = R @ v
            else:  # size3 does not rotate: orientation is in the pose quat
                params[k] = v
        return Token(self.type, pose, params)

    def scale(self, s: float) -> "Token":
        """Uniform scale about origin (positions and vec3 params scale;
        directions and scalar params like diameter must be handled by the
        caller if they are in world units)."""
        pose = None
        if self.pose is not None:
            pose = torch.cat([self.pose[:3] * s, self.pose[3:]])
        params = {}
        schema = _REGISTRY.get(self.type, {})
        for k, v in self.params.items():
            if schema.get(k) in ("vec3", "size3"):
                params[k] = v * s
            elif k in ("diameter", "grip"):
                params[k] = float(v) * s
            else:
                params[k] = v
        return Token(self.type, pose, params)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        params = {k: (v.tolist() if isinstance(v, Tensor) else v)
                  for k, v in self.params.items()}
        return {"type": self.type,
                "pose": None if self.pose is None else self.pose.tolist(),
                "params": params}

    @staticmethod
    def from_dict(d: dict) -> "Token":
        return Token(d["type"], d.get("pose"), dict(d.get("params", {})))


def tokens_to_json(tokens: list[Token]) -> str:
    return json.dumps([t.to_dict() for t in tokens])


def tokens_from_json(s: str) -> list[Token]:
    return [Token.from_dict(d) for d in json.loads(s)]


def filter_known(tokens: list[Token]) -> list[Token]:
    """Drop tokens with unregistered types (warning already emitted)."""
    return [t for t in tokens if t.type in _REGISTRY]

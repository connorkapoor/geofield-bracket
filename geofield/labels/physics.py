"""GPU immersed voxel FEA: field + tokens -> von Mises stress + displacement.

No meshing. The field is voxelized on a regular grid of CUBIC elements over
either a symmetric cube (legacy: res/domain) or an arbitrary tight AABB
(bounds/h) — tight anisotropic domains are essential for plate-like parts,
where a bounding-sphere cube wastes resolution and leaves <2 voxels through
the plate thickness. Elements whose center has f < h/2 are active (sub-voxel
boundary softening via a clamped linear density ramp). Trilinear 8-node hex
elements, isotropic linear elasticity, matrix-free Jacobi-preconditioned CG
restricted to active-element DOFs.

Units: whatever the field is expressed in, interpreted as meters with E in Pa
and forces in newtons -> stresses in Pa. The L-bracket pipeline builds parts
in mm and wraps them with Scaled(field, 1e-3) + token.scale(1e-3) to solve in
true SI.

Boundary conditions come from tokens (fixed_point grips clamp solid nodes;
the load token spreads its force over solid nodes within its grip radius), or
from `extra_fixed` for benchmark-exact clamps.

L-bracket load cases: `l_load_cases` draws engineering-meaningful cases
(point on the free leg / free-leg bolt, direction biased toward gravity and
pull-out); magnitudes are CALIBRATED after a unit-force solve so peak stress
lands at a target fraction of yield (linear elasticity => exact rescale, no
extra solves).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..fields.primitives import Field
from ..tokens.schema import Token

# ---------------------------------------------------------------------------
# Element stiffness for a cube element of side h, E=1 (scaled at solve time)
# ---------------------------------------------------------------------------

_NODE_OFFSETS = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.long)


def _shape_grads(xi: Tensor) -> Tensor:
    """dN/dxi for the 8 trilinear shape functions at natural coords xi [3].
    Returns [8, 3] float64."""
    xi = xi.double()
    signs = _NODE_OFFSETS.double() * 2 - 1
    g = torch.empty(8, 3, dtype=torch.float64)
    for a in range(8):
        s = signs[a]
        g[a, 0] = 0.125 * s[0] * (1 + s[1] * xi[1]) * (1 + s[2] * xi[2])
        g[a, 1] = 0.125 * s[1] * (1 + s[0] * xi[0]) * (1 + s[2] * xi[2])
        g[a, 2] = 0.125 * s[2] * (1 + s[0] * xi[0]) * (1 + s[1] * xi[1])
    return g


def _elasticity_C(nu: float) -> Tensor:
    """6x6 isotropic elasticity for E = 1 (Voigt order xx,yy,zz,xy,yz,zx)."""
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1.0 / (2 * (1 + nu))
    C = torch.zeros(6, 6, dtype=torch.float64)
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _B_matrix(dNdx: Tensor) -> Tensor:
    """Strain-displacement matrix [6, 24] from shape grads [8, 3] (Voigt)."""
    B = torch.zeros(6, 24, dtype=torch.float64)
    for a in range(8):
        dx, dy, dz = dNdx[a]
        c = 3 * a
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c] = dy
        B[3, c + 1] = dx
        B[4, c + 1] = dz
        B[4, c + 2] = dy
        B[5, c] = dz
        B[5, c + 2] = dx
    return B


def element_stiffness(h: float, nu: float) -> tuple[Tensor, Tensor]:
    """(KE [24,24], B0 [6,24]) for a cube element of side h, E=1.
    B0 is the strain-displacement matrix at the element center (for stress)."""
    C = _elasticity_C(nu)
    gp = 1.0 / math.sqrt(3.0)
    KE = torch.zeros(24, 24, dtype=torch.float64)
    detJ = (h / 2) ** 3
    for sx in (-gp, gp):
        for sy in (-gp, gp):
            for sz in (-gp, gp):
                dNdx = _shape_grads(torch.tensor([sx, sy, sz])) * (2.0 / h)
                B = _B_matrix(dNdx)
                KE += B.T @ C @ B * detJ
    B0 = _B_matrix(_shape_grads(torch.zeros(3)) * (2.0 / h))
    return KE, B0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class FEAResult:
    von_mises: Tensor      # [Q]
    displacement: Tensor   # [Q, 3]
    mask: Tensor           # [Q] 1 where the query lies inside the solid
    peak_vm: float
    max_disp: float
    iters: int
    residual: float


class VoxelFEA:
    def __init__(self, res: int = 96, domain: float = 1.05,
                 bounds: tuple | None = None, h: float | None = None,
                 device: str | torch.device | None = None,
                 max_iter: int = 1200, tol: float = 1e-5,
                 max_dim: int = 256, dtype: torch.dtype = torch.float64):
        """dtype: CG working precision. Slender plate structures have stiffness
        condition numbers beyond fp32's reach (CG stalls above ~1e7); fp64 is
        the default and cheap here because only active elements participate."""
        """Either (res, domain) for a symmetric cube, or (bounds=(lo, hi), h)
        for a tight AABB with cubic elements of side h. If h would exceed
        max_dim cells on any axis it is coarsened to fit."""
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if bounds is not None:
            lo = torch.as_tensor(bounds[0], dtype=torch.float32)
            hi = torch.as_tensor(bounds[1], dtype=torch.float32)
            assert h is not None
            span = (hi - lo)
            h = max(h, float(span.max()) / max_dim)
            dims = torch.ceil(span / h).long().clamp_min(2)
            # recentre so the grid covers the AABB symmetrically
            pad = (dims.float() * h - span) / 2
            self.lo = lo - pad
            self.dims = tuple(int(d) for d in dims)
            self.h = float(h)
        else:
            self.lo = torch.full((3,), -domain)
            self.dims = (res, res, res)
            self.h = 2 * domain / res
        self.max_iter = max_iter
        self.tol = tol
        self.dtype = dtype

    # -- grid helpers --------------------------------------------------------

    def _axis_lin(self, i: int, nodes: bool) -> Tensor:
        n = self.dims[i] + (1 if nodes else 0)
        start = float(self.lo[i]) + (0.0 if nodes else self.h / 2)
        return start + self.h * torch.arange(n, device=self.device, dtype=torch.float32)

    def _element_centers(self) -> Tensor:
        cx, cy, cz = (self._axis_lin(i, nodes=False) for i in range(3))
        gx, gy, gz = torch.meshgrid(cx, cy, cz, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1)

    def _node_flat_index(self, ijk: Tensor) -> Tensor:
        ny1, nz1 = self.dims[1] + 1, self.dims[2] + 1
        return (ijk[..., 0] * ny1 + ijk[..., 1]) * nz1 + ijk[..., 2]

    # -- main entry -----------------------------------------------------------

    def solve(self, field: Field, tokens: list[Token], query_x: Tensor,
              extra_fixed=None) -> FEAResult:
        """extra_fixed: optional callable(positions[M,3]) -> bool[M] adding
        Dirichlet-clamped nodes (benchmarks needing exact clamp geometry)."""
        dev = self.device
        nx, ny, nz = self.dims
        h = self.h

        material = next(t for t in tokens if t.type == "material")
        E = float(material.params["E"])
        nu = float(material.params["nu"])
        load_tok = next(t for t in tokens if t.type == "load")
        fixed_toks = [t for t in tokens if t.type == "fixed_point" and t.params.get("fixed")]
        assert fixed_toks or extra_fixed is not None, \
            "physics.solve needs a fixed fixed_point token or extra_fixed"

        centers = self._element_centers()
        with torch.no_grad():
            fvals = _eval_chunked(field, centers.reshape(-1, 3), dev).reshape(nx, ny, nz)
        dens = ((0.5 - fvals / h).clamp(0.0, 1.0))
        active = dens > 0.05
        act_idx = active.nonzero(as_tuple=False)              # [nel, 3]
        nel = act_idx.shape[0]
        if nel == 0:
            raise ValueError("empty solid: no active elements")
        rho = dens[active]

        corners = act_idx.unsqueeze(1) + _NODE_OFFSETS.to(dev).unsqueeze(0)
        node_ids = self._node_flat_index(corners)              # [nel, 8]
        dof = (node_ids.unsqueeze(-1) * 3 + torch.arange(3, device=dev)).reshape(nel, 24)

        n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
        n_dof = n_nodes * 3

        dt = self.dtype
        KE, B0 = element_stiffness(h, nu)
        KE = (KE * E).to(dt).to(dev)
        B0 = B0.to(dt).to(dev)
        C = (_elasticity_C(nu) * E).to(dt).to(dev)
        rho = rho.to(dt)

        lin = [self._axis_lin(i, nodes=True) for i in range(3)]

        def nodes_near(pos: Tensor, radius: float) -> Tensor:
            pos = pos.to(dev)
            lo_i = torch.floor((pos - radius - self.lo.to(dev)) / h).long()
            hi_i = torch.ceil((pos + radius - self.lo.to(dev)) / h).long()
            rng = []
            for a in range(3):
                lo_a = int(lo_i[a].clamp(0, self.dims[a]))
                hi_a = int(hi_i[a].clamp(0, self.dims[a]))
                rng.append(torch.arange(lo_a, hi_a + 1, device=dev))
            gx, gy, gz = torch.meshgrid(*rng, indexing="ij")
            ijk = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
            p = torch.stack([lin[0][ijk[:, 0]], lin[1][ijk[:, 1]],
                             lin[2][ijk[:, 2]]], dim=-1)
            near = torch.linalg.vector_norm(p - pos, dim=-1) <= radius
            ijk, p = ijk[near], p[near]
            if ijk.numel() == 0:
                return torch.zeros(0, dtype=torch.long, device=dev)
            with torch.no_grad():
                solid = _eval_chunked(field, p, dev) < h
            return self._node_flat_index(ijk[solid])

        fixed_dof_mask = torch.zeros(n_dof, dtype=torch.bool, device=dev)
        for t in fixed_toks:
            grip = 1.1 * float(t.params["diameter"])
            ids = nodes_near(t.position, grip)
            if ids.numel() == 0:
                ids = nodes_near(t.position, 2.5 * grip)
            fixed_dof_mask[(ids.unsqueeze(-1) * 3 + torch.arange(3, device=dev)).reshape(-1)] = True
        if extra_fixed is not None:
            gx, gy, gz = torch.meshgrid(*lin, indexing="ij")
            node_pos = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
            sel = extra_fixed(node_pos).reshape(-1)
            ids = sel.nonzero(as_tuple=False).squeeze(-1)
            fixed_dof_mask[(ids.unsqueeze(-1) * 3 + torch.arange(3, device=dev)).reshape(-1)] = True

        active_dof_mask = torch.zeros(n_dof, dtype=torch.bool, device=dev)
        active_dof_mask[dof.reshape(-1)] = True
        free = active_dof_mask & ~fixed_dof_mask

        F = torch.zeros(n_dof, device=dev, dtype=dt)
        load_pos = load_tok.position
        assert load_pos is not None, "load token needs a pose"
        grip = 1.1 * float(load_tok.params.get("grip", 0.12))
        ids = nodes_near(load_pos, grip)
        if ids.numel() == 0:
            ids = nodes_near(load_pos, 3 * grip)
        fvec = load_tok.params["direction"].to(dev).to(dt) \
            * float(load_tok.params["magnitude"])
        per_node = fvec / max(int(ids.numel()), 1)
        for c in range(3):
            F.index_add_(0, ids * 3 + c, per_node[c].expand(ids.shape[0]))
        free_f = free.to(dt)
        F = F * free_f

        rhoKE_diag = rho.unsqueeze(-1) * KE.diagonal().unsqueeze(0)
        diag = torch.zeros(n_dof, device=dev, dtype=dt)
        diag.scatter_add_(0, dof.reshape(-1), rhoKE_diag.reshape(-1))
        inv_diag = torch.where(free, 1.0 / diag.clamp_min(1e-300),
                               torch.zeros_like(diag))

        def matvec(u: Tensor) -> Tensor:
            ue = u[dof]
            fe = (ue @ KE.T) * rho.unsqueeze(-1)
            out = torch.zeros_like(u)
            out.scatter_add_(0, dof.reshape(-1), fe.reshape(-1))
            return out * free_f

        u = torch.zeros(n_dof, device=dev, dtype=dt)
        b = F
        res_vec = b - matvec(u)
        z = inv_diag * res_vec
        p = z.clone()
        rz = (res_vec * z).sum()
        b_norm = torch.linalg.vector_norm(b).clamp_min(1e-30)
        iters = 0
        for iters in range(1, self.max_iter + 1):
            Ap = matvec(p)
            alpha = rz / (p * Ap).sum().clamp_min(1e-30)
            u = u + alpha * p
            res_vec = res_vec - alpha * Ap
            rel = (torch.linalg.vector_norm(res_vec) / b_norm).item()
            if rel < self.tol:
                break
            z = inv_diag * res_vec
            rz_new = (res_vec * z).sum()
            p = z + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        residual = (torch.linalg.vector_norm(res_vec) / b_norm).item()

        ue = u[dof]
        strain = ue @ B0.T
        stress = strain @ C.T
        sx, sy, sz_ = stress.unbind(-1)[:3]
        txy, tyz, tzx = stress.unbind(-1)[3:]
        vm = torch.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz_) ** 2 + (sz_ - sx) ** 2)
                        + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))

        vm_grid = torch.zeros(nx, ny, nz, device=dev, dtype=dt)
        vm_grid[active] = vm

        q = query_x.to(dev)
        with torch.no_grad():
            fq = _eval_chunked(field, q, dev)
            gq = _grad_chunked(field, q, dev)
        mask = (fq < 0).float()
        q_eval = torch.where(mask.unsqueeze(-1) > 0, q, q - fq.unsqueeze(-1) * gq)
        q_eval = q_eval - 0.75 * h * gq * (1 - mask).unsqueeze(-1)

        vm_q = self._trilinear(vm_grid, q_eval, cell_centered=True)
        disp_grid = u.reshape(nx + 1, ny + 1, nz + 1, 3)
        disp_q = torch.stack([
            self._trilinear(disp_grid[..., c], q_eval, cell_centered=False)
            for c in range(3)], dim=-1)

        return FEAResult(
            von_mises=vm_q.cpu(), displacement=disp_q.cpu(), mask=mask.cpu(),
            peak_vm=float(vm.max().item()),
            max_disp=float(torch.linalg.vector_norm(u.reshape(-1, 3), dim=-1).max().item()),
            iters=iters, residual=residual)

    def _trilinear(self, grid: Tensor, x: Tensor, cell_centered: bool) -> Tensor:
        lo = self.lo.to(x.device)
        offs = self.h / 2 if cell_centered else 0.0
        u = (x - lo - offs) / self.h
        dims = torch.tensor(grid.shape[:3], device=x.device, dtype=torch.float32)
        u = torch.minimum(torch.clamp(u, min=torch.zeros(3, device=x.device)),
                          dims - 1 - 1e-6)
        i0 = u.floor().long()
        frac = u - i0.float()
        i1 = torch.minimum(i0 + 1, (dims - 1).long())

        def g(a, b, c):
            return grid[a[:, 0], b[:, 1], c[:, 2]]

        c00 = g(i0, i0, i0) * (1 - frac[:, 0]) + g(i1, i0, i0) * frac[:, 0]
        c10 = g(i0, i1, i0) * (1 - frac[:, 0]) + g(i1, i1, i0) * frac[:, 0]
        c01 = g(i0, i0, i1) * (1 - frac[:, 0]) + g(i1, i0, i1) * frac[:, 0]
        c11 = g(i0, i1, i1) * (1 - frac[:, 0]) + g(i1, i1, i1) * frac[:, 0]
        c0 = c00 * (1 - frac[:, 1]) + c10 * frac[:, 1]
        c1 = c01 * (1 - frac[:, 1]) + c11 * frac[:, 1]
        return c0 * (1 - frac[:, 2]) + c1 * frac[:, 2]


def _eval_chunked(field: Field, x: Tensor, dev, chunk: int = 262144) -> Tensor:
    outs = []
    for i in range(0, x.shape[0], chunk):
        outs.append(field(x[i:i + chunk].to(dev)))
    return torch.cat(outs)


def _grad_chunked(field: Field, x: Tensor, dev, chunk: int = 131072) -> Tensor:
    outs = []
    for i in range(0, x.shape[0], chunk):
        outs.append(field.grad(x[i:i + chunk].to(dev)))
    return torch.cat(outs)


# ---------------------------------------------------------------------------
# Materials + L-bracket load cases
# ---------------------------------------------------------------------------

MATERIALS = {
    "Al6061":    {"E": 68.9e9,  "nu": 0.33, "yield": 276e6,  "density": 2700.0},
    "Steel1018": {"E": 205e9,   "nu": 0.29, "yield": 370e6,  "density": 7870.0},
    "Ti64":      {"E": 113.8e9, "nu": 0.342, "yield": 880e6, "density": 4430.0},
    "PA12":      {"E": 1.7e9,   "nu": 0.39, "yield": 48e6,   "density": 1010.0},
    "ABS":       {"E": 2.3e9,   "nu": 0.35, "yield": 40e6,   "density": 1040.0},
}
L_MATERIALS = ("Al6061", "Steel1018")

# held-out load-position band (fraction of the free leg): excluded from train
OOD_LOAD_BAND = (0.55, 0.70)


def material_token(name: str) -> Token:
    m = MATERIALS[name]
    return Token("material", None, {"name": name, **m})


def _biased_direction(gen: torch.Generator) -> Tensor:
    """60% engineering-biased (40% gravity cone, 20% pull-out cone), 40% uniform.
    Canonical frame: gravity = -z, pull-out (away from wall) = +y."""
    r = torch.rand((), generator=gen).item()
    if r < 0.4:
        d = torch.tensor([0.0, 0.0, -1.0]) + 0.45 * torch.randn(3, generator=gen)
    elif r < 0.6:
        sgn = 1.0 if torch.rand((), generator=gen).item() < 0.5 else -1.0
        d = torch.tensor([0.0, sgn, 0.0]) + 0.45 * torch.randn(3, generator=gen)
    else:
        d = torch.randn(3, generator=gen)
    return d / torch.linalg.vector_norm(d).clamp_min(1e-9)


def l_load_cases(meta: dict, seed: int, n_cases: int = 10,
                 ood_load: bool = False) -> list[dict]:
    """Canonical-frame (mm) load cases for one L-bracket.

    Each case: {pos_mm[3], dir[3], grip_mm, target_yield_fraction, kind}.
    Train draws exclude the OOD_LOAD_BAND of positions along the free leg;
    ood_load=True draws exclusively from it. Geometries with a free-leg bolt
    hole get ~half their cases applied at the bolt.
    """
    gen = torch.Generator().manual_seed(seed ^ 0x10AD)
    Lf, W, tw = meta["Lf"], meta["W"], meta["tw"]
    reach0 = meta.get("gusset", meta.get("ribs", {})).get("a", 0.0) \
        if meta.get("reinforcement", "none") != "none" else 0.0
    y_min_frac = max(0.15, (tw + reach0 + 5.0) / Lf)
    cases = []
    for _ in range(n_cases):
        bolt = (meta.get("free_hole") is not None
                and torch.rand((), generator=gen).item() < 0.5)
        if bolt:
            pos = torch.tensor([0.0, meta["free_hole"]["y"], -meta["tf"] / 2])
            grip = meta["free_hole"]["dia"]
        else:
            for _try in range(64):
                frac = _u01(gen) * (0.95 - y_min_frac) + y_min_frac
                in_band = OOD_LOAD_BAND[0] <= frac <= OOD_LOAD_BAND[1]
                if in_band == ood_load:
                    break
            x = (_u01(gen) * 0.7 - 0.35) * W
            pos = torch.tensor([x, frac * Lf, 0.0])   # on the top surface
            grip = max(0.12 * W, 6.0)
        cases.append({
            "pos_mm": pos,
            "dir": _biased_direction(gen),
            "grip_mm": grip,
            "target_yield_fraction": 0.1 + 1.4 * _u01(gen),
            "kind": "bolt" if bolt else "patch",
        })
    return cases


def _u01(gen) -> float:
    return torch.rand((), generator=gen).item()


class LBracketFEA:
    """Immersed FEA for one (geometry, material): assemble once, solve many
    load cases against the SAME algebraic-multigrid hierarchy.

    Thin-plate structures make Jacobi-CG hopeless (condition numbers beyond
    fp64's reach); PyAMG smoothed aggregation with the 6 rigid-body modes as
    the near-nullspace solves each case in a handful of V-cycles.

    Everything is SI: pass the mm-frame field/tokens and this class scales to
    meters internally. Query resampling matches VoxelFEA's conventions.
    """

    def __init__(self, field_mm: Field, wall_tokens_mm: list[Token], meta: dict,
                 material: str, device=None, h_frac: float = 1 / 3.5,
                 max_dim: int = 220):
        import numpy as np
        import pyamg
        import scipy.sparse as sp

        self.material = material
        mat = MATERIALS[material]
        self.field = field_mm.scale(1e-3)
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        lo = torch.tensor(meta["aabb"]["lo"], dtype=torch.float32) * 1e-3 - 0.004
        hi = torch.tensor(meta["aabb"]["hi"], dtype=torch.float32) * 1e-3 + 0.004
        h = max(min(meta["tw"], meta["tf"]) * h_frac * 1e-3, 5e-4)
        self.grid = VoxelFEA(bounds=(lo, hi), h=h, device=self.device,
                             max_dim=max_dim)
        g = self.grid
        nx, ny, nz = g.dims

        centers = g._element_centers()
        with torch.no_grad():
            fv = _eval_chunked(self.field, centers.reshape(-1, 3),
                               self.device).reshape(nx, ny, nz)
        dens = (0.5 - fv / g.h).clamp(0.0, 1.0)
        self.active = (dens > 0.05).cpu()
        act_idx = self.active.nonzero(as_tuple=False)
        if act_idx.shape[0] == 0:
            raise ValueError("empty solid")
        rho = dens.cpu()[self.active].double()

        corners = act_idx.unsqueeze(1) + _NODE_OFFSETS.unsqueeze(0)
        node_ids = g._node_flat_index(corners)                    # [nel, 8]
        self.dof = (node_ids.unsqueeze(-1) * 3
                    + torch.arange(3)).reshape(-1, 24)            # [nel, 24]
        self.n_dof = (nx + 1) * (ny + 1) * (nz + 1) * 3
        KE, self.B0 = element_stiffness(g.h, mat["nu"])
        self.KE = KE * mat["E"]
        self.C = _elasticity_C(mat["nu"]) * mat["E"]
        self._lin = [g._axis_lin(i, nodes=True).cpu() for i in range(3)]

        # Dirichlet: wall grips
        fixed_mask = torch.zeros(self.n_dof, dtype=torch.bool)
        for t in wall_tokens_mm:
            if t.type != "fixed_point" or not t.params.get("fixed"):
                continue
            tm = t.scale(1e-3)
            ids = self._nodes_near(tm.position, 1.1 * float(tm.params["diameter"]))
            fixed_mask[(ids.unsqueeze(-1) * 3 + torch.arange(3)).reshape(-1)] = True
        assert fixed_mask.any(), "no clamped nodes found at wall grips"

        active_mask = torch.zeros(self.n_dof, dtype=torch.bool)
        active_mask[self.dof.reshape(-1)] = True
        self.free = active_mask & ~fixed_mask
        free_idx = self.free.nonzero(as_tuple=False).squeeze(-1)
        comp = torch.full((self.n_dof,), -1, dtype=torch.long)
        comp[free_idx] = torch.arange(free_idx.shape[0])
        self._comp, self._free_idx = comp, free_idx
        nf = free_idx.shape[0]

        # assemble K_ff (CSR, float64)
        vals = (rho.reshape(-1, 1, 1) * self.KE.unsqueeze(0)).reshape(-1)
        rows = comp[self.dof].unsqueeze(2).expand(-1, 24, 24).reshape(-1)
        cols = comp[self.dof].unsqueeze(1).expand(-1, 24, 24).reshape(-1)
        keep = (rows >= 0) & (cols >= 0)
        K = sp.coo_matrix(
            (vals[keep].numpy(), (rows[keep].numpy(), cols[keep].numpy())),
            shape=(nf, nf)).tocsr()

        # rigid-body near-nullspace at free-dof node coordinates
        node_of_dof = free_idx // 3
        comp3 = (free_idx % 3).numpy()
        ny1, nz1 = ny + 1, nz + 1
        i = node_of_dof // (ny1 * nz1)
        jk = node_of_dof % (ny1 * nz1)
        j, k = jk // nz1, jk % nz1
        p = torch.stack([self._lin[0][i], self._lin[1][j], self._lin[2][k]],
                        dim=-1).double().numpy()
        B = np.zeros((nf, 6))
        for c in range(3):
            B[comp3 == c, c] = 1.0
        rot = [((1, 2), (2, 1)), ((2, 0), (0, 2)), ((0, 1), (1, 0))]
        for m_i, ((a1, a2), (b1, b2)) in enumerate(rot):
            B[comp3 == a1, 3 + m_i] = -p[comp3 == a1, a2]
            B[comp3 == b1, 3 + m_i] = p[comp3 == b1, b2]
        self.ml = pyamg.smoothed_aggregation_solver(
            K, B=B, strength="symmetric", max_coarse=400)
        self.rho = rho

    def _nodes_near(self, pos: Tensor, radius: float) -> Tensor:
        g = self.grid
        pos = pos.cpu()
        lo_i = torch.floor((pos - radius - g.lo) / g.h).long()
        hi_i = torch.ceil((pos + radius - g.lo) / g.h).long()
        rng = [torch.arange(int(lo_i[a].clamp(0, g.dims[a])),
                            int(hi_i[a].clamp(0, g.dims[a])) + 1) for a in range(3)]
        gx, gy, gz = torch.meshgrid(*rng, indexing="ij")
        ijk = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        p = torch.stack([self._lin[0][ijk[:, 0]], self._lin[1][ijk[:, 1]],
                         self._lin[2][ijk[:, 2]]], dim=-1)
        near = torch.linalg.vector_norm(p - pos, dim=-1) <= radius
        ijk, p = ijk[near], p[near]
        if ijk.numel() == 0:
            return torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            solid = _eval_chunked(self.field, p, self.device).cpu() < g.h
        return self.grid._node_flat_index(ijk[solid])

    def query_context(self, query_mm: Tensor) -> tuple[Tensor, Tensor]:
        """(mask, q_eval): inside-solid mask and surface-projected sample
        points in meters — geometry-only, compute ONCE per query set and pass
        to every solve_case."""
        g = self.grid
        q = query_mm * 1e-3
        with torch.no_grad():
            fq = _eval_chunked(self.field, q, self.device).cpu()
            gq = _grad_chunked(self.field, q, self.device).cpu()
        mask = (fq < 0).float()
        q_eval = torch.where(mask.unsqueeze(-1) > 0, q, q - fq.unsqueeze(-1) * gq)
        q_eval = q_eval - 0.75 * g.h * gq * (1 - mask).unsqueeze(-1)
        return mask, q_eval

    def solve_case(self, case: dict, query_mm: Tensor, tol: float = 1e-6,
                   query_ctx=None, force_n: float | None = None
                   ) -> tuple[FEAResult, float]:
        """Unit-force AMG solve, then scale.

        force_n=None (dataset labelling): calibrate F so peak VM hits the
        case's target yield fraction. force_n given (verification): use that
        actual force. Either way the solve itself is done once at 1 N and
        scaled — exact under linear elasticity.
        """
        import numpy as np

        mat = MATERIALS[self.material]
        pos_m = case["pos_mm"] * 1e-3
        ids = self._nodes_near(pos_m, 1.1 * case["grip_mm"] * 1e-3)
        if ids.numel() == 0:
            ids = self._nodes_near(pos_m, 3.3 * case["grip_mm"] * 1e-3)
        assert ids.numel() > 0, "no load nodes found"
        F_full = torch.zeros(self.n_dof, dtype=torch.float64)
        d = case["dir"].double()
        for c in range(3):
            F_full.index_add_(0, ids * 3 + c,
                              (d[c] / ids.numel()).expand(ids.shape[0]))
        b = F_full[self._free_idx].numpy()

        res_hist: list = []
        x = self.ml.solve(b, tol=tol, accel="cg", maxiter=300,
                          residuals=res_hist)
        u = torch.zeros(self.n_dof, dtype=torch.float64)
        u[self._free_idx] = torch.from_numpy(x)
        residual = float(res_hist[-1] / max(res_hist[0], 1e-300))

        ue = u[self.dof]
        stress = (ue @ self.B0.T) @ self.C.T
        sx, sy, sz_, txy, tyz, tzx = stress.unbind(-1)
        vm = torch.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz_) ** 2 + (sz_ - sx) ** 2)
                        + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))
        g = self.grid
        nx, ny, nz = g.dims
        vm_grid = torch.zeros(nx, ny, nz, dtype=torch.float64)
        vm_grid[self.active] = vm

        mask, q_eval = (query_ctx if query_ctx is not None
                        else self.query_context(query_mm))
        vm_q = g._trilinear(vm_grid, q_eval, cell_centered=True)
        disp_grid = u.reshape(nx + 1, ny + 1, nz + 1, 3)
        disp_q = torch.stack([g._trilinear(disp_grid[..., c], q_eval,
                                           cell_centered=False)
                              for c in range(3)], dim=-1)

        peak = float(vm.max())
        F = (float(force_n) if force_n is not None
             else case["target_yield_fraction"] * mat["yield"] / max(peak, 1e-9))
        return FEAResult(
            von_mises=(vm_q * F).float(),
            displacement=(disp_q * F).float(),
            mask=mask,
            peak_vm=peak * F,
            max_disp=float(torch.linalg.vector_norm(
                u.reshape(-1, 3), dim=-1).max()) * F,
            iters=len(res_hist), residual=residual), F


def solve_l_case(field_mm: Field, wall_tokens_mm: list[Token], case: dict,
                 material: str, query_mm: Tensor, device=None,
                 aabb_mm: tuple | None = None, t_min_mm: float | None = None,
                 **_) -> tuple[FEAResult, float]:
    """One-shot convenience wrapper (tests); production reuses LBracketFEA
    across a geometry's cases via generate.make_l_record."""
    meta = {"aabb": {"lo": list(aabb_mm[0]), "hi": list(aabb_mm[1])},
            "tw": float(t_min_mm), "tf": float(t_min_mm)}
    solver = LBracketFEA(field_mm, wall_tokens_mm, meta, material, device=device)
    return solver.solve_case(case, query_mm)

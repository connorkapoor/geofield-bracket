"""Tests for marching cubes export and reconstruction/rotation eval metrics."""
import numpy as np
import torch

from geofield.export.marching_cubes import (field_to_grid, grid_to_mesh,
                                            mesh_to_stl, sample_mesh_surface)
from geofield.eval.reconstruction import chamfer, occupancy_iou
from geofield.fields.primitives import Sphere


def test_marching_cubes_sphere():
    s = Sphere(0.5)
    grid = field_to_grid(lambda p: s(p), res=64)
    verts, faces = grid_to_mesh(grid)
    assert len(verts) > 100 and len(faces) > 100
    r = np.linalg.norm(verts, axis=-1)
    assert abs(r.mean() - 0.5) < 0.01
    assert r.std() < 0.01


def test_surface_sampling_and_chamfer():
    s = Sphere(0.5)
    grid = field_to_grid(lambda p: s(p), res=64)
    verts, faces = grid_to_mesh(grid)
    a = torch.from_numpy(sample_mesh_surface(verts, faces, 2048, seed=0))
    b = torch.from_numpy(sample_mesh_surface(verts, faces, 2048, seed=1))
    assert chamfer(a, b) < 2e-3          # same surface -> tiny Chamfer
    c = a + torch.tensor([0.2, 0.0, 0.0])
    assert chamfer(a, c) > chamfer(a, b) * 10


def test_occupancy_iou():
    s1, s2 = Sphere(0.5), Sphere(0.4)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20000, 3, generator=g) * 0.5
    iou = occupancy_iou(s2(x), s1(x))
    assert 0.4 < iou < 0.7               # volume ratio 0.512, sampled
    assert occupancy_iou(s1(x), s1(x)) == 1.0


def test_stl_writer(tmp_path):
    s = Sphere(0.5)
    grid = field_to_grid(lambda p: s(p), res=32)
    verts, faces = grid_to_mesh(grid)
    path = tmp_path / "sphere.stl"
    mesh_to_stl(verts, faces, path)
    data = path.read_bytes()
    n_tri = int.from_bytes(data[80:84], "little")
    assert n_tri == len(faces)
    assert len(data) == 84 + 50 * n_tri

"""
Magnetic field model tests (Phase 03.1).

test_cuboid_matches_magpylib pins the njit closed-form cuboid field against the
magpylib getB oracle (magnetization in A/m, output Tesla) at capture-region points.
test_field_writes_tesla checks the per-atom Tesla write contract.
"""
import json

import numpy as np
import pytest

from src.atoms import Li6
from src.magnetic_field import CuboidBarMagneticField

SETUP = "setup parameters/Hammel_Setup.json"


def _load_bars():
    with open(SETUP) as f:
        return json.load(f)["Magnetic_Fields"]["dipoles"]


def _build_cuboid(bars):
    field = CuboidBarMagneticField(len(bars))
    for idx, bar in enumerate(bars):
        field.add_dipole(
            idx,
            np.asarray(bar["position"], dtype=np.float64),
            np.asarray(bar["dimension"], dtype=np.float64),
            np.asarray(bar["orientation"], dtype=np.float64),
            float(bar["magnetization"]),
        )
    return field


def _capture_points():
    # Grid inside the capture region, well clear of every bar face
    # (bars at x=±0.042, z=±0.021; here |x|,|y| <= 0.02 so never inside a bar).
    xs = np.linspace(-0.02, 0.02, 3)
    ys = np.linspace(-0.02, 0.02, 3)
    zs = np.linspace(0.0, 0.19, 4)
    pts = [[x, y, z] for x in xs for y in ys for z in zs]
    return np.asarray(pts, dtype=np.float64)


def test_cuboid_matches_magpylib():
    magpy = pytest.importorskip("magpylib")
    assert magpy.__version__.startswith("5"), magpy.__version__

    bars = _load_bars()
    sources = []
    for bar in bars:
        orient = np.asarray(bar["orientation"], dtype=float)
        m_vec = bar["magnetization"] * orient / np.linalg.norm(orient)
        sources.append(
            magpy.magnet.Cuboid(
                position=np.asarray(bar["position"], dtype=float),
                dimension=np.asarray(bar["dimension"], dtype=float),
                magnetization=m_vec,  # A/m -> getB returns Tesla
            )
        )
    coll = magpy.Collection(*sources)

    obs = _capture_points()
    B_ref = coll.getB(obs)

    field = _build_cuboid(bars)
    B_njit, _ = field.field_at_positions(obs)

    assert np.allclose(B_njit, B_ref, rtol=1e-3, atol=1e-6)


def test_field_writes_tesla():
    bars = _load_bars()
    field = _build_cuboid(bars)

    atoms = Li6(1)
    # Off-axis capture point (the z-axis is the quadrupole zero-field line).
    atoms.positions[0] = np.array([0.012, -0.008, 0.02], dtype=np.float64)
    field.calculate_magnetic_field(atoms, 0)

    B = atoms.magnetic_field_vectors[0]
    strength = atoms.magnetic_field_strength[0]
    assert np.all(np.isfinite(B))
    assert 1e-4 < strength < 1.0  # Tesla scale near the MOT, not 1e3 or 1e9
    assert strength == pytest.approx(np.linalg.norm(B))


def test_cuboid_config_wires_and_gradient():
    """Parameters builds CuboidBarMagneticField; central gradient ~0.5 T/m (paper)."""
    from src.parameters import Parameters

    p = Parameters("setup parameters/Hammel_Cuboid_Setup.json")
    assert p.valid, p.errors
    field = p._construct_magnetic_field()
    assert type(field).__name__ == "CuboidBarMagneticField"

    d = 5e-4
    Bp, _ = field.field_at_positions(np.array([[d, 0.0, 0.0]]))
    Bm, _ = field.field_at_positions(np.array([[-d, 0.0, 0.0]]))
    grad = (Bp[0, 1] - Bm[0, 1]) / (2 * d)  # dBy/dx at center
    assert grad == pytest.approx(0.5, abs=0.05)  # paper: 0.50 T/m


def _build_grid_model_from_cuboid(cub, x_axis, y_axis, z_axis):
    from src.magnetic_field import GridFieldModel

    pts = np.array(
        [[x, y, z] for x in x_axis for y in y_axis for z in z_axis],
        dtype=np.float64,
    )
    B, _ = cub.field_at_positions(pts)
    B = B.reshape(len(x_axis), len(y_axis), len(z_axis), 3)
    return GridFieldModel(
        np.ascontiguousarray(x_axis),
        np.ascontiguousarray(y_axis),
        np.ascontiguousarray(z_axis),
        np.ascontiguousarray(B[..., 0]),
        np.ascontiguousarray(B[..., 1]),
        np.ascontiguousarray(B[..., 2]),
        cub.positions,
        cub.half_extents,
        cub.mag_axis,
        cub.mag_signed,
    )


def test_grid_interp():
    cub = _build_cuboid(_load_bars())
    x_axis = np.linspace(0.0, 0.02, 21)
    y_axis = np.linspace(-0.01, 0.01, 21)
    z_axis = np.linspace(0.01, 0.03, 21)
    gm = _build_grid_model_from_cuboid(cub, x_axis, y_axis, z_axis)

    # Exact grid node -> interpolated value equals the stored node value.
    node = np.array([[x_axis[3], y_axis[4], z_axis[5]]], dtype=np.float64)
    Bg, _ = gm.field_at_positions(node)
    Bc, _ = cub.field_at_positions(node)
    assert np.allclose(Bg[0], Bc[0], rtol=1e-9, atol=1e-12)

    # Cell midpoint -> interp matches the exact cuboid within interp tolerance.
    mid = np.array([[
        0.5 * (x_axis[3] + x_axis[4]),
        0.5 * (y_axis[4] + y_axis[5]),
        0.5 * (z_axis[5] + z_axis[6]),
    ]], dtype=np.float64)
    Bg, _ = gm.field_at_positions(mid)
    Bc, _ = cub.field_at_positions(mid)
    assert np.allclose(Bg[0], Bc[0], rtol=2e-2, atol=1e-5)


def test_grid_oob_fallback():
    cub = _build_cuboid(_load_bars())
    x_axis = np.linspace(-0.02, 0.02, 41)
    y_axis = np.linspace(-0.02, 0.02, 41)
    z_axis = np.linspace(0.0, 0.05, 51)
    gm = _build_grid_model_from_cuboid(cub, x_axis, y_axis, z_axis)

    # Point well outside (z past z_last) -> exact analytic cuboid, finite (not clamp).
    out = np.array([[0.005, 0.005, 0.20]], dtype=np.float64)
    Bg, _ = gm.field_at_positions(out)
    Bc, _ = cub.field_at_positions(out)
    assert np.all(np.isfinite(Bg[0]))
    assert np.allclose(Bg[0], Bc[0], rtol=1e-9, atol=1e-12)

    # Continuity across z=z_last: eps inside vs eps outside differ << |B|.
    zf = z_axis[-1]
    Bi, _ = gm.field_at_positions(np.array([[0.005, 0.005, zf - 1e-5]]))
    Bo, _ = gm.field_at_positions(np.array([[0.005, 0.005, zf + 1e-5]]))
    scale = np.linalg.norm(Bi[0]) + 1e-9
    assert np.linalg.norm(Bi[0] - Bo[0]) < 0.05 * scale

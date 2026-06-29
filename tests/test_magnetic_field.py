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

"""Magnetic-field wiring tests (Phase 03.1-03): both new types parse + construct."""
import json

import numpy as np
import pytest

from src.parameters import ParameterError, Parameters

BASE = "setup parameters/Hammel_Cuboid_Setup.json"


def _write_grid(path, nan=False):
    x = np.linspace(-0.02, 0.02, 5)
    y = np.linspace(-0.02, 0.02, 5)
    z = np.linspace(0.0, 0.04, 5)
    shape = (x.size, y.size, z.size)
    Bx, By, Bz = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    if nan:
        Bx[0, 0, 0] = np.nan
    np.savez(str(path), x_axis=x, y_axis=y, z_axis=z, Bx=Bx, By=By, Bz=Bz)


def _grid_config(tmp_path, grid_path):
    cfg = json.load(open(BASE))
    cfg["Magnetic_Fields"] = {
        "type": "GridFieldModel",
        "grid_file": str(grid_path),
        "center_offset": [0.0, 0.0, 0.0],
        "dipoles": json.load(open(BASE))["Magnetic_Fields"]["dipoles"],
    }
    cfg_path = tmp_path / "grid_setup.json"
    json.dump(cfg, open(cfg_path, "w"))
    return str(cfg_path)


def test_construct_new_field_types(tmp_path):
    # Cuboid: builds straight from the default config.
    p = Parameters(BASE)
    assert p.valid, p.errors
    assert type(p._construct_magnetic_field()).__name__ == "CuboidBarMagneticField"

    # Grid: a valid NPZ -> GridFieldModel.
    good = tmp_path / "good.npz"
    _write_grid(good)
    pg = Parameters(_grid_config(tmp_path, good))
    assert pg.valid, pg.errors
    assert type(pg._construct_magnetic_field()).__name__ == "GridFieldModel"

    # Grid: a NaN-injected NPZ -> ParameterError on construct (load-time validation).
    bad = tmp_path / "bad.npz"
    _write_grid(bad, nan=True)
    pb = Parameters(_grid_config(tmp_path, bad))
    assert pb.valid, pb.errors
    with pytest.raises(ParameterError):
        pb._construct_magnetic_field()

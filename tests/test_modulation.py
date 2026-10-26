"""Per-laser power modulation: map shape, gating, parser wiring."""
import json

import numpy as np

from src.modulation import laser_factors, map_factor, validate
from src.parameters import Parameters

BASE = "setup parameters/Hammel_Setup.json"
DT = 1e-5
HALF = 1 / (2 * 256)


def _map(**kw):
    m = dict(shape="square", frequency=256.0, plateau_width=HALF, min=0.0,
             max=1.0, phase=0.0, rest=1.0, t_on=0.0, t_off=np.inf)
    m.update(kw)
    return m


def _steps(t):
    return np.rint(np.asarray(t) / DT).astype(np.int64)


def test_square_levels_and_phase():
    m = _map(min=0.04, t_on=1e-3)
    t = 1e-3 + np.array([0.1, 0.9, 1.1, 1.9]) * HALF
    assert np.allclose(map_factor(t, m), [1.0, 1.0, 0.04, 0.04])
    assert np.allclose(map_factor(t, _map(min=0.04, t_on=1e-3, phase=180)),
                       [0.04, 0.04, 1.0, 1.0])


def test_rest_outside_window():
    m = _map(t_on=2e-3, t_off=4e-3, rest=0.5)
    assert np.allclose(map_factor(np.array([1e-3, 5e-3]), m), 0.5)


def test_laser_gate_wins():
    on, off = _steps([1e-3]), _steps([2e-3])
    f = laser_factors(300, DT, on, off, [_map()], np.array([0]))
    assert f[:100].max() == 0 and f[200:].max() == 0 and f[100] == 1


def test_matches_one_entry_per_window():
    """One map reproduces the chopped config with one entry per on-window.

    Only edges exactly between two steps may land one step apart (np.rint
    ties to even vs midpoint evaluation).
    """
    preload, n_cyc = 9e-3, 16
    end = preload + n_cyc * 2 * HALF
    starts = [preload + (2 * k + 1) * HALF for k in range(n_cyc)]
    windows = [(0.0, preload)] + [(t, t + HALF) for t in starts]
    n = int(_steps(end))
    on = _steps([a for a, _ in windows])
    off = _steps([b for _, b in windows])
    entries = laser_factors(n, DT, on, off, [], np.full(len(windows), -1))
    chop = _map(t_on=preload, phase=180.0, t_off=end)
    f = laser_factors(n, DT, _steps([0.0]), np.array([n + 1]), [chop],
                      np.array([0]))
    bad = np.nonzero(entries.max(axis=1) != f[:, 0])[0]
    edges = np.array([t for w in windows for t in w]) / DT
    ties = np.floor(edges[np.isclose(edges % 1, 0.5)]).astype(int)
    assert set(bad) <= set(ties)


def test_no_map_is_old_mask():
    on, off = _steps([0.0, 1e-3]), _steps([5e-3, 3e-3])
    f = laser_factors(600, DT, on, off, [], np.array([-1, -1]))
    i = np.arange(600)[:, None]
    assert np.array_equal(f, ((on <= i) & (i < off)).astype(float))


def test_validate():
    assert validate(_map()) == []
    assert validate(_map(plateau_width=2 * HALF))
    assert validate(_map(min=0.5, max=0.4))
    assert validate(_map(shape="triangle"))


def _config(tmp_path, maps, refs):
    cfg = json.load(open(BASE))
    cfg["Modulations"] = maps
    for laser, ref in zip(cfg["Lasers"], refs):
        laser["modulation"] = ref
    path = tmp_path / "mod.json"
    json.dump(cfg, open(path, "w"))
    return str(path)


def test_parser_wiring(tmp_path):
    m = dict(name="chop", frequency=256.0, plateau_width=HALF * 1e3,
             min=0.04, phase=180.0, t_on=9.0)
    p = Parameters(_config(tmp_path, [m], [None, "chop"]))
    assert p.valid and not p.errors, p.errors
    assert np.isclose(p.modulations[0]["plateau_width"], HALF)
    assert np.isclose(p.modulations[0]["t_on"], 9e-3)
    idx = p._laser_modulation_index()
    assert idx[0] == -1 and idx[1] == 0 and (idx[2:] == -1).all()


def test_parser_errors(tmp_path):
    m = dict(name="chop", frequency=256.0, plateau_width=5.0)
    p = Parameters(_config(tmp_path, [m, m], ["nope"]))
    msg = " ".join(p.errors)
    assert "plateau_width" in msg and "unique" in msg and "nope" in msg


def test_config_without_section(tmp_path):
    p = Parameters(BASE)
    assert p.modulations == []
    assert (p._laser_modulation_index() == -1).all()

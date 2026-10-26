"""Per-laser power modulation: one factor per laser and step.

The factor combines the laser's t_on/t_off gate with an optional map
("Modulations" in the JSON, referenced by name from a laser). A map is active
in [t_on, t_off) and holds `rest` outside. Inside, the square shape is `max`
for `plateau_width` of every period and `min` otherwise; phase 0 puts a
rising edge at the map's t_on. Factors are evaluated at the step midpoint,
so every edge snaps to the nearest step boundary like the t_on/t_off gate.
An edge exactly between two steps belongs to the later step (the gate's
np.rint sends such ties to the even step), so a map and the equivalent
one-entry-per-window config can differ by one step at those edges.
The table is precomputed as a pure function of the step index, so
checkpoint resume needs no state.
"""

import numpy as np

SHAPES = ("square",)


def square(t, m):
    """Factor of a square map at times t (s), inside its window."""
    period = 1.0 / m["frequency"]
    tau = np.mod(t - m["t_on"] + m["phase"] / 360.0 * period, period)
    return np.where(tau < m["plateau_width"], m["max"], m["min"])


def map_factor(t, m):
    inside = (t >= m["t_on"]) & (t < m["t_off"])
    return np.where(inside, square(t, m), m["rest"])


def laser_factors(n_steps, dt, on_step, off_step, maps, laser_map):
    """(n_steps, n_lasers) power factors.

    0 outside each laser's [on_step, off_step), else its map factor (1 for
    lasers without a map). laser_map[j] is an index into maps or -1.
    """
    i = np.arange(n_steps)
    gate = (on_step[None, :] <= i[:, None]) & (i[:, None] < off_step[None, :])
    t_mid = (i + 0.5) * dt
    f = np.ones((n_steps, len(on_step)))
    for j, k in enumerate(laser_map):
        if k >= 0:
            f[:, j] = map_factor(t_mid, maps[k])
    return np.where(gate, f, 0.0)


def validate(m):
    """Error messages for one parsed map (times in s, frequency in Hz)."""
    errors = []
    if m["shape"] not in SHAPES:
        errors.append(f"unknown shape {m['shape']!r}")
    if m["frequency"] <= 0:
        errors.append("frequency must be > 0")
    elif not 0 < m["plateau_width"] < 1.0 / m["frequency"]:
        errors.append("plateau_width must lie between 0 and one period")
    if not 0 <= m["min"] <= m["max"] <= 1:
        errors.append("need 0 <= min <= max <= 1")
    if not 0 <= m["rest"] <= 1:
        errors.append("rest must lie in [0, 1]")
    if m["t_off"] <= m["t_on"]:
        errors.append("t_off must be greater than t_on")
    return errors

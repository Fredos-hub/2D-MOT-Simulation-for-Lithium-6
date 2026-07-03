"""Shared constants and result-loading helpers for the plotting package."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.constants as scc

REPO_ROOT = Path(__file__).resolve().parents[2]

LI6_MASS = 6.0151228874 * scc.atomic_mass  # kg
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # std -> Gaussian FWHM

HANDEDNESS_LABELS = {-1: "LH", 0: "LIN", 1: "RH"}
POL_COLORS = {"LH": "tab:blue", "RH": "tab:red", "LIN": "tab:green"}
SPECTRUM_LABELS = {
    -1: "LH ($\\sigma^-$)",
    0: "LIN ($\\pi$)",
    1: "RH ($\\sigma^+$)",
}


def output_dir(*subdirs: str) -> Path:
    """Default figure/cache location: <repo>/analysis_output/<subdirs>."""
    path = REPO_ROOT.joinpath("analysis_output", *subdirs)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_data(file_path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def load_config(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "config.json").read_text())


def session_runs(session_dir: str | Path) -> list[Path]:
    """Sorted run_* directories of a session; raises if there are none."""
    session_dir = Path(session_dir)
    runs = sorted(session_dir.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* directories in {session_dir}")
    return runs


def pushbeam_meta(config: dict | str | Path) -> dict | None:
    """Identify the push beam in a run config and return its parameters.

    The push beam is matched by waist 0.6 mm + direction +z (falling back
    to "has t_on" for pulse-scan configs). Returns None when no push beam
    with a known handedness is present.

    Keys: detuning, pol, t_on_ms, t_off_ms, pulse_ms (None when the beam
    has no t_off), power_mW, seed, weight, dt_s.
    """
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    push = [
        L
        for L in config["Lasers"]
        if abs(L["waist"] - 0.0006) < 1e-9 and L["direction"][2] == 1.0
    ]
    if not push:
        push = [L for L in config["Lasers"] if "t_on" in L]
    if not push or push[0]["handedness"] not in HANDEDNESS_LABELS:
        return None
    L = push[0]
    t_on = float(L.get("t_on", 0.0))
    t_off = L.get("t_off")
    sim = config.get("Simulation", {})
    return {
        "detuning": float(L["detuning"]),
        "pol": HANDEDNESS_LABELS[L["handedness"]],
        "t_on_ms": t_on,
        "t_off_ms": None if t_off is None else float(t_off),
        "pulse_ms": None if t_off is None else float(t_off) - t_on,
        "power_mW": round(sum(p["beam_power"] for p in push) * 1000, 3),
        "seed": int(sim.get("random_seed", -1)),
        "weight": float(sim.get("macro_particle_weight", 1.0)),
        "dt_s": (
            None
            if "default_time_step" not in sim
            else sim["default_time_step"] * 1e-6
        ),
    }


def select_alive_ids(
    df: pd.DataFrame,
    ref_time: float,
    match: str = "nearest",
    time_scale: float = 1.0,
    time_col: str = "subjective_time",
) -> tuple[np.ndarray, float]:
    """Atom ids present at `ref_time` (after scaling `time_col`).

    Returns (alive_ids, selected_time). With match='nearest' the closest
    available time is used; 'exact' raises when the time is absent.
    """
    t = df[time_col].astype(float) * time_scale
    times = np.sort(t.unique())
    if match == "exact":
        if not np.any(times == ref_time):
            raise ValueError(
                f"No rows found for {time_col} == {ref_time / time_scale} "
                f"(after scaling). Use match='nearest' or change time_scale."
            )
        selected_time = float(ref_time)
    elif match == "nearest":
        selected_time = float(times[np.argmin(np.abs(times - ref_time))])
    else:
        raise ValueError("match must be 'nearest' or 'exact'")

    alive_ids = df.loc[t == selected_time, "atom_id"].dropna().unique()
    if len(alive_ids) == 0:
        raise ValueError(
            f"No atom IDs found at reference time {selected_time}"
        )
    return alive_ids, selected_time


def load_snapshot(
    csv_path: str | Path,
    step: int,
    exclusion_radius_mm: float,
    transverse_radius_mm: float,
    extrapolate: bool = False,
    dt_s: float | None = None,
    columns: tuple[str, ...] = ("velocity_z", "excitation_count"),
) -> pd.DataFrame:
    """Spatially filtered per-atom snapshot at `step`.

    Keeps atoms with |r| > exclusion_radius AND hypot(x, y) < transverse
    radius. With `extrapolate`, atoms whose last record precedes `step`
    are coasted ballistically (all velocity components held constant) and
    the masks use the extrapolated positions; requires `dt_s`.

    Returns the requested `columns` of the surviving atoms.
    """
    base = ["step", "position_x", "position_y", "position_z"]
    if extrapolate:
        if dt_s is None:
            raise ValueError("extrapolate=True requires dt_s")
        usecols = sorted(
            {
                *base,
                "atom_id",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                *columns,
            }
        )
        df = pd.read_csv(csv_path, usecols=usecols)
        # each atom's row at the latest step <= snapshot step
        snap = (
            df[df["step"] <= step]
            .sort_values("step")
            .groupby("atom_id")
            .tail(1)
        )
        dn = (step - snap["step"]) * dt_s
        x = snap["position_x"] + snap["velocity_x"] * dn
        y = snap["position_y"] + snap["velocity_y"] * dn
        z = snap["position_z"] + snap["velocity_z"] * dn
    else:
        usecols = sorted({*base, *columns})
        df = pd.read_csv(csv_path, usecols=usecols)
        snap = df[df["step"] == step]
        x, y, z = (
            snap["position_x"],
            snap["position_y"],
            snap["position_z"],
        )

    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.hypot(x, y)
    mask = (r > exclusion_radius_mm * 1e-3) & (
        rho < transverse_radius_mm * 1e-3
    )
    return snap.loc[mask, list(columns)]


def crossing_records(
    csv_path: str | Path,
    z_threshold_m: float,
    dt_s: float,
    extrapolate_to_ms: float | None = None,
    include_excitation: bool = False,
) -> list[dict]:
    """Per-atom first crossing of the z = z_threshold plane.

    One dict per atom that crosses, with NO radius / time / frame
    filtering (callers filter):

        t_cross_ms   crossing time since sim start (ms)
        rho_cross_mm transverse radius hypot(x, y) at the crossing (mm)
        vz_cross_m_s v_z at the crossing (m/s)
        t_first_ms   atom's first recorded time (ms) -> "flight" frame
        coasted      1 if ballistically extrapolated past the data, else 0
        exc          excitation count (only with `include_excitation`)

    The crossing step is linearly interpolated between the two bracketing
    rows; rho and v_z are interpolated at the same fraction. With
    `extrapolate_to_ms`, atoms that never crossed within the recorded
    data are coasted ballistically from their last row (velocities held
    constant) and accepted only if the crossing happens by that time.
    """
    usecols = [
        "step",
        "atom_id",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    ]
    if include_excitation:
        usecols.append("excitation_count")
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.sort_values(["atom_id", "step"])
    t_max_s = (
        extrapolate_to_ms * 1e-3 if extrapolate_to_ms is not None else None
    )
    recs = []
    for _, g in df.groupby("atom_id", sort=False):
        z = g["position_z"].to_numpy()
        s = g["step"].to_numpy()
        x, y = g["position_x"].to_numpy(), g["position_y"].to_numpy()
        vz_arr = g["velocity_z"].to_numpy()
        exc_arr = (
            g["excitation_count"].to_numpy() if include_excitation else None
        )
        cross = np.where((z[:-1] < z_threshold_m) & (z[1:] >= z_threshold_m))[
            0
        ]
        if len(cross) > 0:
            i = cross[0]
            frac = (z_threshold_m - z[i]) / (z[i + 1] - z[i])
            xc = x[i] + frac * (x[i + 1] - x[i])
            yc = y[i] + frac * (y[i + 1] - y[i])
            vzc = vz_arr[i] + frac * (vz_arr[i + 1] - vz_arr[i])
            t_cross = (s[i] + frac * (s[i + 1] - s[i])) * dt_s
            exc = exc_arr[i + 1] if include_excitation else None
            coasted = 0
        elif t_max_s is not None:
            # coast from the last recorded row until it crosses (or give up)
            vzc = vz_arr[-1]
            if z[-1] >= z_threshold_m or vzc <= 0:
                continue
            t_last = s[-1] * dt_s
            t_cross = t_last + (z_threshold_m - z[-1]) / vzc
            if t_cross > t_max_s:
                continue
            dt_extra = t_cross - t_last
            xc = x[-1] + g["velocity_x"].to_numpy()[-1] * dt_extra
            yc = y[-1] + g["velocity_y"].to_numpy()[-1] * dt_extra
            exc = exc_arr[-1] if include_excitation else None
            coasted = 1
        else:
            continue
        rec = {
            "t_cross_ms": t_cross * 1e3,
            "rho_cross_mm": float(np.hypot(xc, yc)) * 1e3,
            "vz_cross_m_s": float(vzc),
            "t_first_ms": s[0] * dt_s * 1e3,
            "coasted": coasted,
        }
        if include_excitation:
            rec["exc"] = exc
        recs.append(rec)
    return recs

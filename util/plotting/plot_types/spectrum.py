"""Integrated probe spectrum over all atoms at all timesteps.

Feeds every atom-timestep of a simulation run into the steady-state
spectrum kernel (the same physics as GUI/widgets/spectrum_tab), so slow
atoms that linger in the high-intensity region contribute proportionally
more. Plots the peak-normalized scattering rate vs detuning from the
trap beam for the three probe handednesses. Atom-timesteps outside the
(thin) probe beam contribute zero intensity and are pre-filtered for
speed — this changes nothing physically.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as scc

from ..common import HANDEDNESS_LABELS, POL_COLORS, SPECTRUM_LABELS, output_dir
from ..registry import plot_type


def _lasers_of_type(cfg: dict, kind: str) -> list[dict]:
    return [L for L in cfg["Lasers"] if L.get("type") == kind]


@plot_type(
    "integrated_spectrum",
    input="run",
    description="Peak-normalized probe spectrum integrated over a run",
)
def plot_integrated_spectrum(
    run_dir: str | Path,
    scan_min_MHz: float = -80.0,
    scan_max_MHz: float = 260.0,
    n_bins: int = 300,
    direction: tuple[float, float, float] = (0.0, -1.0, -0.16),
    power_mW: float = 0.005,
    radius_mm: float = 0.5,
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    interaction_name: str = "Lithium18LevelInteraction",
    use_b_field: bool = True,
    step_stride: int = 1,
    save_path: str | Path | None = None,
    show: bool = False,
):
    # Heavy import (Numba JIT) — keep local so importing the plotting
    # package stays cheap.
    from src.spectrum_kernel import (
        BeamConfig,
        _gaussian_intensity,
        build_interaction,
        build_magnetic_field,
        compute_spectrum_scan,
    )

    run_dir = Path(run_dir)
    csv_path = run_dir / "result.csv"
    cfg = json.loads((run_dir / "config.json").read_text())

    traps = _lasers_of_type(cfg, "trap")
    if not traps:
        raise ValueError(
            "No trap laser in config to use as frequency reference."
        )
    trap_freq_Hz = float(traps[0]["beam_frequency"]) * 1e6

    df = pd.read_csv(
        csv_path,
        usecols=[
            "step",
            "position_x",
            "position_y",
            "position_z",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "current_groundstate",
        ],
    )
    if step_stride > 1:
        keep = np.sort(df["step"].unique())[::step_stride]
        df = df[df["step"].isin(keep)]

    positions = df[["position_x", "position_y", "position_z"]].to_numpy(
        np.float64
    )
    velocities = df[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(
        np.float64
    )
    ground_states = df["current_groundstate"].to_numpy(np.int32)

    # Pre-filter atom-timesteps outside the probe beam (zero intensity
    # -> zero rate).
    dvec = np.asarray(direction, np.float64)
    dvec = dvec / np.linalg.norm(dvec)
    wavelength = scc.c / trap_freq_Hz
    intensity = _gaussian_intensity(
        positions,
        np.asarray(origin_m, np.float64),
        dvec,
        radius_mm * 1e-3,
        wavelength,
        1.0,
    )
    inside = intensity > 0.0
    positions, velocities, ground_states = (
        positions[inside],
        velocities[inside],
        ground_states[inside],
    )

    interaction = build_interaction(interaction_name)
    mag_field = (
        build_magnetic_field(cfg["Magnetic_Fields"]) if use_b_field else None
    )
    detunings = np.linspace(scan_min_MHz, scan_max_MHz, n_bins)

    fig, ax = plt.subplots(figsize=(9, 5))
    for h in (-1, 0, 1):
        beam = BeamConfig(
            origin_m=np.asarray(origin_m, np.float64),
            direction=dvec,
            power_W=power_mW * 1e-3,
            frequency_Hz=trap_freq_Hz,
            detuning_offset_rad=0.0,
            handedness=h,
            radius_m=radius_mm * 1e-3,
            use_position=True,
        )
        res = compute_spectrum_scan(
            positions,
            velocities,
            ground_states,
            interaction,
            mag_field,
            beam,
            detunings,
        )
        peak = float(res.rates.max())
        y = res.rates / peak if peak > 0.0 else res.rates
        color = POL_COLORS[HANDEDNESS_LABELS[h]]
        ax.plot(
            res.detunings_MHz,
            y,
            color=color,
            label=SPECTRUM_LABELS[h],
            lw=1.8,
        )
        ax.fill_between(res.detunings_MHz, y, alpha=0.10, color=color)

    # Orientation guides: trap line at 0, repump beam offset.
    ax.axvline(0.0, color="0.5", lw=0.8, ls=":")
    repumps = _lasers_of_type(cfg, "repump")
    if repumps:
        rep_off = (
            float(repumps[0]["beam_frequency"]) * 1e6 - trap_freq_Hz
        ) / 1e6
        ax.axvline(rep_off, color="0.5", lw=0.8, ls=":")
        ax.text(rep_off, 1.02, "repump", color="0.4", ha="center", fontsize=9)
    ax.text(0.0, 1.02, "trap", color="0.4", ha="center", fontsize=9)

    ax.set_xlabel("Detuning from trap beam (MHz)")
    ax.set_ylabel("Normalized scattering rate (peak = 1)")
    ax.set_title(
        f"Integrated probe spectrum — {run_dir.name}  "
        f"($N={len(positions)}$ atom-steps in beam"
        f"{', B-field' if use_b_field else ', B=0'})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is None:
        save_path = (
            output_dir("spectra")
            / f"{run_dir.name}_integrated_spectrum.png"
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path

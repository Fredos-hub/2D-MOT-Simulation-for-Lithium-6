"""Integrated probe spectrum over all atoms at all timesteps.

Feeds every atom-timestep of a simulation run into the steady-state spectrum
kernel (the same physics as GUI/widgets/spectrum_tab), so slow atoms that
linger in the high-intensity region contribute proportionally more. Plots the
peak-normalized scattering rate vs detuning from the trap beam for the three
probe handednesses. Atom-timesteps outside the (thin) probe beam contribute
zero intensity and are pre-filtered for speed — this changes nothing physically.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as scc

from src.spectrum_kernel import (
    BeamConfig,
    _gaussian_intensity,
    build_interaction,
    build_magnetic_field,
    compute_spectrum_scan,
)

HANDEDNESS_LABELS = {
    -1: "LH ($\\sigma^-$)",
    0: "LIN ($\\pi$)",
    1: "RH ($\\sigma^+$)",
}
POL_COLORS = {-1: "tab:blue", 1: "tab:red", 0: "tab:green"}


def _lasers_of_type(cfg: dict, kind: str) -> list[dict]:
    return [L for L in cfg["Lasers"] if L.get("type") == kind]


def plot_integrated_spectrum(
    csv_path: str | Path,
    config_path: str | Path,
    scan_min_MHz: float = -80.0,
    scan_max_MHz: float = 260.0,
    n_bins: int = 300,
    direction=(0.0, -1.0, -0.16),
    power_mW: float = 0.005,
    radius_mm: float = 0.5,
    origin_m=(0.0, 0.0, 0.0),
    interaction_name: str = "Lithium18LevelInteraction",
    use_b_field: bool = True,
    step_stride: int = 1,
    save_path: str | Path | None = None,
    show: bool = False,
):
    csv_path = Path(csv_path)
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())

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

    # Pre-filter atom-timesteps outside the probe beam (zero intensity -> zero rate).
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
        ax.plot(
            res.detunings_MHz,
            y,
            color=POL_COLORS[h],
            label=HANDEDNESS_LABELS[h],
            lw=1.8,
        )
        ax.fill_between(res.detunings_MHz, y, alpha=0.10, color=POL_COLORS[h])

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
        f"Integrated probe spectrum — {csv_path.parent.name}  "
        f"($N={len(positions)}$ atom-steps in beam"
        f"{', B-field' if use_b_field else ', B=0'})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is None:
        save_path = (
            Path(__file__).parent
            / "spectra"
            / f"{csv_path.parent.name}_integrated_spectrum.png"
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    run = repo_root / "simulation_results" / "19_05_26_1" / "run_4"
    plot_integrated_spectrum(run / "result.csv", run / "config.json")

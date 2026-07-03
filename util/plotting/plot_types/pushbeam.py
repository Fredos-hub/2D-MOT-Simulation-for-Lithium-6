"""Per-detuning v_z histograms and scan summaries for push-beam sessions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import (
    FWHM_FACTOR,
    POL_COLORS,
    load_snapshot,
    output_dir,
    pushbeam_meta,
    session_runs,
)
from ..registry import plot_type


def _step_for_detuning(det: float) -> int:
    """Snapshot step per detuning for the 2026-05 detuning-scan campaign:
    -15..13 -> 1.2 ms, 15..23 -> 2.5 ms, 25 -> 3.0 ms
    (default_time_step = 10 us).
    """
    if det <= 13:
        return 119
    if det <= 23:
        return 249
    return 299


@plot_type(
    "pushbeam_detuning_histograms",
    input="session",
    description="Overlaid v_z histograms (LH/RH/LIN) per push detuning",
)
def plot_pushbeam_detuning_histograms(
    session_dir: str | Path,
    exclusion_radius_mm: float = 30.0,
    transverse_radius_mm: float = 5.0,
    bins: int = 40,
    v_range: tuple[float, float] | None = None,
    save_dir: str | Path | None = None,
    show: bool = False,
):
    """One figure per detuning, 3 overlaid v_z histograms (LH/RH/LIN).

    exclusion_radius_mm  — atom must be FURTHER than this from (0,0,0)
    transverse_radius_mm — atom must be CLOSER than this to the z-axis
    """
    session_dir = Path(session_dir)

    data: dict[float, dict[str, np.ndarray]] = {}
    for run in session_runs(session_dir):
        meta = pushbeam_meta(run / "config.json")
        if meta is None:
            continue
        det, pol = meta["detuning"], meta["pol"]
        vz = load_snapshot(
            run / "result.csv",
            _step_for_detuning(det),
            exclusion_radius_mm,
            transverse_radius_mm,
            columns=("velocity_z",),
        )["velocity_z"].to_numpy()
        data.setdefault(det, {})[pol] = vz

    if save_dir is None:
        save_dir = output_dir("pushbeam_histograms", session_dir.name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for det in sorted(data):
        pol_data = data[det]
        step = _step_for_detuning(det)

        if v_range is None:
            non_empty = [v for v in pol_data.values() if len(v) > 0]
            if non_empty:
                cat = np.concatenate(non_empty)
                rng = (float(cat.min()), float(cat.max()))
            else:
                rng = (0.0, 1.0)
        else:
            rng = v_range

        fig, ax = plt.subplots(figsize=(8, 5))
        for pol in ("LH", "RH", "LIN"):
            vz = pol_data.get(pol, np.array([]))
            ax.hist(
                vz,
                bins=bins,
                range=rng,
                alpha=0.5,
                label=f"{pol} (n={len(vz)})",
                color=POL_COLORS[pol],
                histtype="stepfilled",
                edgecolor=POL_COLORS[pol],
            )
        ax.set_xlabel(r"$v_z$ (m/s)")
        ax.set_ylabel("counts")
        ax.set_title(
            rf"Push-beam $\delta/\Gamma = {det:+g}$   "
            rf"(step {step}; |r| > {exclusion_radius_mm} mm, "
            rf"$\rho$ < {transverse_radius_mm} mm)"
        )
        ax.legend()
        fig.tight_layout()

        save_path = save_dir / f"det{int(det):+d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        results.append((fig, det, save_path))
        if not show:
            plt.close(fig)

    if show:
        plt.show()
    return results


@plot_type(
    "pushbeam_mean_vz_vs_detuning",
    input="session",
    description="Mean v_z vs push detuning per polarization, vs experiment",
)
def plot_pushbeam_mean_vz_vs_detuning(
    session_dir: str | Path,
    experimental_csv: str | Path,
    exclusion_radius_mm: float = 30.0,
    transverse_radius_mm: float = 5.0,
    min_excitation_count: int = 300,
    extrapolate_z: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Mean v_z vs push-beam detuning, one curve per polarization
    (LH/RH/LIN), with the experimental reference overlaid.

    Atoms that scattered fewer than `min_excitation_count` photons were
    never really pushed out of the centre and are excluded (physical
    alternative to trimming the low-v_z tail).

    With `extrapolate_z`, atoms that left the box before the snapshot
    step are coasted ballistically so the fast push-out atoms are not
    lost from the average.
    """
    session_dir = Path(session_dir)

    data: dict[float, dict[str, pd.DataFrame]] = {}
    for run in session_runs(session_dir):
        meta = pushbeam_meta(run / "config.json")
        if meta is None:
            continue
        det, pol = meta["detuning"], meta["pol"]
        snap = load_snapshot(
            run / "result.csv",
            _step_for_detuning(det),
            exclusion_radius_mm,
            transverse_radius_mm,
            extrapolate=extrapolate_z,
            dt_s=meta["dt_s"] if extrapolate_z else None,
        )
        data.setdefault(det, {})[pol] = snap

    curves = {}
    for pol in ("LH", "RH", "LIN"):
        dets, means, errs = [], [], []
        for det in sorted(data):
            if pol not in data[det]:
                continue
            sub = data[det][pol]
            vz = sub.loc[
                sub["excitation_count"] >= min_excitation_count,
                "velocity_z",
            ].to_numpy()
            if len(vz) == 0:
                continue
            dets.append(det)
            means.append(vz.mean())
            errs.append(
                vz.std(ddof=1) / np.sqrt(len(vz)) if len(vz) > 1 else 0.0
            )
        if dets:
            curves[pol] = (dets, means, errs)

    exp = pd.read_csv(experimental_csv)
    exp_lookup = {
        round(float(d), 3): float(v)
        for d, v in zip(exp["pb_det"], exp["v_z"], strict=True)
    }

    fig, (ax, axd) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for pol, (dets, means, errs) in curves.items():
        ax.errorbar(
            dets,
            means,
            yerr=errs,
            marker="o",
            capsize=3,
            color=POL_COLORS[pol],
            label=f"{pol} (sim)",
        )
        ddet = [d for d in dets if round(float(d), 3) in exp_lookup]
        dvz = [
            m - exp_lookup[round(float(d), 3)]
            for d, m in zip(dets, means, strict=True)
            if round(float(d), 3) in exp_lookup
        ]
        axd.plot(ddet, dvz, marker="o", color=POL_COLORS[pol])

    ax.plot(
        exp["pb_det"],
        exp["v_z"],
        marker="x",
        linestyle="none",
        color="black",
        markersize=8,
        label="experiment",
    )
    axd.axhline(0, color="0.6", lw=0.8)

    ax.set_ylabel(r"mean $v_z$ (m/s)")
    ax.set_title(
        rf"Mean $v_z$ vs detuning  "
        rf"(|r| > {exclusion_radius_mm} mm, "
        rf"$\rho$ < {transverse_radius_mm} mm, "
        rf"exc $\geq$ {min_excitation_count}"
        + (", z-extrap)" if extrapolate_z else ")")
    )
    ax.legend()
    axd.set_xlabel(r"push-beam detuning $\delta/\Gamma$")
    axd.set_ylabel(r"sim $-$ exp (m/s)")
    fig.tight_layout()

    if save_path is None:
        save_path = (
            output_dir("pushbeam_histograms", session_dir.name)
            / "mean_vz_vs_detuning.png"
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, save_path


@plot_type(
    "pushbeam_velocity_spread",
    input="session",
    description="Velocity FWHM (x/y/z) over time per (detuning, pol)",
)
def plot_pushbeam_velocity_spread_over_time(
    session_dir: str | Path,
    save_dir: str | Path | None = None,
    show: bool = False,
):
    """Velocity spread (Gaussian-equivalent FWHM = 2.355*std) of v_x,
    v_y, v_z over time for each (detuning, polarization), over all atoms.

    Atoms that leave the box are held at their last velocity
    (forward-fill): once outside the beam they coast, so the spread
    doesn't artificially collapse as the fast atoms exit. One figure per
    run.
    """
    session_dir = Path(session_dir)
    runs = session_runs(session_dir)

    if save_dir is None:
        save_dir = output_dir(
            "pushbeam_histograms",
            session_dir.name,
            "velocity_spread_over_time",
        )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    comps = [
        ("velocity_x", r"$v_x$", "tab:blue"),
        ("velocity_y", r"$v_y$", "tab:orange"),
        ("velocity_z", r"$v_z$", "tab:green"),
    ]

    results = []
    for run in runs:
        meta = pushbeam_meta(run / "config.json")
        if meta is None:
            continue
        det, pol = meta["detuning"], meta["pol"]
        df = pd.read_csv(
            run / "result.csv",
            usecols=[
                "step",
                "atom_id",
                "subjective_time",
                "velocity_x",
                "velocity_y",
                "velocity_z",
            ],
        )
        steps = np.sort(df["step"].unique())
        t = (
            df.groupby("step")["subjective_time"]
            .mean()
            .reindex(steps)
            .to_numpy()
            * 1e3
        )  # ms

        fig, ax = plt.subplots(figsize=(8, 5))
        for col, lbl, c in comps:
            # hold each atom's last velocity after it leaves the box
            piv = df.pivot(index="step", columns="atom_id", values=col)
            piv = piv.reindex(steps).ffill()
            fwhm = piv.std(axis=1, ddof=1).to_numpy() * FWHM_FACTOR
            ax.plot(t, fwhm, color=c, label=lbl)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(r"velocity FWHM $\approx 2.355\,\sigma$ (m/s)")
        ax.set_title(
            rf"$\delta/\Gamma = {det:+g}$, {pol} — velocity spread vs time"
        )
        ax.legend()
        fig.tight_layout()

        save_path = save_dir / f"det{int(det):+d}_{pol}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        results.append((det, pol, save_path))
        if not show:
            plt.close(fig)
    return results

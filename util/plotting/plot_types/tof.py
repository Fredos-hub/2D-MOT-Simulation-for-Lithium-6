"""Time-of-flight spectra and push-scan summaries at a z-crossing plane.

Two workflows:
- session-direct: `plot_pushbeam_tof_spectrum` / `..._circular_average`
  read every run's result.csv on each call.
- cached: `build_tof_summary` / `extract_crossings` parse each run once
  into a tidy crossings table; the `plot_tof_from_summary` and
  `plot_pushscan_*` plotters then work from that cache (instant re-plots
  with different filters/bins).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..common import (
    crossing_records,
    load_config,
    output_dir,
    pushbeam_meta,
    session_runs,
)
from ..registry import plot_type

PUSH_LEN_CMAP = plt.get_cmap("viridis")


def _crossing_times_ms(
    csv_path: Path,
    z_threshold_m: float,
    dt_s: float,
    transverse_radius_mm: float,
    time_reference: str,
    extrapolate_to_ms: float | None = None,
    push_start_ms: float = 0.0,
) -> np.ndarray:
    """First-crossing times (ms) of z = z_threshold, kept only for atoms
    with rho < transverse_radius. Thin filter over `crossing_records`.

    time_reference: "lab"    -> time since push start (t_cross - t_on)
                    "flight" -> time since the atom's first-recorded time
    """
    out = []
    for r in crossing_records(
        csv_path, z_threshold_m, dt_s, extrapolate_to_ms
    ):
        if r["rho_cross_mm"] >= transverse_radius_mm:
            continue
        t = r["t_cross_ms"] - (
            r["t_first_ms"] if time_reference == "flight" else push_start_ms
        )
        out.append(t)
    return np.asarray(out)


def _collect_tof_times(
    session_dir,
    z_threshold_m,
    transverse_radius_mm,
    time_reference,
    extrapolate_to_ms,
):
    """data[pol][push_len] = crossing times (ms) across all runs."""
    data: dict[str, dict[float, list]] = {}
    nreps: dict[str, dict[float, int]] = {}
    for run in session_runs(session_dir):
        meta = pushbeam_meta(run / "config.json")
        if meta is None or meta["pulse_ms"] is None:
            continue
        pol, push_len, t_on = meta["pol"], meta["pulse_ms"], meta["t_on_ms"]
        times = _crossing_times_ms(
            run / "result.csv",
            z_threshold_m,
            meta["dt_s"],
            transverse_radius_mm,
            time_reference,
            extrapolate_to_ms,
            push_start_ms=t_on,
        )
        data.setdefault(pol, {}).setdefault(push_len, []).append(times)
        nreps.setdefault(pol, {})[push_len] = (
            nreps.get(pol, {}).get(push_len, 0) + 1
        )
    pooled = {
        pol: {
            pl: (np.concatenate(ch) if ch else np.asarray([]))
            for pl, ch in by_pl.items()
        }
        for pol, by_pl in data.items()
    }
    return pooled, nreps


def _tof_edges(data, bin_width_ms, t_range, extrapolate_to_ms):
    if t_range is not None:
        rng = t_range
    elif extrapolate_to_ms is not None:
        rng = (0.0, float(extrapolate_to_ms))
    else:
        allt = [t for pl in data.values() for t in pl.values() if len(t)]
        rng = (0.0, float(np.concatenate(allt).max())) if allt else (0.0, 1.0)
    return np.arange(rng[0], rng[1] + bin_width_ms, bin_width_ms)


def build_tof_summary(
    session_dir: str | Path,
    out_csv: str | Path,
    z_threshold_m: float = 0.19,
    extrapolate_to_ms: float = 20.0,
) -> Path:
    """Parse every run_* once and cache per-atom z-crossing records to a
    tidy CSV so re-plots don't re-read the (large) result.csv files.
    Built deliberately permissive — NO transverse-radius cut (rho stored,
    filter at plot time) and a generous coast horizon — so the cache
    serves any radius / frame / bin / range. Only `z_threshold_m` and
    `extrapolate_to_ms` are baked in.

    Columns: seed, pol, push_len_ms, t_on_ms, z_cm, t_cross_ms,
             rho_cross_mm, vz_cross_m_s, t_first_ms, coasted
    """
    rows = []
    for run in session_runs(session_dir):
        meta = pushbeam_meta(run / "config.json")
        if meta is None or meta["pulse_ms"] is None:
            continue
        for r in crossing_records(
            run / "result.csv", z_threshold_m, meta["dt_s"], extrapolate_to_ms
        ):
            rows.append(
                {
                    "seed": meta["seed"],
                    "pol": meta["pol"],
                    "push_len_ms": meta["pulse_ms"],
                    "t_on_ms": meta["t_on_ms"],
                    "z_cm": z_threshold_m * 100,
                    **r,
                }
            )
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


@plot_type(
    "tof_from_summary",
    input="summary_csv",
    description="ToF spectra from a build_tof_summary cache",
)
def plot_tof_from_summary(
    summary_csv: str | Path,
    transverse_radius_mm: float = 5.0,
    time_reference: str = "lab",
    bin_width_ms: float = 0.1,
    t_range: tuple[float, float] | None = None,
    extrapolate_to_ms: float | None = None,
    density_weight: bool = False,
    save_dir: str | Path | None = None,
    show: bool = False,
):
    """ToF spectra from a `build_tof_summary` cache (no result.csv
    reads). Same figure as `plot_pushbeam_tof_spectrum`: one per
    polarization, push lengths overlaid, per-run mean counts (pooled
    crossings / distinct seeds).

    density_weight: if True, weight each crossing by 1/v_z (transit time
    through a thin probe) instead of counting it as 1 — i.e. emulate a
    density/fluorescence probe signal rather than a flux-through-plane
    count.
    """
    if time_reference not in ("lab", "flight"):
        raise ValueError(
            f"time_reference must be 'lab' or 'flight', got {time_reference!r}"
        )
    df = pd.read_csv(summary_csv)
    df = df[df["rho_cross_mm"] < transverse_radius_mm].copy()
    df["t"] = df["t_cross_ms"] - (
        df["t_first_ms"] if time_reference == "flight" else df["t_on_ms"]
    )

    # build {pol: {push_len: times}} and seed counts
    data: dict[str, dict[float, np.ndarray]] = {}
    wts: dict[str, dict[float, np.ndarray]] = {}
    nreps: dict[str, dict[float, int]] = {}
    for (pol, pl), grp in df.groupby(["pol", "push_len_ms"]):
        data.setdefault(pol, {})[pl] = grp["t"].to_numpy()
        if density_weight:
            wts.setdefault(pol, {})[pl] = 1.0 / grp["vz_cross_m_s"].to_numpy()
        nreps.setdefault(pol, {})[pl] = grp["seed"].nunique()
    edges = _tof_edges(data, bin_width_ms, t_range, extrapolate_to_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])

    save_dir = (
        Path(save_dir)
        if save_dir
        else Path(summary_csv).parent / "tof_spectrum"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    xlabel = (
        "time of flight (ms)"
        if time_reference == "flight"
        else "time since push start (ms)"
    )
    results = []
    for pol in ("LH", "RH", "LIN"):
        if pol not in data:
            continue
        push_lens = sorted(data[pol])
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, pl in enumerate(push_lens):
            t = data[pol][pl]
            w = wts[pol][pl] if density_weight else None
            counts = np.histogram(t, bins=edges, weights=w)[0] / nreps[pol][pl]
            color = PUSH_LEN_CMAP(k / max(len(push_lens) - 1, 1))
            ax.plot(
                centers,
                counts,
                linestyle="--",
                linewidth=0.9,
                marker="o",
                markersize=3,
                color=color,
                label=f"{pl:g} ms (n={len(t) / nreps[pol][pl]:.0f})",
            )
        ax.set_xlabel(xlabel)
        if density_weight:
            ax.set_ylabel(
                f"probe signal $\\propto$ transit time (1/$v_z$), "
                f"z = {df['z_cm'].iloc[0]:g} cm  [arb., per run]"
            )
        else:
            ax.set_ylabel(
                f"atoms crossing z = {df['z_cm'].iloc[0]:g} cm "
                f"per {bin_width_ms * 1e3:g} us (per run)"
            )
        wtag = "density-weighted" if density_weight else "count"
        ax.set_title(
            rf"ToF spectrum ({wtag}), {pol}  "
            rf"($\rho$ < {transverse_radius_mm:g} mm)"
        )
        ax.legend(title="push length")
        fig.tight_layout()
        rtag = f"r{transverse_radius_mm:g}mm"
        suffix = "_dens" if density_weight else ""
        save_path = save_dir / f"tof_{pol}_{time_reference}_{rtag}{suffix}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        results.append((pol, save_path))
        if not show:
            plt.close(fig)
    if show:
        plt.show()
    return results


@plot_type(
    "tof_spectrum",
    input="session",
    description="ToF spectra per polarization, push lengths overlaid",
)
def plot_pushbeam_tof_spectrum(
    session_dir: str | Path,
    z_threshold_m: float = 0.19,
    transverse_radius_mm: float = 15.0,
    time_reference: str = "lab",
    bin_width_ms: float = 0.1,
    extrapolate_to_ms: float | None = None,
    t_range: tuple[float, float] | None = None,
    save_dir: str | Path | None = None,
    show: bool = False,
):
    """Time-of-flight spectra: atom counts crossing the z =
    `z_threshold_m` plane vs time. One figure per polarization
    (LH/RH/LIN), with the push lengths (pulse t_off) overlaid as
    separate curves.

    transverse_radius_mm — keep atoms with hypot(x, y) < this at the
                           crossing.
    bin_width_ms         — histogram bin width (default 100 us, matching
                           the experiment).
    extrapolate_to_ms    — coast non-crossing atoms ballistically up to
                           this time.
    time_reference       — "lab" (since push start) or "flight" (since
                           injection).
    """
    if time_reference not in ("lab", "flight"):
        raise ValueError(
            f"time_reference must be 'lab' or 'flight', got {time_reference!r}"
        )

    data, nreps = _collect_tof_times(
        session_dir,
        z_threshold_m,
        transverse_radius_mm,
        time_reference,
        extrapolate_to_ms,
    )
    edges = _tof_edges(data, bin_width_ms, t_range, extrapolate_to_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])

    session_dir = Path(session_dir)
    if save_dir is None:
        save_dir = output_dir(
            "pushbeam_histograms", session_dir.name, "tof_spectrum"
        )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    xlabel = (
        "time of flight (ms)"
        if time_reference == "flight"
        else "time since push start (ms)"
    )
    results = []
    for pol in ("LH", "RH", "LIN"):
        if pol not in data:
            continue
        push_lens = sorted(data[pol])
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, pl in enumerate(push_lens):
            t = data[pol][pl]
            counts = np.histogram(t, bins=edges)[0] / nreps[pol][pl]
            color = PUSH_LEN_CMAP(k / max(len(push_lens) - 1, 1))
            ax.plot(
                centers,
                counts,
                linestyle="--",
                linewidth=0.9,
                marker="o",
                markersize=3,
                color=color,
                label=f"{pl:g} ms (n={len(t) / nreps[pol][pl]:.0f})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            f"atoms crossing z = {z_threshold_m * 100:g} cm "
            f"per {bin_width_ms * 1e3:g} us (per run)"
        )
        ax.set_title(
            rf"ToF spectrum, {pol}  "
            rf"($\rho$ < {transverse_radius_mm:g} mm)"
        )
        ax.legend(title="push length")
        fig.tight_layout()

        rtag = f"r{transverse_radius_mm:g}mm"
        save_path = save_dir / f"tof_{pol}_{time_reference}_{rtag}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        results.append((pol, save_path))
        if not show:
            plt.close(fig)
    if show:
        plt.show()
    return results


@plot_type(
    "tof_circular_average",
    input="session",
    description="ToF spectrum averaged over LH and RH polarizations",
)
def plot_pushbeam_tof_circular_average(
    session_dir: str | Path,
    z_threshold_m: float = 0.19,
    transverse_radius_mm: float = 1.0,
    time_reference: str = "lab",
    bin_width_ms: float = 0.1,
    extrapolate_to_ms: float | None = None,
    t_range: tuple[float, float] | None = None,
    save_dir: str | Path | None = None,
    show: bool = False,
):
    """ToF spectrum averaged over the two circular polarizations
    (LH, RH), one curve per push length. Each curve is the per-bin mean
    of the LH and RH histograms (LIN is excluded — it is not circular).
    """
    if time_reference not in ("lab", "flight"):
        raise ValueError(
            f"time_reference must be 'lab' or 'flight', got {time_reference!r}"
        )

    data, nreps = _collect_tof_times(
        session_dir,
        z_threshold_m,
        transverse_radius_mm,
        time_reference,
        extrapolate_to_ms,
    )
    if "LH" not in data or "RH" not in data:
        raise ValueError(
            "need both LH and RH runs to average circular polarizations"
        )
    edges = _tof_edges(data, bin_width_ms, t_range, extrapolate_to_ms)

    session_dir = Path(session_dir)
    if save_dir is None:
        save_dir = output_dir(
            "pushbeam_histograms", session_dir.name, "tof_spectrum"
        )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    centers = 0.5 * (edges[:-1] + edges[1:])
    push_lens = sorted(set(data["LH"]) & set(data["RH"]))
    xlabel = (
        "time of flight (ms)"
        if time_reference == "flight"
        else "time since push start (ms)"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, pl in enumerate(push_lens):
        h_lh = np.histogram(data["LH"][pl], bins=edges)[0] / nreps["LH"][pl]
        h_rh = np.histogram(data["RH"][pl], bins=edges)[0] / nreps["RH"][pl]
        avg = 0.5 * (h_lh + h_rh)
        n_avg = 0.5 * (
            len(data["LH"][pl]) / nreps["LH"][pl]
            + len(data["RH"][pl]) / nreps["RH"][pl]
        )
        color = PUSH_LEN_CMAP(k / max(len(push_lens) - 1, 1))
        ax.plot(
            centers,
            avg,
            linestyle="--",
            linewidth=0.9,
            marker="o",
            markersize=3,
            color=color,
            label=f"{pl:g} ms (n={n_avg:.0f})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        f"atoms crossing z = {z_threshold_m * 100:g} cm "
        f"per {bin_width_ms * 1e3:g} us"
    )
    ax.set_title(
        rf"ToF spectrum, circular avg (LH+RH)/2  "
        rf"($\rho$ < {transverse_radius_mm:g} mm)"
    )
    ax.legend(title="push length")
    fig.tight_layout()

    rtag = f"r{transverse_radius_mm:g}mm"
    save_path = save_dir / f"tof_circ-avg_{time_reference}_{rtag}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


# ── Push-param-scan analysis (2026-06-25 campaign style) ──
# Works on a crossings table built by `extract_crossings` over the runs
# of interest; the plotters below consume that table.


def extract_crossings(
    run_dir: str | Path,
    z_threshold_m: float = 0.19,
    extrapolate_to_ms: float = float("inf"),
) -> pd.DataFrame:
    """Per-atom crossings of z = z_threshold for one run, with the run's
    push-beam metadata attached as columns (pol, det, pulse_ms, t_on_ms,
    power_mW, seed, weight). t_arrival_ms is relative to push start.
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir)
    meta = pushbeam_meta(cfg)
    if meta is None:
        raise ValueError(f"no push beam found in {run_dir / 'config.json'}")
    recs = crossing_records(
        run_dir / "result.csv",
        z_threshold_m,
        meta["dt_s"],
        extrapolate_to_ms=extrapolate_to_ms,
        include_excitation=True,
    )
    rec = pd.DataFrame(recs)
    if rec.empty:
        return rec
    rec = rec.rename(columns={"vz_cross_m_s": "vz_cross"})
    rec["t_arrival_ms"] = rec["t_cross_ms"] - meta["t_on_ms"]
    for k in (
        "t_on_ms",
        "pulse_ms",
        "detuning",
        "pol",
        "power_mW",
        "seed",
        "weight",
    ):
        rec["det" if k == "detuning" else k] = meta[k]
    return rec


def filter_crossings(
    df: pd.DataFrame, rho_max_mm: float = 5.0, exc_min: int = 300
) -> pd.DataFrame:
    """Standard campaign filter: transverse radius + excitation count."""
    return df[(df.rho_cross_mm <= rho_max_mm) & (df.exc >= exc_min)]


def _gauss(x: np.ndarray, A: float, mu: float, sig: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def fit_peak_vz(vz: np.ndarray, binw: float = 2.0) -> tuple:
    """Gaussian fit to the fast peak: window = contiguous bins >=30% of
    the mode bin, tail rejected. Returns (A, mu, sigma); falls back to
    (nan, mode, nan).
    """
    vz = np.asarray(vz)
    if len(vz) < 20:
        return np.nan, (np.median(vz) if len(vz) else np.nan), np.nan
    bins = np.arange(0, max(vz.max() + binw, 12), binw)
    h, e = np.histogram(vz, bins=bins)
    c = 0.5 * (e[:-1] + e[1:])
    i = int(h.argmax())
    peak = h[i]
    mode = c[i]
    thr = 0.3 * peak
    lo = i
    while lo > 0 and h[lo - 1] >= thr:
        lo -= 1
    hi = i
    while hi < len(h) - 1 and h[hi + 1] >= thr:
        hi += 1
    xw, yw = c[lo : hi + 1], h[lo : hi + 1]
    if len(xw) < 4:
        return np.nan, mode, np.nan
    try:
        A, mu, sig = curve_fit(
            _gauss,
            xw,
            yw,
            p0=[peak, mode, max(binw, (xw[-1] - xw[0]) / 4)],
            maxfev=5000,
        )[0]
        sig = abs(sig)
        if (
            not (xw[0] - binw <= mu <= xw[-1] + binw)
            or sig > 80
            or sig < binw / 2
        ):
            return np.nan, mode, np.nan
        return A, mu, sig
    except Exception:
        return np.nan, mode, np.nan


def plot_pushscan_tof(
    crossings: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """ToF spectrum + v_z distribution per pulse length, one curve per
    push start time (crossings pre-filtered with `filter_crossings`).
    """
    d = crossings
    out_dir = Path(out_dir) if out_dir else output_dir("push_param_scans")
    tons = [5.0, 2.0, 0.0]
    col = {5.0: "C0", 2.0: "C1", 0.0: "C2"}
    for pulse in sorted(d.pulse_ms.unique()):
        for kind, field, bins, xl, fn in [
            (
                "ToF spectrum",
                "t_arrival_ms",
                np.arange(0, 8.01, 0.1),
                "arrival time since push (ms)",
                "tof_spectrum",
            ),
            (
                "v_z distribution",
                "vz_cross",
                np.arange(0, 160, 2),
                "v_z at 19 cm (m/s)",
                "vz_dist",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for t in tons:
                s = d[(d.pulse_ms == pulse) & (d.t_on_ms == t)]
                if s.empty:
                    continue
                ax.hist(
                    s[field],
                    bins=bins,
                    weights=s.weight / s.seed.nunique(),
                    histtype="step",
                    color=col[t],
                    label=f"t_on={t:g} ms",
                )
                if field == "vz_cross":
                    A, mu, sg = fit_peak_vz(s.vz_cross.values)
                    if np.isfinite(A) and np.isfinite(sg):
                        scale = s.weight.iloc[0] / s.seed.nunique()
                        xf = np.linspace(0, 158, 400)
                        ax.plot(
                            xf,
                            _gauss(xf, A * scale, mu, sg),
                            color=col[t],
                            ls=":",
                            lw=1.2,
                        )
            ax.set_xlabel(xl)
            ax.set_ylabel("physical atoms / bin (per run)")
            ax.set_title(f"{kind} @19cm - pulse {pulse:g} ms (LH, det+7)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{fn}_pulse{pulse:g}ms.png", dpi=130)
            plt.close(fig)


def _series(dd: pd.DataFrame, xcol: str) -> tuple:
    xs = sorted(dd[xcol].unique())
    mu = []
    sig = []
    cnt = []
    for x in xs:
        s = dd[dd[xcol] == x]
        _, m, sg = fit_peak_vz(s.vz_cross.values)
        mu.append(m)
        sig.append(sg)
        cnt.append(len(s) * s.weight.iloc[0])
    return xs, mu, sig, cnt


def plot_pushscan_power(
    crossings: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Dual-axis power scan: fitted peak v_z + crossing counts."""
    d = crossings
    out_dir = Path(out_dir) if out_dir else output_dir("push_param_scans")
    x, vz, sig, cnt = _series(d, "power_mW")
    fig, a1 = plt.subplots(figsize=(7, 4.5))
    a2 = a1.twinx()
    x = np.asarray(x, float)
    vz = np.asarray(vz, float)
    sig = np.asarray(sig, float)
    a1.plot(x, vz, "o", color="C0")
    a1.fill_between(x, vz - sig, vz + sig, color="C0", alpha=0.15)
    a2.plot(x, cnt, "s", color="C3")
    a1.set_ylim(0, 120)
    a1.set_xlabel("push power (mW)")
    a1.set_ylabel("fitted peak v_z @19cm (m/s)", color="C0")
    a2.set_ylabel("atoms crossing 19 cm", color="C3")
    a1.set_xticks(np.arange(0, 1.21, 0.2))
    a1.grid(alpha=0.3)
    a1.set_title("Push power scan (det+7, LH, 5 ms start)")
    fig.tight_layout()
    fig.savefig(out_dir / "scan_power.png", dpi=130)
    plt.close(fig)


def plot_pushscan_duration(
    crossings: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Dual-axis pulse-duration scan per push start time."""
    d = crossings
    out_dir = Path(out_dir) if out_dir else output_dir("push_param_scans")
    tons = [5.0, 2.0, 0.0]
    mk = {5.0: "o", 2.0: "s", 0.0: "^"}
    col = {5.0: "C0", 2.0: "C1", 0.0: "C2"}
    fig, a1 = plt.subplots(figsize=(8, 5))
    a2 = a1.twinx()
    for t in tons:
        dd = d[d.t_on_ms == t]
        xs = sorted(dd.pulse_ms.unique())
        mu = []
        sg = []
        cm = []
        for p in xs:
            sub = dd[dd.pulse_ms == p]
            _, m, s = fit_peak_vz(sub.vz_cross.values)
            mu.append(m)
            sg.append(s)
            cm.append(len(sub) / sub.seed.nunique() * sub.weight.iloc[0])
        xs = np.asarray(xs, float)
        mu = np.asarray(mu, float)
        sg = np.asarray(sg, float)
        a1.plot(xs, mu, mk[t], color=col[t], label=f"v_z t_on={t:g}ms")
        a1.fill_between(xs, mu - sg, mu + sg, color=col[t], alpha=0.15)
        a2.plot(xs, cm, mk[t], color=col[t], mfc="none", ls="--", alpha=0.6)
    a1.set_xlabel("push pulse duration (ms)")
    a1.set_ylabel("fitted peak v_z @19cm (m/s)")
    a1.set_ylim(0, 120)
    a2.set_ylabel("atoms crossing 19 cm")
    a1.set_xlim(0, 3)
    a1.set_xticks(np.arange(0, 3.01, 0.5))
    a1.grid(alpha=0.3)
    a1.set_title("Pulse-duration scan (det+7, LH) - filled=v_z, open=counts")
    a1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "scan_duration.png", dpi=130)
    plt.close(fig)


def plot_pushscan_detuning(
    crossings: pd.DataFrame,
    out_dir: str | Path | None = None,
) -> None:
    """Dual-axis detuning scan per (polarization, push start time)."""
    d = crossings
    out_dir = Path(out_dir) if out_dir else output_dir("push_param_scans")
    for pol in ("LH", "LIN"):
        for t in (5.0, 2.0):
            dd = d[(d.pol == pol) & (d.t_on_ms == t)]
            if dd.empty:
                continue
            xs, vz, sig, cnt = _series(dd, "det")
            fig, a1 = plt.subplots(figsize=(7, 4.5))
            a2 = a1.twinx()
            xs = np.asarray(xs, float)
            vz = np.asarray(vz, float)
            sig = np.asarray(sig, float)
            a1.plot(xs, vz, "o", color="C0")
            a1.fill_between(xs, vz - sig, vz + sig, color="C0", alpha=0.15)
            a2.plot(xs, cnt, "s", color="C3", mfc="none")
            a1.set_ylim(0, 120)
            a1.set_xlabel("push detuning (Gamma)")
            a1.set_ylabel("fitted peak v_z @19cm (m/s)", color="C0")
            a2.set_ylabel("atoms crossing 19 cm", color="C3")
            a1.set_xticks(np.arange(0, 16, 2))
            a1.grid(alpha=0.3)
            a1.set_title(f"Detuning scan - {pol}, {t:g} ms start")
            fig.tight_layout()
            fig.savefig(out_dir / f"scan_detuning_{pol}_{t:g}ms.png", dpi=130)
            plt.close(fig)

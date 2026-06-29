"""Push-scan crossing analysis at z=19cm: ToF spectra + dual-axis scan summaries.

Extraction reads every run's result.csv once and writes a per-crossing cache
(crossings.csv). Re-run with --reuse to re-plot from cache (instant): filters,
bins, normalization and plot styling all work from the cache; only changing the
crossing plane (Z_M) or the coasting logic needs a rebuild.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parent.parent
SIM = REPO / "simulation_results"
OUT = REPO / "util" / "pushbeam_histograms" / "push_param_scans_2026-06-25"
OUT.mkdir(parents=True, exist_ok=True)
Z_M, EXC_MIN, RHO_MAX_MM = 0.19, 300, 5.0
POL = {-1: "LH", 0: "LIN", 1: "RH"}
USECOLS = [
    "step",
    "atom_id",
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "excitation_count",
]


def extract_crossings(run_dir: Path) -> pd.DataFrame:
    cfg = json.load(open(run_dir / "config.json"))
    sim = cfg["Simulation"]
    dt_ms = sim["default_time_step"] / 1000.0
    weight = float(sim["macro_particle_weight"])
    seed = int(sim.get("random_seed", -1))
    push = [l for l in cfg["Lasers"] if "t_on" in l]
    t_on = float(push[0]["t_on"])
    pulse = round(float(push[0]["t_off"]) - t_on, 3)
    det = float(push[0]["detuning"])
    pol = POL[int(push[0]["handedness"])]
    power_mW = round(sum(l["beam_power"] for l in push) * 1000, 3)
    df = pd.read_csv(run_dir / "result.csv", usecols=USECOLS)
    df.sort_values(["atom_id", "step"], inplace=True)
    cr = df[df.position_z >= Z_M].groupby("atom_id", sort=False).head(1)
    reached = set(cr.atom_id)
    last = df.groupby("atom_id", sort=False).tail(1)
    coast = last[~last.atom_id.isin(reached)]
    rec_r = pd.DataFrame(
        {
            "t_cross_ms": cr.step.to_numpy() * dt_ms,
            "vz_cross": cr.velocity_z.to_numpy(),
            "rho_cross_mm": np.hypot(cr.position_x, cr.position_y).to_numpy()
            * 1e3,
            "exc": cr.excitation_count.to_numpy(),
            "coasted": 0,
        }
    )
    vz = coast.velocity_z.to_numpy()
    zc = coast.position_z.to_numpy()
    ok = (vz > 0) & (zc < Z_M)
    coast = coast[ok]
    vz, zc = vz[ok], zc[ok]
    dtc = (Z_M - zc) / vz  # s
    rec_c = pd.DataFrame(
        {
            "t_cross_ms": coast.step.to_numpy() * dt_ms + dtc * 1e3,
            "vz_cross": vz,
            "rho_cross_mm": np.hypot(
                coast.position_x.to_numpy()
                + coast.velocity_x.to_numpy() * dtc,
                coast.position_y.to_numpy()
                + coast.velocity_y.to_numpy() * dtc,
            )
            * 1e3,
            "exc": coast.excitation_count.to_numpy(),
            "coasted": 1,
        }
    )
    rec = pd.concat([rec_r, rec_c], ignore_index=True)
    rec["t_arrival_ms"] = rec.t_cross_ms - t_on
    for k, v in dict(
        t_on_ms=t_on,
        pulse_ms=pulse,
        det=det,
        pol=pol,
        power_mW=power_mW,
        seed=seed,
        weight=weight,
    ).items():
        rec[k] = v
    return rec


def runs_from_manifest(mf: str | Path, multi: bool = False) -> list:
    out = []
    for _, r in pd.read_csv(mf).iterrows():
        folder = SIM / str(r["batch_folder"])
        if not folder.exists():
            continue
        rd = sorted(folder.glob("run_*")) if multi else [folder / "run_0"]
        out += [d for d in rd if (d / "result.csv").exists()]
    return out


def build() -> pd.DataFrame:
    runs = [
        ("power", d)
        for d in runs_from_manifest(REPO / "pb_powerscan_runs/manifest.csv")
    ]
    runs += [
        ("detuning", d)
        for d in runs_from_manifest(REPO / "pb_detscan_runs/manifest.csv")
    ]
    runs += [
        ("tof", d)
        for d in runs_from_manifest(
            REPO / "tof_runs_delays/manifest.csv", multi=True
        )
    ]
    for f in [
        "16_06_26_0",
        "17_06_26_0",
        "17_06_26_1",
        "17_06_26_2",
        "17_06_26_3",
    ]:
        runs += [
            ("tof", d)
            for d in sorted((SIM / f).glob("run_*"))
            if (d / "result.csv").exists()
        ]
    parts = []
    for ds, d in runs:
        r = extract_crossings(d)
        r["dataset"] = ds
        parts.append(r)
        print("done", ds, d.parent.name, d.name, len(r), flush=True)
    allc = pd.concat(parts, ignore_index=True)
    allc.to_csv(OUT / "crossings.csv", index=False)
    return allc


def filt(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.rho_cross_mm <= RHO_MAX_MM) & (df.exc >= EXC_MIN)]


def _gauss(x: np.ndarray, A: float, mu: float, sig: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def fit_peak_vz(vz: np.ndarray, binw: float = 2.0) -> tuple:
    """Gaussian fit to the fast peak: window = contiguous bins >=30% of the mode
    bin, tail rejected. Returns (A, mu, sigma); falls back to (nan, mode, nan).
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


def plot_tof(allc: pd.DataFrame) -> None:
    d = filt(allc[allc.dataset == "tof"])
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
            fig.savefig(OUT / f"{fn}_pulse{pulse:g}ms.png", dpi=130)
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


def plot_power(allc: pd.DataFrame) -> None:
    d = filt(allc[allc.dataset == "power"])
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
    fig.savefig(OUT / "scan_power.png", dpi=130)
    plt.close(fig)


def plot_duration(allc: pd.DataFrame) -> None:
    d = filt(allc[allc.dataset == "tof"])
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
    fig.savefig(OUT / "scan_duration.png", dpi=130)
    plt.close(fig)


def plot_detuning(allc: pd.DataFrame) -> None:
    d = filt(allc[allc.dataset == "detuning"])
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
            fig.savefig(OUT / f"scan_detuning_{pol}_{t:g}ms.png", dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    allc = (
        pd.read_csv(OUT / "crossings.csv")
        if ("--reuse" in sys.argv and (OUT / "crossings.csv").exists())
        else build()
    )
    plot_tof(allc)
    plot_power(allc)
    plot_duration(allc)
    plot_detuning(allc)
    print("figures in", OUT)

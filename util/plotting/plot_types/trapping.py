"""2D heatmap of trapped-atom counts over a (trap, repump) detuning scan."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as scc
from matplotlib.colors import BoundaryNorm

from ..common import LI6_MASS, output_dir, session_runs
from ..registry import plot_type

SUMMARY_FILE = "_trapped_summary.json"


def _extract_detunings(config_path: Path) -> tuple[float, float]:
    """Return (trap_detuning, repump_detuning) for one run, in gamma."""
    config = json.loads(config_path.read_text())
    trap = {
        L["detuning"] for L in config["Lasers"] if L.get("type") == "trap"
    }
    repump = {
        L["detuning"] for L in config["Lasers"] if L.get("type") == "repump"
    }
    if len(trap) != 1 or len(repump) != 1:
        raise ValueError(
            f"{config_path}: expected one detuning per beam type, got "
            f"trap={trap}, repump={repump}"
        )
    return next(iter(trap)), next(iter(repump))


def _count_atoms(
    csv_path: Path,
    step: int | None,
    max_T_transverse: float,
    max_radius: float,
) -> tuple[int, int]:
    """Return (n_trapped_at_step, n_total_at_step_0)."""
    df = pd.read_csv(
        csv_path,
        usecols=[
            "step",
            "atom_id",
            "position_x",
            "position_y",
            "velocity_x",
            "velocity_y",
        ],
    )
    if step is None:
        step = int(df["step"].max())
    n_total = int(df[df["step"] == 0]["atom_id"].nunique())
    snap = df[df["step"] == step]
    T_perp = (
        LI6_MASS
        * (snap["velocity_x"] ** 2 + snap["velocity_y"] ** 2)
        / (2 * scc.k)
    )
    r = np.hypot(snap["position_x"], snap["position_y"])
    n_trapped = int(((T_perp < max_T_transverse) & (r < max_radius)).sum())
    return n_trapped, n_total


def _build_summary(
    session_dir: Path,
    step: int | None,
    max_T_transverse: float,
    max_radius: float,
) -> pd.DataFrame:
    records = []
    for run in session_runs(session_dir):
        try:
            t, r = _extract_detunings(run / "config.json")
            n_trapped, n_total = _count_atoms(
                run / "result.csv", step, max_T_transverse, max_radius
            )
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"skip {run.name}: {e}")
            continue
        records.append(
            {
                "trap_d": t,
                "repump_d": r,
                "n_trapped": n_trapped,
                "n_total": n_total,
            }
        )

    return pd.DataFrame(records)


def _load_or_build_summary(
    session_dir: Path,
    step: int | None,
    max_T_transverse: float,
    max_radius: float,
    force_rebuild: bool,
) -> pd.DataFrame:
    """Reuse a cached summary if its parameters match, else rebuild."""
    cache_path = session_dir / SUMMARY_FILE
    params = {
        "step": step,
        "max_T_transverse": max_T_transverse,
        "max_radius": max_radius,
    }

    if cache_path.exists() and not force_rebuild:
        cached = json.loads(cache_path.read_text())
        rows = cached.get("rows", [])
        if cached.get("params") == params and rows and "n_total" in rows[0]:
            return pd.DataFrame(rows)

    df = _build_summary(session_dir, step, max_T_transverse, max_radius)
    cache_path.write_text(
        json.dumps(
            {"params": params, "rows": df.to_dict(orient="records")},
            indent=2,
        )
    )
    return df


@plot_type(
    "trapped_atoms_heatmap",
    input="session",
    description="Trapped-atom counts vs (trap, repump) detuning",
)
def plot_trapped_atoms_heatmap(
    session_dir: str | Path,
    step: int | None = None,
    max_T_transverse: float = 1.5e-3,  # K
    max_radius: float = 3e-3,  # m
    interpolation: str = "nearest",
    cmap: str = "viridis",
    levels: int | list[float] | None = None,
    ax: plt.Axes | None = None,
    force_rebuild: bool = False,
    normalize: bool = True,
    save_path: str | Path | None = None,
):
    """Plot trapped-atom counts vs (repump, trap) detuning in gamma.

    Trapped := T_perp = m*(vx^2+vy^2)/(2 k_B) < max_T_transverse
               AND sqrt(x^2+y^2) < max_radius, evaluated at `step`
               (default: each run's final step).
    Axes are inverted so more-negative detunings sit further from the
    origin.
    levels: None        -> continuous colormap.
            int N       -> N discrete color bands spanning the data range.
            list[float] -> explicit boundary values for the bands.
    Caches per-run counts in <session_dir>/_trapped_summary.json keyed
    on the above parameters; pass force_rebuild=True to ignore the cache.
    """
    session_dir = Path(session_dir)
    df = _load_or_build_summary(
        session_dir, step, max_T_transverse, max_radius, force_rebuild
    )

    df = df.assign(
        value=df["n_trapped"] / df["n_total"] if normalize
        else df["n_trapped"]
    )
    pivot = (
        df.groupby(["trap_d", "repump_d"])["value"]
        .mean()
        .unstack("repump_d")
        .sort_index()
        .sort_index(axis=1)
    )

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    x = pivot.columns.values  # repump detunings
    y = pivot.index.values  # trap detunings
    dx = (x[1] - x[0]) / 2 if len(x) > 1 else 0.5
    dy = (y[1] - y[0]) / 2 if len(y) > 1 else 0.5
    extent = (x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy)

    if levels is None:
        norm = None
        used_cmap = cmap
    else:
        vmin = float(np.nanmin(pivot.values))
        vmax = float(np.nanmax(pivot.values))
        if isinstance(levels, int):
            boundaries = np.linspace(vmin, vmax, levels + 1)
            n_colors = levels
        else:
            boundaries = np.asarray(levels, dtype=float)
            n_colors = len(boundaries) - 1
        norm = BoundaryNorm(boundaries, ncolors=n_colors)
        used_cmap = plt.get_cmap(cmap, n_colors)

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation=interpolation,
        cmap=used_cmap,
        norm=norm,
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"Repump detuning $\delta_\mathrm{repump}/\Gamma$")
    ax.set_ylabel(r"Trap detuning $\delta_\mathrm{trap}/\Gamma$")
    ax.set_title(
        rf"Trapped atoms  ($T_\perp$ < {max_T_transverse * 1e3:g} mK, "
        rf"r < {max_radius * 1e3:g} mm)"
    )
    fig.colorbar(
        im,
        ax=ax,
        label="fraction trapped" if normalize else "# trapped atoms",
    )

    if save_path is None and owns_fig:
        save_path = output_dir("heatmaps") / f"{session_dir.name}.png"
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax

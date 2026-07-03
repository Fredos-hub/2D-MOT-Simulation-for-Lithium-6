"""Generic atom-cloud plots from a single result.csv DataFrame."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as scc
from matplotlib.colors import LogNorm

from ..common import read_csv_data, select_alive_ids
from ..registry import plot_type


def _check_columns(df):
    required = [
        "step",
        "atom_id",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "subjective_time",
        "excitation_count",
        "current_groundstate",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def temperature_xy_over_time(
    data,
    alive_time_ms=2,
    atom_mass=scc.u * 6,
    match="nearest",
    time_scale=1.0,
    time_col="subjective_time",
):
    """
    Compute xy-plane temperature over time for the set of atoms alive at
    a reference time.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: atom_id, velocity_x, velocity_y,
        subjective_time
    alive_time_ms : float
        Reference time used to select the set of atom IDs that are
        considered "alive".
    atom_mass : float
        Mass of one atom in kg.
    match : {"nearest", "exact"}
        How to match alive_time_ms to an available time in the data.
    time_scale : float
        Multiply `subjective_time` by this factor before comparing to
        `alive_time_ms`. Use 1000 if subjective_time is in seconds but
        alive_time_ms is in milliseconds.
    time_col : str
        Name of the time column.

    Returns
    -------
    pd.DataFrame with columns:
        time_ms, temperature_K, temperature_stderr_K, n_atoms
    """
    df = data.copy()
    _check_columns(df)

    # Work in milliseconds (or whatever unit alive_time_ms is in)
    df["_time_ms"] = df[time_col].astype(float) * time_scale
    alive_ids, _ = select_alive_ids(
        df, alive_time_ms, match=match, time_col="_time_ms"
    )

    # Restrict all later calculations to those atom IDs
    df = df[df["atom_id"].isin(alive_ids)].copy()

    rows = []
    for t, g in df.groupby("_time_ms", sort=True):
        vx = g["velocity_x"].astype(float).to_numpy()
        vy = g["velocity_y"].astype(float).to_numpy()

        n = len(g)
        if n == 0:
            continue

        # Remove bulk drift so temperature reflects thermal spread
        dvx = vx - vx.mean()
        dvy = vy - vy.mean()

        # Per-particle xy kinetic energy -> temperature estimate
        # (1/2)m(<dvx^2> + <dvy^2>) = k_B T  in 2D
        temp_samples = (atom_mass * (dvx**2 + dvy**2)) / (2.0 * scc.k)
        temperature = temp_samples.mean()

        # Standard error across atoms at this time point
        if n > 1:
            stderr = temp_samples.std(ddof=1) / np.sqrt(n)
        else:
            stderr = np.nan

        rows.append(
            {
                "time_ms": t,
                "temperature_K": temperature,
                "temperature_stderr_K": stderr,
                "n_atoms": n,
            }
        )

    return pd.DataFrame(rows).sort_values("time_ms").reset_index(drop=True)


def plot_temperature_three_runs(
    file_paths,
    alive_time_ms,
    atom_mass,
    match="nearest",
    time_scale=1.0,
    labels=None,
    title="XY-plane Temperature vs Time",
):
    """
    Load three CSVs, compute temperature-time curves for atoms alive at
    `alive_time_ms`, and plot them with uncertainty bands.

    Parameters
    ----------
    file_paths : list[str]
        Exactly three CSV file paths.
    alive_time_ms : float
        Reference time used to define the alive atom set.
    atom_mass : float
        Atom mass in kg.
    match : {"nearest", "exact"}
        How to match the alive reference time.
    time_scale : float
        Scale factor applied to subjective_time before comparison.
    labels : list[str] | None
        Optional labels for the three runs.
    title : str
        Plot title.

    Returns
    -------
    list[pd.DataFrame]
        One temperature-time DataFrame per input file.
    """
    if len(file_paths) != 3:
        raise ValueError(
            "file_paths must contain exactly three CSV file paths."
        )

    if labels is None:
        labels = [f"Run {i + 1}" for i in range(3)]
    if len(labels) != 3:
        raise ValueError("labels must contain exactly three labels.")

    results = []
    fig, ax = plt.subplots(figsize=(9, 6))

    for path, label in zip(file_paths, labels, strict=False):
        df = read_csv_data(path)
        temp_df = temperature_xy_over_time(
            df,
            alive_time_ms=alive_time_ms,
            atom_mass=atom_mass,
            match=match,
            time_scale=time_scale,
        )
        results.append(temp_df)

        x = temp_df["time_ms"].to_numpy() * 1000
        y = temp_df["temperature_K"].to_numpy()
        yerr = temp_df["temperature_stderr_K"].to_numpy()

        ax.plot(x, y, linewidth=2, label=label)
        if np.any(np.isfinite(yerr)):
            ax.fill_between(
                x,
                y - np.nan_to_num(yerr, nan=0.0),
                y + np.nan_to_num(yerr, nan=0.0),
                alpha=0.18,
            )

    fontdict = {"fontsize": 16, "fontfamily": "serif"}

    ax.set_xlabel("Time (ms)", fontdict=fontdict)
    ax.set_ylabel("Temperature in xy plane (K)", fontdict=fontdict)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return results


@plot_type(
    "position_hist",
    input="dataframe",
    description="2D histogram of particle positions in the xy plane",
)
def position_hist(data, bins=500, ax=None, show=True):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 6))

    # draw histogram and capture the QuadMesh (last returned element)
    H, xedges, yedges, mesh = ax.hist2d(
        data["position_x"],
        data["position_y"],
        bins=bins,
        cmap="magma",
        norm=LogNorm(
            vmin=2
        ),  # use vmin >= cmin to avoid warnings; LogNorm needs >0
        cmin=1,  # hide bins with counts < 2
    )

    # put the hist "above" the grid
    mesh.set_zorder(3)
    ax.grid(True, zorder=0)  # grid behind the hist

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_xlim(-0.03, 0.03)
    ax.set_ylim(-0.03, 0.03)
    ax.set_title("2D Histogram of Particle Positions")

    if own:
        fig.colorbar(mesh, ax=ax, label="Number of Particles (log scale)")
        if show:
            plt.show()
    return mesh


@plot_type(
    "zeeman_hist",
    input="dataframe",
    description="2D histogram of y position vs y velocity (Zeeman view)",
)
def plot_zeeman_hist(data, bins=3000):
    fig, ax = plt.subplots(figsize=(8, 6))

    # draw histogram and capture the QuadMesh (last returned element)
    H, xedges, yedges, mesh = ax.hist2d(
        data["position_y"],
        data["velocity_y"],
        bins=bins,
        cmap="magma",
        cmin=1,  # hide bins with counts < 2
    )

    # put the hist "above" the grid
    mesh.set_zorder(3)
    ax.grid(True, zorder=0)  # grid behind the hist

    ax.set_xlabel("y position")
    ax.set_ylabel("y velocity")
    ax.set_title("2D Histogram of Particle Positions and Velocities")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-10, 510)
    fig.colorbar(mesh, ax=ax, label="Number of Particles")
    plt.show()


def plot_state_populations_three_runs(
    file_paths,
    alive_time,
    match="nearest",
    time_scale=1.0,
    labels=None,
    normalize=True,
    figsize=(10, 12),
):
    """
    Plot normalized populations per state vs time for three datasets.
    The cohort is fixed by selecting atom_ids present at `alive_time`.

    Parameters
    ----------
    file_paths : list[str]
        Exactly three CSV file paths.
    alive_time : float
        Reference time used to define the alive cohort.
    match : {'nearest', 'exact'}
        How to match alive_time to available times.
    time_scale : float
        Multiply subjective_time by this factor before comparing to
        alive_time. Use 1000.0 if subjective_time is in seconds and
        alive_time is in ms.
    labels : list[str] | None
        Optional labels for the three panels.
    normalize : bool
        If True, each state's population is divided by the cohort size
        at the reference time (fraction of the original alive cohort).
    figsize : tuple
        Figure size.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per run with columns:
        time, state, count, fraction, cohort_size
    """
    if len(file_paths) != 3:
        raise ValueError("file_paths must contain exactly three file paths.")

    if labels is None:
        labels = [f"Run {i + 1}" for i in range(3)]
    if len(labels) != 3:
        raise ValueError("labels must contain exactly three labels.")

    required_cols = {"atom_id", "subjective_time", "current_groundstate"}
    all_results = []

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for ax, path, panel_label in zip(axes, file_paths, labels, strict=False):
        df = read_csv_data(path).copy()

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        # Convert time to the same unit as alive_time
        df["_time"] = df["subjective_time"].astype(float) * time_scale
        alive_ids, _ = select_alive_ids(
            df, alive_time, match=match, time_col="_time"
        )

        # Restrict to the fixed cohort
        df = df[df["atom_id"].isin(alive_ids)].copy()

        # Cohort size at reference time (constant normalization factor)
        cohort_size = len(alive_ids)

        # Count atoms in each state at each time
        counts = (
            df.groupby(["_time", "current_groundstate"])["atom_id"]
            .nunique()
            .reset_index(name="count")
            .sort_values(["_time", "current_groundstate"])
        )

        # Normalize by the fixed cohort size
        counts["fraction"] = counts["count"] / float(cohort_size)
        counts["cohort_size"] = cohort_size

        all_results.append(
            counts.rename(
                columns={"_time": "time", "current_groundstate": "state"}
            )
        )

        # Plot one line per state
        states = list(pd.unique(counts["current_groundstate"]))
        states = sorted(states, key=lambda x: str(x))

        for state in states:
            sub = counts[counts["current_groundstate"] == state]
            y = sub["fraction"].values if normalize else sub["count"].values
            ax.plot(
                sub["_time"].values, y, linewidth=2, label=f"State {state}"
            )

        ax.set_title(panel_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)

        ax.set_ylabel(
            "Population fraction",
            fontsize=16,
            fontfamily="serif",
        )

    axes[-1].set_xlabel(
        "Time",
        fontsize=16,
        fontfamily="serif",
    )

    plt.tight_layout()
    plt.show()

    return all_results


@plot_type(
    "mean_velocity",
    input="dataframe",
    description="Mean y velocity of all particles over time",
)
def plot_mean_velocity_over_time(data):
    # mean velocity across all particles for each time step
    mean_velocity = data.groupby("subjective_time")["velocity_y"].mean()

    plt.figure(figsize=(8, 6))
    plt.plot(data["subjective_time"].unique(), mean_velocity.values)
    plt.xlabel("Time")
    plt.ylabel("Mean Velocity (y-direction)")
    plt.title("Mean Velocity of Particles Over Time")
    plt.grid()
    plt.show()


@plot_type(
    "mean_velocity_clean",
    input="dataframe",
    description="Mean y velocity of an alive cohort, with alive fraction",
)
def plot_mean_velocity_over_time_clean(
    data, alive_time=0.002, match="nearest", plot_alive_fraction=True
):
    """
    Plot mean velocity over time (y-direction). Optionally restrict the
    analysis to atoms present at `alive_time`, and optionally plot the
    (normalized) fraction of those atoms still present over time on a
    second y-axis.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns 'subjective_time', 'velocity_y', 'atom_id'.
    alive_time : float or None
        If provided, select the set of atom_id values present at this
        time and restrict all computations to those atoms. If match ==
        'nearest', the closest available time is used.
    match : 'exact' or 'nearest'
        How to interpret alive_time if provided.
    plot_alive_fraction : bool
        If True, plot normalized alive fraction on a second y-axis,
        normalized to the atom count at the first time entry.

    Returns
    -------
    dict with keys:
      'mean_velocity' : pandas.Series (indexed by subjective_time)
      'alive_counts'  : pandas.Series or None (indexed by subjective_time)
      'alive_fraction': pandas.Series or None (indexed by subjective_time)
    """
    for col in ("subjective_time", "velocity_y", "atom_id"):
        if col not in data.columns:
            raise ValueError(f"Input DataFrame must contain column '{col}'")

    df = data.copy()

    if alive_time is not None:
        alive_ids, _ = select_alive_ids(df, alive_time, match=match)
        df = df[df["atom_id"].isin(alive_ids)]

    # mean velocity over time for the (possibly filtered) set of atoms
    mean_velocity = (
        df.groupby("subjective_time")["velocity_y"].mean().sort_index()
    )

    alive_counts = None
    alive_fraction = None
    if plot_alive_fraction:
        # count unique atom ids at each time (for this filtered set)
        alive_counts = (
            df.groupby("subjective_time")["atom_id"]
            .nunique()
            .reindex(mean_velocity.index)
            .fillna(0)
            .astype(int)
        )
        # normalize to the count at the first time entry
        if len(alive_counts) == 0:
            denom = 1
        else:
            denom = alive_counts.iloc[0] if alive_counts.iloc[0] > 0 else 1
        alive_fraction = alive_counts / float(denom)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        mean_velocity.index.values,
        mean_velocity.values,
        label="Mean velocity (y)",
        linewidth=2,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Velocity (y-direction)")
    ax.set_title("Mean Velocity of Particles Over Time")
    ax.grid()
    if plot_alive_fraction and alive_fraction is not None:
        ax2 = ax.twinx()
        ax2.plot(
            alive_fraction.index.values,
            alive_fraction.values,
            linestyle="--",
            label="Alive fraction (norm)",
            linewidth=1.5,
        )
        ax2.set_ylabel("Alive fraction (normalized to first time entry)")
        # combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

    return {
        "mean_velocity": mean_velocity,
        "alive_counts": alive_counts,
        "alive_fraction": alive_fraction,
    }

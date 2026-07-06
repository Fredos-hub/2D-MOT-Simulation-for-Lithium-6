"""Physics comparison: diagonalizer vs interpolation interaction models.

Two independent comparisons on the same Li-6 D2 system:

1. Microphysics (no MOT): sweep |B| and compare per-transition line shifts and
   branching ratios across the live diagonalizer, the table diagonalizer, the
   full 18-level fit, and the simple 18-level fit. This is where the known
   interpolation defects surface (silent-zero for |B| > 0.1 T; a GS2/GS3 swap).

2. Dynamics: rerun the identical Hammel_Cuboid MOT scenario (same atom count,
   seed, geometry, step count) for the three fast models and compare cooling
   curve, survival, and total scattering. The LIVE model's dynamics point is
   parsed from its standalone run log (avoids a redundant ~1 h re-run).

Outputs report.md + PNG figures + results.npz to util/diagonalizer_comparison/.
Offline analysis tool — never imported by src/ or the kernel.
"""
from __future__ import annotations

import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import src.diagonalizer_setup as ds  # noqa: E402
import src.interactions as interactions  # noqa: E402
from src.parameters import Parameters  # noqa: E402

OUT_DIR = "util/diagonalizer_comparison"
TABLE_NPZ = "interaction_tables/li6_d2_compare_table.npz"
SETUP = "setup parameters/Hammel_Cuboid_Setup.json"
N_ATOMS = 500
LIVE_LOG = (
    "/tmp/claude-1000/-home-fredo-Schreibtisch-Test-2D-MOT-Simulation-"
    "for-Lithium-6/7813d3c8-ce5f-469d-b5c8-40b5e846a422/tasks/"
    "bxwts8s8x.output"
)
LOG_STEPS = [1] + list(range(25, 301, 25))
POL = {0: "sigma-", 1: "pi", 2: "sigma+"}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Microphysics
# --------------------------------------------------------------------------- #
def build_models():
    """Return {label: instance} for the four models (table built from NPZ)."""
    models = {}
    models["live"] = interactions.Lithium6DiagonalizerInteraction()
    models["18level"] = interactions.Lithium18LevelInteraction()
    models["simple18"] = interactions.SimpleEighteenLevelInteraction()
    with np.load(TABLE_NPZ) as d:
        b_axis = np.ascontiguousarray(d["b_axis"], dtype=np.float64)
        pos = np.ascontiguousarray(d["pos_table"], dtype=np.float64)
        strg = np.ascontiguousarray(d["strength_table"], dtype=np.float64)
    models["table"] = interactions.Lithium6DiagonalizerTableInteraction(
        b_axis, pos, strg
    )
    return models


def sweep_transition(models, pol, gs, es, b_vals):
    """Return {label: (shift_MHz[], branch[])} over b_vals for one line."""
    out = {}
    for label, m in models.items():
        shift = np.full(b_vals.size, np.nan)
        branch = np.full(b_vals.size, np.nan)
        for i, B in enumerate(b_vals):
            try:
                shift[i] = m.calculate_transition_frequency_shift(
                    pol, gs, es, float(B)
                ) / 1e6
                branch[i] = m.calculate_branching_ratio(
                    pol, gs, es, float(B)
                )
            except Exception:
                pass
        out[label] = (shift, branch)
    return out


def microphysics(models):
    """Sweep |B|, write figures, return a divergence summary vs the live model."""
    b_vals = np.linspace(0.0, 0.3, 301)
    summary = []

    # cycling transition GS5 -> ES11, sigma+
    lines = [(2, 5, 11), (2, 2, 6), (2, 3, 6)]
    colors = {"live": "C0", "table": "C1", "18level": "C2", "simple18": "C3"}

    for (pol, gs, es) in lines:
        data = sweep_transition(models, pol, gs, es, b_vals)
        tag = f"GS{gs}-ES{es}-{POL[pol]}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for label, (shift, branch) in data.items():
            ax1.plot(b_vals, shift, color=colors[label], label=label)
            ax2.plot(b_vals, branch, color=colors[label], label=label)
        ax1.set_title(f"Line shift — {tag}")
        ax1.set_xlabel("|B| (T)")
        ax1.set_ylabel("shift (MHz)")
        ax1.axvline(0.1, color="k", ls=":", alpha=0.4)
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8)
        ax2.set_title(f"Branching ratio — {tag}")
        ax2.set_xlabel("|B| (T)")
        ax2.set_ylabel("branching |CG|^2")
        ax2.axvline(0.1, color="k", ls=":", alpha=0.4,
                    label="|B|=0.1 T (interp cutoff)")
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/microphysics_{tag}.png", dpi=110)
        plt.close(fig)

        # divergence vs live in two bands
        live_shift, live_branch = data["live"]
        for label in ("table", "18level", "simple18"):
            s, b = data[label]
            for lo, hi, band in ((0.0, 0.1, "|B|<=0.1T"),
                                 (0.1, 0.3, "|B|>0.1T")):
                mask = (b_vals >= lo) & (b_vals <= hi)
                ds_max = np.nanmax(np.abs(s[mask] - live_shift[mask]))
                db_max = np.nanmax(np.abs(b[mask] - live_branch[mask]))
                summary.append({
                    "line": tag, "model": label, "band": band,
                    "max_shift_diff_MHz": float(ds_max),
                    "max_branch_diff": float(db_max),
                })
    return summary


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
def mean_speed_alive(states):
    st = np.asarray(states.status)
    v = np.asarray(states.velocities)
    alive = st == 1
    if alive.sum() == 0:
        return float("nan")
    return float(np.linalg.norm(v[alive], axis=1).mean())


def run_dynamics(model_name, table_file=None):
    """Run the identical scenario for one model; return an observables dict."""
    with open(SETUP) as f:
        cfg = json.load(f)
    cfg["Simulation"]["interaction"] = model_name
    if table_file is not None:
        cfg["Simulation"]["interaction_table_file"] = table_file
    cfg["Atoms"]["number"] = N_ATOMS
    path = f"/tmp/claude-1000/cmp_{model_name}.json"
    os.makedirs("/tmp/claude-1000", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f)

    p = Parameters(path)
    sim = p.build_simulation()
    steps, speeds, alive_ct = [], [], []
    t0 = time.perf_counter()
    for i in range(sim.current_step, sim.max_step_number):
        cont, states, exc, alive_ids, _ = sim.step(i)
        if (i + 1) in LOG_STEPS:
            st = np.asarray(states.status)
            steps.append(i + 1)
            speeds.append(mean_speed_alive(states))
            alive_ct.append(int((st == 1).sum()))
            log(f"  [{model_name}] step {i+1}/{sim.max_step_number} "
                f"alive={int((st==1).sum())} "
                f"mean|v|={speeds[-1]:.2f} elapsed={time.perf_counter()-t0:.0f}s")
        if not cont:
            break
    st = np.asarray(sim.simulation_atoms.status)
    exc_arr = np.asarray(sim.excitation_counter)
    return {
        "model": model_name,
        "steps": np.array(steps),
        "mean_speed": np.array(speeds),
        "alive": np.array(alive_ct),
        "final_alive": int((st == 1).sum()),
        "final_dead": int((st == 0).sum()),
        "total_scatter": int(exc_arr.sum()),
        "substeps": int(getattr(sim, "total_substeps", -1)),
        "wall_s": time.perf_counter() - t0,
        "min_mean_speed": float(np.nanmin(speeds)) if speeds else float("nan"),
    }


def parse_live_log(path):
    """Extract the live model's dynamics observables from its run log."""
    steps, speeds, alive = [], [], []
    scatter = substeps = None
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    for m in re.finditer(
        r"step\s+(\d+)/\d+\s+alive=\s*(\d+).*?mean\|v\|_alive=\s*"
        r"([\d.]+|nan)", text
    ):
        steps.append(int(m.group(1)))
        alive.append(int(m.group(2)))
        speeds.append(float(m.group(3)) if m.group(3) != "nan" else np.nan)
    ms = re.search(r"total excitation events:\s*(\d+)", text)
    ss = re.search(r"total_substeps.*?:\s*(\d+)|substeps.*?:\s*(\d+)", text)
    if ms:
        scatter = int(ms.group(1))
    if ss:
        substeps = int(ss.group(1) or ss.group(2))
    valid = [s for s in speeds if not np.isnan(s)]
    return {
        "model": "live (Lithium6Diagonalizer)",
        "steps": np.array(steps),
        "mean_speed": np.array(speeds),
        "alive": np.array(alive),
        "final_alive": alive[-1] if alive else -1,
        "final_dead": (N_ATOMS - alive[-1]) if alive else -1,
        "total_scatter": scatter if scatter is not None else -1,
        "substeps": substeps if substeps is not None else -1,
        "wall_s": float("nan"),
        "min_mean_speed": min(valid) if valid else float("nan"),
    }


def dynamics_figures(runs):
    colors = {"live (Lithium6Diagonalizer)": "C0", "table": "C1",
              "Lithium18LevelInteraction": "C2",
              "SimpleEighteenLevelInteraction": "C3",
              "Lithium6DiagonalizerTableInteraction": "C1"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in runs:
        c = colors.get(r["model"], None)
        ax1.plot(r["steps"], r["mean_speed"], marker="o", ms=3,
                 color=c, label=r["model"])
        ax2.plot(r["steps"], r["alive"], marker="o", ms=3,
                 color=c, label=r["model"])
    ax1.set_title("Cooling: mean |v| of surviving atoms")
    ax1.set_xlabel("step")
    ax1.set_ylabel("mean |v| (m/s)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    ax2.set_title("Survival: alive atom count")
    ax2.set_xlabel("step")
    ax2.set_ylabel("alive")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dynamics_cooling_survival.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def write_report(micro, runs):
    lines = ["# Diagonalizer vs interpolation — physics comparison\n"]
    lines.append(
        "Same Li-6 D2 system. Microphysics is a direct |B|-sweep of each "
        "model's transition shift/branching; dynamics reruns the identical "
        f"Hammel_Cuboid MOT scenario ({N_ATOMS} atoms, 300 steps, same seed).\n"
    )

    lines.append("## 1. Microphysics divergence vs live diagonalizer\n")
    lines.append("| line | model | band | max |Δshift| (MHz) | max |Δbranch| |")
    lines.append("|---|---|---|---:|---:|")
    for row in micro:
        lines.append(
            f"| {row['line']} | {row['model']} | {row['band']} | "
            f"{row['max_shift_diff_MHz']:.4g} | {row['max_branch_diff']:.4g} |"
        )
    lines.append(
        "\nFigures: `microphysics_*.png`. The dotted line marks |B| = 0.1 T, "
        "the documented interpolation cutoff (silent-zero above it).\n"
    )

    lines.append("## 2. Dynamics (identical scenario)\n")
    lines.append("| model | final alive/dead | min mean|v| (m/s) | "
                 "total scatter | substeps | wall (s) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in runs:
        lines.append(
            f"| {r['model']} | {r['final_alive']}/{r['final_dead']} | "
            f"{r['min_mean_speed']:.3g} | {r['total_scatter']} | "
            f"{r['substeps']} | "
            f"{'n/a' if np.isnan(r['wall_s']) else f'{r['wall_s']:.0f}'} |"
        )
    lines.append("\nFigure: `dynamics_cooling_survival.png`.\n")
    lines.append(
        "Caveats: dynamics uses the setup's rate-mode injection with a fixed "
        "seed, so initial conditions match but per-atom RNG streams diverge "
        "once scattering differs between models (expected). The live point is "
        "parsed from its standalone run log; the fast models are rerun here.\n"
    )
    with open(f"{OUT_DIR}/report.md", "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("== building comparison table NPZ ==")
    ds.generate_table(ds.li6_d2_constants(), 0.0, 1.0, 256, TABLE_NPZ)

    log("== microphysics sweep ==")
    models = build_models()
    micro = microphysics(models)

    log("== dynamics runs (fast models) ==")
    runs = []
    live = parse_live_log(LIVE_LOG)
    if live is not None:
        runs.append(live)
    runs.append(run_dynamics("Lithium18LevelInteraction"))
    runs.append(run_dynamics("SimpleEighteenLevelInteraction"))
    runs.append(run_dynamics("Lithium6DiagonalizerTableInteraction",
                             table_file=TABLE_NPZ))

    dynamics_figures(runs)
    write_report(micro, runs)

    # machine-readable dump
    np.savez(
        f"{OUT_DIR}/results.npz",
        micro=json.dumps(micro),
        dynamics=json.dumps([
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in r.items()} for r in runs
        ]),
    )
    log("== DONE ==")
    log(f"report: {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()

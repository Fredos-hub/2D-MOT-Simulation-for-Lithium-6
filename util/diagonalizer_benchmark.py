"""Compute-cost benchmark: live vs table vs 18-level interaction (plan 04-06).

Runs the three interaction models on the same Hammel-derived scenario and
reports, per model, the steady-state wall time, the total number of inner
substeps executed, and the derived per-substep cost. One-time costs (import +
sympy setup build, and the Numba JIT warm-up) are measured and reported
SEPARATELY from the steady-state timed region so the steady-state per-substep
numbers are not polluted by first-run compilation.

Reporting only: this is the raw data Phase 5 uses for the keep/retire and
live-vs-table decision (D-03). There is no hard bar and no pass/fail
performance assertion here (D-09).

Run:
    python -m util.diagonalizer_benchmark
    python -m util.diagonalizer_benchmark --atoms 200 --steps 20 --json out.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

# One-time setup cost: importing src.interactions transitively runs the sympy
# get_li6_setup() build (04-03 note). Time the heavy imports once here so the
# steady-state numbers below exclude this fixed startup cost.
_IMPORT_T0 = time.perf_counter()
import numpy as np  # noqa: E402

import src.diagonalizer_setup as ds  # noqa: E402
from src.parameters import Parameters  # noqa: E402

IMPORT_SETUP_S = time.perf_counter() - _IMPORT_T0

LIVE_NAME = "Lithium6DiagonalizerInteraction"
TABLE_NAME = "Lithium6DiagonalizerTableInteraction"
REF_NAME = "Lithium18LevelInteraction"

_HAMMEL = Path("setup parameters/Hammel_Setup.json")


def _write_scenario(tmp_dir, interaction, n_atoms, table_path=None):
    """Derive a fast, deterministic MOT scenario from the Hammel setup.

    Keeps the default field/laser geometry; overrides only the interaction,
    atom count, run length and start conditions so every model runs the same
    controlled workload. Atoms start at the trap center at low speed so
    scattering is guaranteed for any working model.
    """
    cfg = json.loads(_HAMMEL.read_text())
    cfg["Simulation"]["interaction"] = interaction
    cfg["Simulation"]["simulated_time"] = 5.0  # ms; capped by step-loop below
    cfg["Simulation"]["rate_mode"] = False
    if table_path is not None:
        cfg["Simulation"]["interaction_table_file"] = table_path
    cfg["Atoms"]["number"] = int(n_atoms)
    cfg["Atoms"]["start_position"] = [0.0, 0.0, 0.0]
    cfg["Atoms"]["start_velocity"] = [0.0, 1.0, 0.0]
    cfg["Atoms"].pop("sample_file", None)  # uniform defaults, no CSV dependency
    out = Path(tmp_dir) / f"{interaction}_bench.json"
    out.write_text(json.dumps(cfg))
    return str(out)


def benchmark_model(interaction, n_atoms, n_steps, tmp_dir, table_path=None):
    """Benchmark one interaction; return a result dict.

    The warm-up (JIT compilation) is timed and reported separately and is
    strictly OUTSIDE the steady-state timed region.
    """
    cfg = _write_scenario(tmp_dir, interaction, n_atoms, table_path)
    params = Parameters(cfg)
    if not params.valid:
        raise RuntimeError(f"{interaction}: invalid config: {params.errors}")

    build_t0 = time.perf_counter()
    sim = params.build_simulation()
    build_s = time.perf_counter() - build_t0

    # One-time JIT compilation — measured, then excluded from the timed region.
    warm_t0 = time.perf_counter()
    sim.warmup()
    jit_warmup_s = time.perf_counter() - warm_t0

    # Reset the substep counter so warm-up substeps do not count.
    sim.substep_counter[:] = 0

    # Steady-state timed region: fixed workload of n_steps.
    run_t0 = time.perf_counter()
    for i in range(n_steps):
        cont = sim.step(i)[0]
        if not cont:
            break
    steady_s = time.perf_counter() - run_t0

    total_substeps = sim.total_substeps
    scatter = int(sim.excitation_counter.sum())
    us_per_substep = (
        (steady_s / total_substeps) * 1e6 if total_substeps > 0 else float("nan")
    )
    return {
        "model": interaction,
        "build_s": build_s,
        "jit_warmup_s": jit_warmup_s,
        "steady_s": steady_s,
        "total_substeps": total_substeps,
        "us_per_substep": us_per_substep,
        "scatter": scatter,
    }


def run_benchmark(n_atoms=200, n_steps=20, n_nodes=200, out_dir=None):
    """Benchmark live, table and 18-level models on the same scenario.

    Generates the diagonalizer table the table model needs, then runs each
    model and returns a list of per-model result dicts. Pure reporting; no
    performance gate is imposed (D-09).
    """
    tmp = tempfile.mkdtemp(prefix="diag_bench_")
    out_dir = out_dir or tmp
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Generate the |B|-table the table model interpolates.
    table_path = str(Path(out_dir) / "li6_d2_bench_table.npz")
    setup_t0 = time.perf_counter()
    ds.generate_table(ds.li6_d2_constants(), 0.0, 1.0, n_nodes, table_path)
    table_gen_s = time.perf_counter() - setup_t0

    results = []
    for name, tp in (
        (LIVE_NAME, None),
        (TABLE_NAME, table_path),
        (REF_NAME, None),
    ):
        res = benchmark_model(name, n_atoms, n_steps, tmp, tp)
        res["table_gen_s"] = table_gen_s if name == TABLE_NAME else 0.0
        results.append(res)
    return results


def format_report(results):
    """Render the results as a fixed-width text table plus a one-time-cost note."""
    lines = []
    lines.append(
        f"One-time setup (import + sympy build): {IMPORT_SETUP_S:8.3f} s "
        "(shared across all models, per process)"
    )
    lines.append("")
    header = (
        f"{'model':<38}{'steady_s':>10}{'substeps':>12}"
        f"{'us/substep':>13}{'jit_warmup_s':>14}{'scatter':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r['model']:<38}{r['steady_s']:>10.4f}"
            f"{r['total_substeps']:>12d}{r['us_per_substep']:>13.4f}"
            f"{r['jit_warmup_s']:>14.3f}{r['scatter']:>10d}"
        )
    return "\n".join(lines)


def main(argv=None):
    """CLI entry point: run the benchmark and print (optionally dump) results."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atoms", type=int, default=200, help="atoms per model")
    ap.add_argument("--steps", type=int, default=20, help="timed sim steps")
    ap.add_argument("--nodes", type=int, default=200, help="table |B| nodes")
    ap.add_argument("--json", type=str, default=None, help="optional JSON out")
    args = ap.parse_args(argv)

    results = run_benchmark(args.atoms, args.steps, args.nodes)
    print(format_report(results))

    if args.json:
        payload: dict[str, Any] = {
            "import_setup_s": IMPORT_SETUP_S,
            "atoms": args.atoms,
            "steps": args.steps,
            "nodes": args.nodes,
            "results": results,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")
    return results


if __name__ == "__main__":
    main()

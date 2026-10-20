"""Benchmark: Numba prange chunk size for the physics kernel (load balance between threads).

Builds the simulation like BatchSimulationWorker but writes no CSV, sets numba.set_parallel_chunksize in this
(calling) thread, and times the first N steps. Per step: wall time, alive atoms, total excitations.
The kernel RNG is unseeded (per-process entropy), so even two chunk-0 runs differ statistically: judge the
physics against the spread of the repeated chunk-0 runs, not bitwise.
Usage: python -m util.bench_chunksize CONFIG.json --steps 500 --chunk 0 --tag base [--weight-factor 4]
--weight-factor multiplies macro_particle_weight (1/x simulated atoms; a temporary config copy, the original stays as is).
Writes analysis_output/benchmarks/chunksize_<tag>_c<chunk>_<n>.csv
"""
import argparse, json, os, tempfile, time
import numba, numpy as np, pandas as pd
from src.parameters import Parameters

OUT = "/home/fredo/Schreibtisch/Test/2D-MOT-Simulation-for-Lithium-6/analysis_output/benchmarks/"   # mot-src copy: main repo output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--tag", default="bench")
    ap.add_argument("--weight-factor", type=float, default=1.0)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg["Simulation"]["macro_particle_weight"] *= a.weight_factor
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
    sim = Parameters(fh.name).build_simulation()
    os.unlink(fh.name)
    sim.warmup()
    numba.set_parallel_chunksize(a.chunk)
    rows = []
    for i in range(a.steps):
        t0 = time.perf_counter()
        cont, atoms, exc, alive, _ = sim.step(i)
        rows.append((i, time.perf_counter() - t0, alive.size, int(np.sum(exc)), sim.total_substeps))
        if not cont:
            break
    df = pd.DataFrame(rows, columns=["step", "t_step_s", "n_alive", "exc_total", "substeps_total"])
    os.makedirs(OUT, exist_ok=True)
    n = sum(f.startswith(f"chunksize_{a.tag}_c{a.chunk}_") for f in os.listdir(OUT))
    df.to_csv(OUT + f"chunksize_{a.tag}_c{a.chunk}_{n}.csv", index=False)
    print(f"chunk {a.chunk}: {len(df)} steps, {df.t_step_s.sum():.1f} s, last step {df.t_step_s.iloc[-1]:.2f} s, "
          f"alive {df.n_alive.iloc[-1]}, excitations {df.exc_total.iloc[-1]}, substeps {df.substeps_total.iloc[-1]}, "
          f"step_cap {sim.step_cap:g} T, step_max {sim.step_max:g} m")


if __name__ == "__main__":
    main()

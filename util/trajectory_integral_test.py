"""Deterministic sub-step check: integral of the scattering rate along fixed straight paths, no photon events.

Test atoms get a pending optical depth tau0 = 1e6, so they never scatter; the kernel drains tau by rate*dt each
sub-step, hence tau0 - tau_end = integral of rate dt exactly as the kernel computes it (polarisation weights incl.).
Paths x 6 ground states through the real kernel per sub-step rule; reference = uniform 1 um steps, checked against 2 um.
Usage: python -m util.trajectory_integral_test CONFIG.json OUT.csv [--trapezoid]
"""
import sys
import numpy as np, pandas as pd
from src.parameters import Parameters
from src.absorption_and_emission_process import absorption_and_emission_default_timestep as kernel

TAU0, DT = 1e6, 1e-5
RULES = {"old": (0.0, 0.0)}
RULES |= {f"{c:g}G_{m * 1e6:g}um": (c * 1e-4, m) for c in (0.8, 0.4, 0.2, 0.1, 0.05) for m in (1e-4, 5e-5, 2.5e-5)}
RULES |= {"uniform_2um": (1e9, 2e-6), "uniform_1um": (1e9, 1e-6)}     # huge cap -> L = step_max everywhere
MIN_PHOTONS = 0.1                                                      # ignore (path, state) pairs that stay dark


def paths(c):
    """name: (start, velocity, duration) around the push-beam axis c = (x, y)."""
    x, y = c
    return {"P1_oven_y100": ((x, y - 12e-3, 0.0), (0.0, 100.0, 0.0), 240e-6),
            "P2_cross_push_x40": ((x - 2e-3, y, 1e-3), (40.0, 0.0, 0.0), 100e-6),
            "P3_along_z80": ((x, y, 0.0), (0.0, 0.0, 80.0), 1e-3),
            "P4_slow_near_zero": ((x - 0.5e-3, y + 50e-6, 0.0), (1.0, 0.0, 0.0), 1e-3),
            "P5_slow_offcentre": ((x, y + 1e-3, 0.0), (0.5, 0.5, 0.0), 1e-3)}


def integrals(sim, P, cap, smax, trap):
    """(path, ground state, integral of rate dt, sub-steps) for one rule; one atom at a time."""
    a = sim.simulation_atoms
    out = []
    for name, (p0, v, T) in P.items():
        for g in range(sim.simulation_interaction.number_of_ground_states):
            a.positions[0], a.velocities[0], a.groundstates[0] = p0, v, g
            a.status[0], a.pending_optical_depth[0], a.max_step_lengths[0] = 1, TAU0, smax
            sim.trap_dt[0] = 0.0
            exc0, sub0 = sim.excitation_counter[0], sim.total_substeps
            for _ in range(round(T / DT)):
                kernel(np.array([0], dtype=np.int64), a, sim.lasers, sim.simulation_interaction, sim.magnetic_field,
                       sim.excitation_counter, DT, sim.excitation_hist, sim.substep_counter, cap,
                       smax if smax > 0 else 1e-4, trap, sim.trap_rate, sim.trap_dt)
            assert sim.excitation_counter[0] == exc0, f"test atom scattered ({name}, gs {g})"
            out.append((name, g, TAU0 - a.pending_optical_depth[0], sim.total_substeps - sub0))
    return out


def main(config, out_csv, trap):
    sim = Parameters(config).build_simulation()
    probe = [i for i in range(sim.lasers.n_lasers) if abs(sim.lasers.normalized_directions[i, 2]) > 0.5][0]
    P = paths(sim.lasers.origins[probe, :2])
    df = pd.concat([pd.DataFrame(integrals(sim, P, cap, smax, trap), columns=["path", "gs", "integral", "substeps"])
                    .assign(rule=rule) for rule, (cap, smax) in RULES.items()])
    ref = df[df.rule == "uniform_1um"].set_index(["path", "gs"]).integral
    df["ref"] = df.set_index(["path", "gs"]).index.map(ref)
    df["rel_err"] = np.where(df.ref > MIN_PHOTONS, df.integral / df.ref - 1, np.nan)
    df.insert(0, "config", config)
    df.to_csv(out_csv, index=False, float_format="%.6g")
    s = df.groupby(["rule", "path"]).agg(max_abs_rel_err=("rel_err", lambda e: np.nanmax(np.abs(e)) if e.notna().any()
                                                          else np.nan), substeps=("substeps", "sum"))
    pd.set_option("display.width", 200)
    print(config, "trapezoid" if trap else "rectangle (rate at sub-step start)")
    print(s.unstack("path")["max_abs_rel_err"].reindex(list(RULES)).to_string(float_format=lambda v: f"{v:.2e}"))
    print(s.unstack("path")["substeps"].reindex(list(RULES)).to_string())
    print("photons along each path (reference, max over states):")
    print(df[df.rule == "uniform_1um"].groupby("path").ref.max().to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--trapezoid" in sys.argv)

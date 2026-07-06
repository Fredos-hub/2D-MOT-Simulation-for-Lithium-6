"""Regression tests for the diagonalizer prototype (phase 04).

This module pins the udem_code reference outputs (line positions + relative
intensities for the Li-6 D2 line) as the D-08 PRIMARY ground truth and validates
the ``src.diagonalizer_setup`` physics port against them.

Waves:
  * ``test_udem_fixtures_present`` — the committed reference (plan 04-01).
  * matrix / positions / strengths / low-B / label tests — the physics port
    (plan 04-02): H_hfs + B*H_Zeeman factorization, ``Linien`` strengths, the
    diabatic order map, and the D-07 label artifact.
  * selectability / kernel-smoke / table tests — later plans (04-03, 04-04).

The udem reference uses unsorted ``LA.eig`` output while the port uses sorted
``eigh`` + diabatic labels, so line positions and strengths are compared as
order-invariant multisets (the set of physical lines/strengths is what D-08
validates). Regenerate the fixture with:
    python tests/fixtures/generate_udem_reference.py
"""
from pathlib import Path

import numpy as np
import pytest

import src.diagonalizer_setup as ds

FIXTURE = Path(__file__).parent / "fixtures" / "udem_reference.npz"


@pytest.fixture(scope="module")
def udem_reference():
    """Load the committed udem reference NPZ (pos, intensity, b_values)."""
    with np.load(FIXTURE) as data:
        return {
            "pos": data["pos"],
            "intensity": data["intensity"],
            "b_values": data["b_values"],
        }


@pytest.fixture(scope="module")
def li6_matrices():
    """Constant matrices + coupling tensor for the Li-6 D2 line."""
    hg, zg, he, ze, coupling = ds.li6_constant_matrices()
    return {"hg": hg, "zg": zg, "he": he, "ze": ze, "coupling": coupling}


def test_udem_fixtures_present(udem_reference):
    """The reference arrays exist with the expected shapes and are all finite."""
    pos = udem_reference["pos"]
    intensity = udem_reference["intensity"]
    b_values = udem_reference["b_values"]

    n_b = b_values.shape[0]
    assert pos.shape == (n_b, 6, 12), f"pos shape {pos.shape}"
    assert intensity.shape == (n_b, 3, 6, 12), f"intensity shape {intensity.shape}"
    assert np.isfinite(pos).all(), "pos contains non-finite values"
    assert np.isfinite(intensity).all(), "intensity contains non-finite values"


# --- Plan 04-02: physics port (SetMatrix / Linien) --------------------------


def test_matrix_shapes_symmetric(li6_matrices):
    """H_hfs / H_Zeeman are real-symmetric with the (6,6) / (12,12) shapes."""
    m = li6_matrices
    assert m["hg"].shape == (6, 6) and m["zg"].shape == (6, 6)
    assert m["he"].shape == (12, 12) and m["ze"].shape == (12, 12)
    for mat in (m["hg"], m["zg"], m["he"], m["ze"]):
        assert np.allclose(mat, mat.T), "matrix is not symmetric"


def test_matrix_factorization_reproduces_setmatrix(li6_matrices):
    """H_hfs + B*H_Zeeman equals the un-factored SetMatrix at several B."""
    I = ds.LI6_D2["I"]
    g, e = ds.LI6_D2["ground"], ds.LI6_D2["excited"]
    for B in (0.0, 0.05, 1.0):
        full_g = ds.set_matrix(I, g["J"], g["A"], g["B_hfs"], B, g["g_J"])
        full_e = ds.set_matrix(I, e["J"], e["A"], e["B_hfs"], B, e["g_J"])
        assert np.allclose(li6_matrices["hg"] + B * li6_matrices["zg"], full_g)
        assert np.allclose(li6_matrices["he"] + B * li6_matrices["ze"], full_e)


def test_coupling_tensor_shape(li6_matrices):
    """Coupling tensor has the (3, 12, 6) shape and non-trivial entries."""
    coupling = li6_matrices["coupling"]
    assert coupling.shape == (3, 12, 6)
    assert np.any(coupling != 0.0)


def test_positions_vs_udem(udem_reference, li6_matrices):
    """D-08 PRIMARY: line positions reproduce the udem reference at all B.

    Compared as an order-invariant multiset (the set of pairwise level
    differences is independent of the eigen-ordering).
    """
    m = li6_matrices
    for k, B in enumerate(udem_reference["b_values"]):
        eg = ds.solve_at_field(m["hg"], m["zg"], B)[0]
        ea = ds.solve_at_field(m["he"], m["ze"], B)[0]
        mine = np.sort(ds.line_positions(eg, ea).ravel())
        ref = np.sort(udem_reference["pos"][k].ravel())
        assert np.allclose(mine, ref, rtol=1e-6, atol=1.0), f"positions @B={B}"


def test_strengths_vs_udem(udem_reference, li6_matrices):
    """D-08 PRIMARY: transition strengths reproduce the udem reference.

    Per polarization the multiset of |Summe|^2 values is compared. At B=0 the
    manifolds are degenerate, so ``eig`` (udem) and ``eigh`` (port) pick
    different bases inside each subspace; there only the per-polarization total
    (a basis-invariant Frobenius norm) is checked.
    """
    m = li6_matrices
    for k, B in enumerate(udem_reference["b_values"]):
        _, vg = ds.solve_at_field(m["hg"], m["zg"], B)
        _, va = ds.solve_at_field(m["he"], m["ze"], B)
        mine = ds.transition_strengths(m["coupling"], vg, va)
        ref = udem_reference["intensity"][k]
        for q in range(3):
            if B == 0.0:
                assert np.isclose(mine[q].sum(), ref[q].sum(), atol=1e-1)
            else:
                assert np.allclose(
                    np.sort(mine[q].ravel()), np.sort(ref[q].ravel()), atol=1e-6
                ), f"strengths pol={q} @B={B}"


def test_cycling_strength_normalization():
    """Pitfall 3: the cycling transition GS5->ES11 (sigma+) is |CG|^2 = 0.25."""
    assert np.isclose(ds.li6_d2_strength(5, 11, 2, 1e-3), 0.25, atol=1e-6)


def test_vs_fits_lowB():
    """D-08 SECONDARY: low-B ground Zeeman slopes agree with the fits' g_F*mF.

    Checked only where the simple fits are trusted. The F=1/2 / F=3/2 mF=+/-0.5
    pair (GS2/GS3) mixes and crosses at higher field (the documented GS2/GS3
    defect region) so its slope is not asserted here; the stretched states are
    the cleanest check.
    """
    for gs in (0, 1, 4, 5):  # skip GS2/GS3 (mixing/defect region)
        slope = ds.li6_d2_zeeman_slope("ground", gs)
        expected = ds.GROUND_GF[gs] * ds.GROUND_MF[gs]
        assert np.isclose(slope, expected, atol=2e-2), (
            f"GS{gs} slope {slope:.4f} vs g_F*mF {expected:.4f}"
        )


def test_order_map_labels_stretched_states():
    """The diabatic order map keeps the stretched-state labels across the sweep.

    GS5 (mF=+3/2) and ES11 (mF=+5/2) are non-degenerate at every field, so
    their labels must never be reassigned along the sweep. The grid is log-dense
    below ~1 G (2e-4 T) where the excited mixing is most volatile, then linear
    to 1 T.
    """
    s = ds.get_li6_setup()
    b_axis_g, order_g = s["b_axis_g"], s["order_g"]
    b_axis_e, order_e = s["b_axis_e"], s["order_e"]
    assert 0.0 < b_axis_g[0] < 1e-3 and np.isclose(b_axis_g[-1], 1.0)
    assert np.all(np.diff(b_axis_g) > 0)  # sorted, strictly increasing
    assert (b_axis_g <= 2e-4).sum() >= 20  # log-dense through the <~1 G regime
    assert order_g.shape == (len(b_axis_g), 6)
    assert order_e.shape == (len(b_axis_e), 12)
    # highest-mF eigenvector is always the last (largest eigenvalue) in eigh
    for node in range(1, len(b_axis_g)):
        mf = ds._state_mf(
            ds.solve_at_field(s["He_hfs"], s["He_zee"], b_axis_e[node])[1],
            s["I"], s["Je"],
        )
        assert np.isclose(mf[order_e[node, 11]], 2.5, atol=0.2)


def test_label_artifact(tmp_path):
    """D-07 artifact: 18-row index<->|F,mF> table with an F column is emitted."""
    out = tmp_path / "li6_d2_state_labels.csv"
    ds.emit_label_artifact(str(out))
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    header = lines[0].split(",")
    assert "F" in header
    assert len(lines) - 1 == 18  # 6 ground + 12 excited


# --- Plan 04-03: live diagonalizer jitclass ---------------------------------


import inspect  # noqa: E402

import src.interactions as interactions  # noqa: E402

_MODEL_NAME = "Lithium6DiagonalizerInteraction"


def test_selectable():
    """The live model is a real jitclass, zero-arg constructible, 6 GS / 12 ES.

    ``inspect.isclass`` MUST be True: a bare-function/factory would pass the
    zero-arg getattr call at parameters.py but silently vanish from the GUI
    dropdown (which introspects with inspect.isclass), violating D-11.
    """
    cls = getattr(interactions, _MODEL_NAME)
    assert inspect.isclass(cls), "live model must be a class (dropdown-visible)"
    model = cls()  # genuinely zero-arg (parameters.build_simulation contract)
    assert model.number_of_ground_states == 6
    assert model.number_of_excited_states == 12


def test_kernel_smoke():
    """The jitclass compiles under @njit and returns finite floats.

    Exercises calculate_transition_frequency_shift / _saturation_parameter /
    _branching_ratio for representative (pol, gs, es) at several |B|.
    """
    model = getattr(interactions, _MODEL_NAME)()
    for B in (0.0, 0.01, 0.1, 0.5):
        for pol in range(3):
            for gs in (0, 3, 5):
                for es in (0, 6, 11):
                    shift = model.calculate_transition_frequency_shift(
                        pol, gs, es, B
                    )
                    branch = model.calculate_branching_ratio(pol, gs, es, B)
                    sat = model.calculate_saturation_parameter(
                        pol, B, gs, es,
                        0.05,                       # laser_intensity
                        2 * np.pi * 5.8724e6,       # natural_linewidth
                        25.4,                       # saturation_intensity
                        446.799677e12,              # effective_transition_freq
                        0.0,                        # doppler_shift
                        446.799677e12,              # laser_beam_frequency
                        0.0,                        # detuning
                    )
                    assert np.isfinite(shift)
                    assert np.isfinite(branch) and branch >= 0.0
                    assert np.isfinite(sat) and sat >= 0.0


def test_live_cycling_strength_normalization():
    """The live cycling transition GS5->ES11 (sigma+) equals |CG|^2 = 0.25."""
    model = getattr(interactions, _MODEL_NAME)()
    live = model.calculate_branching_ratio(2, 5, 11, 1e-3)
    assert np.isclose(live, 0.25, rtol=1e-6)


def test_live_matches_setup_cycling():
    """Live strength matches the plan-02 setup module at the cycling transition.

    The stretched states GS5/ES11 are unambiguously labeled at every field, so
    the live node-map reorder and the setup's per-B (mF, F) labeling select the
    same eigenvectors and must agree to machine precision (acceptance rtol 1e-6).
    """
    model = getattr(interactions, _MODEL_NAME)()
    for B in (1e-3, 1e-2):
        live = model.calculate_branching_ratio(2, 5, 11, B)
        setup = ds.li6_d2_strength(5, 11, 2, B)
        assert np.isclose(live, setup, rtol=1e-6), (
            f"live {live} vs setup {setup} @B={B}"
        )


# --- Plan 04-03 Task 2: end-to-end MOT run (D-10) ---------------------------


def _write_small_setup(tmp_path, interaction_name):
    """Derive a fast headless MOT scenario from the default Hammel setup.

    Keeps the default field/laser geometry; overrides only the interaction, the
    atom count/start conditions and the run length so the run is quick. Atoms
    start at the trap center at low speed so scattering is guaranteed for any
    working model.
    """
    import json
    from pathlib import Path

    src = Path("setup parameters/Hammel_Setup.json")
    cfg = json.loads(src.read_text())
    cfg["Simulation"]["interaction"] = interaction_name
    cfg["Simulation"]["simulated_time"] = 0.05  # ms -> few default steps
    cfg["Atoms"]["number"] = 8
    cfg["Atoms"]["start_position"] = [0.0, 0.0, 0.0]
    cfg["Atoms"]["start_velocity"] = [0.0, 1.0, 0.0]
    cfg["Atoms"].pop("sample_file", None)  # use uniform defaults, no CSV dep
    out = tmp_path / f"{interaction_name}_small.json"
    out.write_text(json.dumps(cfg))
    return str(out)


def _run_short_mot(config_path, n_steps=4):
    """Build a Simulation and run a few steps; return (atoms, total_scatter)."""
    from src.parameters import Parameters

    params = Parameters(config_path)
    assert params.valid, f"config invalid: {params.errors}"
    sim = params.build_simulation()
    for i in range(n_steps):
        cont = sim.step(i)[0]
        if not cont:
            break
    atoms = sim.simulation_atoms
    total_scatter = int(sim.excitation_counter.sum())
    return atoms, total_scatter


def test_end_to_end_diagonalizer(tmp_path):
    """A short MOT run with the live diagonalizer completes with finite state.

    D-10: real run on the default Hammel geometry with the model selected by
    name; produces finite atom state and non-zero scattering.
    """
    cfg = _write_small_setup(tmp_path, _MODEL_NAME)
    atoms, scatter = _run_short_mot(cfg)
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(atoms.velocities).all()
    assert scatter > 0, "diagonalizer produced no scattering events"


def test_end_to_end_comparable_to_18level(tmp_path):
    """Diagonalizer scattering is qualitatively comparable to the 18-level fit.

    Loose order-of-magnitude sanity bound (NOT equality) — deviations at the
    fit defects are expected (D-08). Also a regression guard that the change is
    additive: the interpolation model still builds and runs.
    """
    cfg_diag = _write_small_setup(tmp_path, _MODEL_NAME)
    cfg_ref = _write_small_setup(tmp_path, "Lithium18LevelInteraction")
    _, scatter_diag = _run_short_mot(cfg_diag)
    _, scatter_ref = _run_short_mot(cfg_ref)

    assert scatter_diag > 0 and scatter_ref > 0
    ratio = scatter_diag / scatter_ref
    assert 1 / 30 < ratio < 30, (
        f"scatter not comparable: diag={scatter_diag}, ref={scatter_ref}"
    )

    # Regression guard: interpolation models remain constructible.
    assert interactions.Lithium18LevelInteraction().number_of_ground_states == 6


# --- Plan 04-04: precomputed-table diagonalizer model -----------------------


_TABLE_NAME = "Lithium6DiagonalizerTableInteraction"


@pytest.fixture(scope="module")
def small_table(tmp_path_factory):
    """Generate a small |B|-table NPZ once and return (path, loaded arrays)."""
    d = tmp_path_factory.mktemp("interaction_tables")
    path = str(d / "li6_d2_table_small.npz")
    ds.generate_table(ds.li6_d2_constants(), 0.0, 1.0, 32, path)
    with np.load(path) as data:
        arrays = {
            "b_axis": np.ascontiguousarray(data["b_axis"], dtype=np.float64),
            "pos_table": np.ascontiguousarray(
                data["pos_table"], dtype=np.float64
            ),
            "strength_table": np.ascontiguousarray(
                data["strength_table"], dtype=np.float64
            ),
        }
    return path, arrays


def test_generate_table_shapes(small_table):
    """generate_table writes b_axis / pos_table / strength_table with shapes."""
    _, a = small_table
    assert a["b_axis"].shape == (32,)
    assert a["pos_table"].shape == (32, 6, 12)
    assert a["strength_table"].shape == (32, 6, 12, 3)
    assert np.isfinite(a["pos_table"]).all()
    assert np.isfinite(a["strength_table"]).all()


def test_table_matches_live(small_table):
    """The table model reproduces the live model AT nodes and interpolates.

    At each stored node the table was filled from the live @njit helpers, so
    shifts/strengths must agree to machine precision (acceptance rtol 1e-6).
    Between nodes the linear interpolation must lie within the bracketing node
    values (smoothness), and the cycling strength stays 0.25.
    """
    _, a = small_table
    table = interactions.Lithium6DiagonalizerTableInteraction(
        a["b_axis"], a["pos_table"], a["strength_table"]
    )
    live = interactions.Lithium6DiagonalizerInteraction()
    b_axis = a["b_axis"]

    for k in (1, 8, 20, 31):
        B = b_axis[k]
        for pol in range(3):
            for gs in (0, 3, 5):
                for es in (0, 6, 11):
                    t_shift = table.calculate_transition_frequency_shift(
                        pol, gs, es, B
                    )
                    l_shift = live.calculate_transition_frequency_shift(
                        pol, gs, es, B
                    )
                    assert np.isclose(t_shift, l_shift, rtol=1e-6, atol=1.0), (
                        f"shift mismatch pol={pol} gs={gs} es={es} B={B}"
                    )
                    t_str = table.calculate_branching_ratio(pol, gs, es, B)
                    l_str = live.calculate_branching_ratio(pol, gs, es, B)
                    assert np.isclose(t_str, l_str, rtol=1e-6, atol=1e-9), (
                        f"strength mismatch pol={pol} gs={gs} es={es} B={B}"
                    )

    # Cycling transition stays normalized through the table.
    assert np.isclose(
        table.calculate_branching_ratio(2, 5, 11, b_axis[3]), 0.25, rtol=1e-6
    )

    # Interpolation between two nodes lies within the bracketing node values.
    k = 10
    B_mid = 0.5 * (b_axis[k] + b_axis[k + 1])
    for pol, gs, es in ((2, 5, 11), (1, 0, 7), (0, 3, 4)):
        v0 = table.calculate_transition_frequency_shift(pol, gs, es, b_axis[k])
        v1 = table.calculate_transition_frequency_shift(
            pol, gs, es, b_axis[k + 1]
        )
        vm = table.calculate_transition_frequency_shift(pol, gs, es, B_mid)
        assert min(v0, v1) - 1.0 <= vm <= max(v0, v1) + 1.0
        assert np.isclose(vm, 0.5 * (v0 + v1), rtol=1e-9, atol=1.0)


def test_validate_table_rejects_malformed(small_table):
    """_validate_table raises ParameterError for missing/mis-shaped/non-finite."""
    from src.parameters import ParameterError, Parameters

    _, a = small_table
    # Missing key.
    with pytest.raises(ParameterError):
        Parameters._validate_table(
            {"b_axis": a["b_axis"], "pos_table": a["pos_table"]}
        )
    # Wrong-shaped pos_table.
    with pytest.raises(ParameterError):
        Parameters._validate_table(
            {
                "b_axis": a["b_axis"],
                "pos_table": a["pos_table"][:, :, :6],
                "strength_table": a["strength_table"],
            }
        )
    # Non-finite entry.
    bad = a["strength_table"].copy()
    bad[0, 0, 0, 0] = np.inf
    with pytest.raises(ParameterError):
        Parameters._validate_table(
            {
                "b_axis": a["b_axis"],
                "pos_table": a["pos_table"],
                "strength_table": bad,
            }
        )
    # A valid table round-trips.
    ok = Parameters._validate_table(a)
    assert ok["pos_table"].shape == (32, 6, 12)


def test_table_end_to_end(tmp_path, small_table):
    """A short MOT run with the table model completes with finite state (D-10)."""
    import json
    from pathlib import Path

    table_path, _ = small_table
    src = Path("setup parameters/Hammel_Setup.json")
    cfg = json.loads(src.read_text())
    cfg["Simulation"]["interaction"] = _TABLE_NAME
    cfg["Simulation"]["interaction_table_file"] = table_path
    cfg["Simulation"]["simulated_time"] = 0.05
    cfg["Atoms"]["number"] = 8
    cfg["Atoms"]["start_position"] = [0.0, 0.0, 0.0]
    cfg["Atoms"]["start_velocity"] = [0.0, 1.0, 0.0]
    cfg["Atoms"].pop("sample_file", None)
    out = tmp_path / "table_small.json"
    out.write_text(json.dumps(cfg))

    atoms, scatter = _run_short_mot(str(out))
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(atoms.velocities).all()
    assert scatter > 0, "table model produced no scattering events"


# --- Plan 04-06: substep instrumentation + benchmark harness ----------------


def test_substep_counter_positive_after_run(tmp_path):
    """The kernel surfaces a positive integer substep count via Simulation.

    Instrumentation (04-06) is additive: after a short MOT run the total
    substep count read off `Simulation.total_substeps` must be a positive
    integer, and the interpolation-model path still runs unchanged.
    """
    cfg = _write_small_setup(tmp_path, "Lithium18LevelInteraction")
    from src.parameters import Parameters

    params = Parameters(cfg)
    assert params.valid, f"config invalid: {params.errors}"
    sim = params.build_simulation()
    for i in range(4):
        if not sim.step(i)[0]:
            break

    total = sim.total_substeps
    assert isinstance(total, int)
    assert total > 0, "substep counter did not advance"
    # The counter is the reduction of the per-thread cells.
    assert total == int(sim.substep_counter.sum())


def test_benchmark_harness_runs(tmp_path):
    """The 04-06 benchmark harness returns finite positive timings per model.

    Tiny workload (few atoms, few steps, coarse table) so the harness stays
    wired in CI without a heavy run. Reporting-only: no performance gate.
    """
    from util.diagonalizer_benchmark import (
        LIVE_NAME,
        REF_NAME,
        TABLE_NAME,
        run_benchmark,
    )

    results = run_benchmark(
        n_atoms=6, n_steps=2, n_nodes=16, out_dir=str(tmp_path)
    )
    names = {r["model"] for r in results}
    assert names == {LIVE_NAME, TABLE_NAME, REF_NAME}
    for r in results:
        assert np.isfinite(r["steady_s"]) and r["steady_s"] > 0.0
        assert r["total_substeps"] > 0
        assert np.isfinite(r["us_per_substep"]) and r["us_per_substep"] > 0.0
        assert np.isfinite(r["jit_warmup_s"]) and r["jit_warmup_s"] >= 0.0

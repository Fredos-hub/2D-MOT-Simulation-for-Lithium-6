"""
Integration tests for checkpoint save/resume wired into BatchSimulationWorker (Phase 2).

Covers: save/restore round-trip fidelity; that an interrupted run leaves a resumable
checkpoint (both stop() and stop_current() paths — D-08 ordering); that a clean run deletes
its checkpoint (D-08); that resume APPENDS to the same result.csv with no duplicate rows (D-06);
and that a resumed run is STATISTICALLY consistent with an uninterrupted run, NOT bit-identical
(D-04/D-05 — Numba @njit RNG is not restorable).

All sims are tiny (Lithium4LevelInteraction, ~10 steps, 5 atoms) and run synchronously
(worker.run() directly, no Qt event loop). simulation_results is redirected to a tmp dir.
"""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import src.batch_worker as batch_worker
from src.batch_worker import BatchSimulationWorker
from src.parameters import Parameters
import src.checkpoint as checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SETUP = REPO_ROOT / "setup parameters" / "Tiecke_Setup.json"


def _tiny_config():
    d = json.loads(BASE_SETUP.read_text())
    d["Atoms"]["number"] = 5
    d["Atoms"]["randomize_ground_state"] = False
    d["Atoms"]["ground_state"] = 0
    d["Simulation"]["interaction"] = "Lithium4LevelInteraction"
    d["Simulation"]["simulated_time"] = 0.1  # ms -> ~10 steps at default_time_step 10us
    return d


@pytest.fixture(autouse=True)
def _repo_root_cwd():
    """Parameters resolves GUI/schema relative to CWD — run from repo root."""
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.fixture
def tiny_input(tmp_path):
    """Write the tiny config into a tmp input dir; return (directory, filename)."""
    fn = "tiny.json"
    (tmp_path / fn).write_text(json.dumps(_tiny_config()))
    return str(tmp_path), fn


@pytest.fixture
def isolated_results(tmp_path, monkeypatch):
    """Redirect simulation_results into a tmp tree so tests don't pollute the repo."""
    results_root = tmp_path / "repo_root"
    results_root.mkdir()
    monkeypatch.setattr(batch_worker, "REPO_ROOT", str(results_root))
    return results_root


def _read_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def _run_with_interrupt(directory, fn, mode):
    """Run a worker synchronously; interrupt via the progress signal once past ~30%.
    mode in {'stop', 'stop_current'}. Returns the run_0 directory."""
    w = BatchSimulationWorker(directory, [fn], checkpoint_interval=0.0)  # save every step

    def on_progress(p):
        if p >= 30:
            if mode == "stop":
                w._stop = True
            else:
                w._stop_current = True

    w.progressChanged.connect(on_progress)
    w.run()
    return os.path.join(w.batch_folder, "run_0")


def test_checkpoint_round_trip(tiny_input):
    """save_checkpoint -> load_checkpoint + restore_atom_state reproduces mutated atom state."""
    directory, fn = tiny_input
    params = Parameters(os.path.join(directory, fn))
    assert params.is_valid()
    sim = params.build_simulation()
    sim.warmup()
    for i in range(3):
        sim.step(i)
    before = {name: np.array(getattr(sim.simulation_atoms, name)).copy()
              for name in checkpoint._ATOM_ARRAYS}
    step_before = int(sim.current_step)

    dest = os.path.join(directory, "ckpt")
    batch_state = {"directory": directory, "file_names": [fn], "current_file_idx": 0,
                   "batch_folder": os.path.join(directory, "batch"), "completed_files": []}
    checkpoint.save_checkpoint(sim, batch_state, dest)

    sim2 = Parameters(os.path.join(directory, fn)).build_simulation()
    arrays, meta = checkpoint.load_checkpoint(dest)
    checkpoint.restore_atom_state(sim2, arrays)
    assert int(meta["current_step"]) == step_before
    for name in checkpoint._ATOM_ARRAYS:
        np.testing.assert_array_equal(np.array(getattr(sim2.simulation_atoms, name)), before[name])


@pytest.mark.parametrize("mode", ["stop", "stop_current"])
def test_interruption_leaves_checkpoint(tiny_input, isolated_results, mode):
    """Both interruption paths leave a resumable checkpoint (D-08 ordering: stop_current too)."""
    directory, fn = tiny_input
    run0 = _run_with_interrupt(directory, fn, mode)
    assert os.path.isfile(os.path.join(run0, "checkpoint.json"))
    assert os.path.isfile(os.path.join(run0, "checkpoint.npz"))


def test_clean_completion_deletes_checkpoint(tiny_input, isolated_results):
    """A run that finishes cleanly leaves no checkpoint (D-08)."""
    directory, fn = tiny_input
    w = BatchSimulationWorker(directory, [fn], checkpoint_interval=0.0)
    w.run()
    run0 = os.path.join(w.batch_folder, "run_0")
    assert not os.path.exists(os.path.join(run0, "checkpoint.json"))
    assert not os.path.exists(os.path.join(run0, "checkpoint.npz"))


def test_resume_appends_without_duplicates(tiny_input, isolated_results):
    """Resume continues the same result.csv with no duplicate (step,atom_id) rows (D-06)."""
    directory, fn = tiny_input
    run0 = _run_with_interrupt(directory, fn, "stop_current")
    csv_path = os.path.join(run0, "result.csv")
    _, rows_before = _read_rows(csv_path)
    max_step_before = max((int(r[0]) for r in rows_before), default=-1)

    BatchSimulationWorker("", [], resume_checkpoint_dir=run0).run()

    header_after, rows_after = _read_rows(csv_path)
    pairs = [(int(r[0]), int(r[1])) for r in rows_after]
    assert len(pairs) == len(set(pairs)), "duplicate (step,atom_id) rows after resume"
    assert all(r[0] != "step" for r in rows_after), "result.csv was re-headered on resume"
    assert max(int(r[0]) for r in rows_after) > max_step_before, "resume did not advance past the checkpoint"


def test_resume_statistically_consistent_not_bit_identical(tiny_input, isolated_results):
    """Resumed run agrees statistically with an uninterrupted run (D-04/D-05), not bit-for-bit."""
    directory, fn = tiny_input
    # (a) uninterrupted full run
    w_full = BatchSimulationWorker(directory, [fn], checkpoint_interval=1e9)
    w_full.run()
    _, full_rows = _read_rows(os.path.join(w_full.batch_folder, "run_0", "result.csv"))

    # (b) interrupt + resume to completion
    run0 = _run_with_interrupt(directory, fn, "stop_current")
    BatchSimulationWorker("", [], resume_checkpoint_dir=run0).run()
    _, resumed_rows = _read_rows(os.path.join(run0, "result.csv"))

    # Same atoms participate and both reach the same final step (structural invariants).
    assert {int(r[1]) for r in full_rows} == {int(r[1]) for r in resumed_rows}
    full_last = max(int(r[0]) for r in full_rows)
    resumed_last = max(int(r[0]) for r in resumed_rows)
    assert full_last == resumed_last
    # Alive-atom count at the final step agrees within a loose tolerance.
    full_final = sum(1 for r in full_rows if int(r[0]) == full_last)
    resumed_final = sum(1 for r in resumed_rows if int(r[0]) == resumed_last)
    assert abs(full_final - resumed_final) <= 2
    # D-05: deliberately NO assertion that full_rows == resumed_rows — Numba @njit RNG is not
    # restorable, so individual trajectories diverge while aggregate statistics are preserved.

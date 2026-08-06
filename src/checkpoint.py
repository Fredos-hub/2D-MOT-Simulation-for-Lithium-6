"""Checkpoint save/load for batch simulations.

A checkpoint is a pair of files written into a run directory:

    checkpoint.npz   — all per-atom state and running counters as NumPy arrays
    checkpoint.json  — batch progress, current_step, RNG state, format version

The two files together let a future Python process reconstruct enough state
to continue an interrupted simulation from where it left off. They are
written by `save_checkpoint()` and read by `load_checkpoint()`. The caller
is responsible for rebuilding the Simulation from the parameter JSON and
then handing the restored arrays back via `restore_atom_state()`.

The RNG caveat: NumPy's global state is captured here. Numba's internal
@njit RNG is *not* shared with NumPy's global state and cannot be restored
bit-perfectly without patching Numba. For Monte Carlo physics this is fine
— aggregate statistics are unchanged, only individual trajectories diverge.
"""

import base64
import json
import os
import pickle

import numpy as np

CHECKPOINT_VERSION = 1

# Per-atom arrays on Li6 that fully describe a Simulation's mutable state.
# Add to this list (and bump CHECKPOINT_VERSION) if Li6 grows fields that
# need to be preserved across a resume.
_ATOM_ARRAYS = (
    "positions",
    "velocities",
    "groundstates",
    "status",
    "time_overshoot",
    "pending_optical_depth",
    "subjective_time",
    "magnetic_field_vectors",
    "magnetic_field_strength",
    "max_step_lengths",
    "location_tags",
)


def save_checkpoint(sim, batch_state, dest_dir):
    """Write a checkpoint pair into dest_dir.

    Parameters
    ----------
    sim : src.simulate.Simulation
        The in-progress simulation. Its `simulation_atoms` jitclass and
        `excitation_counter` / `excitation_hist` / `current_step` are read.
    batch_state : dict
        Keys:
            'directory'         absolute path to input JSON directory
            'file_names'        list[str], the queued JSONs
            'current_file_idx'  int, index into file_names that's in progress
            'batch_folder'      absolute path to the active DD_MM_YY_N folder
            'completed_files'   list[str], file_names already done (optional)
    dest_dir : str
        Where to write checkpoint.npz + checkpoint.json. Created if missing.

    Returns
    -------
    str : absolute path to dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)

    atoms = sim.simulation_atoms
    npz_payload = {
        name: np.asarray(getattr(atoms, name)) for name in _ATOM_ARRAYS
    }
    npz_payload["excitation_counter"] = np.asarray(sim.excitation_counter)
    npz_payload["excitation_hist"] = np.asarray(sim.excitation_hist)

    npz_path = os.path.join(dest_dir, "checkpoint.npz")
    np.savez_compressed(npz_path, **npz_payload)

    # NumPy's global RNG state is a small tuple; pickle then b64-encode so it
    # round-trips through JSON without surprises.
    rng_blob = base64.b64encode(pickle.dumps(np.random.get_state())).decode(
        "ascii"
    )

    meta = {
        "version": CHECKPOINT_VERSION,
        "current_step": int(sim.current_step),
        "n_atoms": int(atoms.n),
        "batch": {
            "directory": batch_state["directory"],
            "file_names": list(batch_state["file_names"]),
            "current_file_idx": int(batch_state["current_file_idx"]),
            "batch_folder": batch_state["batch_folder"],
            "completed_files": list(batch_state.get("completed_files", [])),
        },
        "rng_state_b64": rng_blob,
    }

    json_path = os.path.join(dest_dir, "checkpoint.json")
    # Atomic write: write to .tmp then rename, so a crash mid-write doesn't
    # leave a half-written checkpoint that load_checkpoint would choke on.
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, json_path)

    return dest_dir


def load_checkpoint(checkpoint_dir):
    """Read a checkpoint from checkpoint_dir.

    Returns
    -------
    arrays : dict[str, np.ndarray]
        All payload arrays from checkpoint.npz (atom state + counters).
    meta : dict
        Parsed checkpoint.json. The NumPy RNG state has already been restored
        as a side effect; caller can ignore meta['rng_state_b64'].

    Raises
    ------
    FileNotFoundError : if either checkpoint file is missing.
    ValueError        : if the version doesn't match CHECKPOINT_VERSION.
    """
    json_path = os.path.join(checkpoint_dir, "checkpoint.json")
    npz_path = os.path.join(checkpoint_dir, "checkpoint.npz")

    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint version {meta.get('version')} != "
            f"expected {CHECKPOINT_VERSION}"
        )

    rng_state = pickle.loads(base64.b64decode(meta["rng_state_b64"]))
    np.random.set_state(rng_state)

    with np.load(npz_path) as data:
        arrays = {k: data[k].copy() for k in data.files}

    return arrays, meta


def restore_atom_state(sim, arrays):
    """Write checkpoint arrays into a freshly-built Simulation.

    Call this after constructing `sim` via `Parameters.build_simulation()`
    and before entering the step loop. Mirrors how Li6.set_starting_conditions
    writes into the jitclass attributes.

    Sanity-checks the atom count against `sim.simulation_atoms.n`.
    """
    atoms = sim.simulation_atoms
    n_expected = int(atoms.n)
    n_got = int(arrays["positions"].shape[0])
    if n_got != n_expected:
        raise ValueError(
            f"checkpoint has {n_got} atoms but the rebuilt Simulation has "
            f"{n_expected}. Did the input JSON change?"
        )

    for name in _ATOM_ARRAYS:
        if name not in arrays:
            # Older checkpoint missing a newer field — skip silently and let
            # the field keep its default-constructed value.
            continue
        getattr(atoms, name)[:] = arrays[name]

    sim.excitation_counter[:] = arrays["excitation_counter"]
    sim.excitation_hist[:] = arrays["excitation_hist"]


def find_resumable_checkpoint(batch_folder):
    """Scan a batch folder for a resumable checkpoint.

    Looks for batch_folder/run_*/checkpoint.json. Returns the path to the
    most-recent (highest run_idx) checkpoint directory, or None if no
    checkpoint is found. Returning the directory keeps the caller free to
    load it with `load_checkpoint(path)` whenever it wants.
    """
    if not os.path.isdir(batch_folder):
        return None

    candidates = []
    for name in os.listdir(batch_folder):
        if not name.startswith("run_"):
            continue
        run_dir = os.path.join(batch_folder, name)
        if os.path.isfile(os.path.join(run_dir, "checkpoint.json")):
            try:
                idx = int(name.split("_", 1)[1])
            except ValueError:
                continue
            candidates.append((idx, run_dir))

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]

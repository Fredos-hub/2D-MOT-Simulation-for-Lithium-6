# Codebase Concerns

**Analysis Date:** 2026-06-02

---

## GUI Refactor (Major In-Progress Area)

**State of the refactor:**
The branch `fs/chore/frontend` has deleted ~18 files under the old flat `GUI/widgets/*.py`
layout and replaced them with a new four-subdirectory structure:
- `GUI/widgets/common/` — shared widgets (file table, bar-dipole table, vector input)
- `GUI/widgets/dialogs/` — popup dialogs (edit-all, edit-defaults, validation)
- `GUI/widgets/features/` — top-level tab shells (cockpit, plotting, sample generator, spectrum, incrementor)
- `GUI/widgets/tabs/` — settings-tab implementations (atoms, laser, magnetic field, boundaries, simulation)
- `GUI/models/` — model layer (`FileModel`)
- `GUI/workers/` — background threads (`OvenWorker`)

**Import graph is fully wired up.**
`GUI/main_window.py` imports from the new paths; no import still references any of the 18
deleted old paths (`GUI/widgets/atoms_tab.py`, `GUI/widgets/laser_tab.py`, etc.). The
refactor is structurally complete at the import level.

**Incomplete placeholder — `GUI/widgets/features/plotting.py`:**
- File: `GUI/widgets/features/plotting.py` (10 lines)
- The `PlottingTab` is a shell: it renders only a `QLabel("Plotting Tab - Shell", ...)`.
  No actual plotting functionality is implemented.
- Impact: the Plotting tab in the GUI is non-functional.
- Fix: implement the tab or mark it clearly as a future milestone.

**Checkpoint module has no GUI integration:**
- `src/checkpoint.py` implements `save_checkpoint()`, `load_checkpoint()`, and
  `find_resumable_checkpoint()`, but none of these are called from `src/batch_worker.py`
  or any GUI component.
- Impact: interrupted long-running simulations cannot be resumed; the checkpoint
  infrastructure is dead code.
- Fix: wire `save_checkpoint()` into `BatchSimulationWorker.run()` at regular intervals
  and expose a "Resume" action in `GUI/widgets/features/simulation_cockpit.py`.

---

## Known Physics / Correctness Issues

**FIXME: Zeeman boundary condition — `src/simulate.py:122`**
- The comment `#FIXME: Zeeman Boundaries` at line 122 in `src/simulate.py` indicates the
  y-axis boundary kill is not correctly accounting for the Zeeman slower region.
  Atoms are killed at `|y| >= boundaries[1]` using the same flat boundary as x and z,
  which is likely wrong when a Zeeman field extends further.
- Impact: atoms may be killed prematurely or survive too long in the y direction during
  Zeeman-slower simulations.

**FIXME: Swapped ground states GS2/GS3 in 18-level wrappers — `src/interaction_wrappers/eighteen_level_wrappers.py:817`**
- The comment `#FIXME: Swapped GS2 and GS3. See Julias Plots for Reference` appears in the
  σ+ (`pol==2`) branch of `calculate_transition_strength()`.
  The affected transitions are `GS==3, ES==8` and `GS==2, ES==8`.
- Impact: incorrect transition strengths and therefore wrong scattering rates for σ+
  polarization in the 18-level model for those particular ground/excited state pairs. This
  is a live correctness defect in the primary physics kernel.
- Fix: cross-reference Julia's plots, swap GS indices, and add a regression test pinning
  the corrected values.

**Numerical fragility in `calculate_transition_strength` (`src/interaction_wrappers/eighteen_level_wrappers.py`)**
- The function is 1085 lines of piecewise polynomial/rational curve fits, assembled by
  piecewise if/elif chains covering three magnetic field ranges per transition.
- There are coverage gaps: several branches silently return 0 for inputs outside the fitted
  range (`B > 0.1 T` or `B <= -0.1 T`) when the physical value is non-zero. For example,
  `GS==1, ES==2` returns `trans_strength = 0` for `B > 0.1`.
- `calculate_transition_frequency_shift` (also ~1000 lines) has the same structure.
  The `frequency_shift != 0` guard at line 1072 returns 0 when no branch matched instead
  of raising an error, silently producing wrong physics.
- Impact: silent wrong results for large magnetic fields; hard to audit.
- Fix: add an explicit `else: raise ValueError(...)` or at minimum a guard assert for each
  top-level `(GS, ES, pol)` block to catch out-of-range B.

---

## Tech Debt

**Schema path is CWD-dependent:**
- `src/parameters.py:128` defaults `schema_path` to `"GUI/schema/schema_v1.json"` (a
  relative path).
- `GUI/widgets/features/simulation_cockpit.py:19` builds `SCHEMA_PATH` with
  `os.path.join('GUI/schema', ...)` — also relative.
- Both rely on the process being launched from the repo root (`python -m main`). Calling
  `Parameters()` from a different directory silently raises `FileNotFoundError`.
- Fix: resolve the schema path relative to `__file__` inside `parameters.py`.

**`TODO` in `src/batch_worker.py:287`:**
- `# TODO: Consider moving to loading of the files.`
  Schema validation currently runs at simulation-build time inside `BatchSimulationWorker`,
  not at file-load time, so invalid configs are only detected when a run starts.
- Fix: validate at file-load time in `FileModel` or when files are added to the queue.

**`print()` calls leaking to stdout in production code:**
- `src/simulate.py:58` — `print("Warmup step completed.")`
- `src/simulate.py:91` — `print("No atoms live, simulation stopping.")`
- `src/simulate.py:147` — `print("Maximum step number reached...")`
- `src/batch_worker.py:398` — `print(exc_hist)` after a run ends
- These bypass the GUI log panel (`statusChanged` signal) and are invisible in the cockpit.
- Fix: route all status messages through `self.statusChanged.emit()` in `BatchSimulationWorker`
  and through a logger in `Simulation`.

**`rate_mode` / `macro_particle_weight` / `flux` parsed but not exposed in tests or GUI tabs:**
- These parameters exist in `src/parameters.py` and the JSON schema, but there are no GUI
  controls surfacing them and no tests exercising rate-mode injection logic.

---

## Performance Bottlenecks

**Numba JIT cold-start latency:**
- `@njit(parallel=True)` on `absorption_and_emission_default_timestep`
  (`src/absorption_and_emission_process.py:25`) triggers LLVM compilation on first call.
- `@njit(cache=True)` is set on `calculate_transition_strength`
  (`src/interaction_wrappers/eighteen_level_wrappers.py:10`) which benefits from caching,
  but no `.nbi`/`.nbc` files were found under `src/__pycache__` (cache may not be primed
  in this checkout), meaning the 1–3 minute compile runs on every fresh environment.
- `Simulation.warmup()` (`src/simulate.py:43`) calls the kernel twice to prime it before
  the real run, which is correct but adds wall time before any progress is emitted to the GUI.

**`absorption_and_emission_default_timestep` inner loop complexity:**
- For each alive atom, per timestep, the loop executes:
  `n_excited_states × n_lasers × 3` saturation parameter computations, then
  `n_excited_states × n_ground_states × 3` branching ratio lookups on each absorption
  event. For 18-level × 6 lasers this is ~324 saturation evaluations per atom per
  sub-step. The per-thread workspace allocation (`np.empty` calls outside `prange` at
  lines 60–69) correctly avoids repeated allocation, but the inner while-loop still
  calls `simulation_interaction.calculate_transition_frequency_shift()` for every
  `(excited_state, polarization)` pair each sub-step, even when B has not changed.
  The comment at line 95 notes the Zeeman shift is pre-computed once per sub-step (which
  is already done), but higher-level caching (per-atom, per-step) is not implemented.

---

## Fragile Areas

**`src/interaction_wrappers/eighteen_level_wrappers.py` — monolithic curve-fit file:**
- Files: `src/interaction_wrappers/eighteen_level_wrappers.py` (1085 lines),
  `src/interaction_wrappers/simple_eighteen_level_wrappers.py` (32 lines)
- The 18-level wrapper contains ~200 hand-fitted piecewise polynomials generated from
  Breit-Rabi numerical data. Any edit risks breaking a case that has no assertion.
- Test coverage: `tests/test_18level_excitation_rates.py` covers the excitation-rate sum
  rules and Zeeman-slope convergence but does **not** test individual transition strength
  values for specific `(GS, ES, pol, B)` inputs.
- Safe modification: add regression snapshot tests for each `(GS, ES, pol)` block before
  touching the fitting coefficients.

**`src/batch_worker.py` — silent exception swallowing:**
- Multiple `except Exception: pass` blocks at lines 68, 123, 130, 299, 326, 433.
  Errors during file-handle cleanup, config-write, and run-folder creation are silently
  dropped or emitted only as a GUI string.
- Impact: a failed run folder creation does not abort the batch run; partial results
  accumulate in unpredictable locations.

**Numba JIT boundary: accidental Python-type arguments:**
- `@njit` functions receive jitclass instances and NumPy arrays. If a caller passes a
  Python `None` or a non-contiguous array (e.g. a slice of a C-order array along a
  non-leading axis) Numba raises an opaque `TypingError` or silently reinterprets memory.
- There are no runtime assertions guarding array contiguity or dtype before any JIT call
  in `src/simulate.py` or `src/batch_worker.py`.

---

## Security Considerations

**No security-sensitive concerns identified.**
The application runs entirely locally; there is no network I/O, authentication, or
user-supplied code execution path.

---

## Test Coverage Gaps

**No tests for `src/parameters.py`:**
- Schema validation, `_parse_*` methods, `build_simulation()` path, and error-collection
  logic have zero test coverage. A typo in a field name or a schema change would only
  be caught at runtime.
- Priority: High — `Parameters` is the single entry point for all simulation configuration.

**No tests for `src/simulate.py`:**
- The `Simulation.step()` loop, boundary kill logic, and the
  `#FIXME: Zeeman Boundaries` code path are untested.
- Priority: High — silent incorrect boundary behavior would not be detected.

**No tests for `src/checkpoint.py`:**
- `save_checkpoint()` / `load_checkpoint()` / `restore_atom_state()` are untested, and
  the module is not yet integrated into `BatchSimulationWorker`.
- Priority: Medium — irrelevant until checkpoint integration is wired in, but will need
  tests before it is safe to rely on for resuming production runs.

**No tests for `src/magnetic_field.py`:**
- `IdealQuadrupoleField`, `ZeemanField`, `EllipticalMagneticField`, `DipoleBarMagneticField`
  all have zero tests. Field strength, gradient direction, and step-length calculations
  are untested.
- Priority: Medium.

**No GUI tests of any kind.**

**No tests for the 6-level, simple-18-level, or 4-level interaction models:**
- `tests/test_18level_excitation_rates.py` covers `Lithium18LevelInteraction` only.
  `Lithium6LevelInteraction`, `SimpleEighteenLevelInteraction`, and
  `Lithium4LevelInteraction` are untested.
- Priority: Medium.

---

## Task Backlog (from `todo.md`)

All four open items are physics-analysis tasks, not code tasks:

| Item | Status | Notes |
|------|--------|-------|
| Pushbeam detuning simulations (z_limit = 20 cm) | `[ ]` open | Re-run detuning scan |
| ToF histograms | `[ ]` open | Time-of-flight histograms |
| Velocity distributions at z = 19 cm | `[ ]` open | Post-pushbeam state analysis |
| Spectrum run (spec beam only, from post-pushbeam state) | `[ ]` open | Full spectrum simulation |

Analysis utilities live in `util/pushbeam_detuning_histograms.py` and
`util/integrated_spectrum.py`.

---

## Documentation Discrepancy

**CLAUDE.md lists `FourLevelInteraction`; the actual class is `Lithium4LevelInteraction`:**
- `CLAUDE.md` interaction table entry: `FourLevelInteraction` — 2 ground, 2 excited states.
- Actual class in `src/interactions.py:287`: `Lithium4LevelInteraction` — 1 ground, 3 excited states.
- Both the name and the state count are wrong in the documentation.
- Impact: a developer following CLAUDE.md to configure a JSON file would use
  `"interaction": "FourLevelInteraction"`, which would fail at `getattr(interactions, ...)` in
  `src/parameters.py:356`.

---

*Concerns audit: 2026-06-02*

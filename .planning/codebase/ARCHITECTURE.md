<!-- refreshed: 2026-06-02 -->
# Architecture

**Analysis Date:** 2026-06-02

## System Overview

```text
┌──────────────────────────────────────────────────────────────────┐
│                         Entry Points                             │
│  main.py --GUI → MainWindow        main.py --files → BatchWorker │
└───────────────────┬──────────────────────────┬───────────────────┘
                    │                          │
                    ▼                          │
┌───────────────────────────────┐             │
│     GUI/ (PyQt5 Python)       │             │
│  MainWindow, SimulationCockpit│             │
│  FileModel (in-memory JSON)   │             │
│  SettingsTabsWidget (tabs)    │             │
│  BatchSimulationWorker signal │─────────────┤
└───────────────────────────────┘             │
                    │                          │
                    ▼                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                  src/ Python layer (QThread)                     │
│   Parameters → validate JSON → parse → build_simulation()        │
│   BatchSimulationWorker.run() → drives Simulation.step() loop    │
└───────────┬──────────────────────────────────┬───────────────────┘
            │                                  │
            ▼                                  ▼
┌────────────────────────┐        ┌────────────────────────────────┐
│  Simulation (Python)   │        │  CSV output                    │
│  `src/simulate.py`     │        │  simulation_results/           │
│  step loop, boundary   │        │  DD_MM_YY_N/run_N/result.csv   │
│  checks, atom lifecycle│        │  config.json                   │
└────────────┬───────────┘        └────────────────────────────────┘
             │  passes jitclass instances
             ▼
┌──────────────────────────────────────────────────────────────────┐
│            Numba @njit / @jitclass kernel layer                  │
│                                                                  │
│  Li6 jitclass (src/atoms.py)          ← atom state (ECS)        │
│  LaserComponent jitclass (src/lasers.py)                         │
│  MagneticField jitclasses (src/magnetic_field.py)                │
│  LightAtomInteraction jitclasses (src/interactions.py)           │
│  absorption_and_emission_default_timestep @njit(parallel=True)   │
│    (src/absorption_and_emission_process.py)                      │
│  interaction_wrappers/ @njit helpers (src/interaction_wrappers/) │
└──────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `main.py` | CLI arg parsing; launches GUI or headless batch run | `main.py` |
| `Parameters` | JSON load → schema validation → field parsing → `build_simulation()` | `src/parameters.py` |
| `Simulation` | Step loop, atom activation/kill logic, boundary checks | `src/simulate.py` |
| `BatchSimulationWorker` | QThread driving the Simulation loop; buffered CSV output | `src/batch_worker.py` |
| `Li6` | ECS atom container (jitclass); all per-atom state as SoA arrays | `src/atoms.py` |
| `LaserComponent` | All laser beam parameters packed in SoA arrays (jitclass) | `src/lasers.py` |
| `LightAtomInteraction` | Selectable interaction models (jitclasses) | `src/interactions.py` |
| `MagneticField` | Selectable field models (jitclasses) | `src/magnetic_field.py` |
| `absorption_and_emission_default_timestep` | `@njit(parallel=True)` core physics kernel | `src/absorption_and_emission_process.py` |
| `interaction_wrappers/` | `@njit` helpers for sat-param, transition strength, freq shift | `src/interaction_wrappers/` |
| `checkpoint.py` | Save/restore per-atom state + RNG state to `.npz`/`.json` | `src/checkpoint.py` |
| `MainWindow` | Top-level QMainWindow; tab container | `GUI/main_window.py` |
| `SimulationCockpit` | Primary GUI feature: file management + run control + log | `GUI/widgets/features/simulation_cockpit.py` |
| `FileModel` | In-memory JSON model with dirty-tracking and nested get/set | `GUI/models/file_model.py` |
| `OvenWorker` | QThread generating Maxwell-Boltzmann oven sample CSV files | `GUI/workers/oven_worker.py` |
| `ECSAtoms / MagneticField / ECSLasers / LightAtomInteraction` | Python dummy classes for type hints only; never instantiated | `util/simulation_typing.py` |

## Pattern Overview

**Overall:** Three-layer architecture — GUI (PyQt5), Python orchestration (`src/`), Numba JIT kernel.

**Key Characteristics:**
- The physics kernel runs inside `@njit(parallel=True)` — GPU-free but CPU-parallel via Numba.
- All mutable atom state lives in a single `Li6` jitclass (ECS-like struct-of-arrays). There are no per-atom Python objects.
- Interaction models and magnetic field models are selectable at runtime by name string in the JSON config; `Parameters.build_simulation()` uses `getattr(interactions, name)()` and `getattr(atoms, name)()`.
- `BatchSimulationWorker` is a `QThread` subclass but also works synchronously (CLI calls `worker.run()` directly in the main thread).
- A hard Numba JIT boundary exists: only jitclass instances and typed NumPy arrays may cross into `@njit` functions. `BatchSimulationWorker` never passes jitclass objects to the GUI; it serializes results to NumPy arrays (written to CSV).

## Layers

**CLI / Entry layer:**
- Purpose: Argument parsing; route to GUI or headless batch mode.
- Location: `main.py`
- Contains: `main()`, `start_gui()`, signal wiring for CLI progress.
- Depends on: `src/batch_worker.py`, `GUI/main_window.py`
- Used by: User process (direct invocation).

**GUI layer (PyQt5 Python):**
- Purpose: Interactive parameter editing, simulation launch, plotting, oven sample generation.
- Location: `GUI/`
- Contains: `MainWindow`, feature widgets, settings tabs, dialogs, `FileModel`, `OvenWorker`.
- Depends on: `src/batch_worker.py`, `src/parameters.py` (for schema validation in the cockpit).
- Used by: `main.py` when `--GUI` flag is present.

**Orchestration / Python layer:**
- Purpose: JSON config validation and parsing; Numba object construction; simulation step-loop control; buffered CSV I/O.
- Location: `src/parameters.py`, `src/batch_worker.py`, `src/simulate.py`
- Contains: `Parameters`, `BatchSimulationWorker`, `Simulation`.
- Depends on: all jitclass modules in `src/`, `util/simulation_typing.py`.
- Used by: `main.py` and `GUI/widgets/features/simulation_cockpit.py`.

**Numba JIT kernel layer:**
- Purpose: Per-atom physics (absorption, emission, Doppler, Zeeman, gravity, Monte Carlo).
- Location: `src/atoms.py`, `src/lasers.py`, `src/magnetic_field.py`, `src/interactions.py`, `src/absorption_and_emission_process.py`, `src/interaction_wrappers/`
- Contains: All `@jitclass` and `@njit` code.
- Depends on: NumPy arrays, scipy constants (at import time only), `util/simulation_typing.py` (type hints only).
- Used by: `Simulation.step()` and `Simulation.warmup()`.

**Utilities:**
- Purpose: Type-hint dummy classes (never instantiated in simulation); geometry helpers; analysis scripts.
- Location: `util/`
- Contains: `simulation_typing.py`, `geometry.py`, `integrated_spectrum.py`, `pushbeam_detuning_histograms.py`, `trapped_atoms_heatmap.py`, `wrappers.py`.

## Data Flow

### Primary Simulation Path

1. `main()` parses CLI args (`main.py:83`)
2. `BatchSimulationWorker.__init__()` receives target dir and file list (`src/batch_worker.py:26`)
3. `Parameters.__init__()` loads JSON, validates against `GUI/schema/schema_v1.json`, calls `_parse_*` methods (`src/parameters.py:125`)
4. `Parameters.build_simulation()` constructs all jitclass objects and returns `Simulation` (`src/parameters.py:346`)
5. `Simulation.warmup()` triggers Numba JIT compilation with a dummy atom (`src/simulate.py:43`)
6. `BatchSimulationWorker.run()` iterates `sim.step(i)` in a loop (`src/batch_worker.py:339`)
7. Each `sim.step(i)` activates atoms, calls `absorption_and_emission_default_timestep()`, applies boundaries (`src/simulate.py:60`)
8. `absorption_and_emission_default_timestep()` runs `@njit(parallel=True)` per alive atom: Doppler/Zeeman shifts, saturation parameters, Monte Carlo absorption/emission, position/velocity update (`src/absorption_and_emission_process.py:26`)
9. `BatchSimulationWorker.write_step_results()` buffers CSV lines; flushes at `buffer_size=10000` (`src/batch_worker.py:154`)
10. Output written to `simulation_results/DD_MM_YY_N/run_N/result.csv` + `config.json`

### GUI Simulation Path

Same as above from step 3 onward, but `BatchSimulationWorker` runs in a QThread. Progress/status signals (`progressChanged`, `statusChanged`, `fileFinished`, `finished`) are connected to GUI widgets in `SimulationCockpit`.

### Atom Lifecycle

- Atoms start with `status = -1` (inactive). `time_overshoot` holds the future injection time.
- `Simulation.step()` activates atoms whose `time_overshoot <= default_timestep` by setting `status = 1`.
- Atoms exceeding boundary limits in x/y/z are killed (`status = 0`).
- After `max_step_number` steps all remaining atoms are killed.

**State Management:**
- All per-atom mutable state is on the `Li6` jitclass instance (SoA: `positions`, `velocities`, `status`, `groundstates`, `time_overshoot`, `subjective_time`, `magnetic_field_strength`, `magnetic_field_vectors`).
- `Simulation` holds references to the four jitclass objects and scalar step counters. No global state.

## Key Abstractions

**Li6 jitclass (ECS atom container):**
- Purpose: Struct-of-arrays holding all atom state for `n` atoms. Single instance per simulation.
- Examples: `src/atoms.py`
- Pattern: ECS-like — physics is done by passing the whole container plus an `alive_ids` index array into `@njit` functions. No per-atom Python objects exist.

**LightAtomInteraction (selectable interaction model):**
- Purpose: Encapsulates level structure, Clebsch-Gordan coefficients, branching tables.
- Examples: `src/interactions.py` — `Lithium6LevelInteraction`, `Lithium18LevelInteraction`, `SimpleLithium18LevelInteraction`, `FourLevelInteraction`
- Pattern: Each jitclass exposes `calculate_saturation_parameter()`, `calculate_rate()`, `calculate_transition_frequency_shift()`, `calculate_branching_ratio()`. Selected by name string in JSON; instantiated via `getattr(interactions, name)()`.

**MagneticField (selectable field model):**
- Purpose: Computes `magnetic_field_vectors` and `magnetic_field_strength` on the atom container for a given atom id.
- Examples: `src/magnetic_field.py` — `IdealQuadrupoleField`, `ZeemanField`, `EllipticalMagneticField`, `DipoleBarMagneticField`
- Pattern: Each jitclass exposes `calculate_magnetic_field()`, `calculate_max_step_length()`. Selected by `"type"` in JSON `"Magnetic_Fields"` block.

**util/simulation_typing.py dummy classes:**
- Purpose: Python-level type-hint mirrors of the four jitclass types. Used in `@njit` function signatures and in Python-layer type annotations so IDEs can navigate the interface without loading Numba.
- Pattern: Never instantiated during a real simulation run. `ECSAtoms`, `MagneticField`, `ECSLasers`, `LightAtomInteraction` in `util/simulation_typing.py`.

**FileModel:**
- Purpose: In-memory representation of a JSON parameter file with dirty-tracking and nested get/set.
- Examples: `GUI/models/file_model.py`
- Pattern: QObject with `dirtyChanged` signal. `data()` returns the current dict; `replace()` replaces the whole document.

## Entry Points

**CLI — headless batch:**
- Location: `main.py:83` (`main()` → `BatchSimulationWorker`)
- Triggers: `python -m main --files <file.json> [--target-dir <dir>]`
- Responsibilities: Build params, run step loop synchronously, emit progress to tqdm.

**CLI — GUI:**
- Location: `main.py:33` (`start_gui()`)
- Triggers: `python -m main --GUI [--style light|dark]`
- Responsibilities: Construct QApplication, splash screen, `MainWindow`; enter Qt event loop.

**Simulation step:**
- Location: `src/simulate.py:60` (`Simulation.step(i)`)
- Triggers: Called in a loop by `BatchSimulationWorker.run()`.
- Responsibilities: Activate atoms, run physics kernel, apply boundaries, advance step counter.

## Architectural Constraints

- **Threading:** `BatchSimulationWorker` inherits `QThread`; its `run()` method is called on the worker thread when run from the GUI, or synchronously on the main thread from CLI. The Numba `@njit(parallel=True)` kernel uses Numba's internal thread pool (controlled by `NUMBA_NUM_THREADS`).
- **JIT boundary:** Only `Li6`, `LaserComponent`, any `MagneticField` jitclass, any `LightAtomInteraction` jitclass, typed `np.ndarray` (C-contiguous `float64`/`int32/64`), and Python scalars may be passed into `@njit` functions. Never pass `None` or non-array Python objects.
- **Global state:** No module-level singletons. `REPO_ROOT` in `src/batch_worker.py:16` is a constant (not mutable). Numba has its own internal RNG state that is separate from NumPy's global RNG.
- **Numba compilation cache:** Compiled kernels are stored in `src/__pycache__/` and `src/interaction_wrappers/__pycache__/` as `.nbc`/`.nbi` files. First run takes 1–3 minutes; `Simulation.warmup()` triggers compilation before the main loop.
- **Circular imports:** None detected. `src/` modules import only `util/simulation_typing.py` for type hints; `GUI/` imports from `src/` but `src/` does not import from `GUI/`.

## Anti-Patterns

### Passing jitclass objects to the GUI layer

**What happens:** `BatchSimulationWorker.write_step_results()` is called with `current_atom_states` (the `Li6` jitclass). Attributes are accessed via `current_atom_states.positions[...]` in plain Python.
**Why it's wrong:** Jitclass instances are opaque Numba objects; accessing them in Python after the `@njit` kernel is correct here but any attempt to emit them across a `pyqtSignal` or pickle them will fail.
**Do this instead:** Slice the NumPy arrays from the jitclass into plain `ndarray`s before any signal emission or serialization.

### Boundary kill logic in Python step loop

**What happens:** Boundary checks (x/y/z) are done in `Simulation.step()` in Python after the `@njit` kernel returns (`src/simulate.py:116`).
**Why it's wrong:** This is correct behavior but there is an open `#FIXME: Zeeman Boundaries` comment at line 122 indicating the y-boundary logic may not apply correctly for Zeeman-slower configurations.
**Do this instead:** Implement configuration-aware boundary logic or skip the y-boundary for `ZeemanField` mode.

## Error Handling

**Strategy:** Exceptions are caught at each phase boundary and converted to status messages. `Parameters` collects errors into `self.errors: List[str]`; callers check `params.is_valid()`. `BatchSimulationWorker` emits `statusChanged` for all errors rather than re-raising.

**Patterns:**
- `SchemaValidationError` (raised by `validate_against_schema()`) carries a deduplicated list of human-readable messages suitable for display.
- `ParameterError` is raised by `build_simulation()` for runtime construction failures (bad class name, constructor failure, etc.).
- `BatchSimulationWorker.run()` catches `ParameterError` and continues to the next file in the batch.

## Cross-Cutting Concerns

**Logging:** No logging framework. `BatchSimulationWorker` emits `statusChanged(str)` signals. In CLI mode these are connected to `tqdm.write()`. In GUI mode they feed a `QPlainTextEdit` log with a syntax highlighter.
**Validation:** JSON schema validation via `jsonschema.Draft7Validator` against `GUI/schema/schema_v1.json`. Errors are surfaced as `SchemaValidationError` or displayed in `GUI/widgets/dialogs/validation_dialog.py`.
**Authentication:** Not applicable.

---

*Architecture analysis: 2026-06-02*

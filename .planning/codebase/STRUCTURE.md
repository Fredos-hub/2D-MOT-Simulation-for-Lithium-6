# Codebase Structure

**Analysis Date:** 2026-06-02

## Directory Layout

```
2D-MOT-Simulation-for-Lithium-6/
├── main.py                         # CLI entry point: --GUI or --files batch mode
├── requirements.txt                # Python dependencies
├── CLAUDE.md                       # Project guidance for Claude sessions
├── todo.md                         # Cross-session backlog
├── plots.py                        # Standalone analysis/plotting scripts
├── find_transition_strengths.py    # One-off utility script
├── oven_snapshot_initial_conditions.csv   # Pre-generated oven sample file
├── snapshot_initial_conditions.csv        # Pre-generated snapshot sample file
├── Experimental PB Vel.csv         # Experimental push-beam velocity reference data
├── src/                            # Core simulation (Python + Numba JIT)
│   ├── parameters.py               # JSON → validated config → Simulation factory
│   ├── simulate.py                 # Simulation: step loop, atom activation/kill
│   ├── batch_worker.py             # BatchSimulationWorker (QThread); CSV I/O
│   ├── atoms.py                    # Li6 jitclass (ECS atom container)
│   ├── interactions.py             # LightAtomInteraction jitclasses (4 models)
│   ├── magnetic_field.py           # MagneticField jitclasses (4 models)
│   ├── lasers.py                   # LaserComponent jitclass
│   ├── absorption_and_emission_process.py  # @njit(parallel=True) physics kernel
│   ├── checkpoint.py               # Save/restore atom state + RNG to .npz/.json
│   ├── distributions.py            # Statistical distribution helpers
│   ├── maxwell_boltzmann_sampler.py        # Maxwell-Boltzmann lookup sampler
│   ├── spectrum_kernel.py          # Spectrum calculation kernel
│   └── interaction_wrappers/       # @njit helpers per interaction model
│       ├── common.py               # Shared sat-param + excitation-rate formulas
│       ├── six_level_wrappers.py
│       ├── eighteen_level_wrappers.py
│       ├── simple_eighteen_level_wrappers.py
│       └── four_level_wrappers.py
├── GUI/                            # PyQt5 interface  [UNDER ACTIVE REFACTOR — see note]
│   ├── main_window.py              # MainWindow (QMainWindow): top-level tab shell
│   ├── menu_bar.py                 # CustomMenuBar
│   ├── toolbar.py                  # ToolBar with save/discard/run actions
│   ├── models/
│   │   └── file_model.py           # FileModel: in-memory JSON with dirty-tracking
│   ├── workers/
│   │   └── oven_worker.py          # OvenWorker (QThread): oven sample generation
│   ├── widgets/
│   │   ├── common/                 # Reusable low-level widgets
│   │   │   ├── bar_dipole_table.py
│   │   │   ├── file_table.py       # FileTableWidget (file list with sim state)
│   │   │   └── vector_input_widget.py
│   │   ├── dialogs/                # Modal popup dialogs
│   │   │   ├── edit_all_popup_widget.py
│   │   │   ├── edit_defaults_popup_widget.py
│   │   │   └── validation_dialog.py
│   │   ├── features/               # Full-featured composite tabs
│   │   │   ├── simulation_cockpit.py  # Primary tab: file list, run control, log
│   │   │   ├── incrementor_tab.py  # Batch parameter-sweep generator
│   │   │   ├── plotting.py         # Plotting tab (result visualization)
│   │   │   ├── sample_generator.py # SampleGeneratorTab (oven sample CSV)
│   │   │   └── spectrum_tab.py     # Spectrum viewer tab
│   │   └── tabs/                   # Settings parameter tabs (per JSON section)
│   │       ├── settings_tab_base.py        # Base class for settings tabs
│   │       ├── settings_tabs.py    # SettingsTabsWidget: assembles all tabs
│   │       ├── simulation_tab.py   # Simulation section controls
│   │       ├── atoms_tab.py        # Atoms section controls
│   │       ├── laser_tab.py        # Lasers section controls
│   │       ├── magnetic_field_tab.py       # Magnetic field controls
│   │       └── boundaries_tab.py   # Boundaries controls
│   ├── schema/
│   │   └── schema_v1.json          # JSON schema for parameter validation
│   ├── defaults/                   # Default JSON files loaded by GUI
│   │   ├── lasers/                 # Default laser preset files
│   │   └── magnets/                # Default magnet preset files
│   └── icons/                      # UI icons (PNG)
├── util/                           # Shared helpers; type-hint dummies
│   ├── simulation_typing.py        # Dummy classes for type hints: ECSAtoms,
│   │                               #   MagneticField, ECSLasers, LightAtomInteraction
│   ├── geometry.py                 # @njit geometry helpers (e.g. random_angle_in_sphere)
│   ├── integrated_spectrum.py      # Post-processing: spectrum integration
│   ├── pushbeam_detuning_histograms.py     # Push-beam analysis utility
│   ├── trapped_atoms_heatmap.py    # Heatmap generation from result CSV
│   └── wrappers.py                 # Additional wrapper utilities
├── tests/                          # Test suite
│   ├── test_18level_excitation_rates.py
│   └── test_polarization_geometry.py
├── setup parameters/               # JSON config files for known experimental setups
│   ├── Tiecke_Setup.json
│   ├── Hammel_Setup.json
│   ├── Hannah_PB_LIN_det+2.json
│   ├── Pohl_Setup_Tiecke_Det.json
│   └── (subdirectories for scan variants)
├── simulation_results/             # Runtime output (not committed)
│   └── DD_MM_YY_N/run_N/
│       ├── result.csv
│       └── config.json
└── prototype/                      # Experimental scratch area (not production code)
```

> **GUI refactor note:** The `GUI/widgets/` tree is mid-refactor. The old flat layout (`GUI/widgets/*.py` — `atoms_tab.py`, `bar_dipole_table.py`, `boundaries_tab.py`, `edit_all_popup_widget.py`, `edit_defaults_popup_widget.py`, `file_table.py`, `incrementor_tab.py`, `io_tap.py`, `laser_tab.py`, `magnetic_field_tab.py`, `plotting.py`, `sample_generator.py`, `settings_tabs.py`, `simulation_cockpit.py`, `simulation_tab.py`, `spectrum_tab.py`, `target_dir_popup.py`, `validation_dialog.py`, `vector_input_widget.py`) has been deleted and replaced by the new four-subdirectory layout (`common/`, `dialogs/`, `features/`, `tabs/`). The old `GUI/file_model.py` and `GUI/oven_worker.py` moved to `GUI/models/` and `GUI/workers/`. These changes are untracked (not yet committed). All import paths in existing code already reference the new locations.

## Directory Purposes

**`src/`:**
- Purpose: All simulation logic — Python orchestration, Numba jitclasses, and `@njit` kernels.
- Contains: `Parameters`, `Simulation`, `BatchSimulationWorker`, and all jitclass/`@njit` files.
- Key files: `src/parameters.py`, `src/simulate.py`, `src/batch_worker.py`, `src/atoms.py`, `src/absorption_and_emission_process.py`

**`src/interaction_wrappers/`:**
- Purpose: `@njit` helper functions for per-interaction-model physics calculations.
- Contains: One file per interaction complexity level plus `common.py` (shared formulas).
- Key files: `src/interaction_wrappers/common.py`

**`GUI/`:**
- Purpose: PyQt5 user interface. Does not contain physics logic.
- Contains: Window hierarchy, model classes, worker threads, settings widgets, dialogs.
- Key files: `GUI/main_window.py`, `GUI/widgets/features/simulation_cockpit.py`, `GUI/models/file_model.py`

**`GUI/models/`:**
- Purpose: Data model classes (Qt model/view separation). Currently only `FileModel`.

**`GUI/workers/`:**
- Purpose: QThread subclasses that run blocking work off the GUI thread. Currently only `OvenWorker`.

**`GUI/widgets/common/`:**
- Purpose: Low-level reusable widgets with no simulation-specific logic (tables, vector inputs).

**`GUI/widgets/dialogs/`:**
- Purpose: Modal popup dialogs for editing, validation, and defaults.

**`GUI/widgets/features/`:**
- Purpose: Full composite feature tabs assembled from common widgets and tabs.

**`GUI/widgets/tabs/`:**
- Purpose: Settings tabs that map to top-level JSON sections (`Simulation`, `Atoms`, `Lasers`, `Magnetic_Fields`, `Boundaries`).

**`GUI/schema/`:**
- Purpose: JSON schema file used by both `Parameters` (validation) and the `SimulationCockpit` (live validation in the GUI).
- Key files: `GUI/schema/schema_v1.json`

**`GUI/defaults/`:**
- Purpose: Pre-filled JSON fragments loaded when user adds a new laser or magnet in the GUI.

**`util/`:**
- Purpose: Shared infrastructure — type-hint dummy classes, geometry helpers, post-processing analysis scripts.
- Key files: `util/simulation_typing.py` (critical: all `@njit` signatures import from here)

**`tests/`:**
- Purpose: Pytest test suite for physics calculations.

**`setup parameters/`:**
- Purpose: Reference JSON configuration files for known experimental setups. Used directly with `--files`.

**`simulation_results/`:**
- Purpose: Runtime output directory. Created automatically; not committed. Structure: `DD_MM_YY_N/run_N/result.csv` + `config.json`.

## Key File Locations

**Entry Points:**
- `main.py`: CLI argument parsing; routes to `start_gui()` or direct `BatchSimulationWorker.run()`.
- `GUI/main_window.py`: Top-level Qt window; constructed by `start_gui()`.

**Configuration:**
- `GUI/schema/schema_v1.json`: JSON schema for all parameter files.
- `GUI/defaults/lasers/`, `GUI/defaults/magnets/`: Default presets.
- `setup parameters/`: Example/reference parameter files.

**Core Physics:**
- `src/parameters.py`: JSON → `Simulation` factory.
- `src/simulate.py`: Step loop + atom lifecycle.
- `src/absorption_and_emission_process.py`: `@njit(parallel=True)` kernel — the computational hot path.
- `src/atoms.py`: `Li6` jitclass — all atom state.
- `src/interactions.py`: All four interaction models.
- `src/magnetic_field.py`: All four field models.

**Type Hints:**
- `util/simulation_typing.py`: Python dummy classes mirroring every jitclass interface.

**Output I/O:**
- `src/batch_worker.py`: Buffered CSV writing; `write_step_results()`.
- `src/checkpoint.py`: `.npz`/`.json` checkpoint save/load.

## Naming Conventions

**Files:**
- `snake_case.py` throughout.
- Jitclass modules named after the physical concept: `atoms.py`, `lasers.py`, `magnetic_field.py`, `interactions.py`.
- Wrapper modules follow pattern `<model>_wrappers.py`.

**Directories:**
- `src/` — simulation code; `GUI/` — interface; `util/` — shared utilities; `tests/` — test suite.
- GUI subdirs use noun categories: `models/`, `workers/`, `widgets/common/`, `widgets/dialogs/`, `widgets/features/`, `widgets/tabs/`.

**Classes:**
- Jitclasses: descriptive noun phrases, e.g. `Li6`, `LaserComponent`, `IdealQuadrupoleField`, `Lithium18LevelInteraction`.
- Qt widgets: suffixed with widget type, e.g. `MainWindow`, `SimulationCockpit`, `SettingsTabsWidget`, `FileTableWidget`.

**JSON config keys:**
- Top-level sections: `PascalCase` (`Simulation`, `Atoms`, `Lasers`, `Magnetic_Fields`, `Boundaries`).
- Field keys: `snake_case`.

## Where to Add New Code

**New interaction model:**
- Implement jitclass in `src/interactions.py` following the `Lithium6LevelInteraction` pattern.
- Add `@njit` helpers in a new `src/interaction_wrappers/<name>_wrappers.py`.
- The name string must match `getattr(interactions, name)` — no registration needed.
- Add tests to `tests/`.

**New magnetic field model:**
- Implement jitclass in `src/magnetic_field.py`.
- Add the type string to `src/parameters.py:_parse_magnetic_fields()` and to `GUI/schema/schema_v1.json`.
- Add a GUI control section in `GUI/widgets/tabs/magnetic_field_tab.py`.

**New simulation parameter:**
1. Add parsing in `src/parameters.py:_parse_<section>()`.
2. Update `GUI/schema/schema_v1.json`.
3. Thread it through `Parameters.build_simulation()` into the appropriate jitclass constructor.
4. Add GUI control in `GUI/widgets/tabs/<section>_tab.py`.

**New GUI feature tab:**
- Implement in `GUI/widgets/features/<name>.py` as a `QWidget` subclass.
- Register in `GUI/main_window.py` with `mainTabWidget.addTab(...)`.

**New reusable GUI widget:**
- Implement in `GUI/widgets/common/<name>.py`.

**New dialog:**
- Implement in `GUI/widgets/dialogs/<name>.py`.

**New analysis / post-processing utility:**
- Add to `util/`.

**Tests:**
- Place in `tests/test_<module>.py`.

## Special Directories

**`src/__pycache__/` and `src/interaction_wrappers/__pycache__/`:**
- Purpose: Python bytecode cache and Numba JIT artifact cache (`.nbc`, `.nbi` files).
- Generated: Yes (by Python/Numba at runtime).
- Committed: Yes (Numba cache is committed to speed up CI/first-run; `.nbc`/`.nbi` files are present in git).

**`simulation_results/`:**
- Purpose: Runtime output from all batch runs.
- Generated: Yes (at runtime by `BatchSimulationWorker`).
- Committed: No (in `.gitignore`).

**`prototype/`:**
- Purpose: Experimental scratch area.
- Generated: No.
- Committed: Yes (empty `__init__.py` only).

---

*Structure analysis: 2026-06-02*

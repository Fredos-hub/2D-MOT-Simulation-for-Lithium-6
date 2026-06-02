# Coding Conventions

**Analysis Date:** 2026-06-02

## Naming Patterns

**Files:**
- `src/` modules: `snake_case.py` — e.g., `absorption_and_emission_process.py`, `magnetic_field.py`
- GUI files: `snake_case.py` — e.g., `main_window.py`, `settings_tab_base.py`
- Wrapper files: `<model>_level_wrappers.py` — e.g., `six_level_wrappers.py`, `eighteen_level_wrappers.py`
- Tests: `test_<topic>.py` — e.g., `test_18level_excitation_rates.py`

**Functions (Python layer):**
- Methods and module-level functions: `snake_case` — e.g., `build_simulation`, `validate_against_schema`
- Private/internal methods: leading underscore — e.g., `_parse_simulation`, `_call_status`, `_flatten_error`

**Functions (Numba layer):**
- `@njit` module-level helpers: `snake_case` — e.g., `absorption_and_emission_default_timestep`, `beam_intensity_at_position`
- Private `@njit` helpers: leading underscore — e.g., `_advance_with_gravity`, `_calculate_excitation_rate`

**Classes:**
- `PascalCase` for all classes: `Lithium6LevelInteraction`, `IdealQuadrupoleField`, `BatchSimulationWorker`, `FileModel`
- Jitclass spec arrays: `<classname_snake>_spec` — e.g., `six_level_spec`, `quadrupole_spec`, `atom_spec`

**Constants:**
- Module-level physical/numeric constants: `UPPER_SNAKE_CASE` — e.g., `BOHR_MAGNETON`, `ATOMIC_MASS`, `NATURAL_LINEWIDTH`
- Private threshold constants: leading underscore — e.g., `_B_FINE`, `_B_MID`, `_B_COARSE` in `src/magnetic_field.py`

**Variables:**
- Local and instance variables: `snake_case`
- PyQt signals: `camelCase` matching Qt convention — e.g., `progressChanged`, `statusChanged`, `dirtyChanged`

## Numba JIT Conventions

**`@jitclass` pattern:**

Every jitclass must be preceded by a `_spec` list that explicitly types all attributes using Numba type tokens. The spec is defined at module level before the class:

```python
atom_spec = [
    ('n', int32),
    ('mass', float64),
    ('positions', float64[:, :]),
    ('status', int32[:]),
]

@jitclass(atom_spec)
class Li6:
    ...
```

Used in: `src/atoms.py`, `src/interactions.py`, `src/magnetic_field.py`, `src/lasers.py`

**`@njit` decorators — flags used:**
- `@njit` (plain): most helper functions — e.g., in `src/interaction_wrappers/common.py`, `util/geometry.py`
- `@njit(parallel=True)`: the main physics kernel only — `absorption_and_emission_default_timestep` in `src/absorption_and_emission_process.py`; use `prange` and `get_thread_id()` inside
- `@njit(inline='always')`: small inner-loop helpers called from parallel code — e.g., `_advance_with_gravity`, `calculate_handedness_to_polarization` in `src/absorption_and_emission_process.py`
- `@njit(cache=True)`: used selectively for expensive pure-math helpers where re-compilation cost is significant — e.g., `src/magnetic_field.py` line 530; `src/interaction_wrappers/eighteen_level_wrappers.py` uses `cache=True` throughout

**JIT boundary rule — what may cross into `@njit` / `@jitclass` methods:**
- Any `@jitclass` instance (`Li6`, `LaserComponent`, any `MagneticField`, any `LightAtomInteraction`)
- `np.ndarray` — must be C-contiguous, typed `float64` or `int32`/`int64`
- NumPy scalars, Python `int`, Python `float`
- `None` is NOT allowed

Do NOT pass: Python dicts, lists, strings, `Optional` values, or `QThread` objects across the JIT boundary. `BatchSimulationWorker` always lives in the Python layer; atom state is serialized to NumPy arrays before anything goes to the GUI.

**Type hint proxy pattern:**

The dummy classes in `util/simulation_typing.py` (`ECSAtoms`, `MagneticField`, `ECSLasers`, `LightAtomInteraction`) are used as type annotations in `@njit` function signatures and Python code. They are never instantiated during simulation — the actual jitclasses from `src/atoms.py`, `src/magnetic_field.py`, etc. are used at runtime. This allows IDE tooltips and type checkers to see the interface without violating Numba's type system.

Example from `src/absorption_and_emission_process.py`:
```python
from util.simulation_typing import ECSLasers, ECSAtoms, MagneticField, LightAtomInteraction

@njit(parallel=True)
def absorption_and_emission_default_timestep(atom_ids: np.ndarray,
                                             simulation_atoms: ECSAtoms,
                                             lasers: ECSLasers,
                                             ...
```

**Thread-local workspace pattern (parallel kernels):**

In `absorption_and_emission_default_timestep`, per-thread scratch arrays are allocated once per call at `(nthreads, ...)` shape and indexed with `get_thread_id()`. This avoids false sharing and per-iteration allocation inside `prange`:

```python
nthreads = get_num_threads()
work_intensity = np.empty((nthreads, n_lasers), dtype=np.float64)

for idx in prange(atom_ids.size):
    tid = get_thread_id()
    intensity_at_position = work_intensity[tid]
```

## Code Style

**Formatting:**
- No formatter configuration file (no `.prettierrc`, `pyproject.toml`, or `ruff.toml` at repo root)
- Standard PEP 8 indentation (4 spaces)
- Lines kept readable; no enforced column limit found

**Linting:**
- No `.eslintrc`, `mypy.ini`, or `ruff` config detected
- Type annotations used in Python-layer code (`src/parameters.py`, `util/simulation_typing.py`) but not enforced

**Decorator comment blocks:**

Some older files use horizontal-rule comment blocks as section separators:
```python
###################################################
#           Simple 6-Level Model                  #
###################################################
```
New code (added during the frontend refactor) does not use this style. Prefer short inline comments or docstrings over decorative separators.

## Import Organization

**Order (observed pattern):**
1. Standard library (`import os`, `import math`, `from pathlib import Path`)
2. Third-party (`import numpy as np`, `from numba import njit`, `from PyQt5.QtCore import ...`)
3. Internal (`from src.interactions import ...`, `from util.simulation_typing import ...`)

**Aliases in use:**
- `import numpy as np`
- `import scipy.constants as scc`
- `import src.interaction_wrappers.eighteen_level_wrappers as elw`
- `import src.interaction_wrappers.six_level_wrappers as slw`

No path aliases or `__init__.py` re-exports for convenience imports.

## Parameter Validation

**Schema file:** `GUI/schema/schema_v1.json` (JSON Schema Draft-07)

**Top-level required sections:** `Atoms`, `Magnetic_Fields`, `Lasers`, `Boundaries`, `Simulation`

**Validation flow** (in `src/parameters.py`):
1. `validate_against_schema(config, schema_path)` — runs `Draft7Validator` from `jsonschema`, collects and deduplicates all errors, raises `SchemaValidationError(errors: List[str])` if any fail
2. `Parameters.__init__` catches `SchemaValidationError`, sets `self.valid = False`, stores errors in `self.errors`
3. After validation passes, five `_parse_*` methods populate typed attributes: `_parse_simulation`, `_parse_atoms`, `_parse_magnetic_fields`, `_parse_lasers`, `_parse_boundaries`
4. `build_simulation()` constructs all jitclass objects from parsed attributes

**Custom exceptions:**
- `SchemaValidationError(errors: List[str])` — schema validation failures
- `ParameterError` — post-validation construction errors

**GUI validation entry point:** `GUI/widgets/dialogs/validation_dialog.py`

## Error Handling

**Python layer (non-Numba):**
- Custom exception classes for domain errors: `SchemaValidationError`, `ParameterError` in `src/parameters.py`; `ValueError` and `RuntimeError` for precondition violations in `src/atoms.py`, `src/distributions.py`, `src/maxwell_boltzmann_sampler.py`
- `BatchSimulationWorker` (`src/batch_worker.py`) wraps most operations in broad `except Exception as e` and emits errors via `statusChanged` signal to the GUI — avoids crashing the QThread
- `src/simulate.py` raises bare `Exception("Canceled during warmup.")` for cancellation — this should be a custom type but currently is not

**Numba layer:**
- `@njit` functions do not raise Python exceptions; they return values or silently proceed. Validation is always done in the Python layer before crossing the JIT boundary.

## Logging

**Framework:** `print()` only — no `logging` module

**Patterns:**
- `Simulation.warmup` and `Simulation.step` use `print()` for progress messages (e.g., `"Warmup step completed."`, `"No atoms live, simulation stopping."`)
- `BatchSimulationWorker` uses `self.statusChanged.emit(msg)` to surface messages to the GUI — the Qt signal carries the same strings that would otherwise be logged
- `Parameters._call_status(msg)` calls an optional callback (set to `statusChanged.emit` in batch mode) for validation/parse progress messages

## Comments and Docstrings

**Module docstrings:** present in well-maintained modules — `src/parameters.py` has a full module docstring with Purpose/Usage sections; `src/atoms.py` and `src/distributions.py` have class-level docstrings. Older interaction/wrapper modules have minimal or no module docstrings.

**Method docstrings:** NumPy-style (`Parameters/Raises/Returns` sections) used in `src/parameters.py` and `src/atoms.py`. Most `@njit` functions have a single-line or no docstring (Numba cannot use full docstring introspection).

**Inline comments:** Used liberally in the physics kernel (`src/absorption_and_emission_process.py`) to explain optimizations — e.g., why thread-local workspaces are used, why Zeeman shifts are precomputed outside the laser loop.

**Commented-out code:** Present in `src/atoms.py` (`#if np.all(velocities == 0):` block) and in `src/parameters.py` (some legacy sections). This is acceptable for conditional-compile stubs near the JIT boundary but otherwise should be removed.

## Function Design

**Size:** `@njit` helpers are kept small and single-purpose (10–40 lines typical). The main parallel kernel `absorption_and_emission_default_timestep` is large (~270 lines) by necessity — splitting it would add overhead across the JIT boundary.

**Parameters:** Jitclass methods use positional keyword arguments matching the interface defined in `util/simulation_typing.py`. Python-layer methods use `Optional[...]` type hints with default `None`.

**Return values:** `@njit` functions that update atom state do so in-place on the passed arrays (no return value, or return a plain scalar). Python-layer builders return the constructed object.

## Module Design

**Exports:** No `__all__` declarations. Public interface is implicit.

**Barrel files:** `GUI/widgets/common/__init__.py`, `GUI/widgets/tabs/__init__.py`, etc. exist but are empty — no re-exports.

**Wrapper module pattern:** Each interaction model has a dedicated wrapper module under `src/interaction_wrappers/` containing only `@njit` helpers. Shared formulas live in `src/interaction_wrappers/common.py` — do not duplicate `_calculate_excitation_rate` or `_calculate_saturation_parameter` into a new wrapper.

---

*Convention analysis: 2026-06-02*

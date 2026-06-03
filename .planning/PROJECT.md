# 2D MOT Simulation for Lithium-6

## What This Is

A Monte-Carlo simulation of Lithium-6 atom trajectories through a 2D magneto-optical
trap (MOT) and push-beam region. It models per-atom absorption/emission, Doppler and
Zeeman shifts, gravity, and configurable laser/magnetic-field geometries via a
Numba-JIT physics core, driven from a PyQt5 desktop GUI or a headless CLI. It is a
research tool used to compare simulated atom distributions, push-beam velocities, and
spectra against experimental data.

## Core Value

The simulation produces physically trustworthy Li-6 atom trajectories that can be
relied on when comparing against experimental push-beam and spectrum measurements.

## Requirements

### Validated

<!-- Inferred from existing codebase (see .planning/codebase/). -->

- ✓ JSON-configured simulations validated against a Draft-07 schema — existing
- ✓ Two run modes: GUI (`--GUI`) and headless batch CLI (`--files`) — existing
- ✓ Numba `@njit(parallel=True)` physics kernel (absorption/emission, Doppler, Zeeman, gravity, Monte-Carlo) — existing
- ✓ Selectable interaction models by name (6-level, 18-level, simple-18-level, 4-level) — existing
- ✓ Selectable magnetic-field models (ideal quadrupole, Zeeman, elliptical, dipole-bar) — existing
- ✓ Per-step, per-atom CSV output plus a copy of the config used — existing
- ✓ Maxwell-Boltzmann oven sample generation — existing
- ✓ Parameter-sweep / incrementor batch generation — existing
- ✓ Spectrum and sample-generator GUI tabs with embedded matplotlib — existing
- ✓ Restructured GUI module layout (`models/`, `workers/`, `widgets/{common,dialogs,features,tabs}/`) — existing, uncommitted
- ✓ Checkpoint save/restore module (`src/checkpoint.py`) — existing, not yet integrated

### Active

<!-- This milestone. Strictly ordered: GUI → analysis → new model. -->

**GUI refactor (finish)**
- [ ] Commit the GUI restructuring and verify the app launches with all tabs functional after the move
- [ ] Integrate `src/checkpoint.py` into `BatchSimulationWorker` (periodic save) and expose a Resume action in the simulation cockpit

**Analysis backlog (`todo.md`)**
- [ ] Pushbeam detuning re-run with boundary `z_limit = 20 cm`
- [ ] Time-of-flight (ToF) histograms
- [ ] Velocity distributions at the spec-beam position (z = 19 cm)
- [ ] Spec-beam-only spectrum run started from the post-pushbeam state

**New physics model (feasibility)**
- [ ] Prototype a solver/diagonalizer for the combined interaction Hamiltonian as a new selectable interaction model (interpolation model kept alongside)
- [ ] Compare diagonalizer output against the interpolation model and decide whether interpolation can eventually be retired

### Out of Scope

- Plotting tab implementation — on ice; analysis currently done via `util/` scripts
- s-wave atom-atom scattering implementation — future milestone; partially depends on the diagonalizer machinery
- Light-assisted collisions and reabsorption / radiation trapping — future atom-atom models
- Outright removal of the interpolation model — deferred until the diagonalizer is validated (decision is an outcome of this milestone, not a deliverable)
- Fixing known interpolation-model defects (swapped GS2/GS3, silent-zero for |B| > 0.1 T) and the Zeeman-boundary FIXME — not in scope unless surfaced as blockers by the diagonalizer comparison

## Context

- Brownfield project. Active branch `fs/chore/frontend` carries a large, structurally-complete-but-uncommitted GUI refactor: the old flat `GUI/widgets/*.py` files are deleted and replaced by a four-subdirectory layout (`common/`, `dialogs/`, `features/`, `tabs/`) plus `GUI/models/` and `GUI/workers/`. Import paths already reference the new locations.
- Stack: Python 3.12, Numba 0.63 (`@jitclass` + `@njit(parallel=True)`), PyQt5 5.15, NumPy 2.3, SciPy 1.17, matplotlib 3.10, pandas 3.0, jsonschema. Run always from repo root (`python -m main`).
- The current physics relies on interpolated curve-fits in `src/interaction_wrappers/eighteen_level_wrappers.py` (~1085 lines). The codebase audit flagged a live correctness defect there (`#FIXME: Swapped GS2 and GS3`, line 817) and silent-zero coverage gaps for |B| > 0.1 T — both directly relevant to the diagonalizer comparison, where the diagonalizer may be the more trustworthy reference.
- Test coverage is thin: `tests/` covers only 18-level excitation rates and polarization geometry; `parameters.py`, `simulate.py`, `magnetic_field.py`, and most interaction models are untested. No GUI tests.
- `todo.md` is the cross-session analysis backlog; analysis utilities live in `util/` (`pushbeam_detuning_histograms.py`, `integrated_spectrum.py`, `trapped_atoms_heatmap.py`).

## Constraints

- **Tech stack**: New interaction model must respect the Numba JIT boundary — only jitclass instances and C-contiguous typed NumPy arrays (and scalars) may cross into `@njit` code; never `None` or arbitrary Python objects.
- **Performance**: The physics kernel is the hot path. A per-step Hamiltonian diagonalization could be far costlier than interpolation table lookups; feasibility must weigh compute cost against accuracy.
- **Compatibility**: Selectable-model convention — a new interaction model is added to `src/interactions.py` and chosen by name string in JSON (`getattr(interactions, name)()`), with `@njit` helpers under `src/interaction_wrappers/`.
- **Numba**: First-run JIT compilation takes 1–3 minutes; cache lives in `src/__pycache__/`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Milestone strictly ordered GUI → analysis → new model | User-chosen sequencing; GUI must be stable before research work resumes | — Pending |
| Plotting tab deferred ("on ice") | Not core to current research needs; analysis handled by `util/` scripts | — Pending |
| Diagonalizer added as a new selectable model, interpolation kept | De-risk — validate the new model before retiring the fragile curve-fits | — Pending |
| Diagonalizer scoped as a feasibility spike this milestone | s-wave scattering only partially depends on it; decide before committing to full replacement | — Pending |
| Checkpoint integration included in the GUI-finish phase | `src/checkpoint.py` already exists but is dead code; wiring it enables resuming long runs | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-03 after initialization*

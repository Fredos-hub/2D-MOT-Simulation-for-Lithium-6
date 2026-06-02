# Technology Stack

**Analysis Date:** 2026-06-02

## Languages

**Primary:**
- Python 3.12 - All simulation, GUI, and utility code

**Secondary:**
- JSON (Draft-07 schema) - Configuration and parameter files (`GUI/schema/schema_v1.json`, `setup parameters/*.json`)

## Runtime

**Environment:**
- CPython 3.12.3 (venv at `.venv/`)

**Package Manager:**
- pip (venv-managed)
- Lockfile: `requirements.txt` (unpinned — no versions locked except `jsonschema>=4.0.0`)

## Frameworks

**GUI:**
- PyQt5 5.15.11 — Main application window, settings tabs, workers, signals/slots
- pyqtdarktheme 0.1.7 — Light/dark stylesheet (`qdarktheme.load_stylesheet(...)` in `main.py`)

**JIT Compilation:**
- Numba 0.63.1 — `@jitclass` for all physics objects; `@njit(parallel=True)` for the core physics kernel; `@njit` for helper math functions
  - `numba.experimental.jitclass` used in: `src/atoms.py`, `src/lasers.py`, `src/magnetic_field.py`, `src/interactions.py`
  - `@njit(parallel=True)` + `prange`: `src/absorption_and_emission_process.py` (the hot loop)
  - `@njit(cache=True)`: `src/magnetic_field.py` (one standalone helper)

**Testing:**
- pytest 9.0.3 — Unit tests in `tests/`

**Build/Dev:**
- No build system — run directly via `python -m main`

## Key Dependencies

**Critical:**
- numpy 2.3.5 — Array backbone for all atom state; `float64`/`int32` typed arrays passed across the JIT boundary; `.npz` checkpoint serialization
- numba 0.63.1 — Entire physics layer is JIT-compiled; first-run compilation cache stored in `src/__pycache__/` (`.nbi`/`.nbc` files)
- scipy 1.17.0 — Physical constants (`scipy.constants` imported as `scc` everywhere in `src/`); `MaxwellBoltzmannSampler` uses scipy distributions
- PyQt5 5.15.11 — GUI event loop, `QThread` for `BatchSimulationWorker` and `OvenWorker`
- matplotlib 3.10.8 — Embedded Qt plots via `FigureCanvasQTAgg` (backend `backend_qt5agg`) in `GUI/widgets/features/sample_generator.py` and `GUI/widgets/features/spectrum_tab.py`; standalone analysis in `plots.py` and `util/integrated_spectrum.py`

**Infrastructure:**
- pandas 3.0.0 — Loading initial-condition CSV samples (`pd.read_csv` in `src/parameters.py`); writing atom snapshots (`pd.DataFrame.to_csv`)
- jsonschema 4.26.0 — JSON Draft-7 schema validation of parameter files (`Draft7Validator` in `src/parameters.py`)
- tqdm 4.67.3 — CLI progress bar in `main.py` (batch mode only)

## Configuration

**Environment:**
- No `.env` file or environment variable configuration found in source
- `NUMBA_CACHE_DIR` / `OMP_NUM_THREADS` etc. are not set explicitly; Numba uses its defaults
- Parallelism for `@njit(parallel=True)` uses all available CPU cores by default

**Build:**
- No build config files (`pyproject.toml`, `setup.cfg`, `setup.py` absent at repo root)
- Run always from repo root: `python -m main --GUI`

**Schema:**
- `GUI/schema/schema_v1.json` — JSON Draft-07 schema for simulation parameter files
- `GUI/defaults/lasers/`, `GUI/defaults/magnets/` — default JSON fragments loaded by GUI tabs

## Platform Requirements

**Development:**
- Linux (tested on Linux Mint; DPI scaling attributes set for Qt in `main.py`)
- Python 3.12 required (venv configured for 3.12.3)
- Numba JIT compilation: first run takes 1–3 minutes; cache persists in `src/__pycache__/`

**Production:**
- No deployment target — desktop application, run locally
- Output written to `simulation_results/DD_MM_YY_N/run_N/` relative to repo root

---

*Stack analysis: 2026-06-02*

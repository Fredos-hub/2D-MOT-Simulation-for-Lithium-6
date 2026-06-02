# Testing Patterns

**Analysis Date:** 2026-06-02

## Test Framework

**Runner:**
- pytest 9.0.3
- Config: none — no `pytest.ini`, `pyproject.toml`, or `setup.cfg` at repo root; pytest discovers tests by convention

**Assertion Library:**
- `pytest.approx` for scalar/tuple comparisons
- `numpy.testing.assert_allclose` for array comparisons (primary pattern)

**Run Commands:**
```bash
# Run all tests (from repo root, with the project venv active)
.venv/bin/python -m pytest tests/

# Verbose
.venv/bin/python -m pytest tests/ -v

# Single module
.venv/bin/python -m pytest tests/test_18level_excitation_rates.py

# Coverage (coverage package not confirmed installed; standard invocation)
.venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing
```

Note: Numba JIT compilation on first run takes 1–3 minutes. Subsequent runs use the Numba cache.

## Test File Organization

**Location:** `tests/` directory at repo root (separate from source)

**Structure:**
```
tests/
├── __init__.py
├── test_18level_excitation_rates.py   # physics correctness for Lithium18LevelInteraction
└── test_polarization_geometry.py      # geometric polarization decomposition
```

**Naming:** `test_<topic>.py`; test functions named `test_<what_is_verified>`

## Test Structure

**Suite Organization:**

Module-level constants define the test conditions so every function in a module exercises the same physics regime:

```python
GROUND_STATE = 4
B_FIELD = 0.0
DETUNING = 0.0
LASER_INTENSITY = 0.05   # W/m^2
EXPECTED_RATIO = np.array([1.0, 2.0 / 3.0, 1.0 / 3.0])
RTOL = 0.05

@pytest.fixture(scope="module")
def interaction():
    return Lithium18LevelInteraction()

def test_analytical_excitation_rate_ratio(interaction):
    rates = np.array([_total_excitation_rate(interaction, GROUND_STATE, p) for p in range(3)])
    np.testing.assert_allclose(rates / rates[0], EXPECTED_RATIO, rtol=RTOL,
                               err_msg=f"Analytical ratio mismatch: {rates / rates[0]}")
```

**Patterns:**
- `scope="module"` fixtures for expensive jitclass construction (avoids repeated JIT compilation per test)
- Helper functions prefixed `_` for shared sub-calculations (not collected by pytest)
- Parametrize with `@pytest.mark.parametrize` for symmetric cases (e.g., both stretched states, all handedness/angle combinations)

## Mocking

**Framework:** None — no `unittest.mock`, `pytest-mock`, or monkeypatching found in the test suite.

**Approach:** The tests call real `@jitclass` instances (`Lithium18LevelInteraction`, `SimpleEighteenLevelInteraction`, `Lithium6LevelInteraction`) and real `@njit` functions (`calculate_handedness_to_polarization`, `elw.calculate_transition_strength`) directly. No fakes or stubs.

**What to mock when adding tests:**
- GUI components (PyQt5 widgets) would require `QApplication` setup or mocking; currently no GUI tests exist
- `BatchSimulationWorker` I/O (file creation) should be mocked or redirected to a `tmp_path` fixture

**What NOT to mock:**
- Jitclass instances — the correctness of the compiled physics is exactly what the tests verify
- `@njit` helper functions called from the kernel

## Fixtures and Factories

**Current fixtures:**

```python
# tests/test_18level_excitation_rates.py
@pytest.fixture(scope="module")
def interaction():
    """Single Lithium18LevelInteraction instance shared across the module."""
    return Lithium18LevelInteraction()
```

Some tests in `test_18level_excitation_rates.py` construct their own instances inline (e.g., `test_zeeman_slopes_match_simple_and_full_18level_at_low_field` instantiates `SimpleEighteenLevelInteraction` and `Lithium18LevelInteraction` directly) rather than using fixtures. Both styles appear.

**Test data:** Constants are module-level (`SEED = 42`, `N_TRIALS = 200`), not loaded from files. No CSV fixtures or JSON parameter files used in tests.

**Location:** No `conftest.py` at repo root or in `tests/` — all fixtures are module-local.

## Coverage

**Requirements:** None enforced — no coverage configuration or minimum threshold.

**Actual coverage (as of 2026-06-02):** Thin. The two test modules cover:
- `src/interactions.py` — `Lithium18LevelInteraction`, `SimpleEighteenLevelInteraction`, `Lithium6LevelInteraction` methods: `calculate_saturation_parameter`, `calculate_transition_frequency_shift`, `calculate_branching_ratio`
- `src/interaction_wrappers/eighteen_level_wrappers.py` — `calculate_transition_strength`
- `src/absorption_and_emission_process.py` — `calculate_handedness_to_polarization` only

**Completely untested:**
- `src/simulate.py` — `Simulation` class, full step loop
- `src/batch_worker.py` — `BatchSimulationWorker` (all I/O, CSV buffering, run lifecycle)
- `src/parameters.py` — `Parameters`, `validate_against_schema`, `build_simulation`
- `src/atoms.py` — `Li6.set_starting_conditions`
- `src/magnetic_field.py` — all four field classes
- `src/distributions.py`, `src/maxwell_boltzmann_sampler.py`
- `GUI/` — all widgets, models, workers
- `util/geometry.py`

## Test Types

**Unit Tests:**
- Both existing test files are unit tests: they isolate a single function or jitclass method, control all inputs, and assert on the output.
- Physics invariants tested: dipole sum rules, Lorentzian detuning profile, linearity in intensity, sum-over-decay-channels, Zeeman slope convergence.
- Stochastic test (`test_monte_carlo_excitation_count_ratio`): fixed seed (`np.random.default_rng(42)`), N=200 trials, `rtol=0.05` — loose enough to pass reliably.

**Integration Tests:** None

**E2E Tests:** Not used

## Common Patterns

**Physics invariant testing:**
```python
# Verify a sum rule holds across all states
sums = np.array([
    sum(elw.calculate_transition_strength(gs, ex, pol, B_FIELD)
        for ex in range(12) for pol in range(3))
    for gs in range(6)
])
np.testing.assert_allclose(sums, 0.5, rtol=0.02,
    err_msg=f"per-gs sums: {sums}")
```

**Parametrized geometric limit tests:**
```python
@pytest.mark.parametrize("handedness", [-1, 0, 1])
@pytest.mark.parametrize("angle", np.linspace(0.0, math.pi, 21))
def test_components_sum_to_one(angle, handedness):
    sq_minus, sq_pi, sq_plus = calculate_handedness_to_polarization(angle, handedness)
    assert sq_minus + sq_pi + sq_plus == pytest.approx(1.0, abs=1e-12)
```

**Comparing two interaction models (cross-model consistency):**
```python
def test_zeeman_slopes_match_simple_and_full_18level_at_low_field():
    simple = SimpleEighteenLevelInteraction()
    full = Lithium18LevelInteraction()
    # finite-difference slope comparison at small B
    slope_full = (full.calculate_transition_frequency_shift(..., B_small)
                  - full.calculate_transition_frequency_shift(..., 0.0)) / B_small
    np.testing.assert_allclose(slope_full, slope_simple, rtol=0.05, ...)
```

**Error message pattern:** Always pass `err_msg=` to `assert_allclose` with the actual computed values so failures are self-describing.

---

*Testing analysis: 2026-06-02*

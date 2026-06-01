"""
Sanity-check for Lithium18LevelInteraction excitation rates at |B|=0.

For an atom pinned in ground state 4 (|F=3/2, mF=-3/2>) and resonant light
(detuning = 0, doppler = 0) the per-polarization total excitation rate must
follow the dipole sum rule

    sigma-  :  pi  :  sigma+   =   3 : 2 : 1

i.e. 1 : 2/3 : 1/3 normalized. This holds when s_total << 1 so the
saturation denominator doesn't compress the ratio.
"""
import math
import numpy as np
import pytest

from src.interactions import Lithium18LevelInteraction
from src.interaction_wrappers import eighteen_level_wrappers as elw


# --- Li-6 D2 line constants (only the linewidth enters the rate formula) ---
NATURAL_LINEWIDTH = 2 * math.pi * 5.8724e6      # rad/s
TRANSITION_FREQUENCY = 446.799677e12            # Hz
SATURATION_INTENSITY = 25.4                     # W/m^2 (API parity; unused in formula)

# --- Test conditions ---
GROUND_STATE = 4
B_FIELD = 0.0
DETUNING = 0.0
DOPPLER = 0.0
LASER_INTENSITY = 0.05      # W/m^2 -- keeps max per-transition s ~ 1e-3
INTERVAL_S = 1.0e-3         # 1 ms per trial
N_TRIALS = 200              # repeat to tighten Poisson statistics
SEED = 42

EXPECTED_RATIO = np.array([1.0, 2.0 / 3.0, 1.0 / 3.0])      # sigma-, pi, sigma+  (3 : 2 : 1 sum)
RTOL = 0.05


def _total_excitation_rate(interaction, gs: int, driving_pol: int) -> float:
    """Sum 0.5*s*Gamma/(1+s_total) over all excited states, mirroring common._calculate_excitation_rate."""
    n_ex = interaction.number_of_excited_states
    sats = np.zeros(n_ex)
    for ex in range(n_ex):
        sats[ex] = interaction.calculate_saturation_parameter(
            driving_pol,          # polarization
            B_FIELD,              # magnetic_field_strength
            gs,                   # ground_state
            ex,                   # excited_state
            LASER_INTENSITY,      # laser_intensity
            NATURAL_LINEWIDTH,    # natural_linewidth
            SATURATION_INTENSITY, # saturation_intensity
            TRANSITION_FREQUENCY, # effective_transition_frequency
            DOPPLER,              # doppler_shift
            TRANSITION_FREQUENCY, # laser_beam_frequency
            DETUNING,             # detuning
        )
    s_total = sats.sum()
    return float(np.sum(0.5 * sats * NATURAL_LINEWIDTH / (1.0 + s_total)))


def _monte_carlo_count(rate: float, interval: float, rng: np.random.Generator) -> int:
    """Count Poisson events in `interval` by sampling t_event = -log(r)/rate, like the kernel."""
    count = 0
    t = 0.0
    while True:
        t += -math.log(rng.random()) / rate
        if t > interval:
            return count
        count += 1


@pytest.fixture(scope="module")
def interaction():
    return Lithium18LevelInteraction()


def test_analytical_excitation_rate_ratio(interaction):
    rates = np.array([_total_excitation_rate(interaction, GROUND_STATE, p) for p in range(3)])
    np.testing.assert_allclose(rates / rates[0], EXPECTED_RATIO, rtol=RTOL,
                               err_msg=f"Analytical ratio mismatch: {rates / rates[0]}")


def test_monte_carlo_excitation_count_ratio(interaction):
    rng = np.random.default_rng(SEED)
    rates = [_total_excitation_rate(interaction, GROUND_STATE, p) for p in range(3)]
    counts = np.array([
        sum(_monte_carlo_count(r, INTERVAL_S, rng) for _ in range(N_TRIALS))
        for r in rates
    ], dtype=np.float64)
    np.testing.assert_allclose(counts / counts[0], EXPECTED_RATIO, rtol=RTOL,
                               err_msg=f"MC ratio mismatch: {counts / counts[0]}")


# --- Test 1: linearity in intensity --------------------------------------
def test_saturation_parameter_linear_in_intensity(interaction):
    """At s << 1, saturation parameter is linear in laser intensity."""
    intensities = [1e-4, 2e-4, 4e-4]            # W/m^2 -- well below saturation
    sats = [
        interaction.calculate_saturation_parameter(
            0, B_FIELD, GROUND_STATE, 10,       # sigma-, gs=4, ex=10 (strongest)
            I, NATURAL_LINEWIDTH, SATURATION_INTENSITY,
            TRANSITION_FREQUENCY, DOPPLER, TRANSITION_FREQUENCY, DETUNING,
        )
        for I in intensities
    ]
    assert sats[1] / sats[0] == pytest.approx(2.0, rel=1e-9)
    assert sats[2] / sats[0] == pytest.approx(4.0, rel=1e-9)


# --- Test 2: Lorentzian detuning profile ---------------------------------
def test_lorentzian_detuning_profile(interaction):
    """s(d) / s(0) = 1 / (1 + (2d/Gamma)^2)."""
    detunings = NATURAL_LINEWIDTH * np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    sats = np.array([
        interaction.calculate_saturation_parameter(
            0, B_FIELD, GROUND_STATE, 10,
            LASER_INTENSITY, NATURAL_LINEWIDTH, SATURATION_INTENSITY,
            TRANSITION_FREQUENCY, DOPPLER, TRANSITION_FREQUENCY, d,
        )
        for d in detunings
    ])
    expected = 1.0 / (1.0 + (2.0 * detunings / NATURAL_LINEWIDTH) ** 2)
    np.testing.assert_allclose(sats / sats[3], expected, rtol=1e-6)


# --- Test 8: every excited state has decay channels + total invariant ----
def test_every_excited_state_has_decay_channels(interaction):
    """No excited state may have all decay channels zero at B=0 (catches a zeroed row)."""
    for ex in range(interaction.number_of_excited_states):
        total = sum(
            interaction.calculate_branching_ratio(pol, gs, ex, B_FIELD)
            for gs in range(interaction.number_of_ground_states)
            for pol in range(3)
        )
        assert total > 0.0, f"Excited state {ex} has zero total decay rate at B=0"


def test_18level_per_ground_state_transition_strength_sum_is_half():
    """Excitation sum rule: Sum_{ex, pol} |CG|^2 = 0.5 for every ground state."""
    sums = np.array([
        sum(elw.calculate_transition_strength(gs, ex, pol, B_FIELD)
            for ex in range(12) for pol in range(3))
        for gs in range(6)
    ])
    np.testing.assert_allclose(sums, 0.5, rtol=0.02,
        err_msg=f"per-gs sums: {sums}")


def test_18level_per_excited_state_transition_strength_sum_is_quarter():
    """Deexcitation sum rule: Sum_{gs, pol} |CG|^2 = 0.25 for every excited state."""
    sums = np.array([
        sum(elw.calculate_transition_strength(gs, ex, pol, B_FIELD)
            for gs in range(6) for pol in range(3))
        for ex in range(12)
    ])
    np.testing.assert_allclose(sums, 0.25, rtol=0.02,
        err_msg=f"per-ex sums: {sums}")


def test_zeeman_slopes_match_simple_and_full_18level_at_low_field():
    """At small B (~10 G), d(transition_frequency_shift)/dB should match between
    SimpleEighteenLevelInteraction (pure Zeeman) and Lithium18LevelInteraction
    (HFS + Zeeman). The HFS offset is constant in B and cancels in the slope."""
    from src.interactions import SimpleEighteenLevelInteraction
    simple = SimpleEighteenLevelInteraction()
    full = Lithium18LevelInteraction()

    B_small = 1e-5  # 0.1 Gauss -- truly in the linear-Zeeman regime (E_Zeeman << E_HFS)
    for gs in range(6):
        for ex in range(12):
            for pol in range(3):
                # only allowed transitions (nonzero strength near B=0)
                if elw.calculate_transition_strength(gs, ex, pol, B_small) < 1e-6:
                    continue
                slope_full = (full.calculate_transition_frequency_shift(pol, gs, ex, B_small)
                              - full.calculate_transition_frequency_shift(pol, gs, ex, 0.0)) / B_small
                slope_simple = (simple.calculate_transition_frequency_shift(pol, gs, ex, B_small)
                                - simple.calculate_transition_frequency_shift(pol, gs, ex, 0.0)) / B_small
                # skip transitions with zero Zeeman slope (m_e g_e == m_g g_g)
                if abs(slope_simple) < 1e6:
                    continue
                np.testing.assert_allclose(slope_full, slope_simple, rtol=0.05,
                    err_msg=f"(gs={gs}, ex={ex}, pol={pol}): "
                            f"full={slope_full:.3e}, simple={slope_simple:.3e}")


def test_18level_zeeman_slopes_converge_to_6level_at_high_field():
    """In the Paschen-Back regime (B ~ 0.5 T, Zeeman >> HFS) the set of unique
    Zeeman slopes per polarization for allowed 18-level transitions must coincide
    with the 6-level slopes -- only m_J matters once HFS is perturbative."""
    from src.interactions import Lithium6LevelInteraction
    six = Lithium6LevelInteraction()
    full18 = Lithium18LevelInteraction()

    B_high = 0.5    # Tesla
    dB = 1e-4

    def slope(interaction, gs, ex, pol):
        return (interaction.calculate_transition_frequency_shift(pol, gs, ex, B_high + dB)
                - interaction.calculate_transition_frequency_shift(pol, gs, ex, B_high)) / dB

    def collect(slopes_list, tol):
        """De-duplicate slopes within `tol` (rad/s/T) to get the unique set."""
        unique = []
        for s in slopes_list:
            if not any(abs(s - u) < tol for u in unique):
                unique.append(s)
        return sorted(unique)

    for pol in range(3):
        six_slopes = []
        for gs in range(2):
            for ex in range(4):
                if six.calculate_branching_ratio(pol, gs, ex, B_high) > 1e-6:
                    six_slopes.append(slope(six, gs, ex, pol))

        full_slopes = []
        for gs in range(6):
            for ex in range(12):
                if elw.calculate_transition_strength(gs, ex, pol, B_high) < 1e-6:
                    continue
                full_slopes.append(slope(full18, gs, ex, pol))

        if not six_slopes or not full_slopes:
            continue

        tol = 0.05 * max(abs(s) for s in six_slopes)
        six_unique = collect(six_slopes, tol)
        full_unique = collect(full_slopes, tol)

        np.testing.assert_allclose(full_unique, six_unique, rtol=0.05, atol=tol,
            err_msg=f"pol={pol}: 18-level unique slopes {full_unique} "
                    f"!= 6-level slopes {six_unique}")


# --- Test 9: 3:2:1 sum rule for both stretched states --------------------
@pytest.mark.parametrize("gs, expected_ratio", [
    (4, (1.0, 2.0 / 3.0, 1.0 / 3.0)),   # |F=3/2, mF=-3/2>, stretched
    (5, (1.0 / 3.0, 2.0 / 3.0, 1.0)),   # |F=3/2, mF=+3/2>, mirror-stretched
])
def test_stretched_state_dipole_ratio(gs, expected_ratio):
    """Both stretched F=3/2 ground states obey the 3:2:1 dipole sum ratio (or its mirror)."""
    sums = np.array([
        sum(elw.calculate_transition_strength(gs, ex, pol, B_FIELD) for ex in range(12))
        for pol in range(3)
    ])
    norm = sums / sums.max()
    np.testing.assert_allclose(norm, expected_ratio, rtol=RTOL,
        err_msg=f"gs={gs} dipole ratio mismatch: {norm}")

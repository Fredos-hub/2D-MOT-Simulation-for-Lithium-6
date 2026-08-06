"""Wrapped @njit functions for the absorption/emission physics kernel."""

import math

import numpy as np
import scipy.constants as scc
from numba import get_num_threads, get_thread_id, njit, prange

from util.simulation_typing import (
    ECSAtoms,
    ECSLasers,
    LightAtomInteraction,
    MagneticField,
)

# Local intensities below this are treated as dark. Worst case (resonant,
# two-level) the discarded scattering rate is (Γ/2)·I/I_sat ≈ 0.7 s⁻¹,
# i.e. ≪ 1 event over any realistic flight time.
MIN_INTENSITY = 1e-6  # W/m²


@njit(inline="always")
def _advance_with_gravity(positions, velocities, atom_id, vel, dt):
    """Translate one atom by vel·dt and apply the gravity kick to v_y."""
    positions[atom_id, 0] += vel[0] * dt
    positions[atom_id, 1] += vel[1] * dt
    positions[atom_id, 2] += vel[2] * dt
    velocities[atom_id, 1] -= scc.g * dt


@njit(parallel=True)
def absorption_and_emission_default_timestep(
    atom_ids: np.ndarray,
    simulation_atoms: ECSAtoms,
    lasers: ECSLasers,
    simulation_interaction: LightAtomInteraction,
    magnetic_field: MagneticField,
    excitation_counter: np.ndarray,
    default_timestep: float,
    excitation_hist: np.ndarray = None,
    substep_counter: np.ndarray = None,
) -> None:
    """Advance the given atoms over one default timestep (core physics kernel).

    For each alive atom this samples scattering events, applies recoil, and
    advances position/clock until the default timestep is consumed. Mutates
    ``simulation_atoms`` and ``excitation_counter`` in place.

    Parameters
    ----------
    atom_ids : ndarray
        Indices of the atoms to process.
    simulation_atoms : ECSAtoms
        Atom container; positions/velocities/clocks updated in place.
    lasers : ECSLasers
        Laser beams.
    simulation_interaction : LightAtomInteraction
        Interaction model providing rates and branching ratios.
    magnetic_field : MagneticField
        Field model used for Zeeman shifts and adaptive stepping.
    excitation_counter : ndarray
        Per-atom scattering-event counter, incremented in place.
    default_timestep : float
        Length of the step, in s.
    excitation_hist : ndarray, optional
        Optional per-event histogram buffer.
    substep_counter : ndarray, optional
        Per-thread ``int64`` cells (length >= n_threads). Each inner while-loop
        iteration (one substep) increments ``substep_counter[tid]``. Lightweight
        benchmark instrumentation (plan 04-06): purely additive, touches no
        physics branch; the total is reduced (summed) in the Python driver.
    """
    n_lasers = lasers.n_lasers
    laser_handedness = lasers.handedness
    wave_vectors = lasers.wave_vectors
    transition_frequency = simulation_atoms.transition_frequency
    natural_linewidth = simulation_atoms.natural_linewidth
    atom_mass = simulation_atoms.mass
    saturation_intensity = simulation_atoms.saturation_intensity
    n_ground_states = simulation_interaction.number_of_ground_states
    n_excited_states = simulation_interaction.number_of_excited_states

    # Sparse per-call clock indexed by `idx`, not by `atom_id`. Avoids a
    # zero-fill over n_atoms_total each step (was O(N_total), now O(N_alive)).
    accumulated_times = np.zeros(atom_ids.size, dtype=np.float64)

    # Hoist jitclass attribute lookups so the inner loop touches plain arrays.
    positions_arr = simulation_atoms.positions
    velocities_arr = simulation_atoms.velocities
    groundstates_arr = simulation_atoms.groundstates
    max_step_lengths_arr = simulation_atoms.max_step_lengths
    b_vec_arr = simulation_atoms.magnetic_field_vectors
    b_norm_arr = simulation_atoms.magnetic_field_strength
    pending_tau_arr = simulation_atoms.pending_optical_depth

    # THREAD-LOCAL WORKSPACES: allocate once per call (shape = (nthreads, ...))
    nthreads = get_num_threads()
    work_intensity = np.empty((nthreads, n_lasers), dtype=np.float64)
    work_doppler = np.empty((nthreads, n_lasers), dtype=np.float64)
    work_relI = np.empty((nthreads, n_lasers, 3), dtype=np.float64)
    work_sat = np.empty(
        (nthreads, n_lasers, n_excited_states, 3), dtype=np.float64
    )
    work_exc_rates = np.empty(
        (nthreads, n_lasers, n_excited_states, 3), dtype=np.float64
    )
    # Zeeman shifts depend on (gs, ex, pol, B) — independent of the laser. We
    # precompute the effective transition frequency f₀ + Δ_Zeeman once per
    # while-iter as a 2D table, saving n_lasers× shift evaluations.
    work_eff_trans_freq = np.empty(
        (nthreads, n_excited_states, 3), dtype=np.float64
    )
    work_branch = np.empty(
        (nthreads, n_excited_states, n_ground_states, 3), dtype=np.float64
    )

    for idx in prange(atom_ids.size):
        atom_id = atom_ids[idx]
        tid = get_thread_id()

        # local references into per-thread workspace
        intensity_at_position = work_intensity[tid]
        doppler_shifts = work_doppler[tid]
        relative_intensity_per_polarization = work_relI[tid]
        saturation_parameters = work_sat[tid]
        excitation_rates = work_exc_rates[tid]
        eff_trans_freq = work_eff_trans_freq[tid]
        branching_ratios = work_branch[tid]

        # main loop for this atom
        while accumulated_times[idx] < default_timestep:
            substep_counter[tid] += 1  # additive benchmark instrumentation
            pos = positions_arr[atom_id]
            vel = velocities_arr[atom_id]
            magnetic_field.calculate_magnetic_field(simulation_atoms, atom_id)
            magnetic_field.calculate_max_step_length(simulation_atoms, atom_id)
            atom_ground_state = groundstates_arr[atom_id]
            atom_max_step_length = max_step_lengths_arr[atom_id]
            b_vec = b_vec_arr[atom_id]
            b_norm = b_norm_arr[atom_id]

            # Zeeman shifts are independent of the laser — precompute the
            # effective transition frequency once per (ex, pol) for this atom.
            for ex in range(n_excited_states):
                for pol in range(3):
                    zeeman_shift = simulation_interaction.calculate_transition_frequency_shift(
                        ground_state=atom_ground_state,
                        excited_state=ex,
                        polarization=pol,
                        magnetic_field_strength=b_norm,
                    )
                    eff_trans_freq[ex, pol] = (
                        zeeman_shift + transition_frequency
                    )

            # Per-laser geometry + saturation, accumulating total saturation.
            total_saturation_parameter = 0.0
            for j in range(n_lasers):
                laser_dir = lasers.normalized_directions[j]
                intensity = beam_intensity_at_position(
                    pos,
                    lasers.origins[j],
                    laser_dir,
                    lasers.beam_waists[j],
                    lasers.beam_wavelengths[j],
                    lasers.initial_intensities[j],
                    lasers.refractive_indices[j],
                )
                intensity_at_position[j] = intensity

                if intensity > MIN_INTENSITY:
                    doppler_shifts[j] = (
                        wave_vectors[j, 0] * vel[0]
                        + wave_vectors[j, 1] * vel[1]
                        + wave_vectors[j, 2] * vel[2]
                    )

                    if b_norm <= 0.0:
                        angle_jB = 0.0
                    else:
                        dot = (
                            b_vec[0] * laser_dir[0]
                            + b_vec[1] * laser_dir[1]
                            + b_vec[2] * laser_dir[2]
                        )
                        cosval = dot / b_norm
                        if cosval > 1.0:
                            cosval = 1.0
                        elif cosval < -1.0:
                            cosval = -1.0
                        angle_jB = math.acos(cosval)

                    sq0, sq1, sq2 = calculate_handedness_to_polarization(
                        angle_jB, laser_handedness[j]
                    )

                    # relI depends only on (j, pol), not on ex — hoist out.
                    relI0 = sq0 * intensity
                    relI1 = sq1 * intensity
                    relI2 = sq2 * intensity
                    relative_intensity_per_polarization[j, 0] = relI0
                    relative_intensity_per_polarization[j, 1] = relI1
                    relative_intensity_per_polarization[j, 2] = relI2

                    doppler_j = doppler_shifts[j]
                    laser_freq_j = lasers.beam_frequencies[j]
                    detuning_j = lasers.detunings[j]

                    for ex in range(n_excited_states):
                        for pol in range(3):
                            if pol == 0:
                                relI = relI0
                            elif pol == 1:
                                relI = relI1
                            else:
                                relI = relI2
                            sat = simulation_interaction.calculate_saturation_parameter(
                                polarization=pol,
                                magnetic_field_strength=b_norm,
                                ground_state=atom_ground_state,
                                excited_state=ex,
                                laser_intensity=relI,
                                natural_linewidth=natural_linewidth,
                                saturation_intensity=saturation_intensity,
                                effective_transition_frequency=eff_trans_freq[
                                    ex, pol
                                ],
                                doppler_shift=doppler_j,
                                laser_beam_frequency=laser_freq_j,
                                detuning=detuning_j,
                            )
                            saturation_parameters[j, ex, pol] = sat
                            total_saturation_parameter += sat
                else:
                    doppler_shifts[j] = 0.0
                    relative_intensity_per_polarization[j, 0] = 0.0
                    relative_intensity_per_polarization[j, 1] = 0.0
                    relative_intensity_per_polarization[j, 2] = 0.0
                    for ex in range(n_excited_states):
                        for pol in range(3):
                            saturation_parameters[j, ex, pol] = 0.0

            # Compute excitation rates.
            simulation_interaction.calculate_rate(
                saturation_parameters,
                total_saturation_parameter,
                natural_linewidth,
                excitation_rates,
            )

            # compute total_excitation_rate by accumulation
            total_excitation_rate = 0.0
            for j in range(n_lasers):
                for ex in range(n_excited_states):
                    for pol in range(3):
                        total_excitation_rate += excitation_rates[j, ex, pol]

            remaining_time = default_timestep - accumulated_times[idx]

            # Pending optical-depth quota τ (dimensionless, rate-invariant).
            pending_tau = pending_tau_arr[atom_id]
            has_pending = pending_tau > 0.0

            if total_excitation_rate <= 0.0:
                # Locally dark: no optical depth accrues. Advance ballistically
                # and carry any pending τ unchanged (never resample here).
                motion_dt = magnetic_field.calculate_max_time_step(
                    atom_max_step_length, vel
                )
                dt = motion_dt
                if dt > remaining_time:
                    dt = remaining_time
                    accumulated_times[idx] = default_timestep
                else:
                    accumulated_times[idx] += dt

                _advance_with_gravity(
                    positions_arr, velocities_arr, atom_id, vel, dt
                )
                continue

            # rate > 0: draw τ once per event, then derive the event time from
            # the *current* local rate each substep (τ is carried, not t_event).
            if has_pending:
                tau = pending_tau
            else:
                tau = -math.log(np.random.random())

            t_event = tau / total_excitation_rate

            mean_free_path_length = magnetic_field.calculate_mean_free_path(
                t_event, vel
            )
            motion_dt = magnetic_field.calculate_max_time_step(
                atom_max_step_length, vel
            )

            if math.fabs(mean_free_path_length) >= atom_max_step_length:
                # geometry-limited motion advance; drain τ by rate·dt
                dt = motion_dt
                if dt > remaining_time:
                    dt = remaining_time
                    accumulated_times[idx] = default_timestep
                else:
                    accumulated_times[idx] += dt

                _advance_with_gravity(
                    positions_arr, velocities_arr, atom_id, vel, dt
                )

                new_tau = tau - total_excitation_rate * dt
                if new_tau < 0.0:
                    new_tau = 0.0
                pending_tau_arr[atom_id] = new_tau

                continue

            # event valid
            if t_event <= remaining_time:
                _advance_with_gravity(
                    positions_arr, velocities_arr, atom_id, vel, t_event
                )

                excitation_counter[atom_id] += 1
                accumulated_times[idx] += t_event
                pending_tau_arr[atom_id] = 0.0

                # fast flattened selection over the same excitation_rates buffer
                idx_laser, atom_excited_state, exciting_polarization = (
                    determine_exciting_laser_flat(
                        excitation_rates, total_excitation_rate
                    )
                )

                excitation_hist[atom_ground_state, exciting_polarization] += 1

                # compute branching ratios for selected excited state
                for gs in range(n_ground_states):
                    for pol in range(3):
                        branching_ratios[atom_excited_state, gs, pol] = (
                            simulation_interaction.calculate_branching_ratio(
                                ground_state=gs,
                                excited_state=atom_excited_state,
                                polarization=pol,
                                magnetic_field_strength=b_norm,
                            )
                        )

                # deexcite
                new_ground, emitted_pol = determine_deexcitation_transition(
                    branching_probs=branching_ratios[atom_excited_state]
                )
                groundstates_arr[atom_id] = new_ground

                # Recoil — combined absorption + spontaneous emission. Sampling
                # of the spontaneous-emission unit vector is inlined to avoid
                # the 3-array allocation `random_angle_in_sphere` would do.
                u = np.random.random()
                v = np.random.random()
                ct = 2.0 * u - 1.0
                st = math.sqrt(1.0 - ct * ct)
                phi = 2.0 * math.pi * v
                em_x = st * math.cos(phi)
                em_y = st * math.sin(phi)
                em_z = ct

                abs_coef = scc.hbar / atom_mass
                em_coef = (scc.h * transition_frequency) / (scc.c * atom_mass)
                wv = wave_vectors[idx_laser]
                velocities_arr[atom_id, 0] += abs_coef * wv[0] + em_coef * em_x
                velocities_arr[atom_id, 1] += abs_coef * wv[1] + em_coef * em_y
                velocities_arr[atom_id, 2] += abs_coef * wv[2] + em_coef * em_z

                continue
            else:
                # event beyond this default timestep: drain τ by rate·delta
                # and carry the remainder to the next step.
                delta = remaining_time
                _advance_with_gravity(
                    positions_arr, velocities_arr, atom_id, vel, delta
                )

                new_tau = tau - total_excitation_rate * delta
                if new_tau < 0.0:
                    new_tau = 0.0
                pending_tau_arr[atom_id] = new_tau
                accumulated_times[idx] = default_timestep

        # end while for atom

    return


@njit
def beam_intensity_at_position(
    atom_position: np.ndarray,
    laser_origin: np.ndarray,
    laser_direction: np.ndarray,
    beam_waist: float,
    beam_wavelength: float,
    initial_intensity: float,
    refractive_index: float,
) -> float:
    """Calculate the beam intensity at the given atom position for a Gaussian beam.

    Uses the identity radial² = |diff|² − axial² so we never materialise the
    cross product or call sqrt for the radial component, and we keep width²
    instead of width to skip another sqrt.
    """
    dx = atom_position[0] - laser_origin[0]
    dy = atom_position[1] - laser_origin[1]
    dz = atom_position[2] - laser_origin[2]

    axial_distance = (
        dx * laser_direction[0]
        + dy * laser_direction[1]
        + dz * laser_direction[2]
    )

    diff_sq = dx * dx + dy * dy + dz * dz
    radial_sq = diff_sq - axial_distance * axial_distance
    if radial_sq < 0.0:
        radial_sq = 0.0  # guard against rounding when diff is nearly axial

    waist_sq = beam_waist * beam_waist
    rayleigh_range = (math.pi * waist_sq * refractive_index) / beam_wavelength
    z_over_zR = axial_distance / rayleigh_range
    width_sq = waist_sq * (1.0 + z_over_zR * z_over_zR)

    return initial_intensity * math.exp(-2.0 * radial_sq / width_sq)


@njit
def determine_exciting_laser_flat(excitation_rates, total_excitation_rate):
    """Sample which (laser, excited_state, polarization) excites an atom.

    Draws from the flattened excitation-rate array proportional to rate;
    assumes total_excitation_rate > 0.

    Returns
    -------
    tuple of int
        (laser_index, excited_state, polarization).
    """
    r = np.random.random() * total_excitation_rate
    cum = 0.0
    n_lasers = excitation_rates.shape[0]
    n_excited = excitation_rates.shape[1]
    n_pol = excitation_rates.shape[2]
    for j in range(n_lasers):
        for ex in range(n_excited):
            for pol in range(n_pol):
                cum += excitation_rates[j, ex, pol]
                if cum >= r:
                    return j, ex, pol
    # safety fallback
    return n_lasers - 1, n_excited - 1, n_pol - 1


@njit
def determine_deexcitation_transition(
    branching_probs: np.ndarray,
) -> tuple[int, int]:
    """Determine the spontaneous decay transition based on branching probabilities.

    Parameters
    ----------
    branching_probs : np.ndarray
        A 2D array of shape (n_ground_states, 3) representing the probabilities
        for spontaneous emission from a given excited state into each ground state
        and polarization channel.

    Returns
    -------
    Tuple[int, int]
        - ground_state (int): The index of the destination ground state.
        - polarization (int): The polarization index (0=σ⁻, 1=π, 2=σ⁺) of the emitted photon.
    """
    total = np.sum(branching_probs)
    if total == 0.0:
        raise ValueError("Branching probabilities are all zero.")

    rnd = np.random.uniform() * total
    cumulative = 0.0

    for g in range(branching_probs.shape[0]):
        for pol in range(3):
            cumulative += branching_probs[g, pol]
            if cumulative >= rnd:
                return g, pol

    # Safety fallback (should never happen due to numerical errors)
    return branching_probs.shape[0] - 1, 2


@njit(inline="always")
def calculate_handedness_to_polarization(
    angle_laser_magnetic_field: float, handedness: int
):
    """Squared dipole matrix elements (sigma-, pi, sigma+) for a probe beam.

    Depends on the beam handedness and the angle theta between propagation
    direction and the local B-axis. Returns three scalars to avoid heap
    allocation in the inner loop.
    """
    c = math.cos(angle_laser_magnetic_field)
    s2 = math.sin(angle_laser_magnetic_field) ** 2

    if handedness == 1:
        return 0.25 * (1.0 - c) ** 2, 0.5 * s2, 0.25 * (1.0 + c) ** 2
    elif handedness == -1:
        return 0.25 * (1.0 + c) ** 2, 0.5 * s2, 0.25 * (1.0 - c) ** 2
    # handedness == 0: linear polarisation along k̂. σ± = ½ sin²θ, π = cos²θ.
    return 0.5 * s2, c * c, 0.5 * s2

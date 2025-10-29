# ################################################################# #
#                                                                   #
# Module to hold wrapped njit functions that are excitation related #
#                                                                   #
# ################################################################# #


from numba import njit, prange,get_thread_id, get_num_threads
import numpy as np
import scipy.constants as scc
import math
from util.geometry import random_angle_in_sphere
from util.simulation_typing import ECSLasers, ECSAtoms, MagneticField, LightAtomInteraction
from typing import Tuple


# -----------------------------------------------------------------------------
# Main routine: absorption and emission of photons for multiple atoms interacting with lasers.
# -----------------------------------------------------------------------------

@njit(parallel=True)
def absorption_and_emission_default_timestep(atom_ids: np.ndarray,
                                                      simulation_atoms: ECSAtoms,
                                                      lasers: ECSLasers,
                                                      simulation_interaction: LightAtomInteraction,
                                                      magnetic_field: MagneticField,
                                                      excitation_counter: np.ndarray,
                                                      default_timestep: float,
                                                      debug_counters: np.ndarray = None) -> None:

    n_lasers = lasers.n_lasers
    laser_handedness = lasers.handedness
    wave_vectors = lasers.wave_vectors
    transition_frequency = simulation_atoms.transition_frequency
    natural_linewidth = simulation_atoms.natural_linewidth
    atom_mass = simulation_atoms.mass
    saturation_intensity = simulation_atoms.saturation_intensity
    n_ground_states = simulation_interaction.number_of_ground_states
    n_excited_states = simulation_interaction.number_of_excited_states

    polarization_states = np.array([0, 1, 2], dtype=np.int64)

    n_atoms_total = simulation_atoms.n
    accumulated_times = np.zeros(n_atoms_total, dtype=np.float64)

    # THREAD-LOCAL WORKSPACES: allocate once per call (shape = (nthreads, ...))
    nthreads = get_num_threads()
    # per-thread arrays to avoid allocations inside loop
    work_intensity = np.empty((nthreads, n_lasers), dtype=np.float64)
    work_doppler = np.empty((nthreads, n_lasers), dtype=np.float64)
    work_angles = np.empty((nthreads, n_lasers), dtype=np.float64)
    work_relI = np.empty((nthreads, n_lasers, 3), dtype=np.float64)
    work_sat = np.empty((nthreads, n_lasers, n_excited_states, 3), dtype=np.float64)
    work_exc_rates = np.empty((nthreads, n_lasers, n_excited_states, 3), dtype=np.float64)
    work_exc_trans_freq = np.empty((nthreads, n_lasers, n_excited_states, 3), dtype=np.float64)
    work_branch = np.empty((nthreads, n_excited_states, n_ground_states, 3), dtype=np.float64)

    for idx in prange(atom_ids.size):
        atom_id = atom_ids[idx]
        tid = get_thread_id()

        # local references into per-thread workspace
        intensity_at_position = work_intensity[tid]
        doppler_shifts = work_doppler[tid]
        angles_between_field_and_lasers = work_angles[tid]
        relative_intensity_per_polarization = work_relI[tid]
        saturation_parameters = work_sat[tid]
        excitation_rates = work_exc_rates[tid]
        excitation_transition_frequencies = work_exc_trans_freq[tid]
        branching_ratios = work_branch[tid]

        # zero only the slices we will use (small)
        # it's okay to leave garbage in unused entries; we'll overwrite used ones

        # main loop for this atom
        while accumulated_times[atom_id] < default_timestep:
            pos = simulation_atoms.positions[atom_id]
            vel = simulation_atoms.velocities[atom_id]
            magnetic_field.calculate_magnetic_field(simulation_atoms, atom_id)
            magnetic_field.calculate_max_step_length(simulation_atoms, atom_id)
            atom_ground_state = simulation_atoms.groundstates[atom_id]
            atom_max_step_length = simulation_atoms.max_step_lengths[atom_id]
            b_vec = simulation_atoms.magnetic_field_vectors[atom_id]
            b_norm = simulation_atoms.magnetic_field_strength[atom_id]

            # compute per-laser geometry and fill saturation buf while accumulating total saturation
            total_saturation_parameter = 0.0
            for j in range(n_lasers):
                intensity_at_position[j] = beam_intensity_at_position(pos,
                                                                       lasers.origins[j],
                                                                       lasers.normalized_directions[j],
                                                                       lasers.beam_waists[j],
                                                                       lasers.beam_wavelengths[j],
                                                                       lasers.initial_intensities[j],
                                                                       lasers.refractive_indices[j])
                if intensity_at_position[j] > 0.0:
                    doppler_shifts[j] = wave_vectors[j, 0] * vel[0] + wave_vectors[j, 1] * vel[1] + wave_vectors[j, 2] * vel[2]

                    if b_norm <= 0.0:
                        angles_between_field_and_lasers[j] = 0.0
                    else:
                        dot = b_vec[0] * lasers.normalized_directions[j][0] + b_vec[1] * lasers.normalized_directions[j][1] + b_vec[2] * lasers.normalized_directions[j][2]
                        cosval = dot / b_norm
                        if cosval > 1.0:
                            cosval = 1.0
                        elif cosval < -1.0:
                            cosval = -1.0
                        angles_between_field_and_lasers[j] = np.arccos(cosval)

                    squared_matrix_elements = calculate_handedness_to_polarization(angle_laser_magnetic_field=angles_between_field_and_lasers[j],
                                                                                   handedness=laser_handedness[j])

                    for excited_state in range(n_excited_states):
                        for pol in polarization_states:
                            relI = squared_matrix_elements[pol] * intensity_at_position[j]
                            relative_intensity_per_polarization[j, pol] = relI

                            zeeman_shift = simulation_interaction.calculate_transition_frequency_shift(
                                ground_state=atom_ground_state,
                                excited_state=excited_state,
                                polarization=pol,
                                magnetic_field_strength=b_norm)

                            excitation_transition_frequencies[j, excited_state, pol] = zeeman_shift + transition_frequency

                            sat = simulation_interaction.calculate_saturation_parameter(
                                polarization=pol,
                                magnetic_field_strength=b_norm,
                                ground_state=atom_ground_state,
                                excited_state=excited_state,
                                laser_intensity=relI,
                                natural_linewidth=natural_linewidth,
                                saturation_intensity=saturation_intensity,
                                effective_transition_frequency=excitation_transition_frequencies[j, excited_state, pol],
                                doppler_shift=doppler_shifts[j],
                                laser_beam_frequency=lasers.beam_frequencies[j],
                                detuning=lasers.detunings[j]
                            )
                            saturation_parameters[j, excited_state, pol] = sat
                            total_saturation_parameter += sat
                else:
                    doppler_shifts[j] = 0.0
                    angles_between_field_and_lasers[j] = 0.0
                    for excited_state in range(n_excited_states):
                        for pol in polarization_states:
                            relative_intensity_per_polarization[j, pol] = 0.0
                            saturation_parameters[j, excited_state, pol] = 0.0

            # Compute excitation rates.
            simulation_interaction.calculate_rate(saturation_parameters, total_saturation_parameter, natural_linewidth, excitation_rates)

            # compute total_excitation_rate by accumulation
            total_excitation_rate = 0.0
            for j in range(n_lasers):
                for ex in range(n_excited_states):
                    for pol in polarization_states:
                        total_excitation_rate += excitation_rates[j, ex, pol]

            remaining_time = default_timestep - accumulated_times[atom_id]

            # pending overshoot
            pending = simulation_atoms.time_overshoot[atom_id]
            has_pending = pending > 0.0
            if debug_counters is not None:
                debug_counters[0] += 1

            if (total_excitation_rate <= 0.0) and (not has_pending):
                # motion-limited
                motion_dt = magnetic_field.calculate_max_time_step(atom_max_step_length, vel)
                dt = motion_dt
                if dt > remaining_time:
                    simulation_atoms.time_overshoot[atom_id] = 0.0
                    dt = remaining_time
                    accumulated_times[atom_id] = default_timestep
                else:
                    accumulated_times[atom_id] += dt

                simulation_atoms.positions[atom_id] += vel * dt
                simulation_atoms.velocities[atom_id] += np.array([0.0, -scc.g * dt, 0.0], dtype=np.float64)
                if debug_counters is not None:
                    debug_counters[4] += 1
                continue

            # choose event time (use pending or sample)
            if has_pending:
                t_event = pending
                if debug_counters is not None:
                    debug_counters[1] += 1
            else:
                r = np.random.random()
                t_event = -np.log(r) / total_excitation_rate
                if debug_counters is not None:
                    debug_counters[2] += 1

            mean_free_path_length = magnetic_field.calculate_mean_free_path(t_event, vel)
            motion_dt = magnetic_field.calculate_max_time_step(atom_max_step_length, vel)

            if np.abs(mean_free_path_length) >= atom_max_step_length:
                # geometry-limited motion advance
                dt = motion_dt
                if dt > remaining_time:
                    dt = remaining_time
                    accumulated_times[atom_id] = default_timestep
                else:
                    accumulated_times[atom_id] += dt

                simulation_atoms.positions[atom_id] += vel * dt
                simulation_atoms.velocities[atom_id] += np.array([0.0, -scc.g * dt, 0.0], dtype=np.float64)

                if has_pending:
                    new_pending = pending - dt
                    if new_pending < 0.0:
                        new_pending = 0.0
                    simulation_atoms.time_overshoot[atom_id] = new_pending
                else:
                    if t_event > dt:
                        simulation_atoms.time_overshoot[atom_id] = t_event - dt
                    else:
                        simulation_atoms.time_overshoot[atom_id] = 0.0

                if debug_counters is not None:
                    debug_counters[4] += 1
                continue

            # event valid
            if t_event <= remaining_time:
                simulation_atoms.positions[atom_id] += vel * t_event
                simulation_atoms.velocities[atom_id] += np.array([0.0, -scc.g * t_event, 0.0], dtype=np.float64)

                excitation_counter[atom_id] += 1
                accumulated_times[atom_id] += t_event
                simulation_atoms.time_overshoot[atom_id] = 0.0
                if debug_counters is not None:
                    debug_counters[3] += 1

                # fast flattened selection over the same excitation_rates buffer
                idx_laser, atom_excited_state, exciting_polarization = determine_exciting_laser_flat(excitation_rates, total_excitation_rate)

                # compute branching ratios for selected excited state
                for gs in range(n_ground_states):
                    for pol in polarization_states:
                        branching_ratios[atom_excited_state, gs, pol] = simulation_interaction.calculate_branching_ratio(
                            ground_state=gs, excited_state=atom_excited_state, polarization=pol, magnetic_field_strength=b_norm)

                # deexcite
                new_ground, emitted_pol = determine_deexcitation_transition(branching_probs=branching_ratios[atom_excited_state])
                simulation_atoms.groundstates[atom_id] = new_ground

                # recoil
                absorption_recoil = (scc.hbar / atom_mass) * wave_vectors[idx_laser]
                emission_dir = random_angle_in_sphere()
                k_mag = (scc.h * transition_frequency) / scc.c
                emission_recoil = (1.0 / atom_mass) * emission_dir * k_mag
                simulation_atoms.velocities[atom_id] += absorption_recoil + emission_recoil

                continue
            else:
                # event beyond this default timestep: store overshoot
                delta = remaining_time
                simulation_atoms.positions[atom_id] += vel * delta
                simulation_atoms.velocities[atom_id] += np.array([0.0, -scc.g * delta, 0.0], dtype=np.float64)

                simulation_atoms.time_overshoot[atom_id] = t_event - delta
                accumulated_times[atom_id] = default_timestep
                if debug_counters is not None:
                    debug_counters[5] += 1
                # will be handled next step

        # end while for atom

    return




@njit
def beam_intensity_at_position(atom_position: np.ndarray,
                               laser_origin: np.ndarray,
                               laser_direction: np.ndarray,
                               beam_waist: float,
                               beam_wavelength: float,
                               initial_intensity: float,
                               refractive_index: float) -> float:
    """
    Calculate the beam intensity at the given atom position for a Gaussian beam.
    Returns 0 if the atom is outside the beam.

    Parameters
    ----------
    atom_position : np.ndarray
        Position of the atom.
    laser_origin : np.ndarray
        Origin (starting point) of the laser beam.
    laser_direction : np.ndarray
        Normalized propagation direction of the laser beam.
    beam_waist : float
        Beam waist (radius at focus) of the laser.
    beam_wavelength : float
        Wavelength of the laser beam.
    initial_intensity : float
        Peak intensity at the beam center.
    refractive_index : float
        Refractive index in the medium.

    Returns
    -------
    float
        Local intensity of the laser beam at atom_position if the atom is
        within the beam (i.e. radial distance less than the local beam width),
        or 0 otherwise.
    """
    # Calculate vector from laser origin to atom position.
    diff = atom_position - laser_origin

    # Compute the radial distance: norm(cross(diff, laser_direction))
    radial_distance = np.linalg.norm(np.cross(diff, laser_direction))

    # Axial distance along the beam direction.
    axial_distance =  diff[0]*laser_direction[0] + diff[1]*laser_direction[1] + diff[2]*laser_direction[2]

    # Calculate the Rayleigh range.

    rayleigh_range = (np.pi * beam_waist**2 * refractive_index) / (beam_wavelength)

    # Compute the beam width at the atom's axial position.
    width_at_position = beam_waist * np.sqrt(1 + (axial_distance / (rayleigh_range))**2)

    # If the atom lies within the local beam width, compute intensity.


    intensity = initial_intensity * np.exp(-2* (radial_distance / (width_at_position))**2)
    return intensity



# ------------------------------------------------------------------------------------------
# Helper function to randomly select a laser based on probabilities. only for 1 laser setups
# ------------------------------------------------------------------------------------------

@njit
def determine_exciting_laser(excitation_rates: np.ndarray) -> Tuple[int, int]:
    """
        Determines the indices of the exciting laser and the excited state based on excitation rates.
        Parameters
        ----------
        excitation_rates : np.ndarray
            A 2D array where each element represents the excitation rate for a specific laser and state.
        Returns
        -------
        Tuple[int, int]
            A tuple containing:
            - exciting_laser_index (int): The index of the exciting laser.
            - excited_state (int): The index of the excited state.
   
    """
    
    random_number = np.random.uniform() * np.sum(excitation_rates)
    cumulative_sum = 1e-35

    for laser_index in range(excitation_rates.shape[0]):
        for excitation_rate_index in range(excitation_rates.shape[1]):
            for polarization in range(excitation_rates.shape[2]):

                cumulative_sum  += excitation_rates[laser_index][excitation_rate_index][polarization]

                if cumulative_sum >= random_number:


                    exciting_laser_index = laser_index
                    excited_state =  excitation_rate_index
                    polarization = polarization
                    return exciting_laser_index, excited_state, polarization



@njit
def determine_exciting_laser_flat(excitation_rates, total_excitation_rate):
    # Flattened sampling: return (laser_idx, excited_state, polarization)
    # Assumes total_excitation_rate > 0
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
def determine_deexcitation_transition(branching_probs: np.ndarray) -> Tuple[int, int]:
    """
    Determine the spontaneous decay transition based on branching probabilities.

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



@njit
def calculate_small_wigner_d_matrix_elements(polarization: int, angle_laser_magnetic_field: float) -> np.ndarray:
    """
    Calculate the squared small Wigner d-matrix elements for a spin-1 photon state projection.

    This function projects the polarization state of a laser beam onto the quantization
    axis defined by the magnetic field. The input 'polarization' indicates the initial
    polarization state:
      - 0 for sigma-,
      - 1 for pi,
      - 2 for sigma+.

    The 'angle_laser_magnetic_field' is the angle (in radians) between the laser propagation
    direction and the magnetic field direction.

    Parameters
    ----------
    polarization : int
        The index representing the initial polarization state:
        0 for sigma-, 1 for pi, and 2 for sigma+.
    angle_laser_magnetic_field : float
        The angle (in radians) between the laser propagation direction and the magnetic field
        (quantization axis).

    Returns
    -------
    projected_polarizations : numpy.ndarray
        A 1D array of length 3 containing the squared amplitudes (i.e., intensity fractions) for
        the projected components in the order [sigma-, pi, sigma+].

    Notes
    -----
    The squared amplitudes are obtained by squaring the small Wigner d-matrix elements for j=1.
    For example, for an initial sigma+ state (polarization 2), the projections are:
      - sigma-: (sin(angle/2)**2)**2,
      - pi:      (-sin(angle)/sqrt(2))**2,
      - sigma+: (cos(angle/2)**2)**2.
    """
    # Initialize an array to store the projected intensities
    projected_polarizations = np.zeros(3, dtype=np.float64)


    if polarization == 0:
        # For sigma- polarization (initial state):
        # d_{-1,-1} = cos(angle/2)**2  --> Squared: (cos(angle/2)**2)**2
        projected_polarizations[0] = (math.cos(angle_laser_magnetic_field/2)**2)**2
        
        # d_{0,-1} = -sin(angle)/sqrt(2)  --> Squared: (-sin(angle)/sqrt(2))**2
        projected_polarizations[1] = (-math.sin(angle_laser_magnetic_field) / math.sqrt(2))**2
        
        # d_{+1,-1} = sin(angle/2)**2  --> Squared: (sin(angle/2)**2)**2
        projected_polarizations[2] = (math.sin(angle_laser_magnetic_field/2)**2)**2
        
        return projected_polarizations
    
    if polarization == 1:
        # For pi polarization (initial state):
        # d_{-1,0} = -sin(angle)/sqrt(2)  --> Squared: (-sin(angle)/sqrt(2))**2
        projected_polarizations[0] = (-math.sin(angle_laser_magnetic_field) / math.sqrt(2))**2
        
        # d_{0,0} = cos(angle)  --> Squared: (cos(angle))**2
        projected_polarizations[1] = (math.cos(angle_laser_magnetic_field))**2
        
        # d_{+1,0} = sin(angle)/sqrt(2)  --> Squared: (sin(angle)/sqrt(2))**2
        projected_polarizations[2] = (math.sin(angle_laser_magnetic_field) / math.sqrt(2))**2
        
        return projected_polarizations
    
    if polarization == 2:
        # For sigma+ polarization (initial state):
        # d_{-1,+1} = sin(angle/2)**2  --> Squared: (sin(angle/2)**2)**2
        projected_polarizations[0] = (math.sin(angle_laser_magnetic_field/2)**2)**2
        
        # d_{0,+1} = -sin(angle)/sqrt(2)  --> Squared: (-sin(angle)/sqrt(2))**2
        projected_polarizations[1] = (-math.sin(angle_laser_magnetic_field) / math.sqrt(2))**2
        
        # d_{+1,+1} = cos(angle/2)**2  --> Squared: (cos(angle/2)**2)**2
        projected_polarizations[2] = (math.cos(angle_laser_magnetic_field/2)**2)**2
        
        return projected_polarizations
    






@njit
def calculate_handedness_to_polarization(angle_laser_magnetic_field: float, 
                                       handedness: int) -> np.ndarray:
    """
    Convert handedness (handedness of laser in lab frame) to the corresponding polarization vector.

    Parameters
    ----------
    angle_laser_magnetic_field : float
        The angle (in radians) between the laser propagation direction and the magnetic field.
    polarization : int
        The polarization state (0 for sigma-, 1 for pi, 2 for sigma+).

    Returns
    -------
    np.ndarray
        A 3D vector representing the polarization direction in Cartesian coordinates.
    """

    projected_polarizations = np.zeros(3, dtype=np.float64)
    if handedness == 1:

        #(left-handed):
        projected_polarizations[0] = 1/4 * (1 - math.cos(angle_laser_magnetic_field))**2
        projected_polarizations[1] = 1/2 * math.sin(angle_laser_magnetic_field)**2
        projected_polarizations[2] = 1/4 * (1 + math.cos(angle_laser_magnetic_field))**2

    elif handedness == -1:
        #(right handed):
        projected_polarizations[0] = 1/4 * (1 + math.cos(angle_laser_magnetic_field))**2
        projected_polarizations[1] = 1/2 * math.sin(angle_laser_magnetic_field)**2
        projected_polarizations[2] = 1/4 * (1 - math.cos(angle_laser_magnetic_field))**2

    return projected_polarizations



@njit
def make_lab_polarization(k_hat: np.ndarray, handedness: int):
    # pick two orthonormal transverse axes (u,v) ⟂ k_hat
    # for simplicity, start with u = arbitrary ⟂ k_hat, then v = k_hat × u
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, k_hat)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(k_hat, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(k_hat, u)
    # now RH: (u + i v)/√2, LH: (u - i v)/√2
    if handedness == +1:
        return -(u + 1j*v) / np.sqrt(2)   # right‑handed
    else:
        return +(u - 1j*v) / np.sqrt(2)   # left‑handed


@njit
def decompose_into_sigma_components(k_hat: np.ndarray,
                                    handedness: int,
                                    B_hat: np.ndarray):
    # 1) build eps_lab from (k_hat,handedness)
    eps_lab = make_lab_polarization(k_hat, handedness)

    # 2) same basis construction as before
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, B_hat)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(B_hat, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(B_hat, e1)

    # 3) project eps_lab onto that basis *manually* to avoid mixed‐dtype dot
    eps1 = eps_lab[0]*e1[0] + eps_lab[1]*e1[1] + eps_lab[2]*e1[2]
    eps2 = eps_lab[0]*e2[0] + eps_lab[1]*e2[1] + eps_lab[2]*e2[2]
    eps3 = eps_lab[0]*B_hat[0] + eps_lab[1]*B_hat[1] + eps_lab[2]*B_hat[2]

    # 4) form the Δm amplitudes
    amp_minus1 =  ( eps1 - 1j*eps2) / np.sqrt(2)   # σ⁻ (Δm=−1)
    amp_plus1  = -( eps1 + 1j*eps2) / np.sqrt(2)   # σ⁺ (Δm=+1)
    amp_0      =  eps3                            # π   (Δm=0)

    return np.array([
        np.abs(amp_minus1)**2,
        np.abs(amp_0)**2,
        np.abs(amp_plus1)**2
    ])


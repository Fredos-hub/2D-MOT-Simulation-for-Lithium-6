"""Light-atom interaction jitclasses (6/18/simple-18/4-level Li-6 models)."""

import numpy as np
import scipy.constants as scc
from numba import float32, float64, int32
from numba.experimental import jitclass

import src.interaction_wrappers.common as common
import src.interaction_wrappers.diagonalizer_wrappers as diw
import src.interaction_wrappers.eighteen_level_wrappers as elw
import src.interaction_wrappers.four_level_wrappers as flw
import src.interaction_wrappers.simple_eighteen_level_wrappers as simple_elw
import src.interaction_wrappers.six_level_wrappers as slw

BOHR_MAGNETON = scc.physical_constants["Bohr magneton"][0]


# Simple 6-Level Model

six_level_spec = [
    ("number_of_ground_states", int32),
    ("number_of_excited_states", int32),
    ("mu_B", float64),
    ("allowed_transitions", int32[:, :]),
    ("ground_mJ", float64[:]),
    ("excited_mJ", float64[:]),
    ("branch_table", float64[:, :, :]),
]


@jitclass(six_level_spec)
class Lithium6LevelInteraction:
    """Simple J=1/2 -> J=3/2 Li-6 model: 2 ground, 4 excited, no hyperfine.

    Branching ratios are a static |CG|^2 table; the Zeeman shift uses the
    m_J levels directly.
    """

    def __init__(self):
        self.number_of_ground_states = 2
        self.number_of_excited_states = 4
        self.mu_B = BOHR_MAGNETON

        # Define m_J values explicitly
        self.ground_mJ = np.array([-0.5, +0.5])
        self.excited_mJ = np.array([-1.5, -0.5, +0.5, +1.5])

        # Allowed transitions matrix: (ground_state, excited_state, polarization)
        # Polarization: 0 -> σ-, 1 -> π, 2 -> σ+
        self.allowed_transitions = np.array(
            [
                [0, 0, 0],  # Ground 0 (-1/2) -> Excited 0 (-3/2), σ-
                [1, 1, 0],  # Ground 1 (+1/2) -> Excited 1 (-1/2), σ-
                [0, 1, 1],  # Ground 0 (-1/2) -> Excited 1 (-1/2), π
                [1, 2, 1],  # Ground 1 (+1/2) -> Excited 2 (+1/2), π
                [0, 2, 2],  # Ground 0 (-1/2) -> Excited 2 (+1/2), σ+
                [1, 3, 2],  # Ground 1 (+1/2) -> Excited 3 (+3/2), σ+
            ],
            dtype=np.int32,
        )

        # Build a 3D “branching‐weight” table (excited × ground × pol)
        #    entry = |<g; 1,q | e>|^2 for J'=3/2→J=1/2.
        table = np.zeros(
            (self.number_of_excited_states, self.number_of_ground_states, 3),
            dtype=np.float64,
        )

        # e=0 (mJ'=-3/2) → only (g=0, pol=0) has weight=1
        table[0, 0, 0] = 1.0 / 4

        # e=1 (mJ'=-½):
        #    (g=0, pol=1) → weight=2/3;  (g=1, pol=0) → weight=1/3
        table[1, 0, 1] = (2.0 / 3.0) / 4
        table[1, 1, 0] = (1.0 / 3.0) / 4

        # e=2 (mJ'=+½):
        #    (g=0, pol=2) → weight=1/3;  (g=1, pol=1) → weight=2/3
        table[2, 0, 2] = (1.0 / 3.0) / 4
        table[2, 1, 1] = (2.0 / 3.0) / 4

        # e=3 (mJ'=+3/2) → only (g=1, pol=2) has weight=1
        table[3, 1, 2] = 1.0 / 4

        self.branch_table = table

    def calculate_rate(
        self,
        saturation_parameters,
        total_saturation_parameter,
        natural_linewidth,
        excitation_rates,
    ):
        """Fill excitation_rates in place from the per-transition saturation.

        Delegates to the shared excitation-rate kernel in
        interaction_wrappers/common.py.

        Parameters
        ----------
        saturation_parameters : ndarray
            Per-transition saturation parameters.
        total_saturation_parameter : float
            Sum of the saturation parameters (drives power broadening).
        natural_linewidth : float
            Natural linewidth in rad/s.
        excitation_rates : ndarray
            Output buffer, written in place.
        """
        common._calculate_excitation_rate(
            saturation_parameters=saturation_parameters,
            total_saturation_parameter=total_saturation_parameter,
            natural_linewidth=natural_linewidth,
            excitation_rates=excitation_rates,
        )

    def calculate_saturation_parameter(
        self,
        polarization: int,
        magnetic_field_strength: float,
        ground_state: float,
        excited_state: float,
        laser_intensity: float,
        natural_linewidth: float,
        saturation_intensity: float,
        effective_transition_frequency: float,
        doppler_shift,
        laser_beam_frequency: float,
        detuning: float,
    ):
        """Saturation parameter for one (laser, ground, excited, pol) tuple.

        Looks up the transition strength for the model, then delegates to the
        shared saturation-parameter kernel in interaction_wrappers/common.py.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+), lab frame.
        magnetic_field_strength : float
            Local field magnitude in T.
        ground_state, excited_state : int
            State indices.
        laser_intensity : float
            Local laser intensity in W/m^2.
        natural_linewidth : float
            Natural linewidth in rad/s.
        saturation_intensity : float
            Saturation intensity in W/m^2.
        effective_transition_frequency : float
            Transition frequency including the Zeeman shift, in Hz.
        doppler_shift : float
            Doppler shift in Hz.
        laser_beam_frequency : float
            Laser frequency in Hz.
        detuning : float
            Fixed laser detuning in rad/s.

        Returns
        -------
        float
            Saturation parameter for the transition.
        """
        transition_strength = self.branch_table[excited_state][ground_state][
            polarization
        ]
        saturation_parameter = common._calculate_saturation_parameter(
            effective_transition_frequency=effective_transition_frequency,
            doppler_shift=doppler_shift,
            laser_beam_frequency=laser_beam_frequency,
            detuning=detuning,
            transition_strength=transition_strength,
            laser_intensity=laser_intensity,
            natural_linewidth=natural_linewidth,
        )

        return saturation_parameter

    def calculate_transition_frequency_shift(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Calculate the Zeeman frequency shift for a transition.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+), lab frame.
        ground_state : int
            Ground-state index.
        excited_state : int
            Excited-state index.
        magnetic_field_strength : float
            Field magnitude in T; the sign encodes the field direction.

        Returns
        -------
        float
            Transition frequency shift in Hz (added to the transition frequency).
        """
        transition_energy_shift = slw._calculate_transition_frequency_shift(
            ground_state=ground_state,
            excited_state=excited_state,
            magnetic_field_strength=magnetic_field_strength,
            mu_B=self.mu_B,
            ground_mJ=self.ground_mJ,
            excited_mJ=self.excited_mJ,
        )
        return transition_energy_shift

    def calculate_branching_ratio(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Branching ratio (|CG|^2 weight) for a (ground, excited, pol) tuple.

        Used to distribute spontaneous decay over the allowed transitions.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+).
        ground_state, excited_state : int
            State indices.
        magnetic_field_strength : float
            Local field magnitude in T (used by the field-dependent models).

        Returns
        -------
        float
            Relative branching weight for the transition.
        """
        return self.branch_table[excited_state, ground_state, polarization]


# 18-Level-Code from Julia


eighteen_level_spec = [
    ("number_of_ground_states", int32),
    ("number_of_excited_states", int32),
]


@jitclass(eighteen_level_spec)
class Lithium18LevelInteraction:
    """Full D2 hyperfine manifold for Li-6: 6 ground, 12 excited states.

    Transition strengths and Zeeman shifts come from the empirical
    field-dependent fits in eighteen_level_wrappers.
    """

    def __init__(self):

        self.number_of_ground_states = 6
        self.number_of_excited_states = 12

    def calculate_transition_frequency_shift(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Calculate the Zeeman frequency shift for a transition.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+), lab frame.
        ground_state : int
            Ground-state index.
        excited_state : int
            Excited-state index.
        magnetic_field_strength : float
            Field magnitude in T; the sign encodes the field direction.

        Returns
        -------
        float
            Transition frequency shift in Hz (added to the transition frequency).
        """
        transition_frequency_shift = elw.calculate_transition_frequency_shift(
            GS=ground_state,
            ES=excited_state,
            pol=polarization,
            B=magnetic_field_strength,
        )

        return transition_frequency_shift

    def calculate_rate(
        self,
        saturation_parameters,
        total_saturation_parameter,
        natural_linewidth,
        excitation_rates,
    ):
        """Fill excitation_rates in place from the per-transition saturation.

        Delegates to the shared excitation-rate kernel in
        interaction_wrappers/common.py.

        Parameters
        ----------
        saturation_parameters : ndarray
            Per-transition saturation parameters.
        total_saturation_parameter : float
            Sum of the saturation parameters (drives power broadening).
        natural_linewidth : float
            Natural linewidth in rad/s.
        excitation_rates : ndarray
            Output buffer, written in place.
        """
        common._calculate_excitation_rate(
            saturation_parameters=saturation_parameters,
            total_saturation_parameter=total_saturation_parameter,
            natural_linewidth=natural_linewidth,
            excitation_rates=excitation_rates,
        )

    def calculate_saturation_parameter(
        self,
        polarization: int,
        magnetic_field_strength: float,
        ground_state: float,
        excited_state: float,
        laser_intensity: float,
        natural_linewidth: float,
        saturation_intensity: float,
        effective_transition_frequency: float,
        doppler_shift,
        laser_beam_frequency: float,
        detuning: float,
    ):
        """Saturation parameter for one (laser, ground, excited, pol) tuple.

        Looks up the transition strength for the model, then delegates to the
        shared saturation-parameter kernel in interaction_wrappers/common.py.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+), lab frame.
        magnetic_field_strength : float
            Local field magnitude in T.
        ground_state, excited_state : int
            State indices.
        laser_intensity : float
            Local laser intensity in W/m^2.
        natural_linewidth : float
            Natural linewidth in rad/s.
        saturation_intensity : float
            Saturation intensity in W/m^2.
        effective_transition_frequency : float
            Transition frequency including the Zeeman shift, in Hz.
        doppler_shift : float
            Doppler shift in Hz.
        laser_beam_frequency : float
            Laser frequency in Hz.
        detuning : float
            Fixed laser detuning in rad/s.

        Returns
        -------
        float
            Saturation parameter for the transition.
        """
        transition_strength = elw.calculate_transition_strength(
            GS=ground_state,
            ES=excited_state,
            pol=polarization,
            B=magnetic_field_strength,
        )

        saturation_parameter = common._calculate_saturation_parameter(
            effective_transition_frequency=effective_transition_frequency,
            doppler_shift=doppler_shift,
            laser_beam_frequency=laser_beam_frequency,
            detuning=detuning,
            transition_strength=transition_strength,
            laser_intensity=laser_intensity,
            natural_linewidth=natural_linewidth,
        )

        return saturation_parameter

    def calculate_branching_ratio(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Branching ratio (|CG|^2 weight) for a (ground, excited, pol) tuple.

        Used to distribute spontaneous decay over the allowed transitions.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+).
        ground_state, excited_state : int
            State indices.
        magnetic_field_strength : float
            Local field magnitude in T (used by the field-dependent models).

        Returns
        -------
        float
            Relative branching weight for the transition.
        """
        return elw.calculate_transition_strength(
            GS=ground_state,
            ES=excited_state,
            pol=polarization,
            B=magnetic_field_strength,
        )


# Live Li-6 D2 Diagonalizer Model


diagonalizer_spec = [
    ("number_of_ground_states", int32),
    ("number_of_excited_states", int32),
]


@jitclass(diagonalizer_spec)
class Lithium6DiagonalizerInteraction:
    """Live Li-6 D2 diagonalizer: 6 ground, 12 excited states.

    Shifts and strengths come from diagonalizing H(|B|) = H_hfs + |B|*H_Zeeman
    with np.linalg.eigh inside the JIT boundary (no curve fits). The constant
    matrices, coupling tensor and diabatic order map are frozen module-level
    constants in diagonalizer_wrappers, so this jitclass holds NO array state
    and is thread-safe under prange: no per-instance eigh cache, diagonalized
    fresh on every call (Pitfall 1).
    """

    def __init__(self):
        self.number_of_ground_states = 6
        self.number_of_excited_states = 12

    def calculate_transition_frequency_shift(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Field-induced Zeeman/HFS shift in Hz (added to transition_frequency)."""
        return diw.shift(
            ground_state, excited_state, magnetic_field_strength
        )

    def calculate_rate(
        self,
        saturation_parameters,
        total_saturation_parameter,
        natural_linewidth,
        excitation_rates,
    ):
        """Fill excitation_rates in place via the shared kernel."""
        common._calculate_excitation_rate(
            saturation_parameters=saturation_parameters,
            total_saturation_parameter=total_saturation_parameter,
            natural_linewidth=natural_linewidth,
            excitation_rates=excitation_rates,
        )

    def calculate_saturation_parameter(
        self,
        polarization: int,
        magnetic_field_strength: float,
        ground_state: float,
        excited_state: float,
        laser_intensity: float,
        natural_linewidth: float,
        saturation_intensity: float,
        effective_transition_frequency: float,
        doppler_shift,
        laser_beam_frequency: float,
        detuning: float,
    ):
        """Diagonalize for the transition strength, then delegate to common."""
        transition_strength = diw.strength(
            polarization, ground_state, excited_state, magnetic_field_strength
        )
        return common._calculate_saturation_parameter(
            effective_transition_frequency=effective_transition_frequency,
            doppler_shift=doppler_shift,
            laser_beam_frequency=laser_beam_frequency,
            detuning=detuning,
            transition_strength=transition_strength,
            laser_intensity=laser_intensity,
            natural_linewidth=natural_linewidth,
        )

    def calculate_branching_ratio(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Decay weight = transition strength (same as the fit models)."""
        return diw.strength(
            polarization, ground_state, excited_state, magnetic_field_strength
        )


# Precomputed-Table Li-6 D2 Diagonalizer Model


diagonalizer_table_spec = [
    ("number_of_ground_states", int32),
    ("number_of_excited_states", int32),
    ("b_axis", float64[:]),
    ("pos_table", float64[:, :, :]),
    ("strength_table", float64[:, :, :, :]),
]


@jitclass(diagonalizer_table_spec)
class Lithium6DiagonalizerTableInteraction:
    """Read-only Li-6 D2 diagonalizer: 6 ground, 12 excited states.

    Instead of diagonalizing H(|B|) at every step (the live
    ``Lithium6DiagonalizerInteraction``), this model 1-D-interpolates a
    precomputed |B|-table generated offline by
    ``diagonalizer_setup.generate_table``. The table (line shifts in Hz and
    cycling-normalized strengths) is loaded and validated in ``parameters.py``
    and passed into the constructor as plain float64 arrays — the thread-safe,
    no-runtime-eigh fallback (D-01, D-12).
    """

    def __init__(self, b_axis, pos_table, strength_table):
        self.number_of_ground_states = 6
        self.number_of_excited_states = 12
        self.b_axis = b_axis
        self.pos_table = pos_table
        self.strength_table = strength_table

    def calculate_transition_frequency_shift(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Field-induced shift in Hz, interpolated from the table (pol-free)."""
        lo, hi, f = diw._bracket(self.b_axis, magnetic_field_strength)
        p = self.pos_table
        return (
            p[lo, ground_state, excited_state] * (1.0 - f)
            + p[hi, ground_state, excited_state] * f
        )

    def calculate_rate(
        self,
        saturation_parameters,
        total_saturation_parameter,
        natural_linewidth,
        excitation_rates,
    ):
        """Fill excitation_rates in place via the shared kernel."""
        common._calculate_excitation_rate(
            saturation_parameters=saturation_parameters,
            total_saturation_parameter=total_saturation_parameter,
            natural_linewidth=natural_linewidth,
            excitation_rates=excitation_rates,
        )

    def calculate_saturation_parameter(
        self,
        polarization: int,
        magnetic_field_strength: float,
        ground_state: float,
        excited_state: float,
        laser_intensity: float,
        natural_linewidth: float,
        saturation_intensity: float,
        effective_transition_frequency: float,
        doppler_shift,
        laser_beam_frequency: float,
        detuning: float,
    ):
        """Interpolate the transition strength, then delegate to common."""
        lo, hi, f = diw._bracket(self.b_axis, magnetic_field_strength)
        s = self.strength_table
        transition_strength = (
            s[lo, ground_state, excited_state, polarization] * (1.0 - f)
            + s[hi, ground_state, excited_state, polarization] * f
        )
        return common._calculate_saturation_parameter(
            effective_transition_frequency=effective_transition_frequency,
            doppler_shift=doppler_shift,
            laser_beam_frequency=laser_beam_frequency,
            detuning=detuning,
            transition_strength=transition_strength,
            laser_intensity=laser_intensity,
            natural_linewidth=natural_linewidth,
        )

    def calculate_branching_ratio(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Decay weight = interpolated transition strength."""
        lo, hi, f = diw._bracket(self.b_axis, magnetic_field_strength)
        s = self.strength_table
        return (
            s[lo, ground_state, excited_state, polarization] * (1.0 - f)
            + s[hi, ground_state, excited_state, polarization] * f
        )


# Simple 4-Level Model


four_level_spec = [
    ("number_of_ground_states", int32),
    ("number_of_excited_states", int32),
    ("mu_B", float64),
    ("ground_mJ", float64[:]),
    ("excited_mJ", float64[:]),
    ("allowed_transitions", int32[:, :]),
]


@jitclass(four_level_spec)
class Lithium4LevelInteraction:
    """Minimal test model: 1 ground, 3 excited states (one sigma-/pi/sigma+).

    Allowed transitions have unit strength; all others are zero.
    """

    def __init__(self):
        self.number_of_ground_states = 1
        self.number_of_excited_states = 3

        self.mu_B = BOHR_MAGNETON

        self.ground_mJ = np.array([0], dtype=np.float64)
        self.excited_mJ = np.array([-1, 0, 1], dtype=np.float64)

        self.allowed_transitions = np.array(
            [[1, 0, 0], [1, 1, 1], [1, 2, 2]], dtype=np.int32
        )  # ground state, excited state, polarization

    def calculate_rate(
        self,
        saturation_parameters,
        total_saturation_parameter,
        natural_linewidth,
        excitation_rates,
    ):
        """Fill excitation_rates in place from the per-transition saturation.

        Delegates to the shared excitation-rate kernel in
        interaction_wrappers/common.py.

        Parameters
        ----------
        saturation_parameters : ndarray
            Per-transition saturation parameters.
        total_saturation_parameter : float
            Sum of the saturation parameters (drives power broadening).
        natural_linewidth : float
            Natural linewidth in rad/s.
        excitation_rates : ndarray
            Output buffer, written in place.
        """
        common._calculate_excitation_rate(
            saturation_parameters=saturation_parameters,
            total_saturation_parameter=total_saturation_parameter,
            natural_linewidth=natural_linewidth,
            excitation_rates=excitation_rates,
        )
        return

    def calculate_saturation_parameter(
        self,
        polarization: int,
        magnetic_field_strength: float,
        ground_state: float,
        excited_state: float,
        laser_intensity: float,
        natural_linewidth: float,
        saturation_intensity: float,
        effective_transition_frequency: float,
        doppler_shift,
        laser_beam_frequency: float,
        detuning: float,
    ):
        """Saturation parameter for one (laser, ground, excited, pol) tuple.

        Looks up the transition strength for the model, then delegates to the
        shared saturation-parameter kernel in interaction_wrappers/common.py.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+), lab frame.
        magnetic_field_strength : float
            Local field magnitude in T.
        ground_state, excited_state : int
            State indices.
        laser_intensity : float
            Local laser intensity in W/m^2.
        natural_linewidth : float
            Natural linewidth in rad/s.
        saturation_intensity : float
            Saturation intensity in W/m^2.
        effective_transition_frequency : float
            Transition frequency including the Zeeman shift, in Hz.
        doppler_shift : float
            Doppler shift in Hz.
        laser_beam_frequency : float
            Laser frequency in Hz.
        detuning : float
            Fixed laser detuning in rad/s.

        Returns
        -------
        float
            Saturation parameter for the transition.
        """
        allowed = flw._is_transition_allowed(
            polarization=polarization,
            ground_state=ground_state,
            excited_state=excited_state,
            allowed_transitions=self.allowed_transitions,
        )
        if allowed:
            transition_strength = 1.0
            saturation_parameter = common._calculate_saturation_parameter(
                effective_transition_frequency=effective_transition_frequency,
                doppler_shift=doppler_shift,
                laser_beam_frequency=laser_beam_frequency,
                detuning=detuning,
                transition_strength=transition_strength,
                laser_intensity=laser_intensity,
                natural_linewidth=natural_linewidth,
            )
        else:
            saturation_parameter = 0.0

        return saturation_parameter

    def calculate_branching_ratio(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Branching ratio (|CG|^2 weight) for a (ground, excited, pol) tuple.

        Used to distribute spontaneous decay over the allowed transitions.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+).
        ground_state, excited_state : int
            State indices.
        magnetic_field_strength : float
            Local field magnitude in T (used by the field-dependent models).

        Returns
        -------
        float
            Relative branching weight for the transition.
        """
        return 1

    def calculate_transition_frequency_shift(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Calculate the Zeeman frequency shift for a transition.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+), lab frame.
        ground_state : int
            Ground-state index.
        excited_state : int
            Excited-state index.
        magnetic_field_strength : float
            Field magnitude in T; the sign encodes the field direction.

        Returns
        -------
        float
            Transition frequency shift in Hz (added to the transition frequency).
        """
        transition_frequency_shift = flw._calculate_transition_frequency_shift(
            ground_state=ground_state,
            excited_state=excited_state,
            magnetic_field_strength=magnetic_field_strength,
            mu_B=self.mu_B,
            ground_mJ=self.ground_mJ,
            excited_mJ=self.excited_mJ,
        )
        return transition_frequency_shift


# Simple 18-Level Model


simple_eighteen_level_spec = [
    ("number_of_ground_states", int32),
    ("number_of_excited_states", int32),
    ("ground_mf", float32[:]),
    ("excited_mf", float32[:]),
    ("ground_gf", float32[:]),
    ("excited_gf", float32[:]),
    ("mu_B", float64),
    ("transition_strength_table", float32[:, :, :]),
]


@jitclass(simple_eighteen_level_spec)
class SimpleEighteenLevelInteraction:
    """Simplified 18-level Li-6 model: 6 ground, 12 excited states.

    Uses a static |F, mF> transition-strength table and per-state gF factors
    for the Zeeman shift, avoiding the field-dependent fits of the full
    18-level model.
    """

    def __init__(self):

        self.mu_B = BOHR_MAGNETON

        self.number_of_ground_states = 6
        self.number_of_excited_states = 12

        # Quantum numbers and gF factors for ground and excited states mapped by index (See Julias Thesis Fig. 2.1)
        self.ground_mf = np.array(
            [0.5, 0.5, -0.5, -0.5, -1.5, 1.5], dtype=np.float32
        )  # mF values for ground states
        self.excited_mf = np.array(
            [1.5, 1.5, -0.5, -0.5, -0.5, -1.5, -1.5, 0.5, 0.5, 0.5, -2.5, 2.5],
            dtype=np.float32,
        )  # mF values for excited states

        # Lande g-factors for ground and excited states mapped by index.
        self.ground_gf = np.array(
            [-0.6667, 0.6667, -0.6667, 0.6667, 0.6667, 0.6667],
            dtype=np.float32,
        )  # gF values for ground states
        self.excited_gf = np.array(
            [
                0.8004,
                0.9783,
                0.8004,
                2.2250,
                0.9783,
                0.8004,
                0.9783,
                2.2250,
                0.8004,
                0.9783,
                0.8004,
                0.8004,
            ],
            dtype=np.float32,
        )  # gF values for excited states (indices 3, 7 = 2P3/2 F'=1/2)

        self.transition_strength_table = np.zeros(
            (self.number_of_ground_states, self.number_of_excited_states, 3),
            dtype=np.float32,
        )  # (ground, excited, polarization)
        self._populate_transition_strength_table()

    def calculate_transition_frequency_shift(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Calculate the Zeeman frequency shift for a transition.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+), lab frame.
        ground_state : int
            Ground-state index.
        excited_state : int
            Excited-state index.
        magnetic_field_strength : float
            Field magnitude in T; the sign encodes the field direction.

        Returns
        -------
        float
            Transition frequency shift in Hz (added to the transition frequency).
        """
        transition_frequency_shift = (
            simple_elw._calculate_transition_frequency_shift(
                magnetic_field_strength=magnetic_field_strength,
                mu_B=self.mu_B,
                ground_mf=self.ground_mf[ground_state],
                excited_mf=self.excited_mf[excited_state],
                ground_gf=self.ground_gf[ground_state],
                excited_gf=self.excited_gf[excited_state],
            )
        )

        return transition_frequency_shift

    def calculate_rate(
        self,
        saturation_parameters,
        total_saturation_parameter,
        natural_linewidth,
        excitation_rates,
    ):
        """Fill excitation_rates in place from the per-transition saturation.

        Delegates to the shared excitation-rate kernel in
        interaction_wrappers/common.py.

        Parameters
        ----------
        saturation_parameters : ndarray
            Per-transition saturation parameters.
        total_saturation_parameter : float
            Sum of the saturation parameters (drives power broadening).
        natural_linewidth : float
            Natural linewidth in rad/s.
        excitation_rates : ndarray
            Output buffer, written in place.
        """
        common._calculate_excitation_rate(
            saturation_parameters=saturation_parameters,
            total_saturation_parameter=total_saturation_parameter,
            natural_linewidth=natural_linewidth,
            excitation_rates=excitation_rates,
        )

    def calculate_saturation_parameter(
        self,
        polarization: int,
        magnetic_field_strength: float,
        ground_state: float,
        excited_state: float,
        laser_intensity: float,
        natural_linewidth: float,
        saturation_intensity: float,
        effective_transition_frequency: float,
        doppler_shift,
        laser_beam_frequency: float,
        detuning: float,
    ):
        """Saturation parameter for one (laser, ground, excited, pol) tuple.

        Looks up the transition strength for the model, then delegates to the
        shared saturation-parameter kernel in interaction_wrappers/common.py.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+), lab frame.
        magnetic_field_strength : float
            Local field magnitude in T.
        ground_state, excited_state : int
            State indices.
        laser_intensity : float
            Local laser intensity in W/m^2.
        natural_linewidth : float
            Natural linewidth in rad/s.
        saturation_intensity : float
            Saturation intensity in W/m^2.
        effective_transition_frequency : float
            Transition frequency including the Zeeman shift, in Hz.
        doppler_shift : float
            Doppler shift in Hz.
        laser_beam_frequency : float
            Laser frequency in Hz.
        detuning : float
            Fixed laser detuning in rad/s.

        Returns
        -------
        float
            Saturation parameter for the transition.
        """
        transition_strength = self.transition_strength_table[
            ground_state, excited_state, polarization
        ]

        saturation_parameter = common._calculate_saturation_parameter(
            effective_transition_frequency=effective_transition_frequency,
            doppler_shift=doppler_shift,
            laser_beam_frequency=laser_beam_frequency,
            detuning=detuning,
            transition_strength=transition_strength,
            laser_intensity=laser_intensity,
            natural_linewidth=natural_linewidth,
        )

        return saturation_parameter

    def calculate_branching_ratio(
        self,
        polarization: int,
        ground_state: int,
        excited_state: int,
        magnetic_field_strength: float,
    ):
        """Branching ratio (|CG|^2 weight) for a (ground, excited, pol) tuple.

        Used to distribute spontaneous decay over the allowed transitions.

        Parameters
        ----------
        polarization : int
            0 (sigma-), 1 (pi), 2 (sigma+).
        ground_state, excited_state : int
            State indices.
        magnetic_field_strength : float
            Local field magnitude in T (used by the field-dependent models).

        Returns
        -------
        float
            Relative branching weight for the transition.
        """
        return self.transition_strength_table[
            ground_state, excited_state, polarization
        ]

    def _populate_transition_strength_table(self):
        """Build and attach the transition-strength table (Numba-friendly).

        Shaped (n_ground, n_excited, n_pol). Zero-initialized (forbidden /
        zero-strength transitions), then the 38 nonzero known entries are
        filled and the 6 disallowed transitions are explicitly set to zero.
        """
        ng = self.number_of_ground_states
        ne = self.number_of_excited_states
        npol = 3

        # zero-initialized table (default = forbidden / zero-strength)
        ts = np.zeros((ng, ne, npol), dtype=np.float32)

        # -- fill the known nonzero transitions (38 entries) --
        # Entries follow (GS, ES, Pol, trans.strength) with Polarization: 0 -> σ-, 1 -> π, 2 -> σ+; atom frame specification. See Julia's thesis Fig. 2.1 for mapping of indices to |F,mF> states.
        entries = [
            (0, 3, 0, 0.1477),
            (0, 4, 0, 0.0490),
            (1, 2, 0, 0.0750),
            (1, 3, 0, 0.0047),
            (1, 4, 0, 0.0592),
            (2, 6, 0, 0.1389),
            (3, 5, 0, 0.1505),
            (3, 6, 0, 0.0431),
            (4, 10, 0, 0.2500),
            (5, 7, 0, 0.0142),
            (5, 8, 0, 0.0250),
            (5, 9, 0, 0.0443),
            (0, 7, 1, 0.0748),
            (0, 9, 1, 0.0888),
            (1, 7, 1, 0.0088),
            (1, 8, 1, 0.1523),
            (1, 9, 1, 0.0089),
            (2, 3, 1, 0.0759),
            (2, 4, 1, 0.0913),
            (3, 2, 1, 0.1496),
            (3, 3, 1, 0.0090),
            (3, 4, 1, 0.0070),
            (4, 5, 1, 0.1000),
            (4, 6, 1, 0.0647),
            (5, 0, 1, 0.0980),
            (5, 1, 1, 0.0681),
            (0, 1, 2, 0.1389),
            (1, 0, 2, 0.1505),
            (1, 1, 2, 0.0431),
            (2, 7, 2, 0.1477),
            (2, 9, 2, 0.0490),
            (3, 7, 2, 0.0047),
            (3, 8, 2, 0.0750),
            (3, 9, 2, 0.0592),
            (4, 2, 2, 0.0250),
            (4, 3, 2, 0.0142),
            (4, 4, 2, 0.0443),
            (5, 11, 2, 0.2500),
        ]

        for g, e, pol, s in entries:
            ts[g, e, pol] = s

        # -- explicitly set the 6 not allowed transitions to zero (optional)--
        explicit_zero_allowed = [
            (0, 2, 0),  # GS0 -> ES2, sigma-
            (0, 8, 1),  # GS0 -> ES8, pi
            (0, 0, 2),  # GS0 -> ES0, sigma+
            (2, 5, 0),  # GS2 -> ES5, sigma-
            (2, 2, 1),  # GS2 -> ES2, pi
            (2, 8, 2),  # GS2 -> ES8, sigma+
        ]

        for g, e, pol in explicit_zero_allowed:
            ts[g, e, pol] = 0.0

        # attach to instance
        self.transition_strength_table = ts

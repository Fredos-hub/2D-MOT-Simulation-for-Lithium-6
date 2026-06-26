"""Simulation: step loop, atom activation, boundary checks, and lifecycle."""

import numpy as np
from numba import njit

from src.absorption_and_emission_process import (
    absorption_and_emission_default_timestep,
)
from util.simulation_typing import (
    ECSAtoms,
    ECSLasers,
    LightAtomInteraction,
    MagneticField,
)


class Simulation:
    """Simulate atoms interacting with lasers and a magnetic field.

    Gravity (scipy.constants.g, -y direction) is applied inside
    absorption_and_emission_default_timestep at every time advance.
    """

    def __init__(
        self,
        lasers: ECSLasers,
        magnetic_field: MagneticField,
        simulation_atoms: ECSAtoms,
        simulation_interaction: LightAtomInteraction,
        max_step_number: int,
        step_resolution: int,
        simulated_time: float,
        boundaries: np.ndarray,
        default_timestep: float = 1e-5,
        voxel_size: float = 1e-5,
        laser_t_on: np.ndarray = None,
        laser_t_off: np.ndarray = None,
    ):

        # setup objects
        self.lasers = lasers
        self.magnetic_field = magnetic_field
        self.simulation_atoms = simulation_atoms
        self.simulation_interaction = simulation_interaction
        self.boundaries = boundaries

        # procedural simulation parameters
        self.max_step_number = max_step_number
        self.current_step = 0
        self.simulated_time = simulated_time
        self.step_resolution = step_resolution
        self.default_timestep = default_timestep
        self.voxel_size = voxel_size

        # Laser active intervals [t_on, t_off) in global sim time; default always-on.
        # Edges are snapped to step indices so that e.g. t_on = 3*dt is not missed
        # by one ulp when comparing against i*dt.
        n_lasers = lasers.n_lasers
        t_on = laser_t_on if laser_t_on is not None else np.zeros(n_lasers)
        t_off = (
            laser_t_off
            if laser_t_off is not None
            else np.full(n_lasers, np.inf)
        )
        self.laser_on_step = np.rint(t_on / default_timestep).astype(np.int64)
        self.laser_off_step = np.full(
            n_lasers, np.iinfo(np.int64).max, dtype=np.int64
        )
        finite = np.isfinite(t_off)
        self.laser_off_step[finite] = np.rint(
            t_off[finite] / default_timestep
        ).astype(np.int64)
        self._base_beam_powers = lasers.beam_powers.copy()
        self._base_intensities = lasers.initial_intensities.copy()

        # Counter for absorption/emission events for each atom.
        self.excitation_counter = np.zeros(
            self.simulation_atoms.n, dtype=np.int64
        )
        self.excitation_hist = np.zeros(
            (self.simulation_interaction.number_of_ground_states, 3),
            dtype=np.int64,
        )

    def warmup(self, stop_callback=None):
        """Run two throwaway steps to trigger Numba JIT compilation.

        Parameters
        ----------
        stop_callback : callable, optional
            Polled between steps; if it returns True the warmup raises to
            abort early.
        """
        for _ in range(2):
            if stop_callback and stop_callback():
                raise Exception("Canceled during warmup.")
            self.simulation_atoms.magnetic_field_strength[0] = 0.1
            absorption_and_emission_default_timestep(
                atom_ids=np.array([0], dtype=np.int64),
                simulation_atoms=self.simulation_atoms,
                simulation_interaction=self.simulation_interaction,
                magnetic_field=self.magnetic_field,
                lasers=self.lasers,
                excitation_counter=self.excitation_counter,
                default_timestep=self.default_timestep,
                excitation_hist=self.excitation_hist,
            )
        print("Warmup step completed.")

    def step(self, i):
        """Process a single simulation step.

        Returns a consistent tuple: (cont: bool, current_atom_states, excitation_counter, alive_ids)
        """
        # 0) Activate inactive atoms whose overshoot means they should start now
        # status: -1 = inactive (not yet "born"), 0 = dead, 1 = alive

        inactive_mask = self.simulation_atoms.status == -1
        if np.any(inactive_mask):
            inactive_ids = np.where(inactive_mask)[0]
            # activate those that will have an event within the upcoming default timestep
            to_activate_mask = (
                self.simulation_atoms.time_overshoot[inactive_ids]
                <= self.default_timestep
            )
            ids_to_activate = inactive_ids[to_activate_mask]
            if ids_to_activate.size > 0:
                # mark them alive; keep their time_overshoot value (the inner event loop will use it)
                self.simulation_atoms.status[ids_to_activate] = 1

        # 1) Find alive atoms
        alive_ids = check_if_alive(
            self.simulation_atoms.atom_ids, self.simulation_atoms.status
        )
        if alive_ids.size == 0:
            print("No atoms live, simulation stopping.")
            # return consistent tuple
            return (
                False,
                self.simulation_atoms,
                self.excitation_counter,
                alive_ids,
                self.excitation_hist,
            )

        # 2) Apply laser active intervals: zero inactive beams, restore active ones.
        # Idempotent each step, so this stays correct across checkpoint resume.
        active = (self.laser_on_step <= i) & (i < self.laser_off_step)
        self.lasers.beam_powers[:] = np.where(
            active, self._base_beam_powers, 0.0
        )
        self.lasers.initial_intensities[:] = np.where(
            active, self._base_intensities, 0.0
        )

        # 3) Do physics
        absorption_and_emission_default_timestep(
            atom_ids=alive_ids,
            simulation_atoms=self.simulation_atoms,
            simulation_interaction=self.simulation_interaction,
            magnetic_field=self.magnetic_field,
            lasers=self.lasers,
            excitation_counter=self.excitation_counter,
            default_timestep=self.default_timestep,
            excitation_hist=self.excitation_hist,
        )

        # 5) Advance the step counter
        self.current_step = i + 1

        self.simulation_atoms.subjective_time += self.default_timestep
        # 6) process boundary/time kills (same logic as before)

        # Boundary kills (z)
        z_alive = self.simulation_atoms.positions[alive_ids, 2]
        too_far = np.abs(z_alive) >= self.boundaries[2]
        ids_to_kill_z = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_z] = 0

        # Boundary kills (y)
        # FIXME: Zeeman Boundaries
        y_alive = self.simulation_atoms.positions[alive_ids, 1]
        too_far = np.abs(y_alive) >= self.boundaries[1]
        ids_to_kill_y = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_y] = 0

        # Boundary kills (x)
        x_alive = self.simulation_atoms.positions[alive_ids, 0]
        too_far = np.abs(x_alive) >= self.boundaries[0]
        ids_to_kill_x = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_x] = 0

        # Subtract default_timestep from still-inactive atoms' overshoot
        # (clamp >= 0)
        inactive_mask = self.simulation_atoms.status == -1
        if np.any(inactive_mask):
            leftover = (
                self.simulation_atoms.time_overshoot[inactive_mask]
                - self.default_timestep
            )
            # clamp to zero to avoid negatives
            leftover = np.where(leftover < 0.0, 0.0, leftover)
            self.simulation_atoms.time_overshoot[inactive_mask] = leftover

        # 7) Max‐step check
        if self.current_step >= self.max_step_number:
            print("Maximum step number reached, simulation stopping.")
            self.simulation_atoms.status[:] = 0
            return (
                False,
                self.simulation_atoms,
                self.excitation_counter,
                alive_ids,
                self.excitation_hist,
            )

        # always return consistent tuple
        return (
            True,
            self.simulation_atoms,
            self.excitation_counter,
            alive_ids,
            self.excitation_hist,
        )

    def finalize(self):
        """Finalize the run; currently a no-op."""
        return


@njit
def check_if_alive(atom_ids, statuses):
    """Return only the atom_ids whose status is 1 (alive)."""
    alive_mask = statuses[atom_ids] == 1
    return atom_ids[alive_mask]

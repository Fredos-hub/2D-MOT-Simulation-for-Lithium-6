import numpy as np
from numba import njit
from src.absorption_and_emission_process import absorption_and_emission_default_timestep
from util.simulation_typing import MagneticField, ECSAtoms, LightAtomInteraction, ECSLasers
import time

class Simulation():
    """
    Handles simulation of atoms interacting with lasers and a magnetic field.
    Dead atoms are advanced each step along a straight line and are affected
    by gravity via a -0.5 * g * dt**2 term in z.
    """
    def __init__(self, lasers: ECSLasers,
                 magnetic_field: MagneticField, 
                 simulation_atoms: ECSAtoms, 
                 simulation_interaction: LightAtomInteraction, 
                 max_step_number: int,
                 step_resolution: int,
                 max_live_time: float, 
                 boundaries,
                 default_timestep: float = 1e-5,
                 voxel_size: float = 1e-5,
                 gravity: float = 9.81):   # new gravity parameter

        # setup objects
        self.lasers = lasers
        self.magnetic_field = magnetic_field
        self.simulation_atoms = simulation_atoms
        self.simulation_interaction = simulation_interaction
        self.boundaries = boundaries

        # procedural simulation parameters
        self.max_step_number = max_step_number
        self.current_step = 0
        self.max_live_time = max_live_time
        self.step_resolution = step_resolution
        self.default_timestep = default_timestep
        self.voxel_size = voxel_size
        self.gravity = gravity

        # Counter for absorption/emission events for each atom.
        self.excitation_counter = np.zeros(self.simulation_atoms.n, dtype=np.int64)


    def warmup(self, stop_callback=None):
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
                default_timestep=self.default_timestep
            )

    def step(self, i):
        """
        Process a single simulation step.

        Returns a consistent tuple: (cont: bool, current_atom_states, excitation_counter, alive_ids)
        """

        # ---------- 0) Activate inactive atoms whose overshoot means they should start now ----------
        # status: -1 = inactive (not yet "born"), 0 = dead, 1 = alive






        inactive_mask = (self.simulation_atoms.status == -1)
        if np.any(inactive_mask):
            inactive_ids = np.where(inactive_mask)[0]
            # activate those that will have an event within the upcoming default timestep
            to_activate_mask = self.simulation_atoms.time_overshoot[inactive_ids] <= self.default_timestep
            ids_to_activate = inactive_ids[to_activate_mask]
            if ids_to_activate.size > 0:
                # mark them alive; keep their time_overshoot value (the inner event loop will use it)
                self.simulation_atoms.status[ids_to_activate] = 1
        # -----------------------------------------------------------------------------------------


        # 1) Find alive atoms
        alive_ids = check_if_alive(self.simulation_atoms.atom_ids,
                                self.simulation_atoms.status)
        if alive_ids.size == 0:
            print("No atoms live, simulation stopping.")
            # return consistent tuple
            return False, self.simulation_atoms, self.excitation_counter, alive_ids

 
        # 3) Do physics
        #self.magnetic_field.calculate_magnetic_field(self.simulation_atoms, alive_ids)
        #self.magnetic_field.calculate_max_step_length(self.simulation_atoms, alive_ids)
        absorption_and_emission_default_timestep(
            atom_ids=alive_ids,
            simulation_atoms=self.simulation_atoms,
            simulation_interaction=self.simulation_interaction,
            magnetic_field=self.magnetic_field,
            lasers=self.lasers,
            excitation_counter=self.excitation_counter,
            default_timestep=self.default_timestep
        )


        # 5) Advance the step counter
        self.current_step = i + 1

        self.simulation_atoms.subjective_time += self.default_timestep
        # 6) process boundary/time kills (same logic as before)

        # Boundary kills (z)
        z_alive = self.simulation_atoms.positions[alive_ids, 2]
        too_far = np.abs(z_alive) >= 0.015
        ids_to_kill_z = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_z] = 0

        # Boundary kills (y)
        y_alive = self.simulation_atoms.positions[alive_ids, 1]
        too_far = np.abs(y_alive) >= 0.0271
        ids_to_kill_y = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_y] = 0

        # Boundary kills (x)
        x_alive = self.simulation_atoms.positions[alive_ids, 0]
        too_far = np.abs(x_alive) >= 0.021
        ids_to_kill_x = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_x] = 0


        # ---------- Subtract default_timestep from still-inactive atoms' overshoot (clamp >= 0) ----------
        inactive_mask = (self.simulation_atoms.status == -1)
        if np.any(inactive_mask):
            leftover = self.simulation_atoms.time_overshoot[inactive_mask] - self.default_timestep
            # clamp to zero to avoid negatives
            leftover = np.where(leftover < 0.0, 0.0, leftover)
            self.simulation_atoms.time_overshoot[inactive_mask] = leftover
        # -----------------------------------------------------------------------------------------------



        #Add times for each step (probably in ms)
        # 7) Max‐step check
        if self.current_step >= self.max_step_number:
            print("Maximum step number reached, simulation stopping.")
            self.simulation_atoms.status[:] = 0
            return False, self.simulation_atoms, self.excitation_counter, alive_ids

        # always return consistent tuple
        return True, self.simulation_atoms, self.excitation_counter, alive_ids



        
    def finalize(self):
        return
    




    def step_debug(self, i):
        """
        Process a single simulation step.

        Returns a consistent tuple: (cont: bool, current_atom_states, excitation_counter, alive_ids)
        """

        t0 = time.perf_counter()

        # ---------- 0) Activate inactive atoms whose overshoot means they should start now ----------
        t_act_start = time.perf_counter()
        inactive_mask = (self.simulation_atoms.status == -1)
        if np.any(inactive_mask):
            inactive_ids = np.where(inactive_mask)[0]
            # activate those that will have an event within the upcoming default timestep
            to_activate_mask = self.simulation_atoms.time_overshoot[inactive_ids] <= self.default_timestep
            ids_to_activate = inactive_ids[to_activate_mask]
            if ids_to_activate.size > 0:
                # mark them alive; keep their time_overshoot value (the inner event loop will use it)
                self.simulation_atoms.status[ids_to_activate] = 1
        t_act_end = time.perf_counter()
        print(f"[timing] activation          : {(t_act_end - t_act_start)*1000:.3f} ms")
        # -----------------------------------------------------------------------------------------

        # 1) Find alive atoms
        t_alive_start = time.perf_counter()
        alive_ids = check_if_alive(self.simulation_atoms.atom_ids,
                                self.simulation_atoms.status)
        t_alive_end = time.perf_counter()
        print(f"[timing] collect alive ids   : {(t_alive_end - t_alive_start)*1000:.3f} ms")

        if alive_ids.size == 0:
            print("No atoms live, simulation stopping.")
            # return consistent tuple
            t_end = time.perf_counter()
            print(f"[timing] whole step          : {(t_end - t0)*1000:.3f} ms")
            return False, self.simulation_atoms, self.excitation_counter, alive_ids

        # 3) Do physics
        t_field_start = time.perf_counter()
        self.magnetic_field.calculate_magnetic_field(self.simulation_atoms, alive_ids)
        t_field_end = time.perf_counter()
        print(f"[timing] mag field calc      : {(t_field_end - t_field_start)*1000:.3f} ms")

        t_maxsteplen_start = time.perf_counter()
        self.magnetic_field.calculate_max_step_length(self.simulation_atoms, alive_ids)
        t_maxsteplen_end = time.perf_counter()
        print(f"[timing] max step length calc: {(t_maxsteplen_end - t_maxsteplen_start)*1000:.3f} ms")
        counters = np.zeros(6, dtype=np.int64)
        t_events_start = time.perf_counter()
        absorption_and_emission_default_timestep(
            atom_ids=alive_ids,
            simulation_atoms=self.simulation_atoms,
            simulation_interaction=self.simulation_interaction,
            magnetic_field=self.magnetic_field,
            lasers=self.lasers,
            excitation_counter=self.excitation_counter,
            default_timestep=self.default_timestep
        )
        t_events_end = time.perf_counter()
        print(f"[timing] events (abs/emi)    : {(t_events_end - t_events_start)*1000:.3f} ms")

        # 5) Advance the step counter
        t_stepadv_start = time.perf_counter()
        self.current_step = i + 1
        # Update subjective time
        self.simulation_atoms.subjective_time += self.default_timestep
        t_stepadv_end = time.perf_counter()
        print(f"[timing] step advance/timeupd : {(t_stepadv_end - t_stepadv_start)*1000:.3f} ms")

        # 6) process boundary/time kills
        # Boundary kills (z)
        t_kill_z_start = time.perf_counter()
        z_alive = self.simulation_atoms.positions[alive_ids, 2]
        too_far = np.abs(z_alive) >= 0.021
        ids_to_kill_z = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_z] = 0
        t_kill_z_end = time.perf_counter()
        print(f"[timing] boundary kill (z)   : {(t_kill_z_end - t_kill_z_start)*1000:.3f} ms")

        # Boundary kills (y)
        t_kill_y_start = time.perf_counter()
        y_alive = self.simulation_atoms.positions[alive_ids, 1]
        too_far = np.abs(y_alive) >= 0.0271
        ids_to_kill_y = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_y] = 0
        t_kill_y_end = time.perf_counter()
        print(f"[timing] boundary kill (y)   : {(t_kill_y_end - t_kill_y_start)*1000:.3f} ms")

        # Boundary kills (x)
        t_kill_x_start = time.perf_counter()
        x_alive = self.simulation_atoms.positions[alive_ids, 0]
        too_far = np.abs(x_alive) >= 0.021
        ids_to_kill_x = alive_ids[too_far]
        self.simulation_atoms.status[ids_to_kill_x] = 0
        t_kill_x_end = time.perf_counter()
        print(f"[timing] boundary kill (x)   : {(t_kill_x_end - t_kill_x_start)*1000:.3f} ms")

        # ---------- Subtract default_timestep from still-inactive atoms' overshoot (clamp >= 0) ----------
        t_overshoot_start = time.perf_counter()
        inactive_mask = (self.simulation_atoms.status == -1)
        if np.any(inactive_mask):
            leftover = self.simulation_atoms.time_overshoot[inactive_mask] - self.default_timestep
            # clamp to zero to avoid negatives
            leftover = np.where(leftover < 0.0, 0.0, leftover)
            self.simulation_atoms.time_overshoot[inactive_mask] = leftover
        t_overshoot_end = time.perf_counter()
        print(f"[timing] overshoot decrement : {(t_overshoot_end - t_overshoot_start)*1000:.3f} ms")
        # -----------------------------------------------------------------------------------------------

        t_end = time.perf_counter()
        print(f"[timing] whole step          : {(t_end - t0)*1000:.3f} ms")
        print(f"Debug Counter Output {counters}")
        # 7) Max‐step check
        if self.current_step >= self.max_step_number:
            print("Maximum step number reached, simulation stopping.")
            self.simulation_atoms.status[:] = 0
            return False, self.simulation_atoms, self.excitation_counter, alive_ids

        # always return consistent tuple
        return True, self.simulation_atoms, self.excitation_counter, alive_ids



    def finalize(self):
        return
@njit
def check_if_alive(atom_ids, statuses):
    """
    Filters the atom_ids to include only those with a status of 1.
    """
    alive_mask = statuses[atom_ids] == 1
    return atom_ids[alive_mask]





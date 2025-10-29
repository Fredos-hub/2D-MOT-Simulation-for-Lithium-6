import json
import numpy as np

import scipy.constants as scc
import src.interactions as interactions
from src.magnetic_field import ZeemanField, IdealQuadropoleField, DipoleBarMagneticField, EllipticalMagneticField
import numpy as np
import pandas as pd
from src.simulate import Simulation
import src.atoms as atoms
from src.experimental_setup import SimpleECSBoundaries, LaserComponent
from util.simulation_typing import ECSAtoms, LightAtomInteraction

#Class to parse the Data from the JSON files
class Parameters:
    """
    Loads simulation parameters from a JSON file and unpacks them into class attributes.
    
    This class extracts simulation settings, atom parameters, magnetic field settings,
    laser parameters, boundaries (converting old format to new if necessary), and oven
    parameters (if applicable). The boundaries are converted into a list-of-dictionaries,
    and vector fields are converted to NumPy arrays for further processing.
    """
    def __init__(self, filename: str) -> None:
        # Validate filename.
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string.")
        if not filename.endswith('.json'):
            raise ValueError("File must be a JSON file with a .json extension.")

        # Load the JSON file.
        with open(filename, 'r', encoding='utf-8') as file:
            self.parameters = json.load(file)

        # --- Simulation settings ---
        sim = self.parameters["Simulation"]
        self.default_time_step = np.float64(sim["default_time_step"]*1e-6)
        self.step_resolution = sim["step_resolution"]
        self.max_live_time = sim["max_live_time"] * 1e-3
        self.max_step_number = round(self.max_live_time/self.default_time_step)
        self.interaction = sim["interaction"]
        self.seed = sim["random_seed"]
        self.flux = sim["flux"] * 1e9
        self.macro_particle_weight = sim["macro_particle_weight"]
        self.rate_mode = sim["rate_mode"]


        # --- Atom parameters ---
        #FIXME: HARD FIX, REMOVE ASAP
        self.random_emission = False

        atom_data = self.parameters["Atoms"]
        self.atom_species = atom_data["species"]
        self.atom_number = atom_data["number"]
        self.natural_linewidth = atom_data["natural_linewidth"]*1e6*2*scc.pi
        self.start_position = np.array(atom_data["start_position"], dtype=np.float64)
        self.start_velocity = np.array(atom_data["start_velocity"], dtype=np.float64)
        self.ground_states= atom_data["ground_state"]
        self.randomize_groundstates = atom_data["randomize_ground_state"]
        self.sample_file = atom_data["sample_file"]

        field_data = self.parameters["Magnetic_Fields"]
        self.magnetic_field_type = field_data["type"]

        # Zeeman params
        self.slower_length = field_data.get("slower_length", None)
        self.B_0 = field_data.get("B_0", None)
        self.B_bias = field_data.get("B_bias", None)
        self.delta_B = field_data.get("delta_B", 0.0) / 100.0  # still supported for Zeeman

        # Quadrupole params
        self.field_gradient = field_data.get("field_gradient", None)

        # Elliptical params
        self.g_x = field_data.get("g_x", None)

        if self.magnetic_field_type == "EllipticalMagneticField":
            self.g_y = field_data.get("g_y", None)
            self.g_z = -(self.g_x+self.g_y)
            self.theta_deg = field_data["theta_deg"]  # stored in degrees

        # shared offset (in meters)
        self.offset = np.array(field_data.get("center_offset", [0.0, 0.0, 0.0]),
                               dtype=np.float64) * 1e-3

        if self.magnetic_field_type == "DipoleBarMagneticField":
            self.dipoles = []
            for dipole in field_data["dipoles"]:
                self.dipoles.append({
                    "position": np.array(dipole["position"], dtype=np.float64),
                    "dimension": np.array(dipole["dimension"], dtype=np.float64),
                    "orientation": np.array(dipole["orientation"], dtype=np.float64),
                    "magnetization": dipole["magnetization"],
                })
        # --- Laser parameters ---
        self.lasers = []
        #FIXME: Implement lasers as dicts with key + value
        for laser in self.parameters["Lasers"]:
            self.lasers.append({

                "waist": laser["waist"],
                "origin": np.array(laser["origin"], dtype=np.float64),
                "direction": np.array(laser["direction"], dtype=np.float64),
                "beam_power": laser["beam_power"],
                "beam_frequency": laser["beam_frequency"] *1e6,
                "detuning": laser["detuning"]*self.natural_linewidth,
                "handedness": laser["handedness"]
            })


    def build_simulation(self, status_callback=None, stop_callback=None):

        seed = self.seed
        setup_rng = np.random.default_rng(seed = seed )
        status_callback("Parameters loaded successfully")

        #Initialize Interactions
        status_callback("Loading interactions")
        if hasattr(interactions, self.interaction):
            chosen_interaction = getattr(interactions, self.interaction)
            simulation_interaction: LightAtomInteraction = chosen_interaction()
            status_callback(f"Interaction: {chosen_interaction} initialized")
        else:
            status_callback("failed to initialize interactions")
            # Optional: Exception werfen oder einen Default-Wert setzen
            raise ValueError(f"Interaction class '{self.interaction}' not found!")

        
        # Creation of the ECSAtoms Object:
        status_callback("Creating atoms...")

        # 1) Load sample data if present
        if self.sample_file:
            sample_data = pd.read_csv(self.sample_file)
            subj = sample_data["subjective_time"].to_numpy(dtype=np.float64)
            is_snapshot = np.any(subj != 0.0)
        else:
            sample_data = None
            is_snapshot = False

        # 2) Determine atom_number & start_times
        if sample_data is None or not is_snapshot:
            # either no sample file, or “flux sample” (all subjective_time==0)
            atom_number, start_times = self.find_atom_number_and_start_time(setup_rng)
        else:
            # snapshot: just continue existing ensemble
            atom_number = len(sample_data)
            start_times = subj

        # 3) Instantiate ECSAtoms
        if not hasattr(atoms, self.atom_species):
            status_callback("failed to create atoms")
            raise ValueError(f"Atom species '{self.atom_species}' not found!")
        simulation_atoms = getattr(atoms, self.atom_species)(atom_number)
        status_callback(f"Atoms of type {self.atom_species} created")

        # 4) Initialize positions/velocities/ground-states
        if sample_data is None:
            if self.random_emission == True:
                self.sample_file = "C:\\Users\\frede\\Desktop\\Masterarbeit\\starting condition samples\\Our MOT starting conditions emission -110-623K-6_2_mm_MBDV3.csv"
                sample_data = pd.read_csv(self.sample_file)
                subj = sample_data["subjective_time"].to_numpy(dtype=np.float64)
                is_snapshot = np.any(subj != 0.0)
                atom_number, start_times = self.find_atom_number_and_start_time(setup_rng)

                status_callback("Sampling from whack hotfix…")
                Nfile = len(sample_data)
                idx = (setup_rng.random(atom_number) * Nfile).astype(int).clip(0, Nfile-1)

                pos = sample_data[["x","y","z"]].to_numpy()[idx]         # (N,3)
                sample_vel = sample_data[["vx","vy","vz"]].to_numpy()[idx]  # (N,3)

                # per-row norms as shape (N,1) so broadcasting works
                vel_norms = np.linalg.norm(sample_vel, axis=1, keepdims=True)  # (N,1)

                # avoid division by zero (or very tiny values)
                eps = 1e-12
                zero_mask = (vel_norms <= eps).flatten()
                if zero_mask.any():
                    # choose fallback directions for zero rows (here +x unit vector)
                    sample_vel[zero_mask, :] = np.array([1.0, 0.0, 0.0])
                    vel_norms = np.linalg.norm(sample_vel, axis=1, keepdims=True)

                direction = sample_vel / vel_norms   # (N,3)

                # compute target speed (handles scalar or vector start_velocity)
                target_speed = np.linalg.norm(self.start_velocity)

                vel = direction * target_speed      # (N,3)

                # ground states
                if "ground_state" in sample_data:
                    gs = sample_data["ground_state"].to_numpy(dtype=np.int32)[idx]
                else:
                    n_gs = simulation_interaction.number_of_ground_states
                    gs = setup_rng.integers(0, n_gs, size=atom_number, dtype=np.int32)

                simulation_atoms.set_starting_conditions(pos, vel, gs, start_times)
            else:
                # — simplified uniform init —
                status_callback("Setting simplified initial conditions…")
                pos = np.full((atom_number, 3), self.start_position)
                vel = np.full((atom_number, 3), self.start_velocity)
                if self.randomize_groundstates:
                    n_gs = simulation_interaction.number_of_ground_states
                    gs = setup_rng.integers(0, n_gs, size=atom_number, dtype=np.int32)
                else:
                    gs = np.full(atom_number, self.ground_states, dtype=np.int32)

                simulation_atoms.set_starting_conditions(pos, vel, gs, start_times)
                status_callback("Atoms starting conditions set (simplified)")

        else:
            if is_snapshot:
                # — snapshot continuation —
                status_callback("Loading snapshot initial conditions…")
                simulation_atoms.positions = sample_data[["x","y","z"]].to_numpy()
                simulation_atoms.velocities = sample_data[["vx","vy","vz"]].to_numpy()
                simulation_atoms.time_overshoot = start_times
                simulation_atoms.groundstates = sample_data["ground_state"].to_numpy(dtype=np.int32)
                status_callback("Snapshot conditions applied")
            else:
                # — flux-sample: inject according to rate, sample pos/vel from file —
                status_callback("Sampling from flux-distribution file…")
                Nfile = len(sample_data)
                idx = (setup_rng.random(atom_number) * Nfile).astype(int).clip(0, Nfile-1)

                pos = sample_data[["x","y","z"]].to_numpy()[idx]
                vel = sample_data[["vx","vy","vz"]].to_numpy()[idx]
                if "ground_state" in sample_data:
                    gs = sample_data["ground_state"].to_numpy(dtype=np.int32)[idx]
                else:
                    n_gs = simulation_interaction.number_of_ground_states
                    gs = setup_rng.integers(0, n_gs, size=atom_number, dtype=np.int32)

                simulation_atoms.set_starting_conditions(pos, vel, gs, start_times)
                status_callback("Atoms starting conditions set (flux sample)")

        status_callback("Initializing magnetic field...")

        if self.magnetic_field_type == "ZeemanField":
            B_field = ZeemanField(
                slower_length=self.slower_length,
                B_0=self.B_0,
                B_bias=self.B_bias,
                delta_B=self.delta_B,   # only here
            )

        elif self.magnetic_field_type == "IdealQuadropoleField":
            B_field = IdealQuadropoleField(
                gradient=self.field_gradient,
                offset=self.offset,
            )

        elif self.magnetic_field_type == "EllipticalMagneticField":
            B_field = EllipticalMagneticField(
                g_x=self.g_x,
                g_y=self.g_y,
                g_z = self.g_z,
                theta=self.theta_deg,   # convert degrees → radians
                offset=self.offset,
            )

        elif self.magnetic_field_type == "DipoleBarMagneticField":
            B_field = DipoleBarMagneticField(len(self.dipoles))
            for index, dipole in enumerate(self.dipoles):
                B_field.add_dipole(
                    index,
                    dipole["position"],
                    dipole["dimension"],
                    dipole["orientation"],
                    dipole["magnetization"],
                )
        else:
            raise ValueError(f"Unsupported magnetic field type: {self.magnetic_field_type}")

        status_callback("Magnetic Field Initialized")
        
        # Initialize Lasers.
        status_callback("Initializing lasers...")
        mot_lasers = LaserComponent(len(self.lasers))
        for index in  range (len(self.lasers)):
            mot_lasers.add_laser(
                index,
                self.lasers[index]["waist"],
                self.lasers[index]["origin"],
                self.lasers[index]["direction"],
                self.lasers[index]["beam_power"],
                self.lasers[index]["beam_frequency"],
                self.lasers[index]["detuning"],
                self.lasers[index]["handedness"]
            )

        status_callback("Lasers initialized")

        #FIXME: change how boundaries work
        ecs_boundaries = []
        status_callback("Creating simulation instance...")


        simulation = Simulation(
            lasers=mot_lasers,
            magnetic_field=B_field,
            simulation_atoms=simulation_atoms,
            simulation_interaction = simulation_interaction,
            max_step_number=self.max_step_number,
            step_resolution=self.step_resolution,
            max_live_time=self.max_live_time,
            boundaries=ecs_boundaries,
            default_timestep=self.default_time_step
        )
        status_callback("Simulation instance created")

        return simulation

    def find_atom_number_and_start_time(self, rng):
        """
        Returns (atom_number, start_times_array).
        - If rate mode on: draws exponential inter-arrivals until max_live_time.
        - Else: returns self.atom_number, all-zero times.
        """
        if self.rate_mode:
            rate = self.flux / self.macro_particle_weight
            times = []
            t = 0.0
            while True:
                dt = rng.exponential(1.0 / rate)
                t += dt
                if t > self.max_live_time:
                    break
                times.append(t)
            return len(times), np.array(times, dtype=np.float64)
        else:
            n = self.atom_number
            return n, np.zeros(n, dtype=np.float64)

    def save_to_file(self, filename: str) -> None:
        """
        Save the current parameters to a JSON file.
        
        Updates the internal parameters dictionary for boundaries and then writes it.
        """

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(self.parameters, file, indent=4)

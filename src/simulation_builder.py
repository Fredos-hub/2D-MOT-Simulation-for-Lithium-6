from src.parameters import Parameters
import src.interactions as interactions

from src.magnetic_field import ZeemanField, IdealQuadropoleField, DipoleBarMagneticField
import numpy as np
import csv
import pandas as pd
from src.simulate import Simulation, SimulationCollision
import src.atoms as atoms
from src.experimental_setup import SimpleECSBoundaries, LaserComponent
from util.simulation_typing import ECSAtoms, LightAtomInteraction

def build_simulation(filename: str = None, status_callback=None, stop_callback=None):
    if not filename:
        raise ValueError("No file given")
    
    # Check before starting.
    if stop_callback and stop_callback():
        raise Exception("Canceled before loading parameters.")
    
    status_callback("Loading parameters from JSON...")
    params = Parameters(filename)
    seed = params.seed
    setup_rng = np.random.default_rng(seed = seed )
    status_callback("Parameters loaded successfully")

    #Initialize Interactions
    status_callback("Loading interactions")
    if hasattr(interactions, params.interaction):
        chosen_interaction = getattr(interactions, params.interaction)
        simulation_interaction: LightAtomInteraction = chosen_interaction()
        status_callback(f"Interaction: {chosen_interaction} initialized")
    else:
        status_callback("failed to initialize interactions")
        # Optional: Exception werfen oder einen Default-Wert setzen
        raise ValueError(f"Interaction class '{params.interaction}' not found!")

    
    # Creation of the ECSAtoms Object:
    status_callback("Creating atoms...")

    # 1) Load sample data if present
    if params.sample_file:
        sample_data = pd.read_csv(params.sample_file)
        subj = sample_data["subjective_time"].to_numpy(dtype=np.float64)
        is_snapshot = np.any(subj != 0.0)
    else:
        sample_data = None
        is_snapshot = False

    # 2) Determine atom_number & start_times
    if sample_data is None or not is_snapshot:
        # either no sample file, or “flux sample” (all subjective_time==0)
        atom_number, start_times = find_atom_number_and_start_time(params, setup_rng)
    else:
        # snapshot: just continue existing ensemble
        atom_number = len(sample_data)
        start_times = subj

    # 3) Instantiate ECSAtoms
    if not hasattr(atoms, params.atom_species):
        status_callback("failed to create atoms")
        raise ValueError(f"Atom species '{params.atom_species}' not found!")
    simulation_atoms = getattr(atoms, params.atom_species)(atom_number)
    status_callback(f"Atoms of type {params.atom_species} created")

    # 4) Initialize positions/velocities/ground-states
    if sample_data is None:
        # — simplified uniform init —
        status_callback("Setting simplified initial conditions…")
        pos = np.full((atom_number, 3), params.start_position)
        vel = np.full((atom_number, 3), params.start_velocity)
        if params.randomize_groundstates:
            n_gs = simulation_interaction.number_of_ground_states
            gs = setup_rng.integers(0, n_gs, size=atom_number, dtype=np.int32)
        else:
            gs = np.full(atom_number, params.ground_states, dtype=np.int32)

        simulation_atoms.set_starting_conditions(pos, vel, gs, start_times)
        status_callback("Atoms starting conditions set (simplified)")

    else:
        if is_snapshot:
            # — snapshot continuation —
            status_callback("Loading snapshot initial conditions…")
            simulation_atoms.positions = sample_data[["x","y","z"]].to_numpy()
            simulation_atoms.velocities = sample_data[["vx","vy","vz"]].to_numpy()
            simulation_atoms.subjective_time = start_times
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






    # Initialize Magnetic Field
    status_callback("Initializing magnetic field...")
    if params.magnetic_field_type == "ZeemanField":
        B_field = ZeemanField(
            slower_length=params.slower_length,
            B_0=params.B_0,
            B_bias=params.B_bias,
            delta_B=params.delta_B
        )

    elif params.magnetic_field_type == "IdealQuadropoleField":
        B_field = IdealQuadropoleField(gradient = params.field_gradient,
                                       delta_B=params.delta_B,
                                       offset = params.offset)
        

    elif params.magnetic_field_type == "DipoleBarMagneticField":
        B_field = DipoleBarMagneticField(len(params.dipoles))
        for index in range(len(params.dipoles)):
            B_field.add_dipole(
                index,
                params.dipoles[index]["position"],
                params.dipoles[index]["dimension"], 
                params.dipoles[index]["orientation"],
                params.dipoles[index]["magnetization"])

    else: 
        raise ValueError("Unsupported magnetic field type")
    status_callback("Magnetic Field Initialized")
    
    # Initialize Lasers.
    status_callback("Initializing lasers...")
    mot_lasers = LaserComponent(len(params.lasers))
    for index in  range (len(params.lasers)):
        mot_lasers.add_laser(
            index,
            params.lasers[index]["waist"],
            params.lasers[index]["origin"],
            params.lasers[index]["direction"],
            params.lasers[index]["beam_power"],
            params.lasers[index]["beam_frequency"],
            params.lasers[index]["detuning"],
            params.lasers[index]["handedness"]
        )

    status_callback("Lasers initialized")

    #FIXME: change how boundaries work
    ecs_boundaries = []
    status_callback("Creating simulation instance...")


    output_file = params.parameters["Simulation"]["output_file"]
    simulation = SimulationCollision(
        lasers=mot_lasers,
        magnetic_field=B_field,
        simulation_atoms=simulation_atoms,
        simulation_interaction = simulation_interaction,
        max_step_number=params.max_step_number,
        step_resolution=params.step_resolution,
        max_live_time=params.max_live_time,
        boundaries=ecs_boundaries,
        output_file=output_file
    )
    status_callback("Simulation instance created")

    return simulation






def find_atom_number_and_start_time(params: Parameters, rng):
    """
    Returns (atom_number, start_times_array).
    - If rate mode on: draws exponential inter-arrivals until max_live_time.
    - Else: returns params.atom_number, all-zero times.
    """
    if params.rate_mode:
        rate = params.flux / params.macro_particle_weight
        times = []
        t = 0.0
        while True:
            dt = rng.exponential(1.0 / rate)
            t += dt
            if t > params.max_live_time:
                break
            times.append(t)
        return len(times), np.array(times, dtype=np.float64)
    else:
        n = params.atom_number
        return n, np.zeros(n, dtype=np.float64)

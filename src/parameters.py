"""Parameters loader and simulation builder.

Purpose
-------
This module provides a Parameters class that performs three tasks
in sequence. First, the JSON configuration file is validated against a JSON
schema to ensure required keys and types are present. Second, the validated
configuration is parsed into a compact set of Python attributes and
structures suitable for construction of simulation objects. Third, a Simulation
instance is created while guarding against runtime problems such as missing
classes, constructor changes, or I/O errors.

Usage
-----
- Create an instance with the path to a configuration JSON and the schema file.
- If schema validation fails, a SchemaValidationError is raised and includes
  a list of human readable messages.
- If initialization succeeds, the Parameters instance holds parsed values
  and can construct a Simulation instance by calling build_simulation().
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from jsonschema import Draft7Validator

import src.atoms as atoms
import src.interactions as interactions
from src.lasers import LaserComponent
from src.magnetic_field import (
    DipoleBarMagneticField,
    EllipticalMagneticField,
    IdealQuadrupoleField,
    ZeemanField,
)
from src.simulate import Simulation
from util.simulation_typing import ECSAtoms, LightAtomInteraction

StatusCallback = Optional[Callable[[str], None]]


class SchemaValidationError(Exception):
    """Raised when JSON configuration fails schema validation.

    Attributes
    ----------
    errors : List[str]
        A list of short human readable validation messages suitable for GUI
        display or logging.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "\n".join(errors)
        super().__init__(message)


class ParameterError(Exception):
    """Raised for fatal parameter or construction errors after validation."""


def validate_against_schema(config: dict[str, Any], schema_path: str) -> None:
    """Validate a configuration dictionary against a JSON schema.

    Produces more informative messages for `oneOf` / `anyOf` failures by
    walking the error context. Raises SchemaValidationError with a
    deduplicated, ordered list of messages.
    """
    schema_file = Path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found at {schema_path}")

    with schema_file.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: list(e.path))

    messages: list[str] = []

    def _flatten_error(err) -> None:
        path = ".".join(map(str, err.absolute_path)) or "root"
        messages.append(f"{path}: {err.message}")
        for sub in getattr(err, "context", []):
            sub_path = ".".join(map(str, sub.absolute_path)) or path
            messages.append(f"{sub_path}: {sub.message}")

    for err in errors:
        _flatten_error(err)

    seen = set()
    out: list[str] = []
    for m in messages:
        if m not in seen:
            seen.add(m)
            out.append(m)

    if out:
        raise SchemaValidationError(out)


class Parameters:
    """Configuration container and simulation builder.

    Responsibilities
    ----------------
    - Validate configuration JSON against a schema
    - Parse validated configuration into internal attributes
    - Create a Simulation instance with guarded runtime initialization
    """

    def __init__(
        self,
        filename: str,
        schema_path: str = "GUI/schema/schema_v1.json",
        status_callback: StatusCallback = None,
    ) -> None:
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        if not isinstance(schema_path, str):
            raise TypeError("schema_path must be a string")

        self.filename = filename
        self.schema_path = schema_path
        self.status_callback = status_callback
        self.errors: list[str] = []

        self.valid: bool = True
        self.parameters: dict[str, Any] = {}

        # simulation fields
        self.default_time_step: np.float64 | None = None
        self.step_resolution: int | None = None
        self.simulated_time: float | None = None
        self.max_step_number: int | None = None
        self.interaction: str | None = None
        self.seed: int | None = None
        self.flux: float | None = None
        self.macro_particle_weight: float | None = None
        self.rate_mode: bool | None = None

        # atoms
        self.random_emission: bool = False
        self.atom_species: str | None = None
        self.atom_number: int | None = None
        self.natural_linewidth: float | None = None
        self.start_position: np.ndarray | None = None
        self.start_velocity: np.ndarray | None = None
        self.ground_states: int | None = None
        self.randomize_groundstates: bool | None = None
        self.sample_file: str | None = None
        self.sample_style: str | None = None

        # magnetic field
        self.magnetic_field_type: str | None = None
        self.slower_length: float | None = None
        self.B_0: float | None = None
        self.B_bias: float | None = None
        self.delta_B: float | None = None
        self.delta_B_min: float | None = None
        self.field_gradient: float | None = None
        self.g_x: float | None = None
        self.g_y: float | None = None
        self.g_z: float | None = None
        self.theta_deg: float | None = None
        self.offset: np.ndarray | None = None
        self.dipoles: list[dict[str, Any]] = []

        # lasers and boundaries
        self.lasers: list[dict[str, Any]] = []
        self.boundaries: np.ndarray | None = None

        self._call_status("Loading configuration file")
        try:
            raw = self._load_json_file(self.filename)
        except Exception as exc:
            msg = f"Failed to read configuration file: {exc}"
            self.errors.append(msg)
            self._call_status(msg)
            self.valid = False
            return

        self._call_status("Validating configuration against schema")
        try:
            validate_against_schema(raw, self.schema_path)
        except SchemaValidationError as exc:
            self.errors.extend(exc.errors)
            summary = f"Schema validation failed: {len(self.errors)} error(s)."
            self._call_status("ERROR: " + summary)
            self.valid = False
            self.parameters = raw
            return
        except Exception as exc:
            msg = f"Schema validation failed with unexpected error: {exc}"
            self.errors.append(msg)
            self._call_status(msg)
            self.valid = False
            self.parameters = raw
            return

        self.parameters = raw
        self._call_status("Schema validation succeeded")
        try:
            self._parse_simulation()
            self._parse_atoms()
            self._parse_magnetic_fields()
            self._parse_lasers()
            self._parse_boundaries()
        except Exception as exc:
            msg = f"Configuration parsing failed: {exc}"
            self.errors.append(msg)
            self._call_status(msg)
            self.valid = False
            return

        self._call_status("Configuration parsing completed")

    # Internal utility methods
    def _call_status(self, message: str) -> None:
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass

    def _load_json_file(self, filename: str) -> dict[str, Any]:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {filename}"
            )
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            raise ParameterError(
                f"Failed to read JSON file {filename}: {exc}"
            ) from exc
        return data

    # Parsing helpers
    def _parse_simulation(self) -> None:
        sim = self.parameters["Simulation"]
        self.default_time_step = np.float64(sim["default_time_step"] * 1e-6)
        self.step_resolution = int(sim["step_resolution"])
        self.simulated_time = float(sim["simulated_time"] * 1e-3)
        self.max_step_number = round(
            self.simulated_time / self.default_time_step
        )
        self.interaction = sim["interaction"]
        self.seed = int(sim["random_seed"])
        self.flux = float(sim["flux"] * 1e9)
        self.macro_particle_weight = float(sim["macro_particle_weight"])
        self.rate_mode = bool(sim["rate_mode"])
        # Oven cutoff: stop injecting atoms at this time (ms in JSON), so the run
        # can continue past it to let pushed atoms reach the detector. <=0/absent
        # means inject for the whole simulation (backward-compatible).
        cutoff = sim.get("injection_cutoff_time", None)
        self.injection_cutoff_time = (
            float(cutoff * 1e-3)
            if cutoff and cutoff > 0
            else self.simulated_time
        )

    def _parse_atoms(self) -> None:
        atom_data = self.parameters["Atoms"]
        self.atom_species = atom_data["species"]
        self.atom_number = int(atom_data["number"])
        self.natural_linewidth = float(
            atom_data["natural_linewidth"] * 1e6 * 2 * math.pi
        )
        self.start_position = np.array(
            atom_data["start_position"], dtype=np.float64
        )
        self.start_velocity = np.array(
            atom_data["start_velocity"], dtype=np.float64
        )

        self.ground_states = int(atom_data["ground_state"])
        self.randomize_groundstates = bool(atom_data["randomize_ground_state"])
        self.sample_style = atom_data.get("sample_style", "oven")
        self.sample_file = atom_data.get("sample_file", None)

    def _parse_magnetic_fields(self) -> None:
        field_data = self.parameters["Magnetic_Fields"]
        self.magnetic_field_type = field_data["type"]
        self.offset = (
            np.array(
                field_data.get("center_offset", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            * 1e-3
        )

        if self.magnetic_field_type == "ZeemanField":
            self.slower_length = field_data.get("slower_length", None)
            self.B_0 = field_data.get("B_0", None)
            self.B_bias = field_data.get(
                "B_bias", field_data.get("B_Bias", None)
            )
            self.delta_B = field_data.get("delta_B", 0.0) / 100.0
            self.delta_B_min = field_data.get("delta_B_min", 0.0)

        elif self.magnetic_field_type == "IdealQuadrupoleField":
            self.field_gradient = field_data["field_gradient"]

        elif self.magnetic_field_type == "EllipticalMagneticField":
            self.g_x = field_data["g_x"]
            self.g_y = field_data["g_y"]
            self.g_z = -(self.g_x + self.g_y)
            self.theta_deg = field_data["theta_deg"]

        elif self.magnetic_field_type == "DipoleBarMagneticField":
            self.dipoles = []
            for dipole in field_data["dipoles"]:
                self.dipoles.append(
                    {
                        "position": np.array(
                            dipole["position"], dtype=np.float64
                        ),
                        "dimension": np.array(
                            dipole["dimension"], dtype=np.float64
                        ),
                        "orientation": np.array(
                            dipole["orientation"], dtype=np.float64
                        ),
                        "magnetization": float(dipole["magnetization"]),
                    }
                )
        else:
            raise ParameterError(
                f"Unsupported magnetic field type: {self.magnetic_field_type}"
            )

    def _parse_lasers(self) -> None:
        lasers = self.parameters["Lasers"]
        parsed: list[dict[str, Any]] = []
        for laser in lasers:
            parsed.append(
                {
                    "waist": float(laser["waist"]),
                    "origin": np.array(laser["origin"], dtype=np.float64),
                    "direction": np.array(
                        laser["direction"], dtype=np.float64
                    ),
                    "beam_power": float(laser["beam_power"]),
                    "beam_frequency": float(laser["beam_frequency"]) * 1e6,
                    "detuning": float(laser["detuning"])
                    * self.natural_linewidth,
                    "handedness": int(laser["handedness"]),
                    "type": laser.get("type", "other"),
                    # active interval [t_on, t_off) in seconds; omitted/null t_off = always on
                    "t_on": float(laser.get("t_on", 0.0)) * 1e-3,
                    "t_off": float(laser["t_off"]) * 1e-3
                    if laser.get("t_off") is not None
                    else np.inf,
                }
            )
        for index, entry in enumerate(parsed):
            if entry["t_off"] <= entry["t_on"]:
                self.errors.append(
                    f"Laser #{index}: t_off must be greater than t_on."
                )
        self.lasers = parsed

    def _parse_boundaries(self) -> None:
        b = self.parameters["Boundaries"]
        self.boundaries = (
            np.array(
                [b["x_limit"], b["y_limit"], b["z_limit"]],
                dtype=np.float64,
            )
            * 1e-3
        )

    # Build simulation
    def build_simulation(self) -> Simulation:
        """Create a Simulation instance from parsed parameters."""
        rng = np.random.default_rng(seed=self.seed)

        self._call_status("Initializing interaction")
        try:
            if not hasattr(interactions, self.interaction):
                raise ParameterError(
                    f"Interaction '{self.interaction}' not found in src.interactions"
                )
            interaction_cls = getattr(interactions, self.interaction)
            simulation_interaction: LightAtomInteraction = interaction_cls()
        except Exception as exc:
            msg = f"Failed to initialize interaction: {exc}"
            self._call_status(msg)
            raise ParameterError(msg) from exc

        sample_data: pd.DataFrame | None = None
        if self.sample_file:
            try:
                sample_data = pd.read_csv(self.sample_file)
                self._call_status("Sample file loaded")
            except Exception as exc:
                msg = f"Failed to read sample file {self.sample_file}: {exc}"
                self.errors.append(msg)
                self._call_status(msg)
                sample_data = None

        atom_number, start_times = self._find_atom_number_and_start_time(
            rng,
            sample_data,
            self.sample_style,
        )

        if not hasattr(atoms, self.atom_species):
            msg = f"Atom species '{self.atom_species}' not found in src.atoms"
            self._call_status(msg)
            raise ParameterError(msg)

        simulation_atoms = getattr(atoms, self.atom_species)(atom_number)
        self._call_status(f"Atom container for {self.atom_species} created")

        try:
            self._initialize_atoms_from_sample_or_defaults(
                simulation_atoms,
                sample_data,
                start_times,
                rng,
                simulation_interaction,
            )
        except Exception as exc:
            msg = f"Failed to initialize atom starting conditions: {exc}"
            self._call_status(msg)
            raise ParameterError(msg) from exc

        try:
            B_field = self._construct_magnetic_field()
            self._call_status("Magnetic field constructed")
        except Exception as exc:
            msg = f"Failed to construct magnetic field: {exc}"
            self._call_status(msg)
            raise ParameterError(msg) from exc

        mot_lasers = LaserComponent(len(self.lasers))
        for index, laser in enumerate(self.lasers):
            try:
                mot_lasers.add_laser(
                    index,
                    laser["waist"],
                    laser["origin"],
                    laser["direction"],
                    laser["beam_power"],
                    laser["beam_frequency"],
                    laser["detuning"],
                    laser["handedness"],
                )
            except Exception as exc:
                msg = f"Failed to add laser #{index}: {exc}"
                self.errors.append(msg)
                self._call_status(msg)

        try:
            simulation = Simulation(
                lasers=mot_lasers,
                magnetic_field=B_field,
                simulation_atoms=simulation_atoms,
                simulation_interaction=simulation_interaction,
                max_step_number=self.max_step_number,
                step_resolution=self.step_resolution,
                simulated_time=self.simulated_time,
                boundaries=self.boundaries,
                default_timestep=self.default_time_step,
                laser_t_on=np.array(
                    [l["t_on"] for l in self.lasers], dtype=np.float64
                ),
                laser_t_off=np.array(
                    [l["t_off"] for l in self.lasers], dtype=np.float64
                ),
            )
        except Exception as exc:
            msg = f"Failed to construct Simulation: {exc}"
            self._call_status(msg)
            raise ParameterError(msg) from exc

        self._call_status("Simulation instance created")
        return simulation

    # Helper functions for build
    def _find_atom_number_and_start_time(
        self,
        rng: np.random.Generator,
        sample_data: pd.DataFrame | None,
        sample_style: str | None,
    ) -> tuple[int, np.ndarray]:
        """Return tuple (atom_number, start_times_array).

        Cases:
        1) No sample file: use the values directly from the JSON.
        2) sample_style == "snapshot": sample from the snapshot file with
           replacement allowed if the file has fewer rows than requested.
        3) sample_style == "oven": preserve the current oven-file behavior.
        """
        if not self.sample_file or sample_data is None:
            if self.rate_mode:
                rate = self.flux / self.macro_particle_weight
                times: list[float] = []
                t = 0.0
                while True:
                    dt = rng.exponential(1.0 / rate)
                    t += dt
                    if t > self.injection_cutoff_time:
                        break
                    times.append(t)
                return len(times), np.array(times, dtype=np.float64)

            n = int(self.atom_number)
            return n, np.zeros(n, dtype=np.float64)

        if sample_style == "snapshot":
            n_target = int(self.atom_number)
            n_file = len(sample_data)
            idx = rng.integers(0, n_file, size=n_target)
            subj = sample_data["subjective_time"].to_numpy(dtype=np.float64)[
                idx
            ]
            return n_target, subj

        # oven-style sample file path: keep existing behavior
        if self.rate_mode:
            rate = self.flux / self.macro_particle_weight
            times: list[float] = []
            t = 0.0
            while True:
                dt = rng.exponential(1.0 / rate)
                t += dt
                if t > self.injection_cutoff_time:
                    break
                times.append(t)
            return len(times), np.array(times, dtype=np.float64)

        n = int(self.atom_number)
        return n, np.zeros(n, dtype=np.float64)

    def _initialize_atoms_from_sample_or_defaults(
        self,
        simulation_atoms: ECSAtoms,
        sample_data: pd.DataFrame | None,
        start_times: np.ndarray,
        rng: np.random.Generator,
        simulation_interaction: LightAtomInteraction,
    ) -> None:
        if sample_data is None or not self.sample_file:
            self._initialize_atoms_no_sample(
                simulation_atoms, start_times, rng, simulation_interaction
            )
        elif self.sample_style == "snapshot":
            self._initialize_atoms_snapshot(simulation_atoms, sample_data, rng)
        else:
            self._initialize_atoms_oven(
                simulation_atoms,
                sample_data,
                start_times,
                rng,
                simulation_interaction,
            )

    def _initialize_atoms_no_sample(
        self,
        simulation_atoms: ECSAtoms,
        start_times: np.ndarray,
        rng: np.random.Generator,
        simulation_interaction: LightAtomInteraction,
    ) -> None:
        N = simulation_atoms.n
        if self.random_emission:
            if not self.sample_file:
                msg = "random_emission is enabled but no valid sample file is available"
                self.errors.append(msg)
                self._call_status(msg)
                raise ParameterError(msg)

            sample_hot = pd.read_csv(self.sample_file)
            n_file = len(sample_hot)
            idx = (rng.random(N) * n_file).astype(int).clip(0, n_file - 1)

            pos = sample_hot[["x", "y", "z"]].to_numpy()[idx]
            sample_vel = sample_hot[["vx", "vy", "vz"]].to_numpy()[idx]

            vel_norms = np.linalg.norm(sample_vel, axis=1, keepdims=True)
            eps = 1e-12
            zero_mask = (vel_norms <= eps).flatten()
            if zero_mask.any():
                sample_vel[zero_mask, :] = np.array([1.0, 0.0, 0.0])
                vel_norms = np.linalg.norm(sample_vel, axis=1, keepdims=True)

            direction = sample_vel / vel_norms
            target_speed = np.linalg.norm(self.start_velocity)
            vel = direction * target_speed

            if "current_groundstate" in sample_hot.columns:
                gs = sample_hot["current_groundstate"].to_numpy(
                    dtype=np.int32
                )[idx]
            else:
                n_gs = simulation_interaction.number_of_ground_states
                gs = rng.integers(0, n_gs, size=N, dtype=np.int32)

            simulation_atoms.set_starting_conditions(pos, vel, gs, start_times)
            self._call_status("Atoms initialized via random_emission path")

        else:
            pos = np.full((N, 3), self.start_position, dtype=np.float64)
            vel = np.full((N, 3), self.start_velocity, dtype=np.float64)

            if self.randomize_groundstates:
                n_gs = simulation_interaction.number_of_ground_states
                gs = rng.integers(0, n_gs, size=N, dtype=np.int32)
            else:
                gs = np.full(N, self.ground_states, dtype=np.int32)

            simulation_atoms.set_starting_conditions(pos, vel, gs, start_times)
            self._call_status("Atoms starting conditions set (uniform)")

    def _initialize_atoms_snapshot(
        self,
        simulation_atoms: ECSAtoms,
        sample_data: pd.DataFrame,
        rng: np.random.Generator,
    ) -> None:
        N = simulation_atoms.n
        n_file = len(sample_data)
        idx = rng.integers(0, n_file, size=N)

        sample_pos = np.ascontiguousarray(
            sample_data[["x", "y", "z"]].to_numpy(dtype=np.float64)[idx]
        )
        sample_vel = np.ascontiguousarray(
            sample_data[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(
                dtype=np.float64
            )[idx]
        )
        sample_gs = np.ascontiguousarray(
            sample_data["current_groundstate"].to_numpy(dtype=np.int32)[idx]
        )
        subj = np.ascontiguousarray(
            sample_data["subjective_time"].to_numpy(dtype=np.float64)[idx]
        )
        zero_times = np.zeros(N, dtype=np.float64)

        simulation_atoms.set_starting_conditions(
            sample_pos, sample_vel, sample_gs, zero_times
        )
        simulation_atoms.subjective_time[:] = subj
        simulation_atoms.time_overshoot[:] = 0.0
        write_atoms_to_csv(simulation_atoms, "snapshot_initial_conditions.csv")
        self._call_status("Snapshot initial conditions applied")

    def _initialize_atoms_oven(
        self,
        simulation_atoms: ECSAtoms,
        sample_data: pd.DataFrame,
        start_times: np.ndarray,
        rng: np.random.Generator,
        simulation_interaction: LightAtomInteraction,
    ) -> None:
        N = simulation_atoms.n
        subj = sample_data.get("subjective_time", pd.Series(0.0)).to_numpy(
            dtype=np.float64
        )
        snapshot_flag = bool((subj != 0.0).any())

        if snapshot_flag:
            pos = sample_data[["x", "y", "z"]].to_numpy(dtype=np.float64)
            vel = sample_data[
                ["velocity_x", "velocity_y", "velocity_z"]
            ].to_numpy(dtype=np.float64)

            if "current_groundstate" not in sample_data.columns:
                raise ParameterError(
                    "Sample file missing required column 'current_groundstate'"
                )

            gs = sample_data["current_groundstate"].to_numpy(dtype=np.int32)
            zero_times = np.zeros(len(pos), dtype=np.float64)

            simulation_atoms.set_starting_conditions(pos, vel, gs, zero_times)
            simulation_atoms.subjective_time[:] = subj
            write_atoms_to_csv(
                simulation_atoms, "oven_snapshot_initial_conditions.csv"
            )
            self._call_status("Snapshot initial conditions applied")

        else:
            n_file = len(sample_data)
            idx = (rng.random(N) * n_file).astype(int).clip(0, n_file - 1)

            pos = sample_data[["x", "y", "z"]].to_numpy()[idx]
            vel = sample_data[["vx", "vy", "vz"]].to_numpy()[idx]

            n_gs = simulation_interaction.number_of_ground_states
            gs = rng.integers(0, n_gs, size=N, dtype=np.int32)

            simulation_atoms.set_starting_conditions(pos, vel, gs, start_times)
            write_atoms_to_csv(
                simulation_atoms, "oven_snapshot_initial_conditions.csv"
            )
            self._call_status("Atoms initialized from flux sample file")

    def _construct_magnetic_field(self):
        t = self.magnetic_field_type
        if t == "ZeemanField":
            return ZeemanField(
                slower_length=self.slower_length,
                B_0=self.B_0,
                B_bias=self.B_bias,
                delta_B=self.delta_B,
                delta_B_min=self.delta_B_min,
            )
        elif t == "IdealQuadrupoleField":
            return IdealQuadrupoleField(
                gradient=self.field_gradient, offset=self.offset
            )
        elif t == "EllipticalMagneticField":
            return EllipticalMagneticField(
                g_x=self.g_x,
                g_y=self.g_y,
                g_z=self.g_z,
                theta=self.theta_deg,
                offset=self.offset,
            )
        elif t == "DipoleBarMagneticField":
            B_field = DipoleBarMagneticField(len(self.dipoles))
            for idx, dip in enumerate(self.dipoles):
                B_field.add_dipole(
                    idx,
                    dip["position"],
                    dip["dimension"],
                    dip["orientation"],
                    dip["magnetization"],
                )
            return B_field
        else:
            raise ParameterError(f"Unsupported magnetic field type: {t}")

    # Persistence
    def save_to_file(self, filename: str) -> None:
        """Write back the internal parameters dictionary to disk.

        The method updates the Boundaries values from the internal numpy
        representation so that saved JSON reflects any in-memory changes.
        """
        if self.boundaries is not None:
            self.parameters.setdefault("Boundaries", {})
            self.parameters["Boundaries"]["x_limit"] = float(
                self.boundaries[0] * 1e3
            )
            self.parameters["Boundaries"]["y_limit"] = float(
                self.boundaries[1] * 1e3
            )
            self.parameters["Boundaries"]["z_limit"] = float(
                self.boundaries[2] * 1e3
            )

        path = Path(filename)
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(self.parameters, fh, indent=4)
            self._call_status(f"Configuration saved to {filename}")
        except Exception as exc:
            msg = f"Failed to save configuration to {filename}: {exc}"
            self.errors.append(msg)
            self._call_status(msg)
            raise ParameterError(msg) from exc

    def is_valid(self) -> bool:
        """Return True when the parameters are valid and ready to build.

        True only after validation and parsing completed successfully.
        """
        return getattr(self, "valid", False)

    def get_errors(self) -> list[str]:
        """Return a copy of the current errors list suitable for display."""
        return list(self.errors)


def write_atoms_to_csv(atoms: ECSAtoms, filename: str) -> None:
    """Write per-atom positions, velocities, and state to a CSV file.

    Parameters
    ----------
    atoms : ECSAtoms
        Atom container to serialize.
    filename : str
        Output CSV path.
    """
    data = {
        "atom_id": atoms.atom_ids,
        "x": atoms.positions[:, 0],
        "y": atoms.positions[:, 1],
        "z": atoms.positions[:, 2],
        "velocity_x": atoms.velocities[:, 0],
        "velocity_y": atoms.velocities[:, 1],
        "velocity_z": atoms.velocities[:, 2],
        "current_groundstate": atoms.groundstates,
        "subjective_time": atoms.subjective_time,
        "time_overshoot": atoms.time_overshoot,
        "status": atoms.status,
    }
    pd.DataFrame(data).to_csv(filename, index=False)

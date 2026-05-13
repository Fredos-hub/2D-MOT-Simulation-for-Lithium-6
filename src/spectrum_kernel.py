"""
One-shot spectroscopy scan kernel.

Given a snapshot of atoms (positions, velocities, ground states) and a
probe-beam configuration, this module sweeps a detuning range and returns
the total steady-state scattering rate at each point. The physics is the
same the simulation loop uses (Gaussian beam intensity, Doppler shift,
Zeeman-shifted transition frequencies, saturation-broadened rate), but
without the Monte Carlo time stepping.

The interaction-jitclass methods are issued once during a precompute pass
(detuning = 0, doppler = 0) to extract per-(atom, excited_state, polarization)
coefficients, after which the detuning sweep is pure NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.constants as scc

from src.atoms import Li6
from src.interactions import (
    Lithium6LevelInteraction,
    Lithium18LevelInteraction,
    Lithium4LevelInteraction,
    SimpleEighteenLevelInteraction,
)
from src.magnetic_field import (
    IdealQuadrupoleField,
    ZeemanField,
    DipoleBarMagneticField,
    EllipticalMagneticField,
)


# Same physical constants the Li6 jitclass exposes, duplicated here so the
# kernel does not depend on instantiating an atoms object.
ATOM_NATURAL_LINEWIDTH = 2.0 * math.pi * 5.87e6        # rad/s
ATOM_TRANSITION_FREQUENCY = 446799648.889e6            # Hz, D2 line CoG
ATOM_SATURATION_INTENSITY = (
    math.pi * scc.h * scc.c * ATOM_NATURAL_LINEWIDTH
    / (3.0 * (scc.c / ATOM_TRANSITION_FREQUENCY) ** 3)
)


INTERACTION_BUILDERS = {
    "Lithium6LevelInteraction": Lithium6LevelInteraction,
    "Lithium18LevelInteraction": Lithium18LevelInteraction,
    "Lithium4LevelInteraction": Lithium4LevelInteraction,
    "SimpleEighteenLevelInteraction": SimpleEighteenLevelInteraction,
}


@dataclass
class BeamConfig:
    origin_m: np.ndarray         # shape (3,)
    direction: np.ndarray        # shape (3,) — normalised internally
    power_W: float
    frequency_Hz: float
    detuning_offset_rad: float   # fixed beam detuning (added to every scan point), rad/s
    handedness: int              # -1, 0, +1
    radius_m: float              # 1/e² waist
    use_position: bool           # False: every atom sees peak intensity


@dataclass
class ScanResult:
    detunings_MHz: np.ndarray
    rates: np.ndarray            # raw total scattering rate per detuning [photons/s]
    n_atoms: int
    groundstates: np.ndarray     # sorted unique ground-state indices that contributed
    rates_per_groundstate: np.ndarray  # shape (len(groundstates), len(detunings))
    counts_per_groundstate: np.ndarray  # shape (len(groundstates),)


def build_interaction(name: str):
    cls = INTERACTION_BUILDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown interaction model: {name}")
    return cls()


def build_magnetic_field(config: Optional[dict]):
    """
    Build a magnetic-field jitclass from a parameters.json 'Magnetic_Fields'
    block. Returns None if config is None (caller treats this as B = 0).
    """
    if config is None:
        return None

    t = config["type"]
    offset = np.array(
        config.get("center_offset", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ) * 1e-3

    if t == "IdealQuadrupoleField":
        return IdealQuadrupoleField(gradient=config["field_gradient"], offset=offset)
    if t == "ZeemanField":
        return ZeemanField(
            slower_length=config.get("slower_length"),
            B_0=config.get("B_0"),
            B_bias=config.get("B_bias", config.get("B_Bias")),
            delta_B=config.get("delta_B", 0.0) / 100.0,
            delta_B_min=config.get("delta_B_min", 0.0),
        )
    if t == "EllipticalMagneticField":
        g_x = config["g_x"]
        g_y = config["g_y"]
        return EllipticalMagneticField(
            g_x=g_x,
            g_y=g_y,
            g_z=-(g_x + g_y),
            theta=config["theta_deg"],
            offset=offset,
        )
    if t == "DipoleBarMagneticField":
        dipoles = config["dipoles"]
        field = DipoleBarMagneticField(len(dipoles))
        for idx, d in enumerate(dipoles):
            field.add_dipole(
                idx,
                np.array(d["position"], dtype=np.float64),
                np.array(d["dimension"], dtype=np.float64),
                np.array(d["orientation"], dtype=np.float64),
                float(d["magnetization"]),
            )
        return field

    raise ValueError(f"Unsupported magnetic field type: {t}")


def _handedness_weights(angle: float, handedness: int) -> np.ndarray:
    """
    Squared matrix elements for (σ-, π, σ+) given beam handedness and the
    angle between propagation direction and the local B-field axis. Mirrors
    `calculate_handedness_to_polarization` in absorption_and_emission_process.
    """
    w = np.zeros(3, dtype=np.float64)
    if handedness == 1:
        w[0] = 0.25 * (1.0 - math.cos(angle)) ** 2
        w[1] = 0.5 * math.sin(angle) ** 2
        w[2] = 0.25 * (1.0 + math.cos(angle)) ** 2
    elif handedness == -1:
        w[0] = 0.25 * (1.0 + math.cos(angle)) ** 2
        w[1] = 0.5 * math.sin(angle) ** 2
        w[2] = 0.25 * (1.0 - math.cos(angle)) ** 2
    else:
        # Linear polarization along k̂ (π along the propagation axis).
        # |d¹_{m,0}(θ)|² with θ = ∠(k̂, B̂): σ± = ½ sin²θ, π = cos²θ, sum = 1.
        w[0] = 0.5 * math.sin(angle) ** 2
        w[1] = math.cos(angle) ** 2
        w[2] = 0.5 * math.sin(angle) ** 2
    return w


def _gaussian_intensity(positions, origin, direction, waist, wavelength, peak):
    """Vectorised Gaussian beam intensity. Zero outside the local waist."""
    rel = positions - origin
    z = rel @ direction
    perp = rel - np.outer(z, direction)
    r2 = np.einsum("ij,ij->i", perp, perp)
    zR = math.pi * waist ** 2 / wavelength
    w = waist * np.sqrt(1.0 + (z / zR) ** 2)
    inside = r2 < w ** 2
    intensity = np.zeros_like(z)
    if np.any(inside):
        intensity[inside] = (
            peak * (waist / w[inside]) ** 2
            * np.exp(-2.0 * r2[inside] / w[inside] ** 2)
        )
    return intensity


def compute_spectrum_scan(
    positions: np.ndarray,
    velocities: np.ndarray,
    ground_states: np.ndarray,
    interaction,
    magnetic_field,
    beam: BeamConfig,
    detunings_MHz: np.ndarray,
) -> ScanResult:
    """
    Compute the total scattering rate per detuning across all snapshot atoms.
    """
    n_atoms = positions.shape[0]
    n_ex = int(interaction.number_of_excited_states)
    n_gs = int(interaction.number_of_ground_states)
    Gamma = ATOM_NATURAL_LINEWIDTH

    max_gs = int(np.max(ground_states)) if n_atoms > 0 else -1
    if max_gs >= n_gs:
        raise ValueError(
            f"Snapshot contains groundstate index {max_gs} but the selected "
            f"interaction only supports {n_gs} ground states. Use a richer model."
        )

    direction = np.asarray(beam.direction, dtype=np.float64)
    dnorm = np.linalg.norm(direction)
    if dnorm == 0.0:
        raise ValueError("Beam direction must be non-zero.")
    direction = direction / dnorm

    wavelength = scc.c / beam.frequency_Hz
    k = 2.0 * math.pi / wavelength
    wave_vector = k * direction
    peak_intensity = 2.0 * beam.power_W / (math.pi * beam.radius_m ** 2)

    if beam.use_position:
        I_atom = _gaussian_intensity(
            positions, np.asarray(beam.origin_m, dtype=np.float64),
            direction, beam.radius_m, wavelength, peak_intensity,
        )
    else:
        I_atom = np.full(n_atoms, peak_intensity, dtype=np.float64)

    doppler = velocities @ wave_vector

    # Per-atom B-field via the existing jitclass machinery
    if magnetic_field is None:
        b_vec = np.zeros((n_atoms, 3), dtype=np.float64)
        b_norm = np.zeros(n_atoms, dtype=np.float64)
    else:
        atoms = Li6(n=n_atoms)
        atoms.positions = positions.astype(np.float64).copy()
        for i in range(n_atoms):
            magnetic_field.calculate_magnetic_field(atoms, i)
        b_vec = np.asarray(atoms.magnetic_field_vectors).copy()
        b_norm = np.asarray(atoms.magnetic_field_strength).copy()

    # Angle B vs beam direction (0 where B is undefined)
    angle = np.zeros(n_atoms, dtype=np.float64)
    has_b = b_norm > 0.0
    if np.any(has_b):
        cos_a = (b_vec[has_b] @ direction) / b_norm[has_b]
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angle[has_b] = np.arccos(cos_a)

    sq_matrix = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        sq_matrix[i] = _handedness_weights(angle[i], beam.handedness)
    rel_I = sq_matrix * I_atom[:, None]   # (n_atoms, 3)

    # Precompute eff_trans_freq and S_0 per (atom, excited, polarization)
    eff_trans_freq_Hz = np.zeros((n_atoms, n_ex, 3), dtype=np.float64)
    S_0 = np.zeros((n_atoms, n_ex, 3), dtype=np.float64)

    laser_beam_freq_Hz = beam.frequency_Hz
    gs_arr = np.asarray(ground_states, dtype=np.int32)

    for i in range(n_atoms):
        gs = int(gs_arr[i])
        B = float(b_norm[i])
        for ex in range(n_ex):
            for pol in range(3):
                # Signature: (polarization, ground_state, excited_state, B)
                zeeman_shift = interaction.calculate_transition_frequency_shift(
                    pol, gs, ex, B,
                )
                eff_trans_freq_Hz[i, ex, pol] = ATOM_TRANSITION_FREQUENCY + zeeman_shift
                rI = rel_I[i, pol]
                if rI > 0.0:
                    # Signature: (polarization, B, ground_state, excited_state,
                    #             laser_intensity, natural_linewidth, saturation_intensity,
                    #             effective_transition_frequency, doppler_shift,
                    #             laser_beam_frequency, detuning)
                    sat0 = interaction.calculate_saturation_parameter(
                        pol, B, gs, ex,
                        rI, Gamma, ATOM_SATURATION_INTENSITY,
                        eff_trans_freq_Hz[i, ex, pol], 0.0,
                        laser_beam_freq_Hz, 0.0,
                    )
                    S_0[i, ex, pol] = sat0

    # Back out 0.5 * Ω² from S_0 (the only intensity-dependent factor).
    laser_beam_freq_rad = laser_beam_freq_Hz * 2.0 * math.pi
    eff_trans_freq_rad = eff_trans_freq_Hz * 2.0 * math.pi
    eff_det_at_zero = laser_beam_freq_rad - eff_trans_freq_rad
    rabi_squared_half = S_0 * (eff_det_at_zero ** 2 + 0.25 * Gamma ** 2)

    # One-hot mask over unique ground-state values for the per-GS breakdown.
    unique_gs = np.unique(gs_arr).astype(np.int32)
    gs_onehot = np.zeros((n_atoms, unique_gs.size), dtype=np.float64)
    for k, g in enumerate(unique_gs):
        gs_onehot[:, k] = (gs_arr == g).astype(np.float64)
    counts_per_gs = gs_onehot.sum(axis=0).astype(np.int64)

    detunings_rad = 2.0 * math.pi * np.asarray(detunings_MHz, dtype=np.float64) * 1e6
    rates = np.zeros(detunings_rad.size, dtype=np.float64)
    rates_per_gs = np.zeros((unique_gs.size, detunings_rad.size), dtype=np.float64)
    doppler_b = doppler[:, None, None]

    for d_idx, dr in enumerate(detunings_rad):
        full_det_rad = dr + beam.detuning_offset_rad
        eff_det = laser_beam_freq_rad - doppler_b + full_det_rad - eff_trans_freq_rad
        sat = rabi_squared_half / (eff_det ** 2 + 0.25 * Gamma ** 2)
        total_sat = sat.sum(axis=(1, 2))
        rate = 0.5 * Gamma * sat / (1.0 + total_sat[:, None, None])
        rate_per_atom = rate.sum(axis=(1, 2))
        rates[d_idx] = rate_per_atom.sum()
        rates_per_gs[:, d_idx] = gs_onehot.T @ rate_per_atom

    return ScanResult(
        detunings_MHz=np.asarray(detunings_MHz, dtype=np.float64),
        rates=rates,
        n_atoms=n_atoms,
        groundstates=unique_gs,
        rates_per_groundstate=rates_per_gs,
        counts_per_groundstate=counts_per_gs,
    )

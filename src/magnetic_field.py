"""Magnetic-field jitclasses: ideal quadrupole, Zeeman, elliptical, dipole-bar."""

import math

import numpy as np
from matplotlib import pyplot as plt
from numba import float64, int32, njit
from numba.experimental import jitclass

from util.simulation_typing import ECSAtoms

# Adaptive step-size B-field thresholds for calculate_max_step_length.
# Derived from ~0.5 T/m × reference distances.
_B_FINE = 1.25e-3  # 0.5 T/m × 2.5 mm
_B_MID = 5.0e-3  # 0.5 T/m × 10 mm
_B_COARSE = 1.0e-2  # 0.5 T/m × 20 mm

# Ideal Quadrupole Magnetic Field Class

quadrupole_spec = [("gradient", float64), ("offset", float64[:])]


@jitclass(quadrupole_spec)
class IdealQuadrupoleField:
    """Represents an ideal quadrupole magnetic field with a linear gradient.

    In the x–y plane the field is defined as:

        B = gradient * (y, x, 0)

    (Note: In the calculation function below the roles of x and y are interchanged.
    Here we follow the formulas in the wrapper functions.)
    """

    def __init__(self, gradient: float, offset: np.ndarray) -> None:
        self.gradient = gradient
        self.offset = offset

    def calculate_magnetic_field(
        self, simulation_atoms: ECSAtoms, atom_id: np.ndarray
    ) -> None:
        """Update the atoms' field vector using the ideal-quadrupole formula."""
        calculate_ideal_quadrupole_field(
            self.gradient, self.offset, simulation_atoms, atom_id
        )

    def field_at_positions(
        self, positions: np.ndarray
    ) -> np.ndarray[np.ndarray]:
        """Compute the magnetic field (and its norm) at arbitrary positions.

        Used for plotting.

        Parameters
        ----------
        positions : np.ndarray
            Array of positions with shape (..., 3).

        Returns
        -------
        B : np.ndarray
            Magnetic field vector at the given positions.
        norm : np.ndarray
            Magnitude of the magnetic field.

        Uses the same formulas as in the wrapper functions:
            B_x = gradient * y
            B_y = gradient * x
            B_z = 0
        """
        x = positions[..., 0]
        y = positions[..., 1]
        B_x = self.gradient * y
        B_y = self.gradient * x
        B_z = np.zeros_like(x)
        B = np.stack((B_x, B_y, B_z), axis=-1)
        norm = np.sqrt(B_x**2 + B_y**2 + B_z**2)
        return B, norm

    def calculate_max_step_length(
        self, simulation_atoms: ECSAtoms, atom_id: np.ndarray
    ) -> None:
        """Set the adaptive max step length for the atoms from the local field."""
        # Extract the field strengths for these atom IDs
        B = simulation_atoms.magnetic_field_strength[atom_id]

        if B >= 1e-1:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif B >= _B_FINE:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        else:
            # B < _B_FINE including B == 0: high scattering rate, use fine step
            simulation_atoms.max_step_lengths[atom_id] = 1e-6

        return

    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):
        """Approximate mean free path between excitation events, in m."""
        mean_free_path = mean_excitation_time * math.sqrt(
            atom_velocity[0] ** 2 + atom_velocity[1] ** 2
        )

        return mean_free_path

    def calculate_max_time_step(self, max_step_length, atom_velocity):
        """Largest time step keeping the atom within max_step_length, in s."""
        max_time_step = max_step_length / (
            math.sqrt(atom_velocity[0] ** 2 + atom_velocity[1] ** 2) + 1e-12
        )

        return max_time_step


# Zeeman Slower Magnetic Field Class

zeeman_spec = [
    ("slower_length", float64),
    ("B_0", float64),
    ("B_bias", float64),
    ("delta_B", float64),
    ("delta_B_min", float64),
]


@jitclass(zeeman_spec)
class ZeemanField:
    """Models the magnetic field profile of an ideal Zeeman slower.

    The field strength is given by:

        B(y) = B_0 * sqrt(1 - (y / slower_length)) + B_bias

    where the field is assumed to point along the y-direction.
    """

    def __init__(
        self,
        slower_length: float,
        B_0: float,
        B_bias: float,
        delta_B: float,
        delta_B_min: float,
    ) -> None:

        self.slower_length = slower_length
        self.B_0 = B_0
        self.B_bias = B_bias
        self.delta_B = delta_B
        self.delta_B_min = delta_B_min

    # Method to calculate the magnetic field strength for selected atoms.

    def calculate_magnetic_field(
        self, simulation_atoms: ECSAtoms, atom_ids: np.ndarray
    ) -> None:
        """Update the magnetic field strength for the given atoms."""
        calculate_zeeman_field(
            self.B_0,
            self.slower_length,
            simulation_atoms,
            atom_ids,
            self.B_bias,
        )

    def calculate_max_step_length(
        self, simulation_atoms: ECSAtoms, atom_ids: np.ndarray
    ) -> None:
        """Compute the max step length for the given atoms from their positions."""
        calculate_zeeman_max_step_length(
            simulation_atoms=simulation_atoms,
            atom_ids=atom_ids,
            slower_length=self.slower_length,
            delta_B=self.delta_B,
            delta_B_min=self.delta_B_min,
        )

    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):
        """Approximate mean free path between excitation events, in m."""
        return calculate_zeeman_mean_free_path(
            mean_excitation_time=mean_excitation_time,
            atom_velocity=atom_velocity,
        )

    def calculate_max_time_step(self, max_step_length, atom_velocity):
        """Largest time step keeping the atom within max_step_length, in s."""
        max_time_step = max_step_length / (atom_velocity[1] + 1e-12)

        return max_time_step

    def field_at_positions(
        self, positions: np.ndarray
    ) -> np.ndarray[np.ndarray]:
        """Compute the field vector (assumed along y) and its magnitude.

        Used for plotting.

        Parameters
        ----------
        positions : np.ndarray
            Array of positions with shape (..., 3).

        Returns
        -------
        B : np.ndarray
            Magnetic field vector (with nonzero component only along y).
        norm : np.ndarray
            Magnitude of the magnetic field.

        Uses the formula:
            B(y) = B_0 * sqrt(1 - (y / slower_length))
        """
        y = positions[..., 1]
        field_strength = (
            self.B_0 * np.sqrt(1.0 - (y / self.slower_length)) + self.B_bias
        )
        B = np.zeros_like(positions)
        B[..., 1] = field_strength  # Field along y direction
        norm = np.abs(field_strength)
        return B, norm


# Dipole-Bar Magnetic Field
dipole_spec = [
    ("n_dipoles", int32),
    ("positions", float64[:, :]),  # (n_dipoles,3)
    ("dimensions", float64[:, :]),  # (n_dipoles,3)
    ("volumes", float64[:]),  # (n_dipoles,)
    ("dipole_moment_vectors", float64[:, :]),  # (n_dipoles,3)
    ("mu0_over_4pi", float64),
]


@jitclass(dipole_spec)
class DipoleBarMagneticField:
    """Magnetic field from a superposition of permanent bar (dipole) magnets."""

    def __init__(self, n_dipoles):
        self.n_dipoles = n_dipoles
        self.positions = np.zeros((n_dipoles, 3), dtype=np.float64)
        self.dimensions = np.zeros((n_dipoles, 3), dtype=np.float64)
        self.volumes = np.zeros(n_dipoles, dtype=np.float64)
        self.dipole_moment_vectors = np.zeros((n_dipoles, 3), dtype=np.float64)

        # μ₀/(4π) in SI
        self.mu0_over_4pi = 1e-7

    def add_dipole(self, idx, position, dimension, orientation, magnetization):
        """Call this once per dipole to set its geometry & magnetization."""
        self.positions[idx] = position
        self.dimensions[idx] = dimension

        # volume
        vol = dimension[0] * dimension[1] * dimension[2]
        self.volumes[idx] = vol

        # pre‑scaled dipole moment vector = M∙V ∙ orientation_unit
        norm_orient = np.linalg.norm(orientation)
        orient_unit = orientation / norm_orient
        self.dipole_moment_vectors[idx] = orient_unit * (magnetization * vol)

    def calculate_magnetic_field(self, simulation_atoms, atom_id):
        """Update each atom's field vector and magnitude from the bar dipoles."""
        calculate_bar_dipole_field(
            self.n_dipoles,
            self.positions,
            self.dipole_moment_vectors,
            self.mu0_over_4pi,
            simulation_atoms,
            atom_id,
        )

    def calculate_max_step_length(
        self, simulation_atoms, atom_id: np.ndarray
    ) -> None:
        """Set the adaptive max step length for the atoms from the local field."""
        B = simulation_atoms.magnetic_field_strength[atom_id]

        if B >= _B_COARSE:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif B >= _B_MID:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif B >= _B_FINE:
            simulation_atoms.max_step_lengths[atom_id] = 1e-5
        else:
            # B < _B_FINE including B == 0: high scattering rate, use fine step
            simulation_atoms.max_step_lengths[atom_id] = 1e-6

        return

    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):
        """Approximate mean free path between excitation events, in m."""
        mean_free_path = mean_excitation_time * math.sqrt(
            atom_velocity[0] ** 2
            + atom_velocity[1] ** 2
            + atom_velocity[2] ** 2
        )

        return mean_free_path

    def calculate_max_time_step(self, max_step_length, atom_velocity):
        """Largest time step keeping the atom within max_step_length, in s."""
        max_time_step = max_step_length / (
            math.sqrt(
                atom_velocity[0] ** 2
                + atom_velocity[1] ** 2
                + atom_velocity[2] ** 2
            )
            + 1e-12
        )

        return max_time_step


# Cuboid-Bar Magnetic Field (exact closed-form, no interpolation error)
cuboid_spec = [
    ("n_bars", int32),
    ("positions", float64[:, :]),  # (n_bars,3) bar centers, m
    ("half_extents", float64[:, :]),  # (n_bars,3) = dimension/2 along lab x,y,z
    ("mag_axis", int32[:]),  # lab axis (0/1/2) the magnetization points along
    ("mag_signed", float64[:]),  # signed magnetization magnitude (A/m)
    ("mu0_over_4pi", float64),
]


@jitclass(cuboid_spec)
class CuboidBarMagneticField:
    """Exact field of uniformly-magnetized rectangular bars (surface-charge form).

    Same geometry input as DipoleBarMagneticField, but evaluates the exact
    closed-form cuboid field instead of the point-dipole approximation. The
    component carrying the atan2 term is the bar's magnetization axis; the two
    transverse components carry log terms (axis-aligned ±x/±y/±z magnetization).
    """

    def __init__(self, n_bars):
        self.n_bars = n_bars
        self.positions = np.zeros((n_bars, 3), dtype=np.float64)
        self.half_extents = np.zeros((n_bars, 3), dtype=np.float64)
        self.mag_axis = np.zeros(n_bars, dtype=np.int32)
        self.mag_signed = np.zeros(n_bars, dtype=np.float64)
        self.mu0_over_4pi = 1e-7

    def add_dipole(self, idx, position, dimension, orientation, magnetization):
        """Store one bar's geometry; magnetization must be axis-aligned (±x/±y/±z)."""
        self.positions[idx] = position
        self.half_extents[idx] = dimension * 0.5

        ox = abs(orientation[0])
        oy = abs(orientation[1])
        oz = abs(orientation[2])
        if ox >= oy and ox >= oz:
            axis = 0
        elif oy >= oz:
            axis = 1
        else:
            axis = 2
        self.mag_axis[idx] = axis
        sign = 1.0 if orientation[axis] >= 0.0 else -1.0
        self.mag_signed[idx] = sign * magnetization

    def calculate_magnetic_field(self, simulation_atoms, atom_id):
        """Update one atom's field vector and magnitude (Tesla), written in place."""
        calculate_cuboid_bar_field(
            self.n_bars,
            self.positions,
            self.half_extents,
            self.mag_axis,
            self.mag_signed,
            self.mu0_over_4pi,
            simulation_atoms,
            atom_id,
        )

    def field_at_positions(self, positions):
        """Vectorized B (Tesla) and |B| at (N,3) positions, for offline plotting."""
        n = positions.shape[0]
        B = np.zeros((n, 3), dtype=np.float64)
        norm = np.zeros(n, dtype=np.float64)
        for p in range(n):
            bx, by, bz = _cuboid_field_one_point(
                self.n_bars,
                self.positions,
                self.half_extents,
                self.mag_axis,
                self.mag_signed,
                self.mu0_over_4pi,
                positions[p, 0],
                positions[p, 1],
                positions[p, 2],
            )
            B[p, 0] = bx
            B[p, 1] = by
            B[p, 2] = bz
            norm[p] = math.sqrt(bx * bx + by * by + bz * bz)
        return B, norm

    # The step/time helpers are field-magnitude-driven and identical to the
    # dipole-bar model (Don't-Hand-Roll): reuse them verbatim.
    def calculate_max_step_length(
        self, simulation_atoms, atom_id: np.ndarray
    ) -> None:
        """Set the adaptive max step length for the atoms from the local field."""
        B = simulation_atoms.magnetic_field_strength[atom_id]

        if B >= _B_COARSE:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif B >= _B_MID:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif B >= _B_FINE:
            simulation_atoms.max_step_lengths[atom_id] = 1e-5
        else:
            simulation_atoms.max_step_lengths[atom_id] = 1e-6

        return

    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):
        """Approximate mean free path between excitation events, in m."""
        mean_free_path = mean_excitation_time * math.sqrt(
            atom_velocity[0] ** 2
            + atom_velocity[1] ** 2
            + atom_velocity[2] ** 2
        )

        return mean_free_path

    def calculate_max_time_step(self, max_step_length, atom_velocity):
        """Largest time step keeping the atom within max_step_length, in s."""
        max_time_step = max_step_length / (
            math.sqrt(
                atom_velocity[0] ** 2
                + atom_velocity[1] ** 2
                + atom_velocity[2] ** 2
            )
            + 1e-12
        )

        return max_time_step


# Grid Field Model (trilinear interpolation + analytic out-of-bounds fallback)
grid_spec = [
    ("x_axis", float64[:]),
    ("y_axis", float64[:]),
    ("z_axis", float64[:]),
    ("Bx", float64[:, :, :]),
    ("By", float64[:, :, :]),
    ("Bz", float64[:, :, :]),
    ("x0", float64),
    ("dx", float64),
    ("nx", int32),
    ("y0", float64),
    ("dy", float64),
    ("ny", int32),
    ("z0", float64),
    ("dz", float64),
    ("nz", int32),
    ("n_bars", int32),
    ("bar_positions", float64[:, :]),
    ("bar_half_extents", float64[:, :]),
    ("bar_mag_axis", int32[:]),
    ("bar_mag_signed", float64[:]),
    ("mu0_over_4pi", float64),
]


@jitclass(grid_spec)
class GridFieldModel:
    """Trilinear interpolation on a regular (x,y,z)->B grid (Tesla).

    Inside the grid: trilinear blend of the 8 surrounding nodes. Outside ANY
    axis: the exact analytic cuboid field of the stored bar geometry (D-02 —
    analytic far-field fallback, NOT clamp), so the field is continuous across
    the grid face and finite beyond it.
    """

    def __init__(
        self,
        x_axis,
        y_axis,
        z_axis,
        Bx,
        By,
        Bz,
        bar_positions,
        bar_half_extents,
        bar_mag_axis,
        bar_mag_signed,
    ):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.x0 = x_axis[0]
        self.dx = x_axis[1] - x_axis[0]
        self.nx = x_axis.shape[0]
        self.y0 = y_axis[0]
        self.dy = y_axis[1] - y_axis[0]
        self.ny = y_axis.shape[0]
        self.z0 = z_axis[0]
        self.dz = z_axis[1] - z_axis[0]
        self.nz = z_axis.shape[0]
        self.n_bars = bar_positions.shape[0]
        self.bar_positions = bar_positions
        self.bar_half_extents = bar_half_extents
        self.bar_mag_axis = bar_mag_axis
        self.bar_mag_signed = bar_mag_signed
        self.mu0_over_4pi = 1e-7

    def calculate_magnetic_field(self, simulation_atoms, atom_id):
        """Update one atom's field vector and magnitude (Tesla), in place."""
        calculate_grid_field(
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.Bx,
            self.By,
            self.Bz,
            self.x0,
            self.dx,
            self.nx,
            self.y0,
            self.dy,
            self.ny,
            self.z0,
            self.dz,
            self.nz,
            self.n_bars,
            self.bar_positions,
            self.bar_half_extents,
            self.bar_mag_axis,
            self.bar_mag_signed,
            self.mu0_over_4pi,
            simulation_atoms,
            atom_id,
        )

    def field_at_positions(self, positions):
        """Vectorized B (Tesla) and |B| at (N,3) positions, for offline plotting."""
        n = positions.shape[0]
        B = np.zeros((n, 3), dtype=np.float64)
        norm = np.zeros(n, dtype=np.float64)
        for p in range(n):
            bx, by, bz = _grid_field_one_point(
                self.x_axis,
                self.y_axis,
                self.z_axis,
                self.Bx,
                self.By,
                self.Bz,
                self.x0,
                self.dx,
                self.nx,
                self.y0,
                self.dy,
                self.ny,
                self.z0,
                self.dz,
                self.nz,
                self.n_bars,
                self.bar_positions,
                self.bar_half_extents,
                self.bar_mag_axis,
                self.bar_mag_signed,
                self.mu0_over_4pi,
                positions[p, 0],
                positions[p, 1],
                positions[p, 2],
            )
            B[p, 0] = bx
            B[p, 1] = by
            B[p, 2] = bz
            norm[p] = math.sqrt(bx * bx + by * by + bz * bz)
        return B, norm

    # Field-magnitude-driven step/time helpers — identical to DipoleBar (v1).
    def calculate_max_step_length(
        self, simulation_atoms, atom_id: np.ndarray
    ) -> None:
        """Set the adaptive max step length for the atoms from the local field."""
        B = simulation_atoms.magnetic_field_strength[atom_id]

        if B >= _B_COARSE:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif B >= _B_MID:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif B >= _B_FINE:
            simulation_atoms.max_step_lengths[atom_id] = 1e-5
        else:
            simulation_atoms.max_step_lengths[atom_id] = 1e-6

        return

    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):
        """Approximate mean free path between excitation events, in m."""
        mean_free_path = mean_excitation_time * math.sqrt(
            atom_velocity[0] ** 2
            + atom_velocity[1] ** 2
            + atom_velocity[2] ** 2
        )

        return mean_free_path

    def calculate_max_time_step(self, max_step_length, atom_velocity):
        """Largest time step keeping the atom within max_step_length, in s."""
        max_time_step = max_step_length / (
            math.sqrt(
                atom_velocity[0] ** 2
                + atom_velocity[1] ** 2
                + atom_velocity[2] ** 2
            )
            + 1e-12
        )

        return max_time_step


# NUMBA SPEC
elliptical_spec = [
    ("g_x", float64),  # gradient along principal x' (units: T/m)
    ("g_y", float64),
    ("g_z", float64),  # gradient along principal y' (units: T/m)
    ("offset", float64[:]),  # length-3 array [x0, y0, z0]
    ("theta", float64),  # tilt angle in radians (CCW)
]


# JITCLASS: EllipticalMagneticField
@jitclass(elliptical_spec)
class EllipticalMagneticField:
    """Elliptical, tilted, offset 'quadrupole-like' linear magnetic field.

    Principal-frame definition (x', y'):
        B'_x = g_x * y'
        B'_y = g_y * x'
        B'_z = 0

    Coordinates are relative to 'offset' and rotated by angle 'theta':
        [x'; y'] = R(theta)^T * ([x; y] - [x0; y0])

    Finally, the field is rotated back to lab frame:
        [B_x; B_y] = R(theta) * [B'_x; B'_y]
        B_z = 0  (can be extended with bias if desired)

    Parameters
    ----------
    g_x, g_y : float
        Linear gradients (T/m) controlling ellipticity (g_x != g_y).
    delta_B : float
        Kept for API compatibility; currently not used (set a bias in kernel if needed).
    offset : np.ndarray
        (3,) array defining where the field is zero: e.g., [-1.3e-3, 1.4e-3, 0.0]
    theta : float
        Tilt angle in radians (CCW). Negative leans "left".
    """

    def __init__(
        self,
        g_x: float,
        g_y: float,
        g_z: float,
        offset: np.ndarray,
        theta: float,
    ) -> None:
        self.g_x = g_x
        self.g_y = g_y
        self.g_z = g_z
        self.offset = offset
        self.theta = theta / 180.0 * math.pi

    def calculate_magnetic_field(
        self, simulation_atoms, atom_id: np.ndarray
    ) -> None:
        """Update the atoms' field vectors using the elliptical model.

        Writes magnetic_field_vectors (and strength if present) following the
        tilted, elliptical quadrupole model.
        """
        calculate_elliptical_field(
            self.g_x,
            self.g_y,
            self.g_z,
            self.offset,
            self.theta,
            simulation_atoms,
            atom_id,
        )

    def field_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """Compute the magnetic field (and its norm) at arbitrary positions.

        Uses the same formulas as the kernel.

        Parameters
        ----------
        positions : array_like (..., 3)
            Positions to evaluate the field at.

        Returns
        -------
        B : (..., 3) array
        norm : (...) array
        """
        pos = np.asarray(positions)
        R = _rot2(self.theta)
        RT = R.T

        # broadcast offset
        dx = pos[..., 0] - self.offset[0]
        dy = pos[..., 1] - self.offset[1]
        dz = pos[..., 2] - self.offset[3]
        # to principal frame
        xprime = RT[0, 0] * dx + RT[0, 1] * dy
        yprime = RT[1, 0] * dx + RT[1, 1] * dy

        # field in principal frame
        Bx_p = self.g_x * yprime
        By_p = self.g_y * xprime
        Bz_p = self.g_z * dz

        # back to lab frame
        Bx = R[0, 0] * Bx_p + R[0, 1] * By_p
        By = R[1, 0] * Bx_p + R[1, 1] * By_p
        Bz = Bz_p

        B = np.stack((Bx, By, Bz), axis=-1)
        norm = np.sqrt(Bx * Bx + By * By + Bz * Bz)
        return B, norm

    # Keep the remaining API identical to IdealQuadrupoleField
    def calculate_max_step_length(
        self, simulation_atoms, atom_id: np.ndarray
    ) -> None:
        """Set the adaptive max step length for the atoms from the local field."""
        B = simulation_atoms.magnetic_field_strength[atom_id]

        if B >= _B_MID:
            simulation_atoms.max_step_lengths[atom_id] = 1e-4
        elif B >= _B_FINE:
            simulation_atoms.max_step_lengths[atom_id] = 1e-5
        else:
            # B < _B_FINE including B == 0: high scattering rate, use fine step
            simulation_atoms.max_step_lengths[atom_id] = 1e-6

        return

    def calculate_mean_free_path(self, mean_excitation_time, atom_velocity):
        """Approximate mean free path between excitation events, in m."""
        return mean_excitation_time * math.sqrt(
            atom_velocity[0] ** 2 + atom_velocity[1] ** 2
        )

    def calculate_max_time_step(self, max_step_length, atom_velocity):
        """Largest time step keeping the atom within max_step_length, in s."""
        return max_step_length / (
            math.sqrt(atom_velocity[0] ** 2 + atom_velocity[1] ** 2) + 1e-12
        )


@njit
def calculate_zeeman_field(
    B_0: float,
    slower_length: float,
    simulation_atoms: ECSAtoms,
    atom_ids: np.ndarray,
    B_bias: float,
) -> None:
    """Compute the Zeeman magnetic field strength for selected atoms."""
    y = simulation_atoms.positions[atom_ids, 1]
    field_strength = np.where(
        y < slower_length, B_0 * np.sqrt(1.0 - y / slower_length) + B_bias, 0.0
    )
    simulation_atoms.magnetic_field_strength[atom_ids] = field_strength
    simulation_atoms.magnetic_field_vectors[atom_ids, 1] = field_strength


@njit
def calculate_zeeman_max_step_length(
    simulation_atoms: ECSAtoms,
    atom_ids: np.ndarray,
    slower_length: float,
    delta_B: float,
    delta_B_min: float,
) -> None:
    """Compute the max y step that changes the field by ~0.1% at a position.

    The magnetic field profile is given by:
        B(y) = B_0 * sqrt(1 - y / slower_length)
    and its absolute derivative is:
        |dB/dy| = B_0 / (2 * slower_length) * (1 - y / slower_length)^(-1/2)

    For a relative change of 0.1% in the magnetic field:
        0.001 * B(y) = |dB/dy| * Δy
    Solving for Δy, we obtain:
        Δy = 2 * 0.001 * slower_length * (1 - y/slower_length)

    This Δy is then used as the maximum allowed step size in the y-direction.

    Parameters
    ----------
    simulation_atoms : ECSAtoms
        The simulation atoms object.
    atom_ids : np.ndarray
        Array of atom indices for which to calculate the maximum step length.
    slower_length : float
        The total length of the slower (or characteristic length scale).
    delta_B : float
        Target fractional field change per step (e.g. 0.001 for 0.1%).
    delta_B_min : float
        Lower clamp on the resulting step length.
    """
    # Get the y-positions of the specified atoms
    y = simulation_atoms.positions[atom_ids, 1]

    # Calculate the maximum step size in y such that the magnetic field changes by 0.1%
    delta_y_max = 2 * delta_B * slower_length * (1.0 - y / slower_length)

    # Enforce clamp
    if delta_y_max < delta_B_min:
        delta_y_max = delta_B_min

    # Set the calculated maximum step sizes for these atoms
    simulation_atoms.max_step_lengths[atom_ids] = delta_y_max
    return


@njit
def calculate_zeeman_mean_free_path(
    mean_excitation_time: float, atom_velocity: np.ndarray
) -> float:
    """Mean free path along the slower (y) axis between excitations, in m."""
    return mean_excitation_time * atom_velocity[1]


@njit
def calculate_ideal_quadrupole_field(
    gradient: float, offset, simulation_atoms: ECSAtoms, atom_id: np.ndarray
) -> None:
    """Compute the ideal quadrupole magnetic field for selected atoms."""
    x = simulation_atoms.positions[atom_id, 0] - offset[0]
    y = simulation_atoms.positions[atom_id, 1] - offset[1]
    B_x = gradient * y
    B_y = gradient * x
    B_z = 0
    simulation_atoms.magnetic_field_vectors[atom_id, 0] = B_x
    simulation_atoms.magnetic_field_vectors[atom_id, 1] = B_y
    simulation_atoms.magnetic_field_vectors[atom_id, 2] = B_z

    simulation_atoms.magnetic_field_strength[atom_id] = math.sqrt(
        B_x**2 + B_y**2 + B_z**2
    )


@njit
def calculate_bar_dipole_field(
    n_dipoles,
    positions,
    dipole_moment_vectors,
    mu0_over_4pi,
    atoms: ECSAtoms,
    atom_id,
):
    """Superpose the field of all bar dipoles at one atom, written in place."""
    rx, ry, rz = atoms.positions[atom_id]

    Bx = 0.0
    By = 0.0
    Bz = 0.0

    for d in range(n_dipoles):
        px, py, pz = positions[d]
        dx = rx - px
        dy = ry - py
        dz = rz - pz

        r2 = dx * dx + dy * dy + dz * dz
        if r2 < 1e-24:
            continue

        inv_r = 1.0 / math.sqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r

        ux = dx * inv_r
        uy = dy * inv_r
        uz = dz * inv_r

        mx, my, mz = dipole_moment_vectors[d]

        m_dot_u = mx * ux + my * uy + mz * uz
        factor = mu0_over_4pi * inv_r3

        Bx += factor * (3.0 * ux * m_dot_u - mx)
        By += factor * (3.0 * uy * m_dot_u - my)
        Bz += factor * (3.0 * uz * m_dot_u - mz)

    atoms.magnetic_field_vectors[atom_id, 0] = Bx
    atoms.magnetic_field_vectors[atom_id, 1] = By
    atoms.magnetic_field_vectors[atom_id, 2] = Bz
    atoms.magnetic_field_strength[atom_id] = math.sqrt(
        Bx * Bx + By * By + Bz * Bz
    )


# Small denominator guard for the cuboid log terms (mirrors the r2<1e-24 dipole
# guard). Capture-region atoms are cm from any bar face, so this only protects
# the removable singularities on the magnet's coordinate planes.
_CUBOID_EPS = 1e-30


@njit
def _cuboid_field_one_point(
    n_bars,
    positions,
    half_extents,
    mag_axis,
    mag_signed,
    mu0_over_4pi,
    rx,
    ry,
    rz,
):
    """Exact B (Tesla) at one point from all bars (surface-charge closed form)."""
    Bx = 0.0
    By = 0.0
    Bz = 0.0
    for d in range(n_bars):
        x0 = rx - positions[d, 0]
        y0 = ry - positions[d, 1]
        z0 = rz - positions[d, 2]
        a = half_extents[d, 0]
        b = half_extents[d, 1]
        c = half_extents[d, 2]
        axis = mag_axis[d]
        pref = mu0_over_4pi * mag_signed[d]

        sx = 0.0
        sy = 0.0
        sz = 0.0
        for i in range(2):
            X = x0 - (1.0 - 2.0 * i) * a
            for j in range(2):
                Y = y0 - (1.0 - 2.0 * j) * b
                for k in range(2):
                    Z = z0 - (1.0 - 2.0 * k) * c
                    sign = 1.0 - 2.0 * ((i + j + k) & 1)
                    R = math.sqrt(X * X + Y * Y + Z * Z)
                    # Magnetization-axis component carries the atan2 term; the two
                    # transverse output components carry the log of the *other*
                    # transverse coordinate. Signs/pairing pinned vs the offline
                    # reference oracle (A1).
                    if axis == 0:
                        fx = math.atan2(Y * Z, X * R)
                        fy = 0.5 * math.log(
                            (R - Z + _CUBOID_EPS) / (R + Z + _CUBOID_EPS)
                        )
                        fz = 0.5 * math.log(
                            (R - Y + _CUBOID_EPS) / (R + Y + _CUBOID_EPS)
                        )
                    elif axis == 1:
                        fx = 0.5 * math.log(
                            (R - Z + _CUBOID_EPS) / (R + Z + _CUBOID_EPS)
                        )
                        fy = math.atan2(X * Z, Y * R)
                        fz = 0.5 * math.log(
                            (R - X + _CUBOID_EPS) / (R + X + _CUBOID_EPS)
                        )
                    else:
                        fx = 0.5 * math.log(
                            (R - Y + _CUBOID_EPS) / (R + Y + _CUBOID_EPS)
                        )
                        fy = 0.5 * math.log(
                            (R - X + _CUBOID_EPS) / (R + X + _CUBOID_EPS)
                        )
                        fz = math.atan2(X * Y, Z * R)
                    sx += sign * fx
                    sy += sign * fy
                    sz += sign * fz
        Bx += pref * sx
        By += pref * sy
        Bz += pref * sz
    return Bx, By, Bz


@njit
def calculate_cuboid_bar_field(
    n_bars,
    positions,
    half_extents,
    mag_axis,
    mag_signed,
    mu0_over_4pi,
    atoms: ECSAtoms,
    atom_id,
):
    """Write the exact cuboid-bar field (Tesla) into one atom, in place."""
    rx, ry, rz = atoms.positions[atom_id]
    Bx, By, Bz = _cuboid_field_one_point(
        n_bars,
        positions,
        half_extents,
        mag_axis,
        mag_signed,
        mu0_over_4pi,
        rx,
        ry,
        rz,
    )
    atoms.magnetic_field_vectors[atom_id, 0] = Bx
    atoms.magnetic_field_vectors[atom_id, 1] = By
    atoms.magnetic_field_vectors[atom_id, 2] = Bz
    atoms.magnetic_field_strength[atom_id] = math.sqrt(
        Bx * Bx + By * By + Bz * Bz
    )


@njit
def _trilinear(A, i, j, k, tx, ty, tz):
    """Blend the 8 corners of cell (i,j,k) with fractional weights tx,ty,tz."""
    c00 = A[i, j, k] * (1.0 - tx) + A[i + 1, j, k] * tx
    c10 = A[i, j + 1, k] * (1.0 - tx) + A[i + 1, j + 1, k] * tx
    c01 = A[i, j, k + 1] * (1.0 - tx) + A[i + 1, j, k + 1] * tx
    c11 = A[i, j + 1, k + 1] * (1.0 - tx) + A[i + 1, j + 1, k + 1] * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@njit
def _grid_field_one_point(
    x_axis,
    y_axis,
    z_axis,
    Bx,
    By,
    Bz,
    x0,
    dx,
    nx,
    y0,
    dy,
    ny,
    z0,
    dz,
    nz,
    n_bars,
    bar_positions,
    bar_half_extents,
    bar_mag_axis,
    bar_mag_signed,
    mu0_over_4pi,
    rx,
    ry,
    rz,
):
    """Trilinear B inside the grid; exact analytic cuboid fallback outside (D-02)."""
    inside = (
        rx >= x_axis[0]
        and rx <= x_axis[nx - 1]
        and ry >= y_axis[0]
        and ry <= y_axis[ny - 1]
        and rz >= z_axis[0]
        and rz <= z_axis[nz - 1]
    )
    if not inside:
        return _cuboid_field_one_point(
            n_bars,
            bar_positions,
            bar_half_extents,
            bar_mag_axis,
            bar_mag_signed,
            mu0_over_4pi,
            rx,
            ry,
            rz,
        )
    i = int((rx - x0) / dx)
    if i < 0:
        i = 0
    elif i > nx - 2:
        i = nx - 2
    j = int((ry - y0) / dy)
    if j < 0:
        j = 0
    elif j > ny - 2:
        j = ny - 2
    k = int((rz - z0) / dz)
    if k < 0:
        k = 0
    elif k > nz - 2:
        k = nz - 2
    tx = (rx - x_axis[i]) / dx
    ty = (ry - y_axis[j]) / dy
    tz = (rz - z_axis[k]) / dz
    bx = _trilinear(Bx, i, j, k, tx, ty, tz)
    by = _trilinear(By, i, j, k, tx, ty, tz)
    bz = _trilinear(Bz, i, j, k, tx, ty, tz)
    return bx, by, bz


@njit
def calculate_grid_field(
    x_axis,
    y_axis,
    z_axis,
    Bx,
    By,
    Bz,
    x0,
    dx,
    nx,
    y0,
    dy,
    ny,
    z0,
    dz,
    nz,
    n_bars,
    bar_positions,
    bar_half_extents,
    bar_mag_axis,
    bar_mag_signed,
    mu0_over_4pi,
    atoms: ECSAtoms,
    atom_id,
):
    """Write the interpolated (or fallback) field (Tesla) into one atom, in place."""
    rx, ry, rz = atoms.positions[atom_id]
    bx, by, bz = _grid_field_one_point(
        x_axis,
        y_axis,
        z_axis,
        Bx,
        By,
        Bz,
        x0,
        dx,
        nx,
        y0,
        dy,
        ny,
        z0,
        dz,
        nz,
        n_bars,
        bar_positions,
        bar_half_extents,
        bar_mag_axis,
        bar_mag_signed,
        mu0_over_4pi,
        rx,
        ry,
        rz,
    )
    atoms.magnetic_field_vectors[atom_id, 0] = bx
    atoms.magnetic_field_vectors[atom_id, 1] = by
    atoms.magnetic_field_vectors[atom_id, 2] = bz
    atoms.magnetic_field_strength[atom_id] = math.sqrt(
        bx * bx + by * by + bz * bz
    )


# HELPER: rotation matrix for theta
@njit(cache=True)
def _rot2(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.empty((2, 2), dtype=np.float64)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


# CORE KERNEL: compute B at given atom ids, in-place on simulation_atoms
# Expects simulation_atoms to expose:
#   positions: (N,3)
#   magnetic_field_vectors: (N,3)
#   magnetic_field_strength: (N,)   (if present; optional but common)


@njit
def calculate_elliptical_field(
    g_x: float,
    g_y: float,
    g_z: float,
    offset: np.ndarray,
    theta: float,
    simulation_atoms,
    atom_id,
) -> None:
    """Compute the elliptical (tilted) quadrupole field for one atom, in place.

    theta is in radians. Principal-frame (diagonal) form:
    Bx' = g_x * x', By' = g_y * y', Bz' = g_z * z'.
    """
    c = math.cos(theta)
    s = math.sin(theta)

    # For each requested atom, compute field and assign

    i = atom_id
    x = simulation_atoms.positions[i, 0] - offset[0]
    y = simulation_atoms.positions[i, 1] - offset[1]
    z = simulation_atoms.positions[i, 2] - offset[2]

    # principal-frame coordinates (R^T * (r - offset))
    xprime = c * x + s * y
    yprime = -s * x + c * y

    # principal-frame (diagonal) field
    Bx_p = g_x * xprime
    By_p = g_y * yprime
    Bz_p = g_z * z

    # rotate back: B = R * B'
    B_x = c * Bx_p - s * By_p
    B_y = s * Bx_p + c * By_p
    B_z = Bz_p

    simulation_atoms.magnetic_field_vectors[i, 0] = B_x
    simulation_atoms.magnetic_field_vectors[i, 1] = B_y
    simulation_atoms.magnetic_field_vectors[i, 2] = B_z

    # store strength
    simulation_atoms.magnetic_field_strength[i] = math.sqrt(
        B_x * B_x + B_y * B_y + B_z * B_z
    )


# Plotting Functions


def plot_magnetic_field_vectors(field: IdealQuadrupoleField) -> None:
    """Plot the quadrupole field with vector directions and magnitudes."""
    data_points = np.linspace(-5, 5, 20)
    grid_x, grid_y = np.meshgrid(data_points, data_points)
    positions = np.stack((grid_x, grid_y, np.zeros_like(grid_x)), axis=-1)
    B, norm = field.field_at_positions(positions)
    B_x = B[..., 0]
    B_y = B[..., 1]

    plt.figure()
    quiver = plt.quiver(
        grid_x, grid_y, B_x, B_y, norm, cmap="plasma", scale=50, pivot="middle"
    )
    plt.colorbar(quiver, label="Magnetic Field Magnitude")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Quadrupole Magnetic Field Vectors")
    plt.axis("equal")
    plt.show()


def plot_magnetic_field_streamplot(field: IdealQuadrupoleField) -> None:
    """Plot the quadrupole magnetic field lines as a streamplot."""
    data_points = np.linspace(-0.05, 0.05, 50)
    grid_x, grid_y = np.meshgrid(data_points, data_points)
    positions = np.stack((grid_x, grid_y, np.zeros_like(grid_x)), axis=-1)
    B, _ = field.field_at_positions(positions)
    B_x = B[..., 0]
    B_y = B[..., 1]

    plt.figure()
    plt.streamplot(grid_x, grid_y, B_x, B_y, color="black", density=1.2)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Quadrupole Magnetic Field Lines")
    plt.axis("equal")
    plt.show()


def plot_zeeman_field_vs_y(zeeman_field: ZeemanField) -> None:
    """Plot the Zeeman slower magnetic field strength as a function of y."""
    # Choose y values from 0 to slower_length (adjust as needed)
    y_values = np.linspace(0, zeeman_field.slower_length, 500)
    positions = np.zeros((len(y_values), 3))
    positions[:, 1] = y_values  # set y-coordinate; x and z remain zero
    _, norm = zeeman_field.field_at_positions(positions)

    plt.figure()
    plt.plot(y_values, norm, label="B(y)")
    plt.xlabel("y (m)")
    plt.ylabel("Magnetic Field Strength (T)")
    plt.title("Zeeman Slower Magnetic Field vs. y")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main Block

if __name__ == "__main__":
    zeeman = ZeemanField(slower_length=1, B_0=0.079)
    plot_zeeman_field_vs_y(zeeman)

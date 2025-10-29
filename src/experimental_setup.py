#####################################################################################
#Contains classes to create objects for the several parts of the experimental setup.#
#####################################################################################

import numpy as np
import scipy.constants as scc
from util.simulation_typing import ECSAtoms
import src.distributions as distributions
from numba.experimental import jitclass
from numba import int32, float64, njit, prange
import util.geometry as geometry

#class to construct the shape of the experimental setup.
#===============================================================================
# SimpleECSBoundaries jitclass specification and implementation
#===============================================================================
#
# We support two types of boundaries:
#   - Container (type 0): A cylinder aligned with the y–axis.
#       Fields used:
#         • position (3-vector): center of the cylinder in the x–z plane.
#         • radius: the maximum allowed radial distance.
#         • top_y: the maximum allowed y coordinate (atoms with y >= top_y are killed).
#
#   - Aperture (type 1): A plane with a circular opening.
#       Fields used:
#         • position (3-vector): a point on the aperture plane.
#         • radius: the aperture’s radius.
#         • normal (3-vector): allowed values (1,0,0), (0,1,0), or (0,0,1) (or their negatives).
#         • target_location: the new location id to assign to atoms that pass through.
#         • aperture_buffer: a small buffer distance around the aperture’s plane
#
#   - TODO: Add sphere and square shaped containers
#
# The enforce_boundaries() method loops over atoms and applies both types of checks.
#
spec_boundaries = [
    ('max_boundaries', int32),         # Maximum number of boundaries
    ('count', int32),                  # Current number of boundaries added
    ('types', int32[:]),               # Boundary type: 0 = container, 1 = aperture
    ('positions', float64[:, :]),      # (max_boundaries x 3) boundary position.
    ('radius', float64[:]),            # (max_boundaries) Used by both container and aperture.
    ('top_y', float64[:]),             # (max_boundaries) For container boundaries only.
    ('normal', float64[:, :]),         # (max_boundaries x 3) For aperture boundaries only.
    ('target_location', int32[:]),     # (max_boundaries) For aperture boundaries only.
    ('aperture_buffer', float64[:]),   # (max_boundaries) For aperture boundaries only.
]

@jitclass(spec_boundaries)
class SimpleECSBoundaries:
    def __init__(self, max_boundaries):
        self.max_boundaries = max_boundaries
        self.count = 0
        
        self.types = np.empty(max_boundaries, dtype=np.int32)
        self.positions = np.empty((max_boundaries, 3), dtype=np.float64)
        self.radius = np.empty(max_boundaries, dtype=np.float64)
        self.top_y = np.empty(max_boundaries, dtype=np.float64)
        self.normal = np.empty((max_boundaries, 3), dtype=np.float64)
        self.target_location = np.empty(max_boundaries, dtype=np.int32)
        self.aperture_buffer = np.empty(max_boundaries, dtype=np.float64)
    
    def add_container(self, position, radius, top_y, location_id):
        """
        Add a container boundary (a cylinder aligned with y).
        
        Parameters:
          position : np.ndarray of shape (3,)
                     (x, y, z) where x and z define the center of the cylinder.
          radius   : float, maximum allowed radial distance.
          top_y    : float, atoms with y >= top_y are considered outside.
          location_id : int, identifier for this container.
                      (Note: This field can be used to tag atoms that are within the container.)
        """
        if self.count >= self.max_boundaries:
            raise ValueError("Maximum number of boundaries reached.")
        idx = self.count
        self.types[idx] = 0  # container type
        self.positions[idx, :] = position
        self.radius[idx] = radius
        self.top_y[idx] = top_y
        # For containers the following fields are not used.
        self.normal[idx, :] = 0.0
        self.target_location[idx] = location_id
        self.aperture_buffer[idx] = 0.0
        self.count += 1

    def add_aperture(self, position, radius, normal, target_location, buffer_val):
        """
        Add an aperture boundary.
        
        Parameters:
          position : np.ndarray (3,)
                     A point on the aperture plane.
          radius   : float, the aperture's radius.
          normal   : np.ndarray (3,)
                     The aperture's normal. Allowed: (1,0,0), (0,1,0), (0,0,1) or negatives.
          target_location : int
                     The new location id to assign to an atom passing through.
          buffer_val : float
                     A small buffer distance around the plane.
        """
        # Check that the normal is axis aligned:
        # (Assuming the normal is exactly one of the allowed values)
        if ( (normal[0] != 0.0 and (normal[1] != 0.0 or normal[2] != 0.0)) or
             (normal[1] != 0.0 and (normal[0] != 0.0 or normal[2] != 0.0)) or
             (normal[2] != 0.0 and (normal[0] != 0.0 or normal[1] != 0.0)) ):
            raise ValueError("Aperture normal must be axis aligned: (±1,0,0), (0,±1,0), or (0,0,±1)")
        if self.count >= self.max_boundaries:
            raise ValueError("Maximum number of boundaries reached.")
        idx = self.count
        self.types[idx] = 1  # aperture type
        self.positions[idx, :] = position
        self.radius[idx] = radius
        self.normal[idx, :] = normal
        self.target_location[idx] = target_location
        self.aperture_buffer[idx] = buffer_val
        # top_y not used for apertures.
        self.top_y[idx] = 0.0
        self.count += 1

    def enforce_boundaries(self, atom_ids, positions, statuses, location_tags):
        """
        Enforce all boundaries.
        
        For container boundaries: if an atom is outside the container, mark its status as dead (0).
        For aperture boundaries: if an atom is in the aperture's buffer region and within the aperture's radius,
        update its location tag to the aperture's target location.
        
        Parameters:
          atom_ids      : 1D np.ndarray of atom indices.
          positions     : 2D np.ndarray (N x 3) of atom positions.
          statuses      : 1D np.ndarray of atom statuses (1 = alive, 0 = dead).
          location_tags : 1D np.ndarray of atom location ids.
        """
        for b in range(self.count):
            btype = self.types[b]
            if btype == 0:
                # Container boundary (cylinder aligned along y)
                # Use positions[b,0] and positions[b,2] as the xz–center and top_y[b] as the upper limit.
                center_x = self.positions[b, 0]
                center_z = self.positions[b, 2]
                max_rad = self.radius[b]
                max_y = self.top_y[b]
                for i in prange(len(atom_ids)):
                    aid = atom_ids[i]
                    # Get atom's position
                    x = positions[aid, 0]
                    y = positions[aid, 1]
                    z = positions[aid, 2]
                    dx = x - center_x
                    dz = z - center_z
                    radial = np.sqrt(dx*dx + dz*dz)
                    # Atom is considered outside if its radial distance exceeds the radius
                    # or if its y is greater than or equal to max_y.
                    if radial > max_rad or y >= max_y:
                        statuses[aid] = 0
            elif btype == 1:
                # Aperture boundary: update location id if the atom is in the aperture region.
                # Only one of the three coordinates is relevant (depending on the normal).
                ap_pos = self.positions[b, :]
                ap_radius = self.radius[b]
                buffer_val = self.aperture_buffer[b]
                target_loc = self.target_location[b]
                norm = self.normal[b, :]
                # Determine which axis is relevant. (Assume one component is ±1.)
                # For example, if norm = (0, 1, 0), then the y coordinate is used.
                if norm[0] != 0.0:
                    axis = 0
                elif norm[1] != 0.0:
                    axis = 1
                else:
                    axis = 2
                for i in prange(len(atom_ids)):
                    aid = atom_ids[i]
                    # Check if the atom is in the aperture's "buffer region"
                    # i.e., the distance along the normal axis to the aperture plane is less than the buffer.
                    if np.abs(positions[aid, axis] - ap_pos[axis]) <= buffer_val:
                        # Now check the lateral distance (in the plane) from the aperture center.
                        # For axis = 0, use y and z; axis = 1, use x and z; axis = 2, use x and y.
                        if axis == 0:
                            d1 = positions[aid, 1] - ap_pos[1]
                            d2 = positions[aid, 2] - ap_pos[2]
                        elif axis == 1:
                            d1 = positions[aid, 0] - ap_pos[0]
                            d2 = positions[aid, 2] - ap_pos[2]
                        else:  # axis == 2
                            d1 = positions[aid, 0] - ap_pos[0]
                            d2 = positions[aid, 1] - ap_pos[1]
                        lateral = np.sqrt(d1*d1 + d2*d2)
                        if lateral <= ap_radius:
                            # Only update if the atom hasn't already been updated.
                            if location_tags[aid] != target_loc:
                                location_tags[aid] = target_loc
            else:
                # Unknown boundary type; do nothing.
                pass

#===============================================================================
# End of SimpleECSBoundaries jitclass
#===============================================================================

#class to construct oven to emit atoms


class Oven:
    """
    Oven class for emitting atoms from a circular surface.

    This class uses an externally provided DistributionLookup object to sample velocity
    magnitudes and then generates atom initial conditions (positions and velocities)
    using a wrapper method that calls an njit-compiled function.

    The primary method, `emit_atoms`, either creates a new ECSAtoms container or updates an
    existing one with the generated initial positions and velocities.

    Parameters
    ----------
    distribution_lookup : DistributionLookup
        A preconfigured DistributionLookup object that encapsulates the distribution
        function and parameters (e.g., mass, temperature) needed for sampling velocity magnitudes.
    position : np.ndarray
        The emitter's center position (e.g., the center of the emitting surface).
    direction : np.ndarray
        The normal (direction) vector of the emitter's surface.
    radius : float
        The radius of the circular emitting surface.
    n : int
        Number of atoms to be emitted.

    Methods
    -------
    emit_atoms(atoms: ECSAtoms = None) -> ECSAtoms
        Generates the initial positions and velocity vectors for the atoms using the provided
        distribution and geometry, then returns an ECSAtoms container populated with these values.
        If an ECSAtoms object is provided, its data will be updated; otherwise, a new container is created.
    """
    def __init__(self, distribution_lookup: distributions.DistributionLookup , position: np.ndarray,
                 direction: np.ndarray, radius: float, n: int):
        # Dependency injection: use the externally provided DistributionLookup object.
        self.distribution_lookup = distribution_lookup

        # Geometry and emitter properties.
        self.position = position  # Assumed to be the emitter's center.
        self.direction = direction / np.linalg.norm(direction)
        self.radius = radius
        self.n = n

        # The emitter_offset is derived directly from the emitter center,
        # here using the y-coordinate as the offset.
        self.emitter_offset = np.float64(self.position[1])

        # Precompute (or cache) the lookup table for velocity sampling.
        self.lookup_table = self.distribution_lookup.get_lookup_table()

    def emit_atoms(self, atoms: ECSAtoms = None) -> ECSAtoms:
        """
        Emits atoms from the oven's surface.

        This method performs the following steps:
          1. Samples velocity magnitudes using the injected DistributionLookup object.
          2. Calls the njit-compiled `sample_atom_initial_conditions` function to generate
             initial positions and velocity vectors based on the oven's geometry.
          3. If an ECSAtoms object is provided, its starting positions and velocities are updated;
             otherwise, a new ECSAtoms object is created.
          4. Returns the ECSAtoms container populated with the generated initial conditions.

        Parameters
        ----------
        atoms : ECSAtoms, optional
            An existing ECSAtoms container to be updated with new initial conditions.
            If None, a new ECSAtoms object will be created.

        Returns
        -------
        ECSAtoms
            The atom container with initial positions and velocity vectors set.
        """
        # Sample velocity magnitudes using the DistributionLookup object.
        velocity_magnitudes = self.distribution_lookup.generate_values(self.n)

        # Generate initial conditions (positions and velocities) using the geometry
        # and sampled velocity magnitudes.
        positions, velocities = sample_atom_initial_conditions(self.n, self.radius,
                                                               self.emitter_offset,
                                                               velocity_magnitudes)
        if atoms is None:
            # Create a new ECSAtoms object.
            atoms = ECSAtoms(n=self.n, mass=self.distribution_lookup.mass)

        # Set the initial conditions in the ECSAtoms container.
        atoms.set_starting_conditions(positions, velocities)
        return atoms


# Define the ECS-like LaserComponent class
laser_component_spec = [
    ("n_lasers", int32),
    ('beam_waists', float64[:]),
    ('origins', float64[:, :]),  # Array of shape (n, 3)
    ('normalized_directions', float64[:, :]),  # Array of shape (n, 3)
    ('beam_powers', float64[:]),
    ('beam_frequencies', float64[:]),
    ('detunings', float64[:]),
    ('handedness', int32[:]), # +1 for right handed and -1 for left handed
    ('wave_vectors', float64[:, :]),  # Array of shape (n, 3)
    ('beam_wavelengths', float64[:]),
    ('initial_intensities', float64[:]),
    ('refractive_indices', float64[:])
]

@jitclass(laser_component_spec)
class LaserComponent:
    def __init__(self, n_lasers):
        # Initialize arrays to store laser properties for n lasers
        self.n_lasers = n_lasers
        self.beam_waists = np.zeros(n_lasers, dtype=np.float64)
        self.origins = np.zeros((n_lasers, 3), dtype=np.float64)
        self.normalized_directions = np.zeros((n_lasers, 3), dtype=np.float64)
        self.beam_powers = np.zeros(n_lasers, dtype=np.float64)
        self.beam_frequencies = np.zeros(n_lasers, dtype=np.float64)
        self.detunings = np.zeros(n_lasers, dtype=np.float64)
        self.handedness = np.zeros(n_lasers, dtype=np.int32)
        self.wave_vectors = np.zeros((n_lasers, 3), dtype=np.float64)
        self.beam_wavelengths = np.zeros(n_lasers, dtype=np.float64)
        self.initial_intensities = np.zeros(n_lasers, dtype=np.float64)
        self.refractive_indices = np.ones(n_lasers, dtype = np.float64)

        
    def add_laser(self, index, waist, origin, direction, beam_power, beam_frequency, detuning,handedness):
        """
        Add or update a laser at a specific index in the component.
        """
        self.beam_waists[index] = waist
        self.origins[index, :] = origin
        self.normalized_directions[index, :] = direction / np.linalg.norm(direction)
        self.beam_powers[index] = beam_power
        self.beam_frequencies[index] = beam_frequency
        self.detunings[index] = detuning
        self.handedness[index] = handedness

        # Calculate derived properties
        self.beam_wavelengths[index] = scc.c / beam_frequency
        k = 2 * scc.pi / self.beam_wavelengths[index]
        self.wave_vectors[index, :] = k * self.normalized_directions[index, :]
        self.initial_intensities[index] = 2 * beam_power / (np.pi * waist**2)





# ------------------------------------------------------------------------------
# Atom Generation Function (Outside the Oven Class)
# ------------------------------------------------------------------------------
@njit(parallel=True)
def sample_atom_initial_conditions(n, radius, emitter_offset, velocity_magnitudes):
    """
    Generate initial positions and velocities for atoms emitted from a circular surface.

    For each atom, a random position on the emitter (a circle with the given radius and emitter offset)
    is generated, and a velocity vector is computed using a random direction (within a hemisphere)
    and a pre-sampled velocity magnitude.

    Parameters
    ----------
    n : int
        Number of atoms.
    radius : float
        Radius of the circular emitter.
    emitter_offset : float
        Y-coordinate offset for the emitter surface.
    velocity_magnitudes : np.ndarray
        Array of velocity magnitudes for each atom.

    Returns
    -------
    positions : np.ndarray
        Array of shape (n, 3) containing the initial positions of the atoms.
    velocities : np.ndarray
        Array of shape (n, 3) containing the initial velocity vectors of the atoms.
    """
    positions = np.zeros((n, 3), dtype=np.float64)
    velocities = np.zeros((n, 3), dtype=np.float64)

    # Generate positions on the circular emitter surface.
    for i in prange(n):
        positions[i] = geometry.random_point_on_oven(radius, emitter_offset)

    # Generate velocities using random angles in the hemisphere.
    for i in prange(n):
        theta, phi = geometry.random_angle_in_hemisphere()
        velocities[i] = geometry.polar_to_cartesian_y_axis(velocity_magnitudes[i], theta, phi)

    return positions, velocities




if __name__ == '__main__':
    pass
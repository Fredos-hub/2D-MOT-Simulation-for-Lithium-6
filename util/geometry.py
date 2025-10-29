############################################################################
#Class to handle generating random values for angles and starting positions#
############################################################################

from numba import njit
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional




@njit
def random_angle_in_hemisphere() -> tuple:
    """
    Generate a random direction in a hemisphere aligned with the positive z-axis.

    Returns
    -------
    tuple of float
        A tuple containing:
            - theta: float
                The polar angle in radians.
            - phi: float
                The azimuthal angle in radians.
    """
    # Generate a uniform random number for the polar component.
    u = np.random.uniform(0, 1)
    # Generate a uniform random number for the azimuthal component.
    v = np.random.uniform(0, 1)
    
    # Compute the polar angle using inverse sine to favor directions closer to the z-axis.
    theta = np.arcsin(np.sqrt(u))
    # Compute the azimuthal angle uniformly around the circle.
    phi = 2 * np.pi * v
    return theta, phi


def random_angle_in_oriented_hemisphere(normal_vector: tuple = (0, 1, 0)) -> np.ndarray:
    """
    Generate a random direction in a hemisphere aligned with a given normal vector.

    Parameters
    ----------
    normal_vector : tuple or np.ndarray, optional
        A 3D vector defining the orientation of the hemisphere. Default is (0, 0, 1).

    Returns
    -------
    np.ndarray
        A unit vector (shape (3,)) representing the random direction.
    """
    # Generate a random angle in a hemisphere aligned with the z-axis.
    theta, phi = random_angle_in_hemisphere()
    
    # Convert spherical coordinates (theta, phi) to Cartesian coordinates.
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    random_direction = np.array([x, y, z])

    # Normalize the provided normal vector.
    normal_vector = np.array(normal_vector) / np.linalg.norm(normal_vector)
    # Define the z-axis unit vector.
    z_axis = np.array([0, 0, 1])
    # Compute the rotation matrix that aligns the z-axis to the normal vector.
    rotation_matrix = align_vectors(z_axis, normal_vector)
    # Rotate the random direction so that it aligns with the given normal vector.
    return np.dot(rotation_matrix, random_direction)


def sample_velocities_from_speeds(speeds, *, uniform: bool = True,
                                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Convert speed magnitudes to velocity vectors on the +y hemisphere.

    speeds : (N,) array-like
    uniform : True -> uniform-in-solid-angle; False -> Lambertian / cosine-weighted
    returns: (N,3) array of (vx, vy, vz)
    """
    if rng is None:
        rng = np.random.default_rng()

    s = np.asarray(speeds).ravel()
    N = s.size

    phi = rng.random(N) * 2.0 * np.pi

    if uniform:
        cos_theta = rng.random(N)           # uniform solid angle on hemisphere
    else:
        cos_theta = np.sqrt(rng.random(N)) # Lambertian / cosine-weighted

    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))

    dx = sin_theta * np.cos(phi)
    dy = cos_theta
    dz = sin_theta * np.sin(phi)

    dirs = np.column_stack((dx, dy, dz))
    velocities = s[:, None] * dirs
    return velocities


def align_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Create a rotation matrix to align one vector to another.

    Parameters
    ----------
    vec1 : np.ndarray
        The initial 3D vector (source vector).
    vec2 : np.ndarray
        The target 3D vector (destination vector).

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix that rotates `vec1` to align with `vec2`.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> def align_vectors(vec1, vec2):
    ...     # Function implementation...
    ...     pass
    >>>
    >>> # Define the initial and target vectors
    >>> vec1 = np.array([1, 0, 0])  # Example vector
    >>> vec2 = np.array([0, 1, 0])  # Target vector
    >>>
    >>> # Compute the rotation matrix
    >>> rotation_matrix = align_vectors(vec1, vec2)
    >>>
    >>> # Apply the rotation to align vec1 with vec2
    >>> aligned_vec1 = np.dot(rotation_matrix, vec1)
    >>> print("Aligned Vector:", aligned_vec1)
    Aligned Vector: [0. 1. 0.]
    """
    # Normalize both vectors to unit length.
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Compute the cross product and dot product between vec1 and vec2.
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    s = np.linalg.norm(v)

    # If the vectors are already aligned, return the identity matrix.
    if s == 0:
        return np.eye(3)

    # Create the skew-symmetric cross-product matrix for vector v.
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    # Compute the rotation matrix using Rodrigues' rotation formula.
    rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    return rotation_matrix


@njit
def random_point_on_oven(radius: float, offset: float) -> np.ndarray:
    """
    Select a random point on a circular surface perpendicular to the y-axis.

    Parameters
    ----------
    radius : float
        The radius of the circular surface.
    offset : float
        The offset along the y-axis at which the surface is located.

    Returns
    -------
    np.ndarray
        A point (x, y, z) on the surface, with y equal to the given offset.
    """
    # Compute a random radius with a square-root transformation for uniform area distribution.
    r = radius * np.sqrt(np.random.uniform(0, 1))
    # Generate a random angle between 0 and 2π.
    theta = np.random.uniform(0, 2 * np.pi)
    # Convert polar coordinates to Cartesian coordinates on the XZ-plane.
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    return np.array([x, offset, z])


@njit 
def random_angle_in_sphere() -> np.ndarray:
    """
    Generate a random direction uniformly distributed on the surface of a sphere.

    Returns
    -------
    np.ndarray
        A unit vector (shape (3,)) representing the random direction.
    """
    # Generate uniform random numbers for computing spherical coordinates.
    u = np.random.uniform(0, 1)  # For cos(theta)
    v = np.random.uniform(0, 1)  # For phi

    # Compute the polar angle using the inverse cosine function.
    theta = np.arccos(2 * u - 1)
    # Compute the azimuthal angle uniformly between 0 and 2π.
    phi = 2 * np.pi * v

    # Convert spherical coordinates to Cartesian coordinates.
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])



@njit
def random_emission_in_sphere_directions(n: int) -> np.ndarray:
    """
    Generate n random directions uniformly distributed on the surface of a sphere.
    
    Parameters
    ----------
    n : int
        The number of random unit vectors to generate.
    
    Returns
    -------
    np.ndarray
        An array of shape (n, 3) where each row is a random unit vector.
    """
    directions = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        directions[i, :] = random_angle_in_sphere()
    return directions


@njit
def polar_to_cartesian_2d_y_axis(r: float, theta: float) -> np.ndarray:
    """
    Convert 2D polar coordinates (r, theta) to Cartesian coordinates on the XZ-plane.
    The y-axis remains zero.

    Parameters
    ----------
    r : float
        The radial distance.
    theta : float
        The angle in radians.

    Returns
    -------
    np.ndarray
        Cartesian coordinates as a 3-element array: [x, 0, z].
    """
    # Calculate the x-coordinate.
    x = r * np.cos(theta)
    # Calculate the z-coordinate.
    z = r * np.sin(theta)
    return np.array([x, 0, z])


@njit
def polar_to_cartesian_3(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Convert arrays of 3D polar coordinates (r, theta, phi) to Cartesian coordinates.

    Parameters
    ----------
    r : np.ndarray
        Array of radial distances with shape (n,).
    theta : np.ndarray
        Array of polar angles in radians with shape (n,).
    phi : np.ndarray
        Array of azimuthal angles in radians with shape (n,).

    Returns
    -------
    np.ndarray
        Array of Cartesian coordinates with shape (n, 3), where each row is [x, y, z].
    """
    # Compute the Cartesian x, y, and z components.
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    # Stack the components along the last axis to form an (n, 3) array.
    return np.stack((x, y, z), axis=-1)


@njit
def polar_to_cartesian_y_axis(r: float, theta: float, phi: float) -> np.ndarray:
    """
    Convert polar coordinates (r, theta, phi) to Cartesian coordinates where the angles
    are defined relative to the y-axis (vertical direction).

    Parameters
    ----------
    r : float
        The radial distance (magnitude of the vector).
    theta : float
        The polar angle in radians, measured from the y-axis.
    phi : float
        The azimuthal angle in radians, measured around the y-axis.

    Returns
    -------
    np.ndarray
        Cartesian coordinates as a 3-element array: [x, y, z], where y is the vertical component.
    """
    # Compute the x-coordinate.
    x = r * np.sin(theta) * np.cos(phi)
    # Compute the y-coordinate (vertical component).
    y = r * np.cos(theta)
    # Compute the z-coordinate.
    z = r * np.sin(theta) * np.sin(phi)
    return np.array([x, y, z])


@njit
def random_radius(r: float) -> float:
    """
    Generate a random radius such that every point within a circle is equally likely.

    Parameters
    ----------
    r : float
        The maximum radius of the circle.

    Returns
    -------
    float
        A random radius value.
    """
    # The square-root transformation ensures uniform distribution over the circle's area.
    return r * np.sqrt(np.random.uniform(0, 1))


@njit
def random_angle() -> float:
    """
    Generate a random angle in radians.

    Returns
    -------
    float
        A random angle uniformly distributed between 0 and 2π.
    """
    return np.random.uniform(0, 2 * np.pi)





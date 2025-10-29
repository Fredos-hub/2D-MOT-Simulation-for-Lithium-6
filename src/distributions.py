##########################################################################################################################
#Class to handle different types of distribution calculations like finding speed and direction of atoms emitted from oven#
##########################################################################################################################


import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import math

# ------------------------------------------------------------------------------
# DistributionLookup Class for Caching the Lookup Table
# ------------------------------------------------------------------------------

class DistributionLookup:
    """
    DistributionLookup caches a lookup table for a specified probability distribution.

    This class wraps the functionality of creating and caching a lookup table
    (i.e., mapping CDF bins to velocity values) for a given distribution function.
    The distribution functions themselves remain pure and stateless, while this
    class provides a stateful wrapper to avoid recomputing the lookup table on
    every sampling call.

    Parameters
    ----------
    distribution_function : callable
        The probability density function (PDF) to generate the lookup table.
        Expected signature: f(x, mass, temperature).
    mass : float
        Mass of particles (in kilograms).
    temperature : float
        Temperature of the system (in Kelvin).
    num_bins : int, optional
        Number of bins for the lookup table (default is 100000).
    find_range : bool, optional
        If True, dynamically compute the x-range for the PDF (default is False).
    threshold : float, optional
        Threshold value for determining the PDF integration bounds (default is 1e-12).
    x_range : np.ndarray, optional
        An explicit array of x values over which to compute the lookup table.
    """
    def __init__(self, distribution_function, mass, temperature,
                 num_bins=100000, find_range=False, threshold=1e-12, x_range=None):
        self.distribution_function = distribution_function
        self.mass = mass
        self.temperature = temperature
        self.num_bins = num_bins
        self.find_range = find_range
        self.threshold = threshold
        self.x_range = x_range
        self.lookup_table = None

    def get_lookup_table(self):
        """
        Get the cached lookup table; if not available, create it.

        Returns
        -------
        np.ndarray
            A 2-element array: (cdf_bins, velocity_bins).
        """
        if self.lookup_table is None:
            self.lookup_table = create_lookup_table(
                self.distribution_function,
                self.mass,
                self.temperature,
                num_bins=self.num_bins,
                find_range=self.find_range,
                threshold=self.threshold,
                x_range=self.x_range
            )
        return self.lookup_table

    def generate_values(self, n):
        """
        Generate n velocity samples using the cached lookup table.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        np.ndarray
            Array of velocity values sampled from the distribution.
        """
        lookup_table = self.get_lookup_table()
        return generate_distribution_values(n, self.distribution_function,
                                            self.mass, self.temperature,
                                            lookup_table, self.find_range,
                                            self.threshold, self.x_range)
    

    def plotit(self, distribution_function=None, n_samples=1000000):
        """
        Sample velocities from the distribution and plot a normalized histogram along with:
          - the theoretical model (the distribution function)
          - the mean velocity
          - the most likely velocity

        Parameters
        ----------
        distribution_function : callable, optional
            The probability density function (PDF) to plot. If None, the
            instance's distribution_function is used.
        n_samples : int, optional
            Number of velocity samples to generate (default is 10000).
        bins : int or sequence, optional
            Number of bins or bin specification for the histogram (default is 50).
        """
        # Use the provided distribution_function if given; otherwise, use the one from the instance.
        if distribution_function is None:
            distribution_function = self.distribution_function

        # Sample velocities
        velocities = self.generate_values(n_samples)

        # Compute the theoretical model curve.
        # If an x_range is provided in the instance, use that; otherwise, generate one.
        if self.x_range is not None:
            x_vals = self.x_range
        else:
            # Determine a range based on the sampled values and theoretical most likely velocity.
            # For Maxwell–boltzmann, the most likely velocity is v_mp = sqrt(2 * k_B * T / m).
            k_B = 1.38064852e-23  # boltzmann constant
            v_mp = np.sqrt(2 * k_B * self.temperature / self.mass)
            x_min = 0
            x_max = max(np.max(velocities), 3 * v_mp)
            x_vals = np.linspace(x_min, x_max, 10000)

        # Evaluate the theoretical PDF over x_vals.
        pdf_vals = distribution_function(x_vals, self.mass, self.temperature)

        # Calculate theoretical mean and most likely velocities.
        # For Maxwell–boltzmann speed distribution:
        # Most likely velocity (v_mp) = sqrt(2 * k_B * T / m)
        # Mean velocity (v_mean) = sqrt(8 * k_B * T / (pi * m))
        k_B = 1.38064852e-23  # boltzmann constant
        v_mp = np.sqrt(2 * k_B * self.temperature / self.mass)
        v_mean = np.sqrt(8 * k_B * self.temperature / (np.pi * self.mass))

        # Create the plot.
        plt.figure(figsize=(8, 6))
        # Plot normalized histogram of sampled velocities.
        plt.hist(velocities, bins=int(np.sqrt(n_samples)), density=True, alpha=0.6, label="Sampled distribution")

        # Plot the theoretical model.
        plt.plot(x_vals, pdf_vals, 'r-', lw=2, label="Theoretical model")

        # Mark the mean velocity.
        plt.axvline(v_mean, color='k', linestyle='--', lw=2, label=f'Mean velocity ({v_mean:.2f} m/s)')
        # Mark the most likely velocity.
        plt.axvline(v_mp, color='g', linestyle='--', lw=2, label=f'Most likely velocity ({v_mp:.2f} m/s)')

        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Probability Density")
        plt.title("Maxwell–boltzmann Velocity Distribution")
        plt.legend()
        plt.grid(True)
        plt.show()

# ------------------------------------------------------------------------------
# Pure Distribution Functions (Stateless)
# ------------------------------------------------------------------------------

def maxwell_boltzmann_v2(velocity, mass, temperature, boltzmann_constant=1.38064852e-23):
    """
    Compute the Maxwell-boltzmann probability density (proportional to v^2).

    Parameters
    ----------
    x : float
        Velocity at which to evaluate the distribution.
    mass : float
        Particle mass in kilograms.
    temperature : float
        Temperature in Kelvin.
    boltzmann_constant : float, optional
        boltzmann constant (default: 1.38064852e-23 J/K).

    Returns
    -------
    float
        Probability density at velocity x.
    """
    factor = 4 * np.pi * (mass / (2 * np.pi * boltzmann_constant * temperature)) ** 1.5
    return factor * velocity**2 * np.exp(-mass * velocity**2 / (2 * boltzmann_constant * temperature))

def maxwell_boltzmann_v2(velocity, mass, temperature, boltzmann_constant=1.38064852e-23):

    P_v = 0.5*math.pow((mass/(boltzmann_constant*temperature)), 2)*math.pow(velocity, 3)*math.exp((-mass*math.pow(velocity, 2))/(2*boltzmann_constant*temperature))

    return P_v


def gaussian(x, mu, sigma):
    """
    Compute the Gaussian probability density function.

    Parameters
    ----------
    x : float
        Value at which to evaluate the Gaussian.
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation of the distribution.

    Returns
    -------
    float
        Gaussian probability density at x.
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# ------------------------------------------------------------------------------
# Utility Functions for CDF and Lookup Table Generation
# ------------------------------------------------------------------------------

def find_pdf_range(distribution_function, mass, temperature, threshold=1e-12, initial_step=0.01):
    """
    Dynamically compute the x-range where the PDF falls below a given threshold.

    Parameters
    ----------
    distribution_function : callable
        The PDF to analyze. Expected signature: f(x, mass, temperature).
    mass : float
        Particle mass.
    temperature : float
        Temperature of the system.
    threshold : float, optional
        The threshold value for the PDF (default is 1e-12).
    initial_step : float, optional
        Step size for searching (default is 0.01).

    Returns
    -------
    np.ndarray
        A linearly spaced array of x values covering the range where the PDF is significant.
    """
    # Determine lower bound (starting at 0 and going negative)
    lower_bound = 0
    while distribution_function(lower_bound, mass, temperature) > threshold:
        lower_bound -= initial_step
        if lower_bound < -1e6:
            raise ValueError("Lower bound could not be determined.")

    # Determine upper bound starting at the most probable velocity.
    most_likely_v = np.sqrt(2 * 1.38064852e-23 * temperature / mass)
    upper_bound = most_likely_v
    while distribution_function(upper_bound, mass, temperature) > threshold:
        upper_bound += initial_step
        if upper_bound > 1e6:
            raise ValueError("Upper bound could not be determined.")

    return np.linspace(lower_bound, upper_bound, 10000)


def calculate_cdf(distribution_function, mass, temperature, find_range=False, threshold=1e-12, x_range=None):
    """
    Calculate the cumulative distribution function (CDF) for a given distribution.

    Parameters
    ----------
    distribution_function : callable
        The PDF to integrate. Expected signature: f(x, mass, temperature).
    mass : float
        Particle mass.
    temperature : float
        Temperature of the system.
    find_range : bool, optional
        Whether to dynamically determine the x-range (default is False).
    threshold : float, optional
        Threshold for determining integration bounds (default is 1e-12).
    x_range : np.ndarray, optional
        An explicit array of x values over which to compute the CDF. If provided, this
        range is used directly. Otherwise, the range is determined based on the find_range flag.

    Returns
    -------
    tuple
        (cdf_function, x_range) where cdf_function maps x values to their CDF values.
    """
    if x_range is None:
        if find_range:
            x_range = find_pdf_range(distribution_function, mass, temperature, threshold)
        else:
            # Default fixed range if not dynamically computed and not provided.
            x_range = np.linspace(0, 6000, 10000)
    pdf_values = np.array([distribution_function(x, mass, temperature) for x in x_range])
    # Normalize PDF
    pdf_values /= np.trapz(pdf_values, x_range)
    cdf_values = cumulative_trapezoid(pdf_values, x_range, initial=0)
    cdf_values /= cdf_values[-1]
    cdf_function = interp1d(x_range, cdf_values, bounds_error=False, fill_value=(0, 1))
    return cdf_function, x_range


def invert_cdf(distribution_function, mass, temperature, find_range=False, threshold=1e-12, x_range=None):
    """
    Numerically invert the CDF for a given distribution function.

    Parameters
    ----------
    distribution_function : callable
        The PDF function (signature: f(x, mass, temperature)).
    mass : float
        Particle mass.
    temperature : float
        Temperature of the system.
    find_range : bool, optional
        Whether to dynamically compute the x-range (default is False).
    threshold : float, optional
        Threshold for the PDF integration bounds (default is 1e-12).
    x_range : np.ndarray, optional
        An explicit array of x values over which to compute the CDF. If provided, this range is used.

    Returns
    -------
    callable
        A function mapping CDF values to corresponding x (velocity) values.
    """
    cdf_function, x_range = calculate_cdf(distribution_function, mass, temperature,
                                           find_range, threshold, x_range)
    cdf_values = cdf_function(x_range)
    inverse_cdf_function = interp1d(cdf_values, x_range, bounds_error=False, fill_value="extrapolate")
    return inverse_cdf_function


def create_lookup_table(distribution_function, mass, temperature, num_bins=100000,
                        find_range=False, threshold=1e-12, x_range=None):
    """
    Create a lookup table mapping CDF values to velocity values.

    Parameters
    ----------
    distribution_function : callable
        The PDF function to use (signature: f(x, mass, temperature)).
    mass : float
        Particle mass.
    temperature : float
        Temperature of the system.
    num_bins : int, optional
        Number of bins in the lookup table (default is 100000).
    find_range : bool, optional
        Whether to compute the x-range dynamically (default is False).
    threshold : float, optional
        Threshold for the PDF bounds (default is 1e-12).
    x_range : np.ndarray, optional
        An explicit array of x values to be used for computing the lookup table.

    Returns
    -------
    np.ndarray
        A 2-element array: (cdf_bins, velocity_bins).
    """
    inv_cdf = invert_cdf(distribution_function, mass, temperature, find_range, threshold, x_range)
    cdf_bins = np.linspace(0, 1, num_bins)
    velocity_bins = inv_cdf(cdf_bins)
    return np.array((cdf_bins, velocity_bins))


def generate_distribution_values(n, distribution_function, mass, temperature,
                                 lookup_table=None, find_range=False, threshold=1e-12, x_range=None):
    """
    Generate velocity values by inverse transform sampling using a lookup table.

    Parameters
    ----------
    n : int
        Number of velocity samples.
    distribution_function : callable
        The PDF function (signature: f(x, mass, temperature)).
    mass : float
        Particle mass.
    temperature : float
        Temperature of the system.
    lookup_table : np.ndarray, optional
        A precomputed lookup table; if not provided, one is created.
    find_range : bool, optional
        Whether to compute the x-range dynamically (default is False).
    threshold : float, optional
        PDF threshold for integration bounds (default is 1e-12).
    x_range : np.ndarray, optional
        An explicit x_range to use when generating the lookup table.

    Returns
    -------
    np.ndarray
        Array of velocity values sampled from the distribution.
    """
    if lookup_table is None or lookup_table.size == 0:
        lookup_table = create_lookup_table(distribution_function, mass, temperature,
                                           num_bins=100000, find_range=find_range,
                                           threshold=threshold, x_range=x_range)
    cdf_bins, velocity_bins = lookup_table
    pvals = np.random.uniform(0, 1, n)
    num_bins = len(velocity_bins)
    indices = (pvals * (num_bins - 1)).astype(int)
    return velocity_bins[indices]



    
if __name__ == '__main__':


    pass
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

class MaxwellBoltzmannLookupSampler:
    """
    Inverse‐transform Maxwell–Boltzmann sampler with explicit lookup table creation
    and support for v^2 or v^3 PDFs, plus plotting utilities.

    Usage:
        sampler = MaxwellBoltzmannLookupSampler(mass_u, temperature,
                                                pdf='v2', num_bins=100000, v_max=None)
        sampler.generate_lookup_table()
        speeds = sampler.sample_speeds(n)
        sampler.plot_speeds(speeds, bins=100)

    Parameters
    ----------
    mass_u : float
        Mass of a single atom in atomic mass units (u).
    temperature : float
        Temperature in Kelvin.
    pdf : {'v2', 'v3'}, optional
        Power‐law exponent in PDF: v^2 (speed) or v^3 (momentum-like), default 'v2'.
    num_bins : int, optional
        Number of points in the lookup table (default 100000).
    v_max : float or None, optional
        Maximum speed to consider (m/s). If None, computed as 5*most probable speed.
    """
    def __init__(self, mass_u: float, temperature: float, pdf: str = 'Maxwell-Boltzmann-Distribution v3', num_bins: int = 1_000_000, v_max: float = None):

        self.k_B = 1.38064852e-23  # Boltzmann constant in J/K
        self.mass = mass_u * 1.66053906660e-27
        self.temperature = temperature
        self.pdf_type = pdf

        if self.pdf_type == 'Maxwell-Boltzmann-Distribution v2':
            self.v_mp = np.sqrt(2 * self.k_B * self.temperature / self.mass)
            self.v_mean = np.sqrt((8 * self.k_B * self.temperature) / (np.pi * self.mass))

        elif self.pdf_type == 'Maxwell-Boltzmann-Distribution v3':
            self.v_mp = np.sqrt(3.0 * self.k_B * self.temperature / self.mass)
            self.v_mean = (3.0 * np.sqrt(2.0 * np.pi) / 4.0) * np.sqrt(self.k_B * self.temperature / self.mass)


        self.v_max = v_max or 5 * self.v_mp
        self.num_bins = num_bins
        self._inv_cdf = None


    def _mb_pdf(self, v):
        base = np.exp(-self.mass * v**2 / (2 * self.k_B * self.temperature))
        if self.pdf_type == 'Maxwell-Boltzmann-Distribution v2':
            factor = 4 * np.pi * (self.mass / (2 * np.pi * self.k_B * self.temperature))**1.5
            return factor * v**2 * base
        elif self.pdf_type == 'Maxwell-Boltzmann-Distribution v3':
            # proper normalization constant for v^3 * exp(-m v^2 / (2 k T))
            factor = 2.0 * (self.mass / (2.0 * self.k_B * self.temperature))**2.0
            return factor * v**3 * base
        else:
            raise ValueError("pdf must be 'Maxwell-Boltzmann-Distribution v2' or 'Maxwell-Boltzmann-Distribution v3'")

    def generate_lookup_table(self):
        """Compute and cache the inverse‐CDF lookup table."""
        v_grid = np.linspace(0, self.v_max, self.num_bins)
        pdf = self._mb_pdf(v_grid)
        pdf /= np.trapz(pdf, v_grid)
        cdf = cumulative_trapezoid(pdf, v_grid, initial=0)
        cdf /= cdf[-1]
        self._inv_cdf = interp1d(cdf, v_grid,
                                 bounds_error=False,
                                 fill_value=(0, self.v_max))
        # store for plotting
        self._v_grid = v_grid
        self._pdf = pdf

    def sample_speeds(self, n: int, v_min: float = 0.0) -> np.ndarray:
        """
        Sample n speeds from the cached lookup table. Call generate_lookup_table() first.
        """
        if self._inv_cdf is None:
            raise RuntimeError("Lookup table not generated. Call generate_lookup_table() first.")
        u = np.random.uniform(0, 1, size=n)
        speeds = self._inv_cdf(u)
        # rejection for lower bound
        if v_min > 0:
            mask = speeds < v_min
            while mask.any():
                u_new = np.random.uniform(0, 1, size=mask.sum())
                speeds[mask] = self._inv_cdf(u_new)
                mask = speeds < v_min
        return speeds



    def calculate_probability_fraction(self):

        # build grid from 0 to v_max
        v_grid = np.linspace(0, 10_000, 500_000)
        # evaluate PDF on grid
        pdf = self._mb_pdf(v_grid)
        # normalize PDF
        pdf /= np.trapz(pdf, v_grid)
        # compute CDF
        cdf = cumulative_trapezoid(pdf, v_grid, initial=0)

        # index for v_max
        idx_max = np.searchsorted(v_grid, self.v_max)
        # probability from 0 to v_max
        return float(cdf[idx_max])



    def plot_speeds(self, speeds: np.ndarray, bins: int = 50, capture_v: float = 92.0,
                    inset_frac: float = 0.20, show_inset: bool = True):
        """
        Plot histogram of sampled speeds overlaid with a fitted theoretical PDF.

        Features:
        - legend without box
        - inset zoom up to `capture_v`
        - histogram bars colored differently for v <= capture_v and v > capture_v
        - fits the theoretical PDF shape to the histogram using curve_fit (single amplitude parameter)
        - prints reduced chi-squared of the fit

        Parameters
        ----------
        speeds : np.ndarray
            Array of sampled speeds.
        bins : int
            Number of histogram bins.
        capture_v : float
            Velocity (m/s) used to color bins and to define the inset zoom x-limit.
        inset_frac : float
            Fractional size of the inset (e.g. 0.30 => 30% width/height).
        show_inset : bool
            If False, no inset is drawn.
        """


        if not hasattr(self, '_pdf'):
            raise RuntimeError("Lookup table data missing. Call generate_lookup_table() first.")

        # clip capture_v to valid range
        capture_v = float(capture_v)
        capture_v = max(0.0, min(capture_v, float(self.v_max)))

        fig, ax = plt.subplots(figsize=(8, 6))

        # compute histogram as COUNTS (so we can compute proper uncertainties),
        # but also produce the density for plotting
        hist_counts, edges = np.histogram(speeds, bins=bins, range=(0, self.v_max), density=False)
        N = len(speeds)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = (edges[1:] - edges[:-1])
        hist_density = hist_counts / (N * widths)   # density for plotting

        # split masks for coloring
        mask_below = centers <= capture_v
        mask_above = ~mask_below

        # For plotting, draw >capture_v group first
        if mask_above.any():
            ax.bar(centers[mask_above], hist_density[mask_above],
                   width=widths[mask_above], align='center',
                   alpha=0.65, color='C1', label='Sampled velocities')

        if mask_below.any():
            ax.bar(centers[mask_below], hist_density[mask_below],
                   width=widths[mask_below], align='center',
                   alpha=0.65, color='C0', label=f'Samples ≤ {capture_v:.0f} m/s')

        # Fitting: fit a single amplitude 'a' so model_density = a * pdf_interp(v)
        pdf_interp = interp1d(self._v_grid, self._pdf, bounds_error=False, fill_value=0.0)

        # prepare sigma for density: sigma_density = sqrt(counts) / (N * width)
        sigma_density = np.sqrt(hist_counts) / (N * widths)
        # avoid zeros in sigma (for empty bins) by setting a reasonable small uncertainty
        zero_sigma_mask = sigma_density == 0
        if zero_sigma_mask.any():
            # set to 1 count equivalent: sigma_density = sqrt(1)/(N*width)
            sigma_density[zero_sigma_mask] = 1.0 / (N * widths[zero_sigma_mask])

        # Only fit bins where the pdf_interp is > 0 or where we have counts (prevents trivial zeros)
        fit_mask = (pdf_interp(centers) > 0) | (hist_counts > 0)
        if fit_mask.sum() < 3:
            # Not enough bins to fit robustly; fall back to plotting theoretical pdf
            print("Not enough bins for fitting; plotting theoretical PDF without fit.")
            ax.plot(self._v_grid, self._pdf, 'r-', lw=2, label=r'Maxwell-Boltzmann distribution (theory)')
            ax.axvline(self.v_mean, label=f"Mean velocity {self.v_mean:.0f} m/s", ls=":", color="b")
            ax.axvline(self.v_mp,   label=f"Most likely velocity {self.v_mp:.0f} m/s", ls="-.", color="b")
            ax.axvline(capture_v,   label=rf"$v_c$ ≈ {capture_v:.0f} m/s", ls=":", color="black")
            ax.set_xlim(0, self.v_max)
            ax.set_xlabel('velocity magnitude (m/s)')
            ax.set_ylabel('probability density (a.u.)')
            ax.legend(frameon=False)
            ax.grid(True, linestyle=':', alpha=0.3)
            plt.tight_layout()
            plt.show()
            return fig, ax

        try:
            def model_density(x, a):
                # curve_fit will pass x=centers; return density per unit velocity
                return a * pdf_interp(x)

            p0 = [1.0]
            popt, pcov = curve_fit(model_density,
                                   centers[fit_mask],            # x
                                   hist_density[fit_mask],       # y
                                   p0=p0,
                                   sigma=sigma_density[fit_mask],
                                   absolute_sigma=True,
                                   maxfev=10000)

            a_fit = popt[0]
            # model density on centers for plotting/chi2
            model_den_centers = model_density(centers, a_fit)

            # reduced chi-squared: sum((y - y_model)^2 / sigma^2) / dof
            residuals = hist_density[fit_mask] - model_den_centers[fit_mask]
            chisq = np.sum((residuals / sigma_density[fit_mask])**2)
            dof = fit_mask.sum() - len(popt)
            red_chisq = chisq / dof if dof > 0 else np.nan

            # print reduced chi-squared
            print(f"Fit successful: reduced chi-squared = {red_chisq:.4g} (chi2={chisq:.4g}, dof={dof})")

            # overlay fitted curve (density units)
            ax.plot(self._v_grid, a_fit * self._pdf, 'r-', lw=2, label=r'Fitted Maxwell-Boltzmann (shape scaled)')


        except Exception as e:
            # If fit fails, plot theoretical pdf and report the error
            print("Fit failed with error:", e)
            ax.plot(self._v_grid, self._pdf, 'r-', lw=2, label=r'Maxwell-Boltzmann distribution (theory)')
            red_chisq = np.nan

        # vertical lines and labels
        ax.axvline(self.v_mean, label=f"Mean velocity {self.v_mean:.0f} m/s", ls=":", color="b")
        ax.axvline(self.v_mp,   label=f"Most likely velocity {self.v_mp:.0f} m/s", ls="-.", color="b")
        ax.axvline(capture_v,   label=rf"$v_c$ ≈ {capture_v:.0f} m/s", ls=":", color="black")

        ax.set_xlim(0, self.v_max)
        ax.set_xlabel('Velocity magnitude (m/s)')
        ax.set_ylabel('Probability density (a.u.)')

        # legend without box
        ax.legend(frameon=False)

        ax.grid(True, linestyle=':', alpha=0.3)

        # inset: zoom from 0 to capture_v
        if show_inset and capture_v > 0:
            cap = capture_v + 20
            # make inset axes (relative size)
            w = f"{inset_frac*100:.0f}%"
            h = w
            axins = inset_axes(ax, width=w, height=h, loc='lower right', borderpad=5)

            # plot only the portion up to cap, drawing the two bar groups separately
            mask_cap = centers <= cap
            if mask_cap.any():
                mask_below_inset = mask_cap & (centers <= capture_v)
                mask_above_inset = mask_cap & (centers > capture_v)

                if mask_above_inset.any():
                    axins.bar(centers[mask_above_inset], hist_density[mask_above_inset],
                              width=widths[mask_above_inset], align='center',
                              alpha=0.65, color='C1')
                if mask_below_inset.any():
                    axins.bar(centers[mask_below_inset], hist_density[mask_below_inset],
                              width=widths[mask_below_inset], align='center',
                              alpha=0.65, color='C0')

            # theoretical (or fitted) curve cropped
            pdf_mask = self._v_grid <= cap
            if pdf_mask.any():
                # If fit succeeded, try to show fitted curve in inset; else show theoretical
                try:
                    if 'a_fit' in locals() and not np.isnan(a_fit):
                        axins.plot(self._v_grid[pdf_mask], a_fit * self._pdf[pdf_mask], 'r-', lw=1.25)
                    else:
                        axins.plot(self._v_grid[pdf_mask], self._pdf[pdf_mask], 'r-', lw=1.25)
                except Exception:
                    axins.plot(self._v_grid[pdf_mask], self._pdf[pdf_mask], 'r-', lw=1.25)

            # vertical lines inside the inset (optionally)
            axins.axvline(self.v_mean if self.v_mean <= cap else cap, ls=":", color="b", lw=0.8)
            axins.axvline(self.v_mp   if self.v_mp   <= cap else cap, ls="-.", color="b", lw=0.8)
            axins.axvline(cap, ls=":", color="black", lw=0.8)

            axins.set_xlim(0, cap)
            # choose a reasonable ylim (cover both hist and pdf)
            y_candidate = 0.0
            if mask_cap.any():
                y_candidate = max(y_candidate, hist_density[mask_cap].max())
            if pdf_mask.any():
                # if fitted, consider fitted peak
                if 'a_fit' in locals() and not np.isnan(a_fit):
                    y_candidate = max(y_candidate, (a_fit * self._pdf[pdf_mask]).max())
                else:
                    y_candidate = max(y_candidate, self._pdf[pdf_mask].max())
            if y_candidate > 0:
                axins.set_ylim(0, y_candidate * 1.12)

            axins.set_title(f"zoom to {cap:.0f} m/s", fontsize=12)
            axins.tick_params(labelsize=12)

            # draw connector lines between inset and the region (nice visual cue)
            try:
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            except Exception:
                # mark_inset can occasionally fail depending on Matplotlib version; ignore quietly
                pass

        plt.tight_layout()
        plt.show()
        return fig, ax


if __name__ == "__main__":
    sampler = MaxwellBoltzmannLookupSampler(mass_u = 6.015, temperature = 623, v_max= 5000, pdf ='Maxwell-Boltzmann-Distribution v3')
    sampler.generate_lookup_table()
    speeds = sampler.sample_speeds(100000000)
    sampler.plot_speeds(speeds, bins=round(np.sqrt(1000000)), inset_frac=0.4)

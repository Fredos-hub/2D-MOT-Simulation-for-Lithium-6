import csv
import os

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import util.geometry as geometry
from src.maxwell_boltzmann_sampler import MaxwellBoltzmannLookupSampler


class OvenWorker(QThread):
    progress = pyqtSignal(int)  # emits 0–100
    finished = pyqtSignal(str)  # emits output filename
    error = pyqtSignal(str)  # emits an error message; finished is NOT emitted
    cancelled = pyqtSignal()  # emits when the user aborts

    # Safety cap: with a geometry that rejects (almost) all atoms the collect
    # loop would otherwise spin forever. Abort once this many are generated.
    MAX_GENERATED = 100_000_000

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self._cancel = False

    def cancel(self):
        """Request cooperative cancellation; the run loop checks this flag."""
        self._cancel = True

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.error.emit(str(e))

    def _run(self):
        # Unpack & convert geometry to meters
        m_u = self.params["atom_mass"]
        pdf_str = self.params["distribution"]
        temp = self.params["temperature"]
        vmin, vmax = self.params["vmin"], self.params["vmax"]

        r_oven_mm, y_oven_mm = self.params["oven_geometry"]
        oven_r, oven_y = r_oven_mm * 1e-3, y_oven_mm * 1e-3

        # apertures: sorted & in meters
        apertures = np.array(
            sorted(self.params["apertures"], key=lambda ap: ap[1])
        )
        apertures[:, 0] *= 1e-3  # radii → m
        apertures[:, 1] *= 1e-3  # heights → m

        output_file = self.params["output_file"]
        n_target = self.params["num_atoms"]

        # Init sampler
        sampler = MaxwellBoltzmannLookupSampler(
            m_u, temp, pdf=pdf_str, num_bins=100_000, v_max=vmax
        )
        sampler.generate_lookup_table()
        distribution_fraction = sampler.calculate_probability_fraction()

        total_generated = 0
        n_collected = 0
        batch_size = 10_000

        cancelled = False
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # write header (no acceptance ratio per row)
            writer.writerow(
                [
                    "x",
                    "y",
                    "z",
                    "vx",
                    "vy",
                    "vz",
                    "subjective_time",
                    "excitation_counter",
                    "current_groundstate",
                ]
            )

            # generate and collect
            while n_collected < n_target:
                if self._cancel:
                    cancelled = True
                    break
                total_generated += batch_size
                if total_generated > self.MAX_GENERATED:
                    raise RuntimeError(
                        f"Generated {total_generated:,} atoms but only "
                        f"{n_collected}/{n_target} passed the apertures. "
                        f"The oven/aperture geometry rejects (almost) all atoms — "
                        f"check the configuration."
                    )

                # positions on oven mouth (disk)
                u_rad = np.random.rand(batch_size)
                theta_r = 2 * np.pi * np.random.rand(batch_size)
                r_vals = oven_r * np.sqrt(u_rad)
                xs = r_vals * np.cos(theta_r)
                zs = r_vals * np.sin(theta_r)
                positions = np.stack(
                    (xs, np.full(batch_size, oven_y), zs), axis=1
                )

                # speeds + directions
                speeds = sampler.sample_speeds(batch_size, v_min=vmin)
                velocities = geometry.sample_velocities_from_speeds(
                    speeds, uniform=False
                )

                # aperture filtering
                alive = np.ones(batch_size, bool)
                pos = positions.copy()
                vel = velocities

                for r_ap, y_ap in apertures:
                    dy = y_ap - pos[:, 1]
                    t = dy / vel[:, 1]
                    alive &= (vel[:, 1] > 0) & (t >= 0)
                    if not alive.any():
                        break

                    x_at = pos[:, 0] + vel[:, 0] * t
                    z_at = pos[:, 2] + vel[:, 2] * t
                    alive &= (x_at**2 + z_at**2) <= (r_ap**2)
                    if not alive.any():
                        break

                    # update survivors positions
                    pos[alive, 0] = x_at[alive]
                    pos[alive, 1] = y_ap
                    pos[alive, 2] = z_at[alive]

                # collect survivors
                survivors = np.where(alive)[0]
                if survivors.size:
                    take = survivors[: max(0, n_target - n_collected)]
                    pv = np.hstack([pos[take], vel[take]])
                    zeros = np.zeros((pv.shape[0], 3), dtype=pv.dtype)
                    rows = np.hstack([pv, zeros])
                    writer.writerows(rows)

                    n_collected += take.size
                    pct = int(n_collected / n_target * 100)
                    self.progress.emit(pct)

            if not cancelled:
                # write summary stats as columns at end
                writer.writerow([])  # blank line separator
                writer.writerow(["distribution_fraction", "capture_ratio"])
                capture_ratio = (
                    n_collected / total_generated
                    if total_generated > 0
                    else 0.0
                )
                writer.writerow([distribution_fraction, capture_ratio])

        if cancelled:
            # Drop the incomplete file so a partial sample is never picked up.
            try:
                os.remove(output_file)
            except OSError:
                pass
            self.cancelled.emit()
            return

        # done
        self.progress.emit(100)
        self.finished.emit(output_file)

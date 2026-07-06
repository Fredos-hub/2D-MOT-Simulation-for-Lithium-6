"""Offline generator for the udem_code reference fixture (D-08 PRIMARY ground truth).

Run once to (re)create ``tests/fixtures/udem_reference.npz``. It imports the
out-of-repo reference implementation ``udem_code_energy_gs_es.Linien`` (a sibling
clone), evaluates it for the Li-6 D2 line at a fixed set of sample B values, and
stores the returned line positions and relative intensities as a numeric-only NPZ.

Dev-only: this script and its sympy dependency are NEVER imported by ``src/``.
The generated NPZ is a committed test fixture, not a generated table (those live
under the gitignored ``interaction_tables/``).
"""
import sys
from pathlib import Path

import numpy as np
import scipy.constants as scc

# Sibling clone of the reference code (not a package on the path by default).
UDEM_DIR = "/home/fredo/Schreibtisch/Test/Zeemanslower"

# Li-6 D2 reference constants (from udem_code_energy_gs_es.__main__).
I = 1
J_GS, J_ES = 1 / 2, 3 / 2
G_J_GS, G_J_ES = 2.002, 1.335
A_GS, A_ES = 150e6, -1.15e6      # magnetic-dipole constant [Hz]
B_GS, B_ES = 0.0e6, -0.1e6       # electric-quadrupole constant [Hz]
HAUF, ISOVER = 1, 0

# Sample B values [Tesla] spanning weak-field to Paschen-Back regimes.
B_VALUES = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0])


def _load_reference():
    """Import Linien and inject the h_Planck global the reference only sets in __main__."""
    if UDEM_DIR not in sys.path:
        sys.path.insert(0, UDEM_DIR)
    import udem_code_energy_gs_es as udem
    # SetMatrix / Linien read a module-global h_Planck that is only defined inside
    # the reference's __main__ block. Inject it so the imported functions work.
    udem.h_Planck = scc.hbar * 2 * np.pi
    return udem


def generate():
    udem = _load_reference()
    pos_all = np.empty((len(B_VALUES), 6, 12))
    intensity_all = np.empty((len(B_VALUES), 3, 6, 12))
    for k, B in enumerate(B_VALUES):
        pos, intensity = udem.Linien(
            I, J_GS, J_ES, A_GS, A_ES, B_GS, B_ES, G_J_ES, G_J_GS, B, HAUF, ISOVER
        )
        pos_all[k] = np.asarray(pos, dtype=np.float64)
        intensity_all[k] = np.asarray(intensity, dtype=np.float64)
    return pos_all, intensity_all


def main():
    pos_all, intensity_all = generate()
    out = Path(__file__).parent / "udem_reference.npz"
    np.savez(out, pos=pos_all, intensity=intensity_all, b_values=B_VALUES)
    print(f"wrote {out}  pos={pos_all.shape}  intensity={intensity_all.shape}")


if __name__ == "__main__":
    main()

"""Plotting package for simulation results.

Plot functions live in `plot_types/` grouped by family and are directly
importable; the registry + PlotShell offer name-based dispatch:

    from util.plotting import PlotShell
    shell = PlotShell(session_dir="simulation_results/01_07_26_0")
    shell.plot("tof_spectrum", transverse_radius_mm=5.0)

CLI: python -m util.plotting list
"""

from . import plot_types  # noqa: F401  (imports register all plot types)
from .registry import get_plot, list_plots, plot_type
from .shell import PlotShell

__all__ = ["PlotShell", "get_plot", "list_plots", "plot_type"]

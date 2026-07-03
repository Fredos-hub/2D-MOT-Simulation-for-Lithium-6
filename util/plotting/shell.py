"""PlotShell: dispatch registered plot types with shared session context."""

import inspect
from pathlib import Path

import pandas as pd

from .registry import PlotType, get_plot


class PlotShell:
    """Holds per-session context (data location, save/show policy) and
    dispatches registered plot types by name.

    >>> shell = PlotShell(session_dir="simulation_results/01_07_26_0")
    >>> shell.plot("tof_spectrum", transverse_radius_mm=5.0)
    >>> shell.plot("position_hist", data="path/to/result.csv")
    """

    def __init__(
        self,
        session_dir: str | Path | None = None,
        save_dir: str | Path | None = None,
        show: bool = False,
    ):
        self.session_dir = Path(session_dir) if session_dir else None
        self.save_dir = Path(save_dir) if save_dir else None
        self.show = show

    def plot(self, name: str, data=None, **kwargs):
        """Run the plot type `name`. `data` is resolved per input kind:

        dataframe    DataFrame or result.csv path (required)
        run          run directory (required)
        session      session directory (defaults to self.session_dir)
        summary_csv  cache CSV path (required)

        The shell's `show` / `save_dir` are injected as defaults when
        the target function accepts them; explicit kwargs win.
        """
        pt = get_plot(name)
        target = self._resolve_input(pt, data)

        params = inspect.signature(pt.func).parameters
        if "show" in params:
            kwargs.setdefault("show", self.show)
        if self.save_dir is not None and "save_dir" in params:
            kwargs.setdefault("save_dir", self.save_dir)

        return pt.func(target, **kwargs)

    def _resolve_input(self, pt: PlotType, data):
        if pt.input == "dataframe":
            if isinstance(data, pd.DataFrame):
                return data
            if data is None:
                raise ValueError(
                    f"{pt.name!r} needs a result.csv path or DataFrame"
                )
            return pd.read_csv(data)
        if pt.input == "session":
            data = data if data is not None else self.session_dir
            if data is None:
                raise ValueError(
                    f"{pt.name!r} needs a session directory (pass `data` "
                    f"or construct the shell with session_dir=...)"
                )
            return Path(data)
        # "run" and "summary_csv": a required path
        if data is None:
            raise ValueError(f"{pt.name!r} needs a {pt.input} path")
        return Path(data)

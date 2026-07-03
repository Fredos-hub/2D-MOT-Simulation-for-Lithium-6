"""Plot-type registry: name -> (function, input kind, description).

Registered functions take the resolved input as their first positional
argument, according to their input kind:

    dataframe    a loaded result.csv DataFrame
    run          a run directory (contains result.csv + config.json)
    session      a session directory (contains run_* subdirectories)
    summary_csv  a cache CSV built by an extraction step
"""

from collections.abc import Callable
from dataclasses import dataclass

INPUT_KINDS = ("dataframe", "run", "session", "summary_csv")


@dataclass(frozen=True)
class PlotType:
    name: str
    func: Callable
    input: str
    description: str


_REGISTRY: dict[str, PlotType] = {}


def plot_type(name: str, input: str, description: str = ""):
    """Decorator registering a plot function under `name`."""
    if input not in INPUT_KINDS:
        raise ValueError(
            f"input must be one of {INPUT_KINDS}, got {input!r}"
        )

    def decorator(func: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"duplicate plot type {name!r}")
        desc = description or (func.__doc__ or "").split("\n")[0]
        _REGISTRY[name] = PlotType(name, func, input, desc)
        return func

    return decorator


def get_plot(name: str) -> PlotType:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown plot type {name!r}. Known: {known}")
    return _REGISTRY[name]


def list_plots() -> list[PlotType]:
    return sorted(_REGISTRY.values(), key=lambda p: p.name)

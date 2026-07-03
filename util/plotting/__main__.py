"""CLI for the plotting shell.

    python -m util.plotting list
    python -m util.plotting <plot> <data> [--show] [--save-dir D] \\
        [-p key=value ...]

Parameter values are parsed as Python literals when possible
(e.g. -p bins=80 -p v_range="(0,150)"), otherwise kept as strings.
"""

import argparse
import ast

from .registry import list_plots
from .shell import PlotShell


def _parse_param(item: str) -> tuple[str, object]:
    if "=" not in item:
        raise argparse.ArgumentTypeError(
            f"expected key=value, got {item!r}"
        )
    key, raw = item.split("=", 1)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return key, value


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m util.plotting",
        description="Run a registered plot type on simulation output.",
    )
    parser.add_argument(
        "plot", help="plot type name, or 'list' to show all"
    )
    parser.add_argument(
        "data",
        nargs="?",
        help="input path (result.csv / run dir / session dir / summary csv)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument(
        "-p",
        "--param",
        action="append",
        default=[],
        type=_parse_param,
        metavar="KEY=VALUE",
        help="extra keyword argument for the plot function (repeatable)",
    )
    args = parser.parse_args()

    if args.plot == "list":
        for pt in list_plots():
            print(f"{pt.name:32} [{pt.input:11}] {pt.description}")
        return

    shell = PlotShell(save_dir=args.save_dir, show=args.show)
    result = shell.plot(args.plot, data=args.data, **dict(args.param))
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()

# 2D-MOT-Simulation-for-Lithium-6

Framework to simulate behavior of ⁶Li atoms in a two-dimensional magneto-optical trap (2D MOT).

## Releases

Tagged versions follow [Semantic Versioning](https://semver.org/):

- **`v1.1.0`** — current release. Adds rate mode, laser active intervals,
  diagonalizer interaction tables, Cuboid and grid magnetic-field models, and a
  restructured GUI.
- **`v1.0.0`** — first tagged release (renamed from the earlier `v1.0` tag).
- **`original-state`** — the state of the simulation at the end of the thesis work.

## Description

This project provides a framework for simulating the behavior of lithium-6 (⁶Li) atoms in a two-dimensional magneto-optical trap (2D MOT). It includes a graphical user interface (GUI) for simulation configuration and controls.

## Documentation

A **User Manual** is included as `manual/user_manual.pdf`. It covers the GUI tab
by tab, running simulations (GUI and CLI), the output file format, and the
current known issues and limitations.

## Requirements & Supported Python

* The simulation has been tested and runs stably under **Python 3.12**.
* Install the required Python dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

It is recommended to use a virtual environment (venv or conda) to isolate dependencies.

## Running the GUI

After installing the dependencies, start the GUI from the repository root directory with:

```bash
python -m main --GUI
python -m main --GUI --style dark   # dark theme
```

This will launch the interactive application. Make sure you are running the command from the project root so module imports resolve correctly.

## Command-line Interface (CLI)

Batch simulations can be run headless, without the GUI:

```bash
python -m main --files "setup parameters/Hammel_Setup.json" --target-dir my_results
```

Pass one or more parameter files with `--files`; results are written under
`--target-dir`. See the User Manual for the full option list.

## Contributing

If you want to add features, tests, or documentation, please contact me under frederic@staudt.werkhaeuser.de

## License

This project is published under the **GNU General Public License v3.0 (GPL-3.0)**. See the included `LICENSE` file for the full license text and terms.


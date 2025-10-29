# 2D-MOT-Simulation-for-Lithium-6

Framework to simulate behavior of ⁶Li atoms in a two-dimensional magneto-optical trap (2D MOT).

## Branches

* **main** — immutable snapshot of the repository at the time the Master’s thesis was handed in. This branch reflects the exact code, configuration and results included in the submission.
* **develop** — active development branch intended for ongoing improvements (GUI, features, tests, documentation).

---

## Description

This project provides a framework for simulating the behavior of lithium-6 (⁶Li) atoms in a two-dimensional magneto-optical trap (2D MOT). It includes a graphical user interface (GUI) for interactive control and visualization of simulations.

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
python -m main
```

This will launch the interactive application. Make sure you are running the command from the project root so module imports resolve correctly.

## Command-line Interface (CLI)

* Running simulations through CLI commands is **currently not supported**.
* The primary entry point for now is the GUI. CLI support may be added in a future update on the `develop` branch.

## Contributing

If you want to add features, tests, or documentation, please work on the `develop` branch and open a pull request when ready.

## License

This project is published under the **GNU General Public License v3.0 (GPL-3.0)**. See the included `LICENSE` file for the full license text and terms.

You may also include an SPDX short identifier where appropriate: `SPDX-License-Identifier: GPL-3.0-or-later`.

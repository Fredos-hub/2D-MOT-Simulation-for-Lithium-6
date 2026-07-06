"""
GUI construction smoke test for the post-refactor GUI package layout.

YMainWindow and all five top-level tabs (Simulation Cockpit, Sample Generator,
Incrementor, Plotting, Spectrum) construct without raising under a headless Qt
platform. The settings tabs (simulation / atoms / lasers / magnetic field /
boundaries) are built inside SimulationCockpit, so constructing MainWindow
exercises the whole tab tree in one shot.

Also statically guards against the deleted flat GUI/widgets/*.py modules and the
moved GUI/file_model.py / GUI/oven_worker.py ever being imported again (D-03).

Runs headless via QT_QPA_PLATFORM=offscreen and with the repo root as CWD, because
SimulationCockpit builds its schema path relative to the working directory
(GUI/schema/schema_v1.json) and raises FileNotFoundError otherwise.
"""
import os
import re
from pathlib import Path

# Headless Qt must be selected before any PyQt5 import opens a display connection.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]  # tests/ is one level under repo root
GUI_DIR = REPO_ROOT / "GUI"

EXPECTED_TABS = [
    "Simulation Cockpit",
    "Sample Generator",
    "Diagonalizer Tables",
    "Incrementor",
    "Plotting",
    "Spectrum",
]

# Diagonalizer models added in phase 04; both must auto-appear in the
# interaction dropdown (populated by introspection over src.interactions).
DIAGONALIZER_MODELS = [
    "Lithium6DiagonalizerInteraction",
    "Lithium6DiagonalizerTableInteraction",
]

# Old paths deleted/moved by the GUI refactor. None of these may be imported again.
OLD_WIDGET_MODULES = [
    "atoms_tab", "bar_dipole_table", "boundaries_tab", "edit_all_popup_widget",
    "edit_defaults_popup_widget", "file_table", "incrementor_tab", "io_tap",
    "laser_tab", "magnetic_field_tab", "plotting", "sample_generator",
    "settings_tabs", "simulation_cockpit", "simulation_tab", "spectrum_tab",
    "target_dir_popup", "validation_dialog", "vector_input_widget",
]


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for the whole session (PyQt5 forbids two instances)."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def repo_root_cwd():
    """Run with repo root as CWD so SimulationCockpit's relative schema path resolves."""
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield REPO_ROOT
    finally:
        os.chdir(prev)


def test_main_window_and_all_tabs_construct(qapp, repo_root_cwd):
    """MainWindow + every top-level tab construct without raising under offscreen Qt."""
    from GUI.main_window import MainWindow

    window = MainWindow(qapp)
    try:
        tab_widget = window.mainTabWidget
        assert tab_widget.count() == len(EXPECTED_TABS)
        titles = [tab_widget.tabText(i) for i in range(tab_widget.count())]
        assert titles == EXPECTED_TABS
    finally:
        window.close()


def test_diagonalizer_tab_present(qapp, repo_root_cwd):
    """The 'Diagonalizer Tables' generator tab is registered on MainWindow."""
    from GUI.main_window import MainWindow

    window = MainWindow(qapp)
    try:
        tab_widget = window.mainTabWidget
        titles = [tab_widget.tabText(i) for i in range(tab_widget.count())]
        assert "Diagonalizer Tables" in titles
        # The tab widget renders and exposes the busy/cancel lifecycle hooks.
        tab = window.diagonalizerGeneratorTab
        assert hasattr(tab, "is_busy") and hasattr(tab, "cancel_and_wait")
        assert not tab.is_busy()
    finally:
        window.close()


def test_diagonalizer_models_in_interaction_dropdown(qapp, repo_root_cwd):
    """Both new models auto-appear in the interaction dropdown (introspection)."""
    from GUI.widgets.tabs.simulation_tab import SimulationSettingsTab

    tab = SimulationSettingsTab()
    try:
        combo = tab.interactionCombo
        names = [combo.itemText(i) for i in range(combo.count())]
        for model in DIAGONALIZER_MODELS:
            assert model in names, f"{model} missing from interaction dropdown"
    finally:
        tab.deleteLater()


def _noncomment_lines(path):
    """Yield source lines of a .py file, skipping full-line comments."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith("#"):
            continue
        yield raw


def test_no_imports_of_deleted_old_paths():
    """Static guard (D-03): no GUI source imports a deleted flat-layout module."""
    patterns = [re.compile(rf"GUI\.widgets\.{re.escape(n)}\b") for n in OLD_WIDGET_MODULES]
    patterns += [re.compile(r"GUI\.file_model\b"), re.compile(r"GUI\.oven_worker\b")]

    offenders = []
    for py_file in GUI_DIR.rglob("*.py"):
        for line in _noncomment_lines(py_file):
            if any(p.search(line) for p in patterns):
                offenders.append(f"{py_file.relative_to(REPO_ROOT)}: {line.strip()}")

    assert not offenders, "Imports of deleted old GUI paths found:\n" + "\n".join(offenders)

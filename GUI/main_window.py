# main_window.py
import os
from PyQt5.QtWidgets import QMainWindow, QTabWidget
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QSize

# Import new shells
from GUI.widgets.simulation_cockpit import SimulationCockpit
from GUI.widgets.plotting import PlottingTab
from GUI.toolbar import ToolBar
from GUI.menu_bar import CustomMenuBar
from GUI.widgets.sample_generator import SampleGeneratorTab
from GUI.widgets.incrementor_tab import IncrementorTab


class MainWindow(QMainWindow):
    BASELINE_DPI = 96.0        # baseline to compare against (Windows/web standard)
    MIN_POINT_SIZE = 9.0       # smallest readable point size
    MAX_POINT_SIZE = 28.0      # optional cap to avoid enormous UI

    def __init__(self, app):
        super().__init__()
        self.app = app

        # store the original (unscaled) app font point size once
        base_font = app.font()
        self.base_point_size = base_font.pointSizeF()
        if self.base_point_size <= 0:  # fallback if font returns -1 or 0
            self.base_point_size = 10.0

        # Determine initial screen. Prefer windowHandle().screen() if available,
        # otherwise fall back to primaryScreen().
        screen = None
        if self.windowHandle() is not None and self.windowHandle().screen() is not None:
            screen = self.windowHandle().screen()
        else:
            screen = app.primaryScreen()

        # Compute and apply scale for the initial screen
        self.scale_factor = 1.0
        if screen is not None:
            self.apply_scale_for_screen(screen)

        self.setWindowTitle("⁶Li Simulation")
        icon_path = os.path.join(os.path.dirname(__file__), "icons/simulation_logo_5.png")
        self.setWindowIcon(QIcon(icon_path))

        # Set minimum size proportionally using current scale_factor
        self.setMinimumSize(QSize(int(800 * self.scale_factor), int(600 * self.scale_factor)))
        self.setWindowState(Qt.WindowMaximized)

        self.toolBar = ToolBar(self)
        self.addToolBar(self.toolBar)
        self.menuBar = CustomMenuBar(self)
        self.setMenuBar(self.menuBar)

        self.mainTabWidget = QTabWidget()
        self.mainTabWidget.setMovable(True)
        self.setCentralWidget(self.mainTabWidget)

        self.simulationCockpitTab = SimulationCockpit(self)
        # Single‐model buttons
        self.simulationCockpitTab.fileDirtyChanged.connect(lambda dirty: self.toolBar.save_action.setEnabled(dirty))
        self.simulationCockpitTab.fileDirtyChanged.connect(lambda dirty: self.toolBar.discard_action.setEnabled(dirty))

        # Any‐model buttons
        self.simulationCockpitTab.anyDirtyChanged.connect(lambda dirty: self.toolBar.save_all_action.setEnabled(dirty))
        self.simulationCockpitTab.anyDirtyChanged.connect(lambda dirty: self.toolBar.discard_all_action.setEnabled(dirty))

        # Tab for creating samples with the Sample Generator
        self.SampleGeneratorTab = SampleGeneratorTab(self)
        # Tab for generating parameter files quickly
        self.incrementorTab = IncrementorTab(self)
        self.plottingTab = PlottingTab(self)

        self.mainTabWidget.addTab(self.simulationCockpitTab, "Simulation Cockpit")
        self.mainTabWidget.addTab(self.SampleGeneratorTab, "Sample Generator")
        self.mainTabWidget.addTab(self.incrementorTab, "Incrementor")
        self.mainTabWidget.addTab(self.plottingTab, "Plotting")

        # ToolBar button wiring
        self.toolBar.load_action.triggered.connect(self.simulationCockpitTab.open_directory)
        self.toolBar.new_action.triggered.connect(self.simulationCockpitTab.create_new_file)
        self.toolBar.save_action.triggered.connect(self.simulationCockpitTab.save_file)
        self.toolBar.save_all_action.triggered.connect(self.simulationCockpitTab.save_all)
        self.toolBar.run_action.triggered.connect(self.simulationCockpitTab.run_simulation_from_file_table)
        self.toolBar.discard_action.triggered.connect(self.simulationCockpitTab.discard_changes)
        self.toolBar.discard_all_action.triggered.connect(self.simulationCockpitTab.discard_all_changes)
        self.toolBar.pause_action.triggered.connect(self.simulationCockpitTab.pause_simulation)
        self.toolBar.resume_action.triggered.connect(self.simulationCockpitTab.resume_simulation)
        self.toolBar.cancel_action.triggered.connect(self.simulationCockpitTab.cancel_simulation)
        self.simulationCockpitTab.simulationStateChanged.connect(self._on_simulation_state_changed)

        # Handle dynamic scaling when moving between screens.
        # windowHandle() can be None until the widget is shown; guard it.
        if self.windowHandle() is not None:
            self.windowHandle().screenChanged.connect(self.onScreenChanged)
        else:
            # Connect once the window handle exists (in case __init__ runs before show)
            self.windowHandleChanged.connect(self._onWindowHandleAvailable)

    def _onWindowHandleAvailable(self):
        if self.windowHandle() is not None:
            self.windowHandle().screenChanged.connect(self.onScreenChanged)

    def apply_scale_for_screen(self, screen):
        """Compute scale from screen DPI and apply to app font and geometry."""
        if screen is None:
            return

        dpi = screen.logicalDotsPerInch()
        if dpi <= 0:
            dpi = self.BASELINE_DPI

        scale = dpi / self.BASELINE_DPI
        self.scale_factor = scale

        # compute new point size from original base_point_size
        new_point = self.base_point_size * scale
        # clamp to sensible bounds
        new_point = max(self.MIN_POINT_SIZE, min(self.MAX_POINT_SIZE, new_point))

        font = QFont(self.app.font())  # copy current app font
        font.setPointSizeF(new_point)
        self.app.setFont(font)

        # Update minimum size proportional to scale
        self.setMinimumSize(QSize(int(800 * self.scale_factor), int(600 * self.scale_factor)))

        # Force layout relayout
        self.adjustSize()
        self.updateGeometry()

    def _on_simulation_state_changed(self, state: str):
        """Update toolbar button states when simulation state changes."""
        tb = self.toolBar
        if state == "running":
            tb.run_action.setEnabled(False)
            tb.pause_action.setEnabled(True)
            tb.resume_action.setEnabled(False)
            tb.cancel_action.setEnabled(True)
        elif state == "paused":
            tb.run_action.setEnabled(False)
            tb.pause_action.setEnabled(False)
            tb.resume_action.setEnabled(True)
            tb.cancel_action.setEnabled(True)
        else:  # "idle"
            tb.run_action.setEnabled(True)
            tb.pause_action.setEnabled(False)
            tb.resume_action.setEnabled(False)
            tb.cancel_action.setEnabled(False)

    def onScreenChanged(self, screen):
        """Slot called when the window moves to another screen."""
        self.apply_scale_for_screen(screen)

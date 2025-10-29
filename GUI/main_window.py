# main_window.py
import os
from PyQt5.QtWidgets import QMainWindow, QTabWidget
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QSize

# Import your new shells
from GUI.widgets.simulation_cockpit import SimulationCockpit
from GUI.widgets.plotting import PlottingTab
from GUI.toolbar import ToolBar
from GUI.menu_bar import CustomMenuBar
from GUI.widgets.sample_generator import SampleGeneratorTab
from GUI.widgets.incrementor_tab import IncrementorTab

class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app

        # Retrieve base resolution
        base_w, base_h = app.property('baseResolution')
        # Current screen resolution
        screen = app.primaryScreen()
        cur_w = screen.availableGeometry().width()
        cur_h = screen.availableGeometry().height()
        # Compute scale factor relative to baseline
        factor_w = cur_w / base_w
        factor_h = cur_h / base_h
        self.scale_factor = min(factor_w, factor_h)

        # Apply scaling to application font
        base_font = app.font()
        scaled_point = base_font.pointSizeF() * self.scale_factor
        base_font.setPointSizeF(scaled_point)
        app.setFont(base_font)

        self.setWindowTitle("⁶Li Simulation")
        icon_path = os.path.join(os.path.dirname(__file__), "icons/simulation_logo_5.png")
        self.setWindowIcon(QIcon(icon_path))
        # Set minimum size proportionally
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

        # any‐file buttons
        self.simulationCockpitTab.anyDirtyChanged.   \
            connect(lambda any_dirty: self.toolBar.save_all_action.setEnabled(any_dirty))
        self.simulationCockpitTab.anyDirtyChanged.   \
            connect(lambda any_dirty: self.toolBar.discard_all_action.setEnabled(any_dirty))

        #Tab for creating samples with the Sample Generator
        self.SampleGeneratorTab = SampleGeneratorTab(self)
   

        #Tab for generating incriminating parameter files quickly
        self.incrementorTab = IncrementorTab(self)


        self.plottingTab = PlottingTab(self)
        self.mainTabWidget.addTab(self.simulationCockpitTab, "Simulation Cockpit")
        self.mainTabWidget.addTab(self.SampleGeneratorTab, "Sample Generator")
        self.mainTabWidget.addTab(self.incrementorTab, "Incrementor")
        self.mainTabWidget.addTab(self.plottingTab, "Plotting")

        self.toolBar.load_action.triggered.connect(self.simulationCockpitTab.open_directory)
        self.toolBar.new_action.triggered.connect(self.simulationCockpitTab.create_new_file)
        self.toolBar.save_action.triggered.connect(self.simulationCockpitTab.save_file)
        self.toolBar.save_all_action.triggered.connect(self.simulationCockpitTab.save_all)
        self.toolBar.run_action.triggered.connect(self.simulationCockpitTab.run_simulation_from_file_table)
        self.toolBar.discard_action.triggered.connect(self.simulationCockpitTab.discard_changes)
        self.toolBar.discard_all_action.triggered.connect(self.simulationCockpitTab.discard_all_changes)


        # Handle dynamic scaling when moving between screens
        self.windowHandle().screenChanged.connect(self.onScreenChanged)

    def onScreenChanged(self, screen):
        # Recompute factor and apply
        base_w, base_h = self.app.property('baseResolution')
        cur_w = screen.availableGeometry().width()
        cur_h = screen.availableGeometry().height()
        factor_w = cur_w / base_w
        factor_h = cur_h / base_h
        self.scale_factor = min(factor_w, factor_h)

        # Update font
        font = self.app.font()
        # derive original font size by dividing current by old factor
        orig_size = font.pointSizeF() / self.scale_factor
        new_size = orig_size * self.scale_factor
        font.setPointSizeF(new_size)
        self.app.setFont(font)

        # Update minimum size
        self.setMinimumSize(QSize(int(800 * self.scale_factor), int(600 * self.scale_factor)))
        # Trigger layouts
        self.adjustSize()
        self.updateGeometry()
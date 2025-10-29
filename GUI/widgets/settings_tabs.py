
from PyQt5.QtWidgets import (
    QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal

from PyQt5.QtWidgets import QStyledItemDelegate, QTabWidget, QSizePolicy, QVBoxLayout
from GUI.widgets.simulation_tab import SimulationSettingsTab
from GUI.widgets.atoms_tab import AtomsSettingsTab
from GUI.widgets.laser_tab import LasersSettingsTab
from GUI.widgets.magnetic_field_tab import MagneticFieldSettingsTab
from GUI.widgets.boundaries_tab import BoundariesSettingsTab


class ReadOnlyDelegate(QStyledItemDelegate):
    """A delegate that prevents editing of cells."""
    def createEditor(self, parent, option, index):
        return None
    
class SettingsTabsWidget(QWidget):
    """
    A widget containing tabs for various settings categories.
    Tabs will stretch to fill available horizontal space.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # Create the QTabWidget
        self.tabs = QTabWidget()
        # add the real tab pages
        self.simTab = SimulationSettingsTab()
        self.simTab.setMaximumWidth(800)
        self.atomsTab = AtomsSettingsTab()
        self.atomsTab.setMaximumWidth(800)
        self.lasersTab = LasersSettingsTab()
        #self.lasersTab.setMaximumWidth(800)
        self.magFieldTab = MagneticFieldSettingsTab()


        self.boundTab = BoundariesSettingsTab()
        self.boundTab.setMaximumWidth(800)

        for widget, title in [
            (self.simTab, "Simulation Settings"),
            (self.atomsTab, "Atoms"),
            (self.lasersTab, "Lasers"),
            (self.magFieldTab, "Magnetic Field"),
            (self.boundTab, "Boundaries")
        ]:
            self.tabs.addTab(widget, title)

        self.tabs.tabBar().setExpanding(True)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.tabs)

    def setModel(self, model):
        """Give the FileModel to each child tab."""
        # only tabs that implement setModel will accept it
        for tab in (self.simTab, self.atomsTab, self.lasersTab, 
                    self.magFieldTab, self.boundTab):
            tab.setModel(model)



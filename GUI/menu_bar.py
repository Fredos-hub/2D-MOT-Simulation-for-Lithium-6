from PyQt5.QtWidgets import QMenuBar, QMenu, QAction
from PyQt5.QtGui import QIcon
import os


class CustomMenuBar(QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.create_menus()

    def create_menus(self):
        # File Menu with file related actions.
        file_menu = self.addMenu("File")

        self.load_action = QAction(QIcon(), "Load", self)
        self.save_action = QAction(QIcon(), "Save", self)
        self.save_action.setEnabled(False)  # Initially disabled
        self.save_as_action = QAction(QIcon(), "Save As...", self)
        self.save_as_action.setEnabled(False)
        self.exit_action = QAction(QIcon(), "Exit", self)

        file_menu.addAction(self.load_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        # Simulation Menu with simulation control actions.
        simulation_menu = self.addMenu("Simulation")
        self.run_action = QAction(QIcon(), "Run", self)
        self.pause_action = QAction(QIcon(), "Pause", self)
        self.resume_action = QAction(QIcon(), "Resume", self)
        self.cancel_action = QAction(QIcon(), "Cancel", self)
        simulation_menu.addAction(self.run_action)
        simulation_menu.addAction(self.pause_action)
        simulation_menu.addAction(self.resume_action)
        simulation_menu.addAction(self.cancel_action)

        io_menu = self.addMenu("I/O Settings")
        self.edit_target_directory = QAction(QIcon(), "Set Target Directory", self)
        self.sim_output = QAction(QIcon(), "Set Simulation Output", self)
        io_menu.addAction(self.edit_target_directory)
        io_menu.addAction(self.sim_output)









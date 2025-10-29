from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QLineEdit, QPushButton, QMessageBox, QTableWidgetItem
)
from PyQt5.QtCore import Qt
import os
class ChooseTargetDir(QWidget):
    """
    Popup for editing all selected lasers’ numeric properties at once,
    and writing them back into both the table AND the model.
    """
    def __init__(self, parent_tab):
        """
        parent_tab: instance of LasersSettingsTab
        """
        super().__init__(flags=Qt.Window)
        self.parent_tab = parent_tab
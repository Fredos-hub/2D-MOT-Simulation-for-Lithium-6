from PyQt5.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal

class BoundariesSettingsTab(QWidget):


    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._init_ui()

    def _init_ui(self):
        layout = QFormLayout(self)
        self.xLimitSpin = QDoubleSpinBox()
        self.xLimitSpin.setMaximumWidth(280)
        self.xLimitSpin.valueChanged.connect(self._on_x_limit_changed)
        layout.addRow("X Limit (mm):", self.xLimitSpin)

    def setModel(self, model):
        #self._model = model

        #self._model.blockSignals(True)         
        #val = model.get("Boundaries", "x_limit", default=0.0)
        #self.xLimitSpin.setValue(val)



        #self._model.blockSignals(False) 
        pass
    def _on_x_limit_changed(self, v):
        #self._model.set(v, "Boundaries", "x_limit")
        #self.settingsChanged.emit()
        pass
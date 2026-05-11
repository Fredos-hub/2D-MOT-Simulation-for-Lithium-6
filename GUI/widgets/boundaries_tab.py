from PyQt5.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QLocale
class BoundariesSettingsTab(QWidget):


    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._init_ui()

    def _init_ui(self):

        layout = QFormLayout(self)
        self.XLimitSpin = QDoubleSpinBox()
        self.XLimitSpin.setLocale(QLocale(QLocale.C))
        self.XLimitSpin.setMaximumWidth(280)
        self.XLimitSpin.setMaximum(5000)
        self.XLimitSpin.valueChanged.connect(lambda v: self._update_model('x_limit', v))
        layout.addRow("X Limit (mm):", self.XLimitSpin)

        self.YLimitSpin = QDoubleSpinBox()
        self.YLimitSpin.setLocale(QLocale(QLocale.C))
        self.YLimitSpin.setMaximumWidth(280)
        self.YLimitSpin.setMaximum(5000)
        self.YLimitSpin.valueChanged.connect(lambda v: self._update_model('y_limit', v))
        layout.addRow("Y Limit (mm):", self.YLimitSpin)


        self.ZLimitSpin = QDoubleSpinBox()
        self.ZLimitSpin.setLocale(QLocale(QLocale.C))
        self.ZLimitSpin.setMaximumWidth(280)
        self.ZLimitSpin.setMaximum(5000)
        self.ZLimitSpin.valueChanged.connect(lambda v: self._update_model('z_limit', v))
        layout.addRow("Z Limit (mm):", self.ZLimitSpin)


    def setModel(self, model):
        self._model = model
        for w in (self.XLimitSpin, self.YLimitSpin, self.ZLimitSpin):
            w.blockSignals(True)
        try:
            self.XLimitSpin.setValue(model.get("Boundaries", "x_limit", default=0.0))
            self.YLimitSpin.setValue(model.get("Boundaries", "y_limit", default=0.0))
            self.ZLimitSpin.setValue(model.get("Boundaries", "z_limit", default=0.0))
        finally:
            for w in (self.XLimitSpin, self.YLimitSpin, self.ZLimitSpin):
                w.blockSignals(False)

    def _update_model(self, key, value):
        if not self._model:
            return
        self._model.set(value, 'Boundaries', key)
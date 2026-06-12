from PyQt5.QtWidgets import QFormLayout, QDoubleSpinBox
from PyQt5.QtCore import QLocale
from GUI.widgets.tabs.settings_tab_base import SettingsTab, signals_blocked


class BoundariesSettingsTab(SettingsTab):
    SECTION = "Boundaries"

    def _init_ui(self):
        layout = QFormLayout(self)
        self.XLimitSpin = self._make_spin()
        layout.addRow("X Limit (mm):", self.XLimitSpin)
        self.YLimitSpin = self._make_spin()
        layout.addRow("Y Limit (mm):", self.YLimitSpin)
        self.ZLimitSpin = self._make_spin()
        layout.addRow("Z Limit (mm):", self.ZLimitSpin)

    def _make_spin(self):
        spin = QDoubleSpinBox()
        spin.setLocale(QLocale(QLocale.C))
        spin.setMaximumWidth(280)
        spin.setMaximum(5000)
        return spin

    def _connect_signals(self):
        self.XLimitSpin.valueChanged.connect(lambda v: self._update_model('x_limit', v))
        self.YLimitSpin.valueChanged.connect(lambda v: self._update_model('y_limit', v))
        self.ZLimitSpin.valueChanged.connect(lambda v: self._update_model('z_limit', v))

    def setModel(self, model):
        self._model = model
        with signals_blocked(self.XLimitSpin, self.YLimitSpin, self.ZLimitSpin):
            self.XLimitSpin.setValue(model.safe_get("Boundaries", "x_limit", default=0.0))
            self.YLimitSpin.setValue(model.safe_get("Boundaries", "y_limit", default=0.0))
            self.ZLimitSpin.setValue(model.safe_get("Boundaries", "z_limit", default=0.0))

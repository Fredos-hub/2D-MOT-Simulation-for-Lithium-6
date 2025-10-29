from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QDoubleSpinBox,
    QLabel
)
from PyQt5.QtCore import QLocale
import inspect
import src.magnetic_field as magnetic_field
from GUI.widgets.bar_dipole_table import BarDipolesTable
from GUI.widgets.vector_input_widget import VectorInputWidget
import math

# drop-in replacement for QDoubleSpinBox with auto-precision display
class AutoPrecisionDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, maximum=1e6, step=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRange(-maximum, maximum)
        self.setSingleStep(step)
        # allow up to 12 decimal places internally
        self.setDecimals(12)

    def textFromValue(self, val):
        s = super().textFromValue(val)
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s


class MagneticFieldSettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._field_widgets = []   # (labelItem, fieldItem, key)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        self.layout = QFormLayout(self)
        self.fieldSelectionCombo = QComboBox()
        names, _ = zip(*inspect.getmembers(magnetic_field, inspect.isclass))
        self.fieldSelectionCombo.addItems(
            ["No Magnetic Field"] +
            [n for n in names if n not in ("ECSAtoms",)]
        )
        self.fieldSelectionCombo.setMaximumWidth(280)
        self.layout.addRow("Field Type Selection:", self.fieldSelectionCombo)

    def _connect_signals(self):
        self.fieldSelectionCombo.currentTextChanged.connect(
            lambda t: self._update_model('type', t)
        )
        self.fieldSelectionCombo.currentTextChanged.connect(
            self._on_magnetic_field_type_changed
        )

    def setModel(self, model):
        self._model = model
        try:
            saved = model.get("Magnetic_Fields", "type", default="No Magnetic Field")
        except Exception:
            saved = "No Magnetic Field"

        self.fieldSelectionCombo.blockSignals(True)
        self.fieldSelectionCombo.setCurrentText(saved)
        self.fieldSelectionCombo.blockSignals(False)

        self._on_magnetic_field_type_changed(saved)

    def _clear_field_widgets(self):
        for lbl_item, fld_item, key in self._field_widgets:
            w_lbl = lbl_item.widget()
            w_fld = fld_item.widget()
            self.layout.removeWidget(w_lbl)
            w_lbl.deleteLater()
            self.layout.removeWidget(w_fld)
            w_fld.deleteLater()
        self._field_widgets.clear()

    def _on_magnetic_field_type_changed(self, field_type):
        self._clear_field_widgets()

        if field_type == "No Magnetic Field":
            return

        def add_param(key, text, default, maximum, step):
            spin = AutoPrecisionDoubleSpinBox(maximum=maximum, step=step)
            spin.setLocale(QLocale(QLocale.C))
            spin.setMaximumWidth(280)

            try:
                val = self._model.get("Magnetic_Fields", key, default=default)
            except Exception:
                val = default

            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

            spin.valueChanged.connect(lambda v, k=key: self._update_model(k, v))

            lbl = QLabel(text)
            self.layout.addRow(lbl, spin)
            self._field_widgets.append((
                self.layout.itemAt(self.layout.rowCount()-1, QFormLayout.LabelRole),
                self.layout.itemAt(self.layout.rowCount()-1, QFormLayout.FieldRole),
                key
            ))

        def add_angle_param(key, text, default):
            """
            Add an angle parameter in degrees.
            Stored in model in degrees, converted to radians later in simulation.
            """
            spin = AutoPrecisionDoubleSpinBox(maximum=360.0, step=0.1)
            spin.setLocale(QLocale(QLocale.C))
            spin.setMaximumWidth(280)

            try:
                val = self._model.get("Magnetic_Fields", key, default=default)
            except Exception:
                val = default

            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

            spin.valueChanged.connect(lambda v, k=key: self._update_model(k, v))

            lbl = QLabel(text)
            self.layout.addRow(lbl, spin)
            self._field_widgets.append((
                self.layout.itemAt(self.layout.rowCount()-1, QFormLayout.LabelRole),
                self.layout.itemAt(self.layout.rowCount()-1, QFormLayout.FieldRole),
                key
            ))

        def add_vector_param(key, text, default):
            vec_widget = VectorInputWidget(initial_value=default, parent=self)
            try:
                val = self._model.get("Magnetic_Fields", key, default=default)
            except Exception:
                val = default
            vec_widget.setMaximumWidth(280)
            vec = default
            try:
                if isinstance(val, (list, tuple)) and len(val) == 3:
                    vec = [float(x) for x in val]
                elif isinstance(val, str):
                    parts = [p.strip() for p in val.split(',')]
                    if len(parts) == 3:
                        vec = [float(p) for p in parts]
                    else:
                        vec = default
            except Exception:
                vec = default
            vec_widget.setVector(vec)
            vec_widget.vectorChanged.connect(lambda v, k=key: self._update_model(k, v))

            lbl = QLabel(text)
            self.layout.addRow(lbl, vec_widget)
            self._field_widgets.append((
                self.layout.itemAt(self.layout.rowCount()-1, QFormLayout.LabelRole),
                self.layout.itemAt(self.layout.rowCount()-1, QFormLayout.FieldRole),
                key
            ))

        # Quadrupole (no deltaB anymore)
        if field_type in ("IdealQuadropoleField", "IdealQuadrupoleField"):
            add_param(
                key="field_gradient",
                text="Gradient (T/m):",
                default=0.5,
                maximum=10.0,
                step=1e-4
            )
            add_vector_param(
                key="center_offset",
                text="Center offset (mm):",
                default=[0.0, 0.0, 0.0]
            )

        # Elliptical field (new class)
        elif field_type == "EllipticalMagneticField":
            add_param(
                key="g_x",
                text="Gradient g_x (T/m):",
                default=0.5,
                maximum=10.0,
                step=1e-4
            )
            add_param(
                key="g_y",
                text="Gradient g_y (T/m):",
                default=0.5,
                maximum=10.0,
                step=1e-4
            )
            add_angle_param(
                key="theta_deg",
                text="Tilt angle θ (degrees):",
                default=0.0
            )
            add_vector_param(
                key="center_offset",
                text="Center offset (mm):",
                default=[0.0, 0.0, 0.0]
            )

        elif field_type == "DipoleBarMagneticField":
            label = QLabel("Bar-Dipoles:")
            table = BarDipolesTable(self._model, parent=self)
            self.layout.addRow(label, table)
            self._field_widgets.append((
                self.layout.itemAt(self.layout.rowCount()-1, QFormLayout.LabelRole),
                self.layout.itemAt(self.layout.rowCount()-1, QFormLayout.FieldRole),
                'dipoles'
            ))

    def _update_model(self, key, value):
        if not self._model:
            return
        if key != 'dipoles':
            self._model.set(value, 'Magnetic_Fields', key)

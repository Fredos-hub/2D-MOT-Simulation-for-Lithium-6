import inspect

from PyQt5.QtCore import QLocale
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QLabel

import src.magnetic_field as magnetic_field
from GUI.widgets.common.bar_dipole_table import BarDipolesTable
from GUI.widgets.common.vector_input_widget import VectorInputWidget
from GUI.widgets.tabs.settings_tab_base import SettingsTab, signals_blocked


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
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s


class MagneticFieldSettingsTab(SettingsTab):
    SECTION = "Magnetic_Fields"

    def _init_ui(self):
        self._field_widgets = []  # (labelItem, fieldItem, key)
        self.layout = QFormLayout(self)
        self.fieldSelectionCombo = QComboBox()
        names, _ = zip(
            *inspect.getmembers(magnetic_field, inspect.isclass), strict=False
        )
        self.fieldSelectionCombo.addItems(
            ["No Magnetic Field"]
            + [n for n in names if n not in ("ECSAtoms",)]
        )
        self.fieldSelectionCombo.setMaximumWidth(280)
        self.layout.addRow("Field Type Selection:", self.fieldSelectionCombo)

    def _connect_signals(self):
        self.fieldSelectionCombo.currentTextChanged.connect(
            self._on_field_type_changed
        )

    def setModel(self, model):
        self._model = model
        saved = model.safe_get(
            "Magnetic_Fields", "type", default="No Magnetic Field"
        )
        with signals_blocked(self.fieldSelectionCombo):
            self.fieldSelectionCombo.setCurrentText(saved)
        # Load: rebuild the param widgets without writing back to the model.
        self._rebuild_field_widgets(saved)

    def _on_field_type_changed(self, field_type):
        """User changed the field type: persist it, then rebuild the param widgets."""
        self._update_model("type", field_type)
        self._rebuild_field_widgets(field_type)

    def _clear_field_widgets(self):
        for lbl_item, fld_item, _key in self._field_widgets:
            w_lbl = lbl_item.widget()
            w_fld = fld_item.widget()
            self.layout.removeWidget(w_lbl)
            w_lbl.deleteLater()
            self.layout.removeWidget(w_fld)
            w_fld.deleteLater()
        self._field_widgets.clear()

    def _rebuild_field_widgets(self, field_type):
        self._clear_field_widgets()

        if field_type == "No Magnetic Field":
            return

        def add_param(key, text, default, maximum, step):
            spin = AutoPrecisionDoubleSpinBox(maximum=maximum, step=step)
            spin.setLocale(QLocale(QLocale.C))
            spin.setMaximumWidth(280)

            val = self._model.safe_get("Magnetic_Fields", key, default=default)

            with signals_blocked(spin):
                spin.setValue(val)

            spin.valueChanged.connect(
                lambda v, k=key: self._update_model(k, v)
            )

            lbl = QLabel(text)
            self.layout.addRow(lbl, spin)
            self._field_widgets.append(
                (
                    self.layout.itemAt(
                        self.layout.rowCount() - 1, QFormLayout.LabelRole
                    ),
                    self.layout.itemAt(
                        self.layout.rowCount() - 1, QFormLayout.FieldRole
                    ),
                    key,
                )
            )

        def add_angle_param(key, text, default):
            """
            Add an angle parameter in degrees.
            Stored in model in degrees, converted to radians later in simulation.
            """
            spin = AutoPrecisionDoubleSpinBox(maximum=360.0, step=0.1)
            spin.setLocale(QLocale(QLocale.C))
            spin.setMaximumWidth(280)

            val = self._model.safe_get("Magnetic_Fields", key, default=default)

            with signals_blocked(spin):
                spin.setValue(val)

            spin.valueChanged.connect(
                lambda v, k=key: self._update_model(k, v)
            )

            lbl = QLabel(text)
            self.layout.addRow(lbl, spin)
            self._field_widgets.append(
                (
                    self.layout.itemAt(
                        self.layout.rowCount() - 1, QFormLayout.LabelRole
                    ),
                    self.layout.itemAt(
                        self.layout.rowCount() - 1, QFormLayout.FieldRole
                    ),
                    key,
                )
            )

        def add_vector_param(key, text, default):
            vec_widget = VectorInputWidget(initial_value=default, parent=self)
            val = self._model.safe_get("Magnetic_Fields", key, default=default)
            vec_widget.setMaximumWidth(280)
            vec = default
            try:
                if isinstance(val, (list, tuple)) and len(val) == 3:
                    vec = [float(x) for x in val]
                elif isinstance(val, str):
                    parts = [p.strip() for p in val.split(",")]
                    if len(parts) == 3:
                        vec = [float(p) for p in parts]
                    else:
                        vec = default
            except Exception:
                vec = default
            vec_widget.setVector(vec)
            vec_widget.vectorChanged.connect(
                lambda v, k=key: self._update_model(k, v)
            )

            lbl = QLabel(text)
            self.layout.addRow(lbl, vec_widget)
            self._field_widgets.append(
                (
                    self.layout.itemAt(
                        self.layout.rowCount() - 1, QFormLayout.LabelRole
                    ),
                    self.layout.itemAt(
                        self.layout.rowCount() - 1, QFormLayout.FieldRole
                    ),
                    key,
                )
            )

        # Quadrupole (no deltaB anymore)
        if field_type in ("IdealQuadropoleField", "IdealQuadrupoleField"):
            add_param(
                key="field_gradient",
                text="Gradient (T/m):",
                default=0.5,
                maximum=10.0,
                step=1e-4,
            )
            add_vector_param(
                key="center_offset",
                text="Center offset (mm):",
                default=[0.0, 0.0, 0.0],
            )

        # Elliptical field (new class)
        elif field_type == "EllipticalMagneticField":
            add_param(
                key="g_x",
                text="Gradient g_x (T/m):",
                default=0.5,
                maximum=10.0,
                step=1e-4,
            )
            add_param(
                key="g_y",
                text="Gradient g_y (T/m):",
                default=0.5,
                maximum=10.0,
                step=1e-4,
            )
            add_angle_param(
                key="theta_deg", text="Tilt angle θ (degrees):", default=0.0
            )
            add_vector_param(
                key="center_offset",
                text="Center offset (mm):",
                default=[0.0, 0.0, 0.0],
            )
        elif field_type == "ZeemanField":
            add_param(
                key="slower_length",
                text="Length of the Slower (m)",
                default=0.7,
                maximum=2,
                step=0.005,
            )

            add_param(
                key="B_0",
                text="Maximum Magnetic Field (T)",
                default=0.079,
                maximum=1,
                step=0.0005,
            )

            add_param(
                key="B_bias",
                text="Bias Field (T)",
                default=0.0000,
                maximum=1,
                step=0.0005,
            )
            add_param(
                key="delta_B",
                text="Max. Change in Magnetic Field (%)",
                default=0.1,
                maximum=10,
                step=0.001,
            )
            add_param(
                key="delta_B_min",
                text="Min. Change in Magnetic Field (T)",
                default=1e-5,
                maximum=1e-3,
                step=1e-5,
            )

        elif field_type == "DipoleBarMagneticField":
            label = QLabel("Bar-Dipoles:")
            table = BarDipolesTable(self._model, parent=self)
            self.layout.addRow(label, table)
            self._field_widgets.append(
                (
                    self.layout.itemAt(
                        self.layout.rowCount() - 1, QFormLayout.LabelRole
                    ),
                    self.layout.itemAt(
                        self.layout.rowCount() - 1, QFormLayout.FieldRole
                    ),
                    "dipoles",
                )
            )

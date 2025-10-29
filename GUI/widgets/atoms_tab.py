import os
import inspect
import scipy.constants as scc
from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QComboBox, QCheckBox, QSpinBox, QMessageBox, QPushButton, QFileDialog
)
from PyQt5.QtCore import Qt, QDir
from GUI.widgets.vector_input_widget import VectorInputWidget
import src.atoms as atoms


class AtomsSettingsTab(QWidget):
    """
    Refactored Atom settings tab using FileModel:
      - species selection updates model.
      - number of atoms stored under Atoms.number.
      - mass, transition_frequency, natural_linewidth are read-only fields.
      - start_position and start_velocity use VectorInputWidget.
      - ground_state combo dynamically sized from interaction's number_of_ground_states.
      - randomize_ground_state stored under Atoms.randomize_ground_state.
      - sample_file stored under Atoms.sample_file; disables velocity and ground-state fields when set.
    """


    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._make_species_instances()
        self._init_ui()
        self._connect_signals()

    def _make_species_instances(self):
        # load all ECSAtoms subclasses
        self.species_instances = {}
        for name, cls in inspect.getmembers(atoms, inspect.isclass):
            try:
                inst = cls(n=1)
                self.species_instances[name] = inst
            except Exception:
                pass

    def _init_ui(self):
        layout = QFormLayout(self)
        # Species
        self.speciesCombo = QComboBox()
        self.speciesCombo.addItems(list(self.species_instances.keys()))
        layout.addRow("Atom Species:", self.speciesCombo)
        self.speciesCombo.setMaximumWidth(225)
        # Number
        self.numAtomsSpin = QSpinBox()
        self.numAtomsSpin.setRange(1, 10_000_000)
        layout.addRow("Number of Atoms:", self.numAtomsSpin)
        self.numAtomsSpin.setMaximumWidth(225)
        # Mass (read-only)
        self.massDisplay = QLineEdit()
        self.massDisplay.setReadOnly(True)
        layout.addRow("Mass (u):", self.massDisplay)
        self.massDisplay.setMaximumWidth(225)
        # Transition frequency (read-only)
        self.transitionFreqDisplay = QLineEdit()
        self.transitionFreqDisplay.setReadOnly(True)
        layout.addRow("Transition Frequency (MHz):", self.transitionFreqDisplay)
        self.transitionFreqDisplay.setMaximumWidth(225)
        # Natural linewidth (read-only)
        self.naturalLinewidthDisplay = QLineEdit()
        self.naturalLinewidthDisplay.setReadOnly(True)
        layout.addRow("Natural Linewidth (MHz) 2π ×", self.naturalLinewidthDisplay)
        self.naturalLinewidthDisplay.setMaximumWidth(225)

        # Start pos/vel
        self.startPosWidget = VectorInputWidget()
        layout.addRow("Start Position (m):", self.startPosWidget)
        self.startVelWidget = VectorInputWidget()
        layout.addRow("Start Velocity (m/s):", self.startVelWidget)
        self.startPosWidget.setMaximumWidth(225)
        self.startVelWidget.setMaximumWidth(225)

        # Ground state
        self.groundStateCombo = QComboBox()
        self.groundStateCombo.addItems([str(i) for i in range(6)])
        layout.addRow("Initial Ground State:", self.groundStateCombo)
        self.groundStateCombo.setMaximumWidth(225)

        # Randomize ground state toggle
        self.randomizeCheckbox = QCheckBox("Randomize Ground State")
        layout.addRow(self.randomizeCheckbox)

        # Sample file
        self.sample_line = QLineEdit()
        self.browse_sample_btn = QPushButton("Browse…")
        self.browse_sample_btn.clicked.connect(self._browse_sample)
        self.sample_line.setMaximumWidth(225)
        self.browse_sample_btn.setMaximumWidth(225)
        layout.addRow("Sample file:", self.sample_line)
        layout.addRow("Browse Sample File...", self.browse_sample_btn)

    def _connect_signals(self):
        # Species selection
        self.speciesCombo.currentTextChanged.connect(self._on_species_changed)
        # Number of atoms
        self.numAtomsSpin.valueChanged.connect(lambda v: self._update_model('number', v))
        # Start position / velocity — vectorChanged now emits the full [x,y,z] list
        self.startPosWidget.vectorChanged.connect(lambda vec: self._update_model('start_position', vec))
        self.startVelWidget.vectorChanged.connect(lambda vec: self._update_model('start_velocity', vec))
        # Initial ground state index
        self.groundStateCombo.currentTextChanged.connect(lambda text: self._update_model('ground_state', int(text)))
        # Randomize checkbox
        self.randomizeCheckbox.stateChanged.connect(lambda state: self._update_model('randomize_ground_state', state == Qt.Checked))
        self.randomizeCheckbox.toggled.connect(self.groundStateCombo.setDisabled)
        # Sample file change
        self.sample_line.textChanged.connect(self._on_sample_changed)
        self.sample_line.textChanged.connect(lambda path: self._update_model('sample_file', path))

    def setModel(self, model):
        self._model = model
        self._model.blockSignals(True) 
        for w in (
            self.speciesCombo,
            self.numAtomsSpin,
            self.startPosWidget,
            self.startVelWidget,
            self.groundStateCombo,
            self.randomizeCheckbox,
            self.sample_line
        ):
            w.blockSignals(True)

        try:
            # Populate fields from model
            self.speciesCombo.setCurrentIndex(
                self.speciesCombo.findText(
                    model.safe_get('Atoms', 'species', default=self.speciesCombo.itemText(0))
                )
            )
            inst = self.species_instances.get(self.speciesCombo.currentText(), None)
            if inst is None:
                raise ValueError(f"Species '{self.speciesCombo.currentText()}' not found.")
            self.massDisplay.setText(str(inst.mass_u))
            self.transitionFreqDisplay.setText(f"{inst.transition_frequency/1e6:.1f}")
            lw = inst.natural_linewidth/(2*scc.pi*1e6)
            self.naturalLinewidthDisplay.setText(f"{lw:.2f}")

            self.numAtomsSpin.setValue(
                model.safe_get('Atoms', 'number', default=1)
            )
            self.startPosWidget.setVector(
                model.safe_get('Atoms', 'start_position', default=[0,0,0])
            )
            self.startVelWidget.setVector(
                model.safe_get('Atoms', 'start_velocity', default=[0,0,0])
            )

            # Ground‐state combo
            gs = model.safe_get('Atoms', 'ground_state', default=0)
            idx = self.groundStateCombo.findText(str(gs))
            if idx >= 0:
                self.groundStateCombo.setCurrentIndex(idx)

            # Randomized flag
            rnd = model.safe_get('Atoms', 'randomize_ground_state', default=False)
            self.randomizeCheckbox.setChecked(rnd)

            # Sample file
            sample = model.safe_get('Atoms', 'sample_file', default='')
            self.sample_line.setText(sample)
            # Reflect disable states
            self._update_sample_field_states(bool(sample))

        except Exception as e:
            QMessageBox.warning(
                self,
                "Load Error",
                f"Some atom settings failed to load.\n\n{e}"
            )

        finally:
            for w in (
                self.speciesCombo,
                self.numAtomsSpin,
                self.startPosWidget,
                self.startVelWidget,
                self.groundStateCombo,
                self.randomizeCheckbox,
                self.sample_line
            ):
                w.blockSignals(False)

            self._model.blockSignals(False) 


    def _browse_sample(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Sample", filter="CSV files (*.csv)")
        if path:
            rel = QDir().relativeFilePath(path)
            self.sample_line.setText(rel)
            # write to model
            self._update_model('sample_file', rel)
            # disable relevant fields
            self._update_sample_field_states(True)

    def _on_sample_changed(self, path):
        populated = bool(path)
        # update model
        self._update_model('sample_file', path)
        # reflect UI state
        self._update_sample_field_states(populated)

    def _update_sample_field_states(self, populated):
        self.startPosWidget.setDisabled(populated)
        # Disable velocity when sample is set
        self.startVelWidget.setDisabled(populated)
        # Disable ground-state controls when sample is set
        self.randomizeCheckbox.setDisabled(populated)
        self.groundStateCombo.setDisabled(populated or self.randomizeCheckbox.isChecked())

    def _update_model(self, key, value):
        if not self._model:
            return
        self._model.set(value, 'Atoms', key)

    def _on_species_changed(self, text):
        self._update_model('species', text)
        inst = self.species_instances[self.speciesCombo.currentText()]
        self.massDisplay.setText(str(inst.mass_u))
        self.transitionFreqDisplay.setText(f"{inst.transition_frequency/1e6:.1f}")
        lw = inst.natural_linewidth/(2*scc.pi*1e6)
        self.naturalLinewidthDisplay.setText(f"{lw:.2f}")
        self.setModel(self._model)

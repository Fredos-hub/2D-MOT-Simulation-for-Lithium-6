
import inspect
from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox,QMessageBox,QRadioButton,QGroupBox,
    QComboBox)
from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QIntValidator
import src.interactions as interactions

class SimulationSettingsTab(QWidget):
    """
    Tab for simulation run settings, backed by a FileModel.
    Fields are loaded from the model on setModel(), and changes are
    written back via model.set(...).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._setup_ui()
        self._connect_signals()
    def _setup_ui(self):
        layout = QFormLayout(self)



        #defaultStepSpin
        self.defaultStepSpin = QDoubleSpinBox()
        self.defaultStepSpin.setRange(0, 100)
        self.defaultStepSpin.setMaximumWidth(284)
        self.defaultStepSpin.setLocale(QLocale(QLocale.C))
        self.defaultStepSpin.setSuffix(" μs")     
        self.defaultStepSpin.setDecimals(1)
        self.defaultStepSpin.setSingleStep(0.1)
        layout.addRow("Default Time Step:", self.defaultStepSpin)

        # Step resolution
        self.resolutionSpin = QSpinBox()
        self.resolutionSpin.setRange(1, 1_000_000)
        self.resolutionSpin.setMaximumWidth(284)
        layout.addRow("Step Resolution:", self.resolutionSpin)

        # Max live time (ms)
        self.maxTimeSpin = QDoubleSpinBox()
        # force “.” as decimal point:
        self.maxTimeSpin.setLocale(QLocale(QLocale.C))
        self.maxTimeSpin.setRange(0.0, 1e6)
        self.maxTimeSpin.setDecimals(1)
        self.maxTimeSpin.setMaximumWidth(284)
        layout.addRow("Max Live Time (ms):", self.maxTimeSpin)

        # Random Seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1000000)
        self.seed_spin.setMaximumWidth(284)
        layout.addRow("Random Seed", self.seed_spin)

        # Interaction class
        self.interactionCombo = QComboBox()
        names = [name for name, obj in inspect.getmembers(interactions, inspect.isclass)]
        self.interactionCombo.addItems(names)
        self.interactionCombo.setMaximumWidth(284)
        layout.addRow("Interaction:", self.interactionCombo)

        # --- Rate Mode Settings Group ---
        self.rateGroup = QGroupBox("Rate Mode Settings")   
        rate_layout = QFormLayout(self.rateGroup)
        self.rateGroup.setMaximumWidth(500)
        # Enable/Disable Rate Mode
        self.rateRadioBtn = QRadioButton("Enable")
        rate_layout.addRow("Rate Mode:", self.rateRadioBtn)

        # Flux Spin Box (10^7 1/s)
        self.fluxSpin = QDoubleSpinBox()
        self.fluxSpin.setRange(0.0, 1e3)  # Represented in 10^7 1/s
        self.fluxSpin.setDecimals(2)
        self.fluxSpin.setSingleStep(0.1)
        self.fluxSpin.setLocale(QLocale(QLocale.C))
        self.fluxSpin.setMaximumWidth(500)
        self.fluxSpin.setSuffix(" ×10⁹ 1/s")
        rate_layout.addRow("Flux:", self.fluxSpin)

        # Macroparticle Factor Spin Box
        self.macroparticleSpin = QSpinBox()
        self.macroparticleSpin.setRange(1, 1_000_000)
        self.macroparticleSpin.setValue(5000)
        self.macroparticleSpin.setMaximumWidth(500)
        rate_layout.addRow("Macroparticle Factor:", self.macroparticleSpin)

        # Estimated number of simulated atoms (read-only)
        self.estimatedAtomsEdit = QLineEdit()
        self.estimatedAtomsEdit.setReadOnly(True)
        self.estimatedAtomsEdit.setPlaceholderText("—")  # or "0"
        self.estimatedAtomsEdit.setMaximumWidth(500)
        rate_layout.addRow("Est. # of Simulated Atoms:", self.estimatedAtomsEdit)

        layout.addRow(self.rateGroup)
        # --- End Rate Settings Group ---




    def _connect_signals(self):

        # commit changes on edit/value change


        self.defaultStepSpin.valueChanged.connect(lambda v: self._update_model('default_time_step', v))
        # Connect Signal for the Step Resolution SpinBox
        self.resolutionSpin.valueChanged.connect(lambda v: self._update_model('step_resolution', v))

        # Connect Signals for the Maximum Simulation Time SpinBox
        self.maxTimeSpin.valueChanged.connect(lambda v: self._update_model('max_live_time', v))
        self.maxTimeSpin.valueChanged.connect(self._estimate_atom_count)


        #Connect Signal for the Interaction Selection ComboBox
        self.interactionCombo.currentTextChanged.connect(lambda t: self._update_model('interaction', t))

        # Connect Signal for the Random Seed SpinBox
        self.seed_spin.valueChanged.connect(lambda v: self._update_model("random_seed", v))

        # Connect Signal for the Rate Mode RadioButton
        self.rateRadioBtn.toggled.connect(self._on_rate_mode_toggled)
        self.rateRadioBtn.toggled.connect(lambda v: self._update_model("rate_mode",v))

        # Connect Signals for the Flux SpinBox
        self.fluxSpin.valueChanged.connect(lambda v: self._update_model("flux", v))
        self.fluxSpin.valueChanged.connect(self._estimate_atom_count)

        # Connect Signals for the Macroparticle SpinBox
        self.macroparticleSpin.valueChanged.connect(lambda v: self._update_model("macro_particle_weight", v))
        self.macroparticleSpin.valueChanged.connect(self._estimate_atom_count)   


    def setModel(self, model):
        self._model = model

        # block signals
        self._model.blockSignals(True)
        for w in (
            self.resolutionSpin,
            self.maxTimeSpin,
            self.defaultStepSpin,
            self.interactionCombo,
            self.rateRadioBtn,
            self.fluxSpin,
            self.macroparticleSpin
        ):
            w.blockSignals(True)

        try:
            # Populate
            self.defaultStepSpin.setValue(model.safe_get('Simulation', 'default_time_step', default=10))

            self.resolutionSpin.setValue(model.safe_get('Simulation', 'step_resolution', default=10))

            self.maxTimeSpin.setValue(model.safe_get('Simulation', 'max_live_time', default=3))


            seed = model.safe_get("Simulation", "random_seed", default = 0)
            self.seed_spin.setValue(seed)

            rate_mode = model.safe_get("Simulation", "rate_mode", default = False)
            self.rateRadioBtn.setChecked(rate_mode)

            self.fluxSpin.setValue(
                model.safe_get("Simulation", "flux", default = 3))
            
            self.macroparticleSpin.setValue(
                model.safe_get("Simulation", "macro_particle_weight", default = 5000))

            interaction = model.safe_get(
                'Simulation', 'interaction', default='Lithium18LevelInteraction'
            )
            if interaction:
                idx = self.interactionCombo.findText(interaction)
                if idx >= 0:
                    self.interactionCombo.setCurrentIndex(idx)

        except Exception as e:
            # Show a warning and continue with defaults
            QMessageBox.warning(
                self,
                "Load Error",
                f"Some simulation settings failed to load.\n\n{e}"
            )



        finally:
            # unblock signals

            for w in (
                self.resolutionSpin,
                self.defaultStepSpin,
                self.maxTimeSpin,
                self.interactionCombo,
                self.rateRadioBtn,
                self.fluxSpin,
                self.macroparticleSpin
            ):
                w.blockSignals(False)
            self._model.blockSignals(False)
            self._estimate_atom_count()   


    def _update_model(self, key, value):
        if not self._model:
            return
        self._model.set(value, 'Simulation', key)




    def _on_rate_mode_toggled(self, checked):
        # Enable/disable just the Flux and Macroparticle spin boxes
        self.fluxSpin.setEnabled(checked)
        self.macroparticleSpin.setEnabled(checked)



    def _estimate_atom_count(self):

        rate = self.fluxSpin.value() * 1e9
        simulation_time = self.maxTimeSpin.value() * 1e-3
        macro_particle_factor = self.macroparticleSpin.value()
        number_of_atoms = round(rate*simulation_time/macro_particle_factor)


        self.estimatedAtomsEdit.setText(str(number_of_atoms))

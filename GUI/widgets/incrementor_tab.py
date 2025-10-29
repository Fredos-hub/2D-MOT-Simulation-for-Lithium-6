import json
from pathlib import Path
from PyQt5.QtCore import QLocale, Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton, QListWidget,
    QFileDialog, QRadioButton, QButtonGroup, QDoubleSpinBox, QMessageBox,
    QFormLayout, QComboBox, QSizePolicy
)

from GUI.widgets.vector_input_widget import VectorInputWidget

class IncrementorTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Left panel: settings ---
        settings_box = QGroupBox("Increment Settings")
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setSpacing(8)  # tighter spacing between sections

        # Help button at top right
        help_btn = QPushButton("?")
        help_btn.setFixedSize(24, 24)
        help_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        help_btn.setToolTip("Show help information")
        help_btn.clicked.connect(self._show_help)
        settings_layout.addWidget(help_btn, alignment=Qt.AlignRight)

        # Exclusive radios
        self.radio_group = QButtonGroup(self)
        # --- Atoms ---
        self.radio_atoms = QRadioButton("Atoms → Start Velocity")
        self.radio_group.addButton(self.radio_atoms, 0)
        settings_layout.addWidget(self.radio_atoms)
        # Vector inputs
        self.vec_group = QWidget()
        vec_layout = QFormLayout(self.vec_group)
        self.from_vec = VectorInputWidget()
        self.to_vec   = VectorInputWidget()
        self.step_vec = VectorInputWidget()
        vec_layout.addRow("From (vx, vy, vz):", self.from_vec)
        vec_layout.addRow("To   (vx, vy, vz):", self.to_vec)
        vec_layout.addRow("Step (vx, vy, vz):", self.step_vec)
        settings_layout.addWidget(self.vec_group)

        # --- Waist ---
        self.radio_waist = QRadioButton("Lasers → Waist (mm)")
        self.radio_group.addButton(self.radio_waist, 1)
        settings_layout.addWidget(self.radio_waist)
        self.waist_group = QWidget()
        waist_layout = QFormLayout(self.waist_group)
        self.waist_scope = QComboBox()
        self.waist_scope.addItems(["All", "Trap only", "Repump only"])
        waist_layout.addRow("Laser Scope:", self.waist_scope)
        self.from_waist = QDoubleSpinBox(); self.from_waist.setDecimals(1)
        self.to_waist   = QDoubleSpinBox(); self.to_waist.setDecimals(1)
        self.step_waist = QDoubleSpinBox(); self.step_waist.setDecimals(1)
        for spin in (self.from_waist, self.to_waist, self.step_waist):
            spin.setRange(-1e6,1e6); spin.setSingleStep(0.1)
            spin.setLocale(QLocale(QLocale.C))
        waist_layout.addRow("From (mm):", self.from_waist)
        waist_layout.addRow("To  (mm):", self.to_waist)
        waist_layout.addRow("Step(mm):", self.step_waist)
        settings_layout.addWidget(self.waist_group)

        # --- Power ---
        self.radio_power = QRadioButton("Lasers → Power (mW)")
        self.radio_group.addButton(self.radio_power, 2)
        settings_layout.addWidget(self.radio_power)
        self.power_group = QWidget()
        power_layout = QFormLayout(self.power_group)
        self.power_scope = QComboBox()
        self.power_scope.addItems(["All", "Trap only", "Repump only"])
        power_layout.addRow("Laser Scope:", self.power_scope)
        self.from_power = QDoubleSpinBox(); self.from_power.setDecimals(1)
        self.to_power   = QDoubleSpinBox(); self.to_power.setDecimals(1)
        self.step_power = QDoubleSpinBox(); self.step_power.setDecimals(1)
        for spin in (self.from_power, self.to_power, self.step_power):
            spin.setRange(-1e6,1e6); spin.setSingleStep(0.1)
            spin.setLocale(QLocale(QLocale.C))
        power_layout.addRow("From (mW):", self.from_power)
        power_layout.addRow("To  (mW):", self.to_power)
        power_layout.addRow("Step(mW):", self.step_power)
        settings_layout.addWidget(self.power_group)

        # --- Detuning ---
        self.radio_detune = QRadioButton("Lasers → Detuning (Γ)")
        self.radio_group.addButton(self.radio_detune, 3)
        settings_layout.addWidget(self.radio_detune)
        self.detune_group = QWidget()
        detune_layout = QFormLayout(self.detune_group)
        self.detune_scope = QComboBox()
        self.detune_scope.addItems(["All", "Trap only", "Repump only"])
        detune_layout.addRow("Laser Scope:", self.detune_scope)
        self.from_detune = QDoubleSpinBox(); self.from_detune.setDecimals(1)
        self.to_detune   = QDoubleSpinBox(); self.to_detune.setDecimals(1)
        self.step_detune = QDoubleSpinBox(); self.step_detune.setDecimals(1)
        for spin in (self.from_detune, self.to_detune, self.step_detune):
            spin.setRange(-1e6,1e6); spin.setSingleStep(0.1)
            spin.setLocale(QLocale(QLocale.C))
        detune_layout.addRow("From (Γ):", self.from_detune)
        detune_layout.addRow("To  (Γ):", self.to_detune)
        detune_layout.addRow("Step(Γ):", self.step_detune)
        settings_layout.addWidget(self.detune_group)

        # Generate
        self.generate_btn = QPushButton("Generate Files")
        self.generate_btn.clicked.connect(self._on_generate)
        settings_layout.addWidget(self.generate_btn)

        main_layout.addWidget(settings_box)

        # --- Right panel: files ---
        files_box = QGroupBox("JSON Files")
        files_layout = QVBoxLayout(files_box)
        btns = QHBoxLayout()
        add_btn = QPushButton("Add Files…")
        remove_btn = QPushButton("Remove Selected")
        add_btn.clicked.connect(self._add_files)
        remove_btn.clicked.connect(self._remove_selected)
        btns.addWidget(add_btn); btns.addWidget(remove_btn)
        files_layout.addLayout(btns)
        self.file_list = QListWidget()
        files_layout.addWidget(self.file_list)
        main_layout.addWidget(files_box)

        # Signals
        self.radio_group.buttonClicked[int].connect(self._update_enabled)
        self._update_enabled(0)

    def _show_help(self):
        QMessageBox.information(
            self, "Incrementor Help",
            "This tab allows you to generate variations of your JSON configurations by sweeping either atom start velocities or laser parameters.\n\n"
            "• Select one or more JSON files on the right.\n"
            "• Choose a mode on the left: vector sweep (velocity) or scalar sweep (waist, power, detuning).\n"
            "• Enter ranges: For velocity, specify From/To/Step for each component; leave Step=0 to fix that component.\n"
            "• For lasers, pick a scope (All/Trap/Repump) and set From/To/Step.\n"
            "• Click Generate and pick a target folder—files will be output with the parameter values in their names."
        )

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select JSON Files", "", "JSON Files (*.json)")
        for p in paths:
            if not any(self.file_list.item(i).text() == p for i in range(self.file_list.count())):
                self.file_list.addItem(p)

    def _remove_selected(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _update_enabled(self, idx):
        sel = self.radio_group.checkedId()
        groups = [self.vec_group, self.waist_group, self.power_group, self.detune_group]
        scopes = [None, self.waist_scope, self.power_scope, self.detune_scope]
        for i, grp in enumerate(groups): grp.setEnabled(i == sel)
        for i in (1,2,3): scopes[i].setEnabled(i == sel)

    def _on_generate(self):
        # 1) Gather files
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            QMessageBox.warning(self, "No files", "Please add at least one JSON file.")
            return

        # 2) Mode and count
        idx = self.radio_group.checkedId()
        if idx == 0:
            from_vals = [float(e.text()) for e in self.from_vec.edits]
            to_vals   = [float(e.text()) for e in self.to_vec.edits]
            step_vals = [float(e.text()) for e in self.step_vec.edits]
            value_lists = []
            for f, t, s in zip(from_vals, to_vals, step_vals):
                if s > 0 and t >= f:
                    n_steps = int((t - f) / s) + 1
                    value_lists.append([f + k * s for k in range(n_steps)])
                else:
                    value_lists.append([f])
            total = len(files) * len(value_lists[0]) * len(value_lists[1]) * len(value_lists[2])
        else:
            widget_map = {1: (self.from_waist, self.to_waist, self.step_waist),
                          2: (self.from_power, self.to_power, self.step_power),
                          3: (self.from_detune, self.to_detune, self.step_detune)}
            f_w, t_w, s_w = widget_map[idx]
            f, t, s = f_w.value(), t_w.value(), s_w.value()
            if s <= 0 or t < f:
                QMessageBox.warning(self, "Invalid range", "Check that From ≤ To and Step > 0.")
                return
            count = int((t - f) / s) + 1
            total = len(files) * count

        # 3) Select output folder
        target_dir = QFileDialog.getExistingDirectory(self, "Select Target Directory")
        if not target_dir: return

        # 4) Confirm
        yn = QMessageBox.question(self, "Confirm",
                                  f"About to generate {total} files into:\n{target_dir}\nProceed?",
                                  QMessageBox.Yes | QMessageBox.No)
        if yn != QMessageBox.Yes: return

        # 5) Generate
        for path in files:
            templ = json.loads(Path(path).read_text())
            base = Path(path).stem
            if idx == 0:
                for vx in value_lists[0]:
                    for vy in value_lists[1]:
                        for vz in value_lists[2]:
                            cfg = json.loads(json.dumps(templ))
                            cfg['Atoms']['start_velocity'] = [vx, vy, vz]
                            json_fname = f"{base}_vx{vx:.1f}_vy{vy:.1f}_vz{vz:.1f}.json"
                            csv_fname = json_fname.replace('.json', '.csv')
                            cfg['output_file'] = csv_fname
                            Path(target_dir, json_fname).write_text(json.dumps(cfg, indent=4))
            else:
                scopes = [self.waist_scope, self.power_scope, self.detune_scope]
                unit_map = {1: 'mm', 2: 'mW', 3: 'Gamma'}
                key_map  = {1: 'waist', 2: 'beam_power', 3: 'detuning'}
                div_map  = {1: 1e3, 2: 1e3, 3: 1.0}
                choice = scopes[idx - 1].currentIndex()
                unit   = unit_map[idx]
                key    = key_map[idx]
                div    = div_map[idx]
                for n in range(count):
                    val = f + n * s
                    cfg = json.loads(json.dumps(templ))
                    for las in cfg.get('Lasers', []):
                        typ = las.get('type', '')
                        if choice == 1 and typ != 'trap': continue
                        if choice == 2 and typ != 'repump': continue
                        las[key] = val / div

                    if idx == 3 and choice in (1, 2):
                        # replace “detuning” with “trap” or “repump”
                        name_key = 'trap' if choice == 1 else 'repump'
                    else:
                        name_key = key_map[idx]
                    json_fname = f"{base}_{name_key}_{val:.1f}{unit}.json"
                    csv_fname = json_fname.replace('.json', '.csv')
                    cfg['Simulation']['output_file'] = csv_fname
                    Path(target_dir, json_fname).write_text(json.dumps(cfg, indent=4))
        QMessageBox.information(self, "Done", f"Generated {total} files.")

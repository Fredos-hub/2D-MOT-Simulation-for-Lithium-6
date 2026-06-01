import json
from itertools import product
from pathlib import Path
from PyQt5.QtCore import QLocale, Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton, QListWidget,
    QFileDialog, QRadioButton, QButtonGroup, QDoubleSpinBox, QMessageBox,
    QFormLayout, QComboBox, QSizePolicy, QListWidgetItem
)

from GUI.widgets.vector_input_widget import VectorInputWidget

_SCOPE_LABEL = {0: "All", 1: "Trap", 2: "Repump"}

class IncrementorTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Left column: settings + queue + generate ---
        left_col = QVBoxLayout()

        settings_box = QGroupBox("Configure Sweep")
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

        # Add Sweep
        self.add_sweep_btn = QPushButton("Add Sweep")
        self.add_sweep_btn.clicked.connect(self._on_add_sweep)
        settings_layout.addWidget(self.add_sweep_btn)

        left_col.addWidget(settings_box)

        # ---- Queue box ----
        queue_box = QGroupBox("Queued Sweeps")
        queue_layout = QVBoxLayout(queue_box)
        self.queue_list = QListWidget()
        queue_layout.addWidget(self.queue_list)
        q_btns = QHBoxLayout()
        self.remove_sweep_btn = QPushButton("Remove Selected")
        self.clear_sweeps_btn = QPushButton("Clear All")
        self.remove_sweep_btn.clicked.connect(self._on_remove_sweep)
        self.clear_sweeps_btn.clicked.connect(self._on_clear_sweeps)
        q_btns.addWidget(self.remove_sweep_btn)
        q_btns.addWidget(self.clear_sweeps_btn)
        queue_layout.addLayout(q_btns)
        left_col.addWidget(queue_box)

        # Generate button at the very bottom of the left column
        self.generate_btn = QPushButton("Generate Files")
        self.generate_btn.clicked.connect(self._on_generate)
        left_col.addWidget(self.generate_btn)

        main_layout.addLayout(left_col)

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
            "Generate variations of JSON configs by sweeping atom velocity or "
            "laser parameters along one or more axes.\n\n"
            "Workflow:\n"
            "• Add the JSON template files on the right.\n"
            "• On the left, configure a sweep (velocity vx/vy/vz, or laser waist/power/detuning).\n"
            "• Click Add Sweep → it appears in the Queued Sweeps list.\n"
            "• Repeat for any further axes you want combined.\n"
            "• Click Generate Files → produces the Cartesian product:\n"
            "      len(files) × ∏ (values per queued sweep)\n\n"
            "Velocity: each axis with Step > 0 becomes its own queued entry (vx, vy, vz independent).\n"
            "Laser sweeps: conflict if same parameter+scope, or if one is All and the other is Trap/Repump only.\n"
            "Generate refuses an empty queue."
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

    # ---------------- Sweep queue ----------------

    def _queued_sweeps(self):
        return [self.queue_list.item(i).data(Qt.UserRole)
                for i in range(self.queue_list.count())]

    @staticmethod
    def _range_values(f, t, s):
        if s <= 0 or t < f:
            return []
        n = int((t - f) / s) + 1
        return [f + k * s for k in range(n)]

    def _on_add_sweep(self):
        idx = self.radio_group.checkedId()
        if idx == 0:
            self._add_velocity_sweeps()
        else:
            self._add_laser_sweep(idx)

    def _add_velocity_sweeps(self):
        froms = [float(e.text()) for e in self.from_vec.edits]
        tos   = [float(e.text()) for e in self.to_vec.edits]
        steps = [float(e.text()) for e in self.step_vec.edits]
        axis_kinds = ('vx', 'vy', 'vz')
        candidates = []
        for f, t, s, kind in zip(froms, tos, steps, axis_kinds):
            if s <= 0:
                continue
            vals = self._range_values(f, t, s)
            if not vals:
                continue
            candidates.append({"kind": kind, "scope": None, "values": vals,
                               "from": f, "to": t, "step": s})
        if not candidates:
            QMessageBox.warning(self, "Nothing to add",
                                "No velocity axis has Step > 0 with From ≤ To.")
            return
        existing = self._queued_sweeps()
        conflicts = [c['kind'] for c in candidates if self._conflicts(existing, c)]
        if conflicts:
            QMessageBox.warning(self, "Conflict",
                                f"Already queued: {', '.join(conflicts)}. "
                                f"Remove the existing entries first.")
            return
        for c in candidates:
            self._enqueue(c)

    def _add_laser_sweep(self, idx):
        widget_map = {
            1: (self.from_waist,  self.to_waist,  self.step_waist,  self.waist_scope,  'waist'),
            2: (self.from_power,  self.to_power,  self.step_power,  self.power_scope,  'power'),
            3: (self.from_detune, self.to_detune, self.step_detune, self.detune_scope, 'detuning'),
        }
        f_w, t_w, s_w, scope_w, kind = widget_map[idx]
        f, t, s = f_w.value(), t_w.value(), s_w.value()
        vals = self._range_values(f, t, s)
        if not vals:
            QMessageBox.warning(self, "Invalid range",
                                "Check that From ≤ To and Step > 0.")
            return
        sweep = {"kind": kind, "scope": scope_w.currentIndex(),
                 "values": vals, "from": f, "to": t, "step": s}
        if self._conflicts(self._queued_sweeps(), sweep):
            QMessageBox.warning(
                self, "Conflict",
                f"A {kind} sweep with overlapping scope is already queued. "
                f"Remove it first or pick a non-overlapping scope."
            )
            return
        self._enqueue(sweep)

    def _enqueue(self, sweep):
        item = QListWidgetItem(self._format_label(sweep))
        item.setData(Qt.UserRole, sweep)
        self.queue_list.addItem(item)

    def _on_remove_sweep(self):
        for item in self.queue_list.selectedItems():
            self.queue_list.takeItem(self.queue_list.row(item))

    def _on_clear_sweeps(self):
        self.queue_list.clear()

    @staticmethod
    def _conflicts(existing, new):
        for e in existing:
            if e['kind'] != new['kind']:
                continue
            if new['kind'] in ('vx', 'vy', 'vz'):
                return True
            if e['scope'] == new['scope']:
                return True
            if 0 in (e['scope'], new['scope']):   # "All" overlaps everything
                return True
        return False

    @staticmethod
    def _format_label(sweep):
        k = sweep['kind']
        n = len(sweep['values'])
        rng = f"{sweep['from']} → {sweep['to']} step {sweep['step']}"
        if k in ('vx', 'vy', 'vz'):
            return f"Velocity · {k} · {rng}  ({n} vals)"
        unit = {'waist': 'mm', 'power': 'mW', 'detuning': 'Γ'}[k]
        scope = _SCOPE_LABEL.get(sweep['scope'], '?')
        return f"{k.capitalize()} · {scope} · {rng} {unit}  ({n} vals)"

    @staticmethod
    def _apply_sweep(cfg, sweep, value):
        k = sweep['kind']
        if k in ('vx', 'vy', 'vz'):
            axis = {'vx': 0, 'vy': 1, 'vz': 2}[k]
            atoms = cfg.setdefault('Atoms', {})
            sv = list(atoms.get('start_velocity', [0.0, 0.0, 0.0]))
            sv += [0.0] * max(0, 3 - len(sv))
            sv[axis] = value
            atoms['start_velocity'] = sv
            return
        key = {'waist': 'waist', 'power': 'beam_power', 'detuning': 'detuning'}[k]
        div = {'waist': 1e3, 'power': 1e3, 'detuning': 1.0}[k]
        scope = sweep['scope']
        for las in cfg.get('Lasers', []):
            typ = las.get('type', '')
            if scope == 1 and typ != 'trap':
                continue
            if scope == 2 and typ != 'repump':
                continue
            las[key] = value / div

    @staticmethod
    def _tag_for(sweep, value):
        k = sweep['kind']
        if k == 'vx': return f"vx{value:.1f}"
        if k == 'vy': return f"vy{value:.1f}"
        if k == 'vz': return f"vz{value:.1f}"
        unit = {'waist': 'mm', 'power': 'mW', 'detuning': 'Gamma'}[k]
        name = {'waist': 'waist', 'power': 'beam_power', 'detuning': 'detuning'}[k]
        scope = sweep['scope']
        if k == 'detuning' and scope in (1, 2):
            prefix = 'trap' if scope == 1 else 'repump'
            return f"{prefix}_{value:.1f}{unit}"
        if scope == 0:
            return f"{name}_{value:.1f}{unit}"
        scope_prefix = 'trap' if scope == 1 else 'repump'
        return f"{scope_prefix}_{name}_{value:.1f}{unit}"

    # ---------------- Generate ----------------

    def _on_generate(self):
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            QMessageBox.warning(self, "No files", "Please add at least one JSON file.")
            return

        sweeps = self._queued_sweeps()
        if not sweeps:
            QMessageBox.warning(
                self, "Empty queue",
                "Add at least one sweep with the Add Sweep button before generating."
            )
            return

        per_axis = [len(s['values']) for s in sweeps]
        total_combos = 1
        for n in per_axis:
            total_combos *= n
        total = len(files) * total_combos

        target_dir = QFileDialog.getExistingDirectory(self, "Select Target Directory")
        if not target_dir:
            return

        yn = QMessageBox.question(
            self, "Confirm",
            f"About to generate {total} files into:\n{target_dir}\n"
            f"({len(files)} input × {total_combos} combinations from {len(sweeps)} sweep axis/axes)\n"
            f"Proceed?",
            QMessageBox.Yes | QMessageBox.No
        )
        if yn != QMessageBox.Yes:
            return

        written = 0
        for path in files:
            templ = json.loads(Path(path).read_text())
            base = Path(path).stem
            for combo in product(*(s['values'] for s in sweeps)):
                cfg = json.loads(json.dumps(templ))
                tags = []
                for sweep, value in zip(sweeps, combo):
                    self._apply_sweep(cfg, sweep, value)
                    tags.append(self._tag_for(sweep, value))
                fname = f"{base}_{'_'.join(tags)}.json"
                Path(target_dir, fname).write_text(json.dumps(cfg, indent=4))
                written += 1
        QMessageBox.information(self, "Done", f"Generated {written} files.")

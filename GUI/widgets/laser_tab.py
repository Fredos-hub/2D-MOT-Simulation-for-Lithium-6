from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QComboBox, QMessageBox, QHeaderView, QSizePolicy
)
from PyQt5.QtCore import Qt
from GUI.widgets.vector_input_widget import VectorInputWidget
from GUI.widgets.edit_all_popup_widget import EditAllPopup


class LasersSettingsTab(QWidget):
    """
    Widget to display and edit an array of laser configurations
    according to the provided JSON schema for Lasers.
    """
    TYPE_OPTIONS = ["unspecified", "repump", "trap"]
    HELICITY_OPTIONS = ["-1", "+1"]

    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self._model = None
        self.currentRow = None
        self._init_ui()
        if model is not None:
            self.setModel(model)
        self.popup = None

    def _init_ui(self):
        self.setWindowTitle("Laser Properties Editor")
        layout = QHBoxLayout(self)

        # Left panel buttons
        panel = QWidget()
        panelLayout = QVBoxLayout(panel)
        for text, slot in [
            ("Add Trapping Laser", lambda: self._add_new_laser('trap')),
            ("Add Repump Laser", lambda: self._add_new_laser('repump')),
            ("Add Lasers from Preset", lambda: self._add_new_laser(None)),
            ("Save Configuration as Preset", self._save_preset_if_needed),
            ("Edit Laser Defaults", self._edit_defaults),
            ("Edit All Selected", self._edit_all_selected)
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            panelLayout.addWidget(btn)
        panelLayout.addStretch()
        panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Table with built-in scrollbars
        self.table = QTableWidget(0, 8)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Allow horizontal scrolling when needed
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.horizontalHeader().setDefaultSectionSize(140)



        # Assemble layout
        layout.addWidget(panel)
        layout.addWidget(self.table)

        # Column headers
        headers = [
            "Type", "Frequency (MHz)", "Detuning (Γ)",
            "Power (mW)", "Waist (mm)", "Origin",
            "Direction", "Handedness"
        ]
        self.table.setHorizontalHeaderLabels(headers)

        # Data column configuration
        self.columnConfig = {
            1: {'key': 'beam_frequency', 'factor': 1,    'format': '{:.3f}'},
            2: {'key': 'detuning',       'factor': 1,    'format': '{:.1f}'},
            3: {'key': 'beam_power',     'factor': 1e3, 'format': '{:.1f}'},
            4: {'key': 'waist',          'factor': 1e3, 'format': '{:.1f}'}
        }

        # Connect cell editing and selection
        self.table.cellClicked.connect(lambda r, c: setattr(self, 'currentRow', r))
        self.table.itemChanged.connect(self._handle_item_changed)

    def setModel(self, model):
        self._model = model
        self._model.blockSignals(True)
        self.table.blockSignals(True)
        try:
            self.table.clearContents()
            laserList = self._model.get('Lasers', default=[]) or []
            self.table.setRowCount(len(laserList))
            for idx, cfg in enumerate(laserList):
                self._populate_row(idx, cfg)
            self.table.setVerticalHeaderLabels([f"L{n}" for n in range(len(laserList))])
        finally:
            self.table.blockSignals(False)
            self._model.blockSignals(False)

    def _populate_row(self, row, cfg):
        # Type
        combo = QComboBox()
        combo.addItems(self.TYPE_OPTIONS)
        combo.setCurrentText(cfg.get('type', 'unspecified'))
        combo.currentTextChanged.connect(lambda val, r=row: self._update_model(r, 'type', val))
        self.table.setCellWidget(row, 0, combo)
        # Numeric
        for col, info in self.columnConfig.items():
            val = cfg.get(info['key'], 0.0) * info['factor']
            item = QTableWidgetItem(info['format'].format(val))
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
        # Vectors
        for key, col in [('origin', 5), ('direction', 6)]:
            w = VectorInputWidget(cfg.get(key, [0,0,0]), self)
            w.vectorChanged.connect(lambda vec, r=row, k=key: self._update_model(r, k, vec))
            self.table.setCellWidget(row, col, w)
        # Handedness
        heli = QComboBox()
        heli.addItems(self.HELICITY_OPTIONS)
        txt = f"{cfg.get('handedness', 1):+d}"
        idx = self.HELICITY_OPTIONS.index(txt) if txt in self.HELICITY_OPTIONS else 1
        heli.setCurrentIndex(idx)
        heli.currentTextChanged.connect(lambda v, r=row: self._update_model(r, 'handedness', int(v)))
        self.table.setCellWidget(row, 7, heli)

    def _handle_item_changed(self, item):
        if not self._model: return
        row, col = item.row(), item.column()
        if col in self.columnConfig:
            info = self.columnConfig[col]
            try:
                val = float(item.text()) / info['factor']
            except ValueError:
                QMessageBox.warning(self, 'Invalid', f'"{item.text()}" not a number')
                return
            lst = list(self._model.get('Lasers', default=[]) or [])
            if row < len(lst):
                lst[row][info['key']] = val
                self._model.set(lst, 'Lasers')

    def _add_new_laser(self, kind):
        if not self._model: return
        defaults = self._model.get('Lasers_defaults', {}) or {}
        cfg = {**defaults, 'type': kind or defaults.get('type', 'unspecified')}
        lst = list(self._model.get('Lasers', default=[]) or [])
        lst.append(cfg)
        self._model.set(lst, 'Lasers')
        self.setModel(self._model)


    def _update_model(self, rowIndex, keyName, value):
        """Update a specific property on one laser and mark dirty"""
        if not self._model:
            return
        laserList = self._model.get('Lasers', default=[]) or []
        if 0 <= rowIndex < len(laserList):
            newList = list(laserList)
            newList[rowIndex] = dict(newList[rowIndex])
            newList[rowIndex][keyName] = value
            self._model.set(newList, 'Lasers')

    def _edit_all_selected(self):
        self.popup = EditAllPopup(self)
        self.popup.show()
        pass

    # Placeholder slots
    def _save_preset_if_needed(self): pass
    def _edit_defaults(self): pass

if __name__ == "__main__":
    pass

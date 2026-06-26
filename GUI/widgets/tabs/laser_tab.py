import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from GUI.widgets.common.vector_input_widget import VectorInputWidget
from GUI.widgets.dialogs.edit_all_popup_widget import EditAllPopup
from GUI.widgets.dialogs.edit_defaults_popup_widget import EditDefaultsPopup
from GUI.widgets.tabs.settings_tab_base import signals_blocked


class LasersSettingsTab(QWidget):
    """
    Widget to display and edit an array of laser configurations
    according to the provided JSON schema for Lasers.
    """

    TYPE_OPTIONS = ["unspecified", "repump", "trap"]
    HELICITY_OPTIONS = ["-1", "0", "+1"]

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

        # Edit defaults button
        edit_defaults_btn = QPushButton("Edit Laser Defaults")
        edit_defaults_btn.clicked.connect(self._edit_defaults)
        panelLayout.addWidget(edit_defaults_btn)

        # Separator under Edit defaults
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        panelLayout.addWidget(sep1)

        # Add/remove buttons
        add_trap_btn = QPushButton("Add Trapping Laser")
        add_trap_btn.clicked.connect(lambda: self._add_new_laser("trap"))
        panelLayout.addWidget(add_trap_btn)

        add_repump_btn = QPushButton("Add Repump Laser")
        add_repump_btn.clicked.connect(lambda: self._add_new_laser("repump"))
        panelLayout.addWidget(add_repump_btn)

        remove_btn = QPushButton("Remove Selected Laser")
        remove_btn.clicked.connect(lambda: self._remove_selected())
        panelLayout.addWidget(remove_btn)

        # Separator under Remove Selected
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        panelLayout.addWidget(sep2)

        # Edit all selected
        edit_all_btn = QPushButton("Edit All Selected")
        edit_all_btn.clicked.connect(self._edit_all_selected)
        panelLayout.addWidget(edit_all_btn)

        # Separator under Edit All Selected
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setFrameShadow(QFrame.Sunken)
        panelLayout.addWidget(sep3)

        # Handedness help text (consistent with table label)
        handedness_label = QLabel(
            "Handedness: \n|  -1 → RH  |  0 → LIN  |  +1 → LH  |"
        )
        handedness_label.setWordWrap(True)
        panelLayout.addWidget(handedness_label)

        panelLayout.addStretch()
        panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Table with built-in scrollbars
        self.table = QTableWidget(0, 10)
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
            "Type",
            "Frequency (MHz)",
            "Detuning (Γ)",
            "Power (mW)",
            "Waist (mm)",
            "Origin",
            "Direction",
            "Handedness",
            "t_on (ms)",
            "t_off (ms)",
        ]
        self.table.setHorizontalHeaderLabels(headers)

        # Data column configuration; 'optional' columns map empty cell <-> null
        self.columnConfig = {
            1: {"key": "beam_frequency", "factor": 1, "format": "{:.3f}"},
            2: {"key": "detuning", "factor": 1, "format": "{:.1f}"},
            3: {"key": "beam_power", "factor": 1e3, "format": "{:.1f}"},
            4: {"key": "waist", "factor": 1e3, "format": "{:.1f}"},
            8: {"key": "t_on", "factor": 1, "format": "{:.2f}"},
            9: {
                "key": "t_off",
                "factor": 1,
                "format": "{:.2f}",
                "optional": True,
                "default": None,
            },
        }

        # Connect cell editing and selection
        self.table.cellClicked.connect(
            lambda r, c: setattr(self, "currentRow", r)
        )
        self.table.itemChanged.connect(self._on_item_changed)

    def setModel(self, model):
        self._model = model
        with signals_blocked(self.table):
            self.table.clearContents()
            laserList = self._model.get("Lasers", default=[]) or []
            self.table.setRowCount(len(laserList))
            for idx, cfg in enumerate(laserList):
                self._populate_row(idx, cfg)
            self.table.setVerticalHeaderLabels(
                [f"L{n}" for n in range(len(laserList))]
            )

    def _populate_row(self, row, cfg):
        # Type
        combo = QComboBox()
        combo.addItems(self.TYPE_OPTIONS)
        combo.setCurrentText(cfg.get("type", "unspecified"))
        combo.currentTextChanged.connect(
            lambda val, r=row: self._update_model(r, "type", val)
        )
        self.table.setCellWidget(row, 0, combo)
        # Numeric
        for col, info in self.columnConfig.items():
            raw = cfg.get(info["key"], info.get("default", 0.0))
            text = (
                ""
                if raw is None
                else info["format"].format(raw * info["factor"])
            )
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
        # Vectors
        for key, col in [("origin", 5), ("direction", 6)]:
            w = VectorInputWidget(cfg.get(key, [0, 0, 0]), self)
            w.vectorChanged.connect(
                lambda vec, r=row, k=key: self._update_model(r, k, vec)
            )
            self.table.setCellWidget(row, col, w)
        # Handedness
        heli = QComboBox()
        heli.addItems(self.HELICITY_OPTIONS)
        txt = f"{cfg.get('handedness', 1):+d}"
        idx = (
            self.HELICITY_OPTIONS.index(txt)
            if txt in self.HELICITY_OPTIONS
            else 1
        )
        heli.setCurrentIndex(idx)
        heli.currentTextChanged.connect(
            lambda v, r=row: self._update_model(r, "handedness", int(v))
        )
        self.table.setCellWidget(row, 7, heli)

    def _on_item_changed(self, item):
        if not self._model:
            return
        row, col = item.row(), item.column()
        if col in self.columnConfig:
            info = self.columnConfig[col]
            text = item.text().strip()
            if info.get("optional") and not text:
                val = None
            else:
                try:
                    val = float(text) / info["factor"]
                except ValueError:
                    QMessageBox.warning(
                        self, "Invalid", f'"{item.text()}" not a number'
                    )
                    return
            lst = list(self._model.get("Lasers", default=[]) or [])
            if row < len(lst):
                lst[row][info["key"]] = val
                self._model.set(lst, "Lasers")

    def _remove_selected(self):
        """Remove the currently highlighted dipole row."""
        if not self._model:
            return
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(
                self, "Remove Laser", "No row selected to remove."
            )
            return
        lst = list(self._model.get("Lasers", default=[]) or [])
        if 0 <= row < len(lst):
            lst.pop(row)
            self._model.set(lst, "Lasers")
            self.setModel(self._model)

    def _update_model(self, rowIndex, keyName, value):
        """Update a specific property on one laser and mark dirty"""
        if not self._model:
            return
        laserList = self._model.get("Lasers", default=[]) or []
        if 0 <= rowIndex < len(laserList):
            newList = list(laserList)
            newList[rowIndex] = dict(newList[rowIndex])
            newList[rowIndex][keyName] = value
            self._model.set(newList, "Lasers")

    def _edit_all_selected(self):
        self.popup = EditAllPopup(self)
        self.popup.show()
        pass

    # Placeholder slots
    def _edit_defaults(self):
        self.popup = EditDefaultsPopup(self)
        self.popup.show()
        pass

    def _add_new_laser(self, kind):
        """
        Add a new laser using defaults. Prefer JSON defaults files if present,
        otherwise fall back to model-provided defaults (Lasers_defaults).
        """
        if not self._model:
            return

        # Try to load defaults from GUI/defaults/lasers/<kind>_default.json
        file_defaults = {}
        try:
            # assumes this file lives in GUI/widgets/<thisfile>.py -> go up to GUI then defaults/lasers
            defaults_dir = (
                Path(__file__).resolve().parents[1] / "defaults" / "lasers"
            )
            fname = f"{kind}_default.json"
            p = defaults_dir / fname
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    file_defaults = json.load(f)
        except Exception:
            # ignore errors and fall back to model defaults
            file_defaults = {}

        # fallback to model-supplied defaults
        model_defaults = self._model.get("Lasers_defaults", {}) or {}

        # prefer file defaults if available, otherwise use model defaults
        defaults = file_defaults or model_defaults

        cfg = {**defaults, "type": kind or defaults.get("type", "unspecified")}
        lst = list(self._model.get("Lasers", default=[]) or [])
        lst.append(cfg)
        self._model.set(lst, "Lasers")
        self.setModel(self._model)


if __name__ == "__main__":
    pass

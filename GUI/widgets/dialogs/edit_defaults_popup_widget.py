import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EditDefaultsPopup(QWidget):
    """
    Popup for editing the numeric defaults for trap/repump lasers.

    Files (relative to the GUI package root):
      GUI/defaults/lasers/trap_default.json
      GUI/defaults/lasers/repump_default.json
    """

    FILE_MAP = {"Trap": "trap_default.json", "Repump": "repump_default.json"}

    # fields we display (key in JSON -> (label text, unit string))
    NUMERIC_FIELDS = [
        ("beam_frequency", "Frequency (Hz)"),
        ("detuning", "Detuning (Γ)"),
        ("beam_power", "Power (W)"),
        ("waist", "Waist (m)"),
    ]

    def __init__(self, parent=None):
        super().__init__(flags=Qt.Window)
        self.parent = parent
        self.setWindowTitle("Edit Laser Defaults")
        self.resize(480, 220)

        # locate defaults dir relative to this file: assume GUI/widgets/<thisfile>
        gui_root = (
            Path(__file__).resolve().parents[1]
        )  # adjust if your structure differs
        self.defaults_dir = gui_root / "defaults" / "lasers"

        self.current_name = None
        self.current_data = {}

        layout = QVBoxLayout(self)

        # selector (Trap / Repump)
        sel_h = QHBoxLayout()
        sel_h.addWidget(QLabel("Edit defaults for:"))
        self.selector = (
            QLineEdit()
        )  # hidden hack replaced below by combo for clarity
        from PyQt5.QtWidgets import QComboBox

        self.selector = QComboBox()
        self.selector.addItems(list(self.FILE_MAP.keys()))
        self.selector.currentTextChanged.connect(self._on_selector_changed)
        sel_h.addWidget(self.selector)
        sel_h.addStretch()
        layout.addLayout(sel_h)

        # fields area: each row will be (checkbox, label, lineedit)
        self.fields = {}  # key -> (checkbox, lineedit)
        for key, label_text in self.NUMERIC_FIELDS:
            row = QHBoxLayout()
            chk = QCheckBox()
            lbl = QLabel(f"{label_text}:")
            edit = QLineEdit()
            edit.setEnabled(False)
            edit.setPlaceholderText("value")
            chk.toggled.connect(edit.setEnabled)
            row.addWidget(chk)
            row.addWidget(lbl)
            row.addWidget(edit)
            row.addStretch()
            layout.addLayout(row)
            self.fields[key] = (chk, edit)

        # Save / Cancel buttons
        btn_h = QHBoxLayout()
        btn_h.addStretch()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.close)
        btn_h.addWidget(save_btn)
        btn_h.addWidget(cancel_btn)
        layout.addLayout(btn_h)

        # load initial selection
        self.selector.setCurrentIndex(0)
        self._on_selector_changed(self.selector.currentText())
        self.show()

    def _on_selector_changed(self, txt):
        """Load JSON for the selected defaults and populate fields."""
        filename = self.FILE_MAP.get(txt)
        if not filename:
            return
        file_path = self.defaults_dir / filename
        if not file_path.exists():
            QMessageBox.warning(
                self, "File missing", f"Defaults file not found:\n{file_path}"
            )
            # clear current_data and UI
            self.current_data = {}
            for chk, edit in self.fields.values():
                chk.setChecked(False)
                edit.clear()
                edit.setEnabled(False)
            return

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(
                self, "Read error", f"Could not read defaults:\n{e}"
            )
            return

        self.current_name = txt
        self.current_data = data

        # Populate the edits with values from file (but keep them disabled until checked)
        for key, (chk, edit) in self.fields.items():
            val = data.get(key, "")
            # prefer string representation; for floats use repr/str
            edit.setText("" if val is None else str(val))
            chk.setChecked(False)
            edit.setEnabled(False)

    def _save(self):
        """Write changed fields back to the JSON file. Unchecked fields are left as-is."""
        if not self.current_name:
            QMessageBox.warning(self, "No selection", "No defaults selected.")
            return
        filename = self.FILE_MAP[self.current_name]
        file_path = self.defaults_dir / filename

        # prepare a copy to mutate
        new_data = dict(self.current_data)

        changed = False
        for key, (chk, edit) in self.fields.items():
            if chk.isChecked():
                txt = edit.text().strip()
                if txt == "":
                    QMessageBox.warning(
                        self, "Empty value", f"Field {key} is empty."
                    )
                    return
                # try to parse number (float preferred). If int-like, keep as int.
                try:
                    if "." in txt or "e" in txt.lower():
                        val = float(txt)
                    else:
                        # try int then fallback to float
                        try:
                            val = int(txt)
                        except ValueError:
                            val = float(txt)
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid",
                        f"'{txt}' is not a number for '{key}'.",
                    )
                    return
                new_data[key] = val
                changed = True

        if not changed:
            QMessageBox.information(
                self, "No changes", "No fields were selected for update."
            )
            return

        # write file
        try:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(new_data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(
                self, "Write error", f"Could not write defaults:\n{e}"
            )
            return

        QMessageBox.information(
            self, "Saved", f"Defaults saved to {file_path}"
        )
        # Optionally tell parent to reload defaults if they care:
        if self.parent and hasattr(self.parent, "on_defaults_updated"):
            try:
                self.parent.on_defaults_updated(self.current_name, new_data)
            except Exception:
                pass
        self.close()

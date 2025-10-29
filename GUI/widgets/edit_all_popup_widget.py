from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QLineEdit, QPushButton, QMessageBox, QTableWidgetItem
)
from PyQt5.QtCore import Qt

class EditAllPopup(QWidget):
    """
    Popup for editing all selected lasers’ numeric properties at once,
    and writing them back into both the table AND the model.
    """
    def __init__(self, parent_tab):
        """
        parent_tab: instance of LasersSettingsTab
        """
        super().__init__(flags=Qt.Window)
        self.parent_tab = parent_tab
        self.table = parent_tab.table
        self.col_cfg = parent_tab.columnConfig  # { colIndex: { 'key':..., 'factor':..., … } }

        self.setWindowTitle("Edit All Selected Lasers")
        layout = QVBoxLayout(self)

        # get selected rows
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            QMessageBox.warning(self, "No Selection", "Please select at least one row first.")
            self.close()
            return

        first_row = sel[0].row()
        self.fields = {}  # map colIndex -> (checkbox, line_edit)

        # build a UI row for each numeric column in columnConfig
        for col, info in self.col_cfg.items():
            header = self.table.horizontalHeaderItem(col).text()
            h = QHBoxLayout()

            chk = QCheckBox()
            lbl = QLabel(header)
            # seed initial text from the first selected row
            existing_item = self.table.item(first_row, col)
            existing = existing_item.text() if existing_item else ""
            edit = QLineEdit(existing)
            edit.setEnabled(False)

            # only enable the QLineEdit when its checkbox is checked
            chk.toggled.connect(edit.setEnabled)

            h.addWidget(chk)
            h.addWidget(lbl)
            h.addWidget(edit)
            layout.addLayout(h)

            self.fields[col] = (chk, edit)

        # Apply / Cancel
        btn_h = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        apply_btn.clicked.connect(self.apply_changes)
        cancel_btn.clicked.connect(self.close)
        btn_h.addStretch()
        btn_h.addWidget(apply_btn)
        btn_h.addWidget(cancel_btn)
        layout.addLayout(btn_h)

        self.setLayout(layout)
        self.resize(450, 250)
        self.show()

    def apply_changes(self):
        # figure out which columns we're updating
        updates = {}
        for col, (chk, edit) in self.fields.items():
            if chk.isChecked():
                txt = edit.text().strip()
                try:
                    val = float(txt)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input",
                                        f"‘{txt}’ is not a number for column {col}.")
                    return
                updates[col] = val

        if not updates:
            QMessageBox.information(self, "Nothing to Do",
                                    "No fields were checked for updating.")
            return

        # gather all selected rows
        rows = [i.row() for i in self.table.selectionModel().selectedRows()]

        # for each row and each selected column:
        for row in rows:
            for col, new_val in updates.items():
                info = self.col_cfg[col]
                # convert back to the model-units
                model_val = new_val / info['factor']
                # 1) update the model
                self.parent_tab._update_model(row, info['key'], model_val)

                # 2) update the table cell text
                text = info['format'].format(new_val)
                item = self.table.item(row, col)
                if item:
                    item.setText(text)
                else:
                    new_item = QTableWidgetItem(text)
                    new_item.setFlags(new_item.flags() | Qt.ItemIsEditable)
                    self.table.setItem(row, col, new_item)

        self.close()
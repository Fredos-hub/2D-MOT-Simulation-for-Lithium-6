import json
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QPlainTextEdit, QPushButton, QLabel, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QTextCursor
from jsonschema import Draft7Validator


class ValidationDiffDialog(QDialog):
    """
    Shows schema validation errors for a FileModel alongside an editable JSON view.
    Clicking an error navigates to the relevant key in the editor.
    Saving writes back through the model (marks it clean).
    """

    def __init__(self, model, schema: dict, parent=None):
        super().__init__(parent)
        self.model = model
        self.schema = schema
        self.validator = Draft7Validator(schema)
        self.setWindowTitle(f"Validation — {model.filepath.name}")
        self.resize(980, 640)
        self._setup_ui()
        self._load_from_model()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.summaryLabel = QLabel("")
        self.summaryLabel.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.summaryLabel)

        splitter = QSplitter(Qt.Horizontal)

        # Left: error tree
        self.errorTree = QTreeWidget()
        self.errorTree.setHeaderLabels(["Path", "Issue", "Expected / Hint"])
        self.errorTree.setColumnWidth(0, 160)
        self.errorTree.setColumnWidth(1, 260)
        self.errorTree.header().setStretchLastSection(True)
        self.errorTree.itemClicked.connect(self._on_error_clicked)
        splitter.addWidget(self.errorTree)

        # Right: JSON editor + Format button
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        fmt_btn = QPushButton("Format JSON")
        fmt_btn.setMaximumWidth(130)
        fmt_btn.clicked.connect(self._format_json)
        right_layout.addWidget(fmt_btn, alignment=Qt.AlignLeft)
        self.jsonEditor = QPlainTextEdit()
        self.jsonEditor.setFont(QFont("Monospace", 10))
        right_layout.addWidget(self.jsonEditor)
        splitter.addWidget(right)

        splitter.setSizes([380, 600])
        layout.addWidget(splitter)

        # Buttons row
        btn_row = QHBoxLayout()
        self.saveBtn = QPushButton("Save to File")
        self.saveBtn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self.saveBtn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Debounced re-validation timer
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self._revalidate)
        self.jsonEditor.textChanged.connect(self._timer.start)

    # ------------------------------------------------------------------
    # Loading / validation
    # ------------------------------------------------------------------

    def _load_from_model(self):
        text = json.dumps(self.model._current, indent=2)
        self.jsonEditor.blockSignals(True)
        self.jsonEditor.setPlainText(text)
        self.jsonEditor.blockSignals(False)
        self._revalidate()

    def _revalidate(self):
        self.errorTree.clear()
        text = self.jsonEditor.toPlainText()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            self.summaryLabel.setText(f"⚠  JSON parse error: {e}")
            self.summaryLabel.setStyleSheet("color: red; font-weight: bold; padding: 4px;")
            item = QTreeWidgetItem(["(parse error)", str(e), ""])
            item.setForeground(1, QColor("red"))
            self.errorTree.addTopLevelItem(item)
            self.saveBtn.setEnabled(False)
            return

        self.saveBtn.setEnabled(True)
        errors = sorted(
            self.validator.iter_errors(data),
            key=lambda e: list(e.absolute_path)
        )

        if not errors:
            self.summaryLabel.setText("✓  No validation errors — file is valid")
            self.summaryLabel.setStyleSheet("color: green; font-weight: bold; padding: 4px;")
            item = QTreeWidgetItem(["", "✓  File matches schema", ""])
            item.setForeground(1, QColor("green"))
            self.errorTree.addTopLevelItem(item)
            return

        self.summaryLabel.setText(f"⚠  {len(errors)} validation error(s)")
        self.summaryLabel.setStyleSheet("color: red; font-weight: bold; padding: 4px;")

        for err in errors:
            path_str = " → ".join(str(p) for p in err.absolute_path) or "(root)"
            hint = self._build_hint(err)
            item = QTreeWidgetItem([path_str, err.message, hint])
            item.setForeground(0, QColor("#cc4400"))
            item.setForeground(1, QColor("#990000"))
            item.setData(0, Qt.UserRole, err)
            self.errorTree.addTopLevelItem(item)

    def _build_hint(self, err) -> str:
        if err.validator == "required":
            missing = err.message.split("'")[1] if "'" in err.message else ""
            default = self._schema_default_for(list(err.absolute_path) + [missing])
            if default is not None:
                return f'Add: "{missing}": {json.dumps(default)}'
            return f'Add missing field: "{missing}"'
        if err.validator == "type":
            expected = err.schema.get("type", "?")
            got = type(err.instance).__name__
            return f"Expected {expected}, got {got}"
        if err.validator == "enum":
            return f"One of: {err.schema.get('enum', [])}"
        if err.validator == "additionalProperties":
            return "Remove unknown property"
        if err.validator in ("minimum", "maximum"):
            bound = err.schema.get(err.validator)
            return f"{err.validator}: {bound}"
        return ""

    def _schema_default_for(self, path: list):
        """Walk schema properties to find the 'default' at the given path."""
        node = self.schema
        for key in path:
            if isinstance(node, dict):
                props = node.get("properties", {})
                if key in props:
                    node = props[key]
                else:
                    return None
            else:
                return None
        return node.get("default") if isinstance(node, dict) else None

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_error_clicked(self, item, _column):
        err = item.data(0, Qt.UserRole)
        if err is None:
            return
        path = list(err.absolute_path)
        # For 'required' errors the path points to the parent object; use the missing key
        if err.validator == "required" and "'" in err.message:
            key = err.message.split("'")[1]
        elif path:
            key = str(path[-1])
        else:
            return
        text = self.jsonEditor.toPlainText()
        search = f'"{key}"'
        idx = text.find(search)
        if idx >= 0:
            cursor = self.jsonEditor.textCursor()
            cursor.setPosition(idx)
            cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
            self.jsonEditor.setTextCursor(cursor)
            self.jsonEditor.setFocus()
            self.jsonEditor.ensureCursorVisible()

    def _format_json(self):
        text = self.jsonEditor.toPlainText()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        formatted = json.dumps(data, indent=2)
        pos = self.jsonEditor.textCursor().position()
        self.jsonEditor.blockSignals(True)
        self.jsonEditor.setPlainText(formatted)
        self.jsonEditor.blockSignals(False)
        cursor = self.jsonEditor.textCursor()
        cursor.setPosition(min(pos, len(formatted)))
        self.jsonEditor.setTextCursor(cursor)
        self._revalidate()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self):
        text = self.jsonEditor.toPlainText()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Invalid JSON", f"Cannot save invalid JSON:\n{e}")
            return
        try:
            self.model._current = data
            self.model.save()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{e}")
            return
        self.accept()

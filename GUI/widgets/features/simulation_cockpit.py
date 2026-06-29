import json
import os
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from PyQt5.QtCore import QRegularExpression, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

import src.batch_worker as batch_worker
from GUI.models.file_model import FileModel
from GUI.widgets.common.file_table import FileSimState, FileTableWidget
from GUI.widgets.tabs.settings_tabs import SettingsTabsWidget
from src import checkpoint
from src.batch_worker import BatchSimulationWorker

schema_version = 1

SCHEMA_PATH = os.path.join("GUI/schema", f"schema_v{str(schema_version)}.json")


class LogHighlighter(QSyntaxHighlighter):
    """
    Highlight log lines:
      - ERROR:  -> red + bold
      - WARNING: -> orange + bold
      - lines containing 'Building' -> green + bold + pale background (stands out)
    """

    def __init__(self, document) -> None:
        super().__init__(document)

        # error format (red + bold)
        self.error_format = QTextCharFormat()
        self.error_format.setForeground(QColor("red"))
        self.error_format.setFontWeight(QFont.Bold)

        # warning format (orange + bold)
        self.warning_format = QTextCharFormat()
        self.warning_format.setForeground(QColor("orange"))
        self.warning_format.setFontWeight(QFont.Bold)

        # building format (green + bold + pale background)
        self.building_format = QTextCharFormat()
        self.building_format.setForeground(QColor(0, 100, 0))  # dark green
        self.building_format.setFontWeight(QFont.Bold)
        # pale green background to make it stand out
        self.building_format.setBackground(QColor(225, 245, 225))

        # Regexes
        # whole-line ERROR/WARNING
        self.error_re = QRegularExpression(r"^(ERROR:.*)$")
        self.warning_re = QRegularExpression(r"^(WARNING:.*)$")
        # any line containing the word "Building" (case-sensitive to match your log)
        # matches strings like "---------------Building Tiecke_Setup.json (1/2)------------------"
        self.building_re = QRegularExpression(r".*\bBuilding\b.*")

    def highlightBlock(self, text: str) -> None:
        # If it's a building line, highlight it first (so it visibly stands out).
        # This also avoids the "ERROR" style overriding the building style if both occurred.
        itb = self.building_re.globalMatch(text)
        while itb.hasNext():
            m = itb.next()
            start = m.capturedStart(0)
            length = m.capturedLength(0)
            if start >= 0 and length > 0:
                self.setFormat(start, length, self.building_format)

        # apply error formatting
        it = self.error_re.globalMatch(text)
        while it.hasNext():
            m = it.next()
            start = m.capturedStart(1)
            length = m.capturedLength(1)
            if start >= 0 and length > 0:
                self.setFormat(start, length, self.error_format)

        # apply warning formatting
        it2 = self.warning_re.globalMatch(text)
        while it2.hasNext():
            m = it2.next()
            start = m.capturedStart(1)
            length = m.capturedLength(1)
            if start >= 0 and length > 0:
                self.setFormat(start, length, self.warning_format)


class AlignDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index) -> None:
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class SimulationCockpit(QWidget):
    fileDirtyChanged = pyqtSignal(bool)
    anyDirtyChanged = pyqtSignal(bool)
    simulationStateChanged = pyqtSignal(str)  # "idle" | "running" | "paused"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.opened_directory = None
        self.models = {}
        self.simulation_running_flag = False
        self.simulation_queue = []  # list of (row, filename)
        self.batch_worker = None
        self._schema_cache = None  # parsed JSON schema, loaded once on demand
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Middle: FileTable + Detail panel
        interaction_layout = QHBoxLayout()

        # Left side
        left_side = QGroupBox()
        left_layout = QVBoxLayout()
        left_side.setLayout(left_layout)
        self.selectedDirLabel = QLabel("Selected Directory: None")
        left_layout.addWidget(self.selectedDirLabel)
        self.fileTable = FileTableWidget(self)
        self.fileTable.setMaximumWidth(900)
        delegate = AlignDelegate(self.fileTable.table)
        self.fileTable.table.setItemDelegateForColumn(2, delegate)
        left_layout.addWidget(self.fileTable)
        interaction_layout.addWidget(left_side)

        # Right side: Settings
        right_side = QGroupBox()
        right_layout = QVBoxLayout()
        right_side.setLayout(right_layout)
        self.settingsLabel = QLabel("Settings")
        right_layout.addWidget(self.settingsLabel)
        self.settings_tabs = SettingsTabsWidget(self)
        self.settings_tabs.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )
        right_layout.addWidget(self.settings_tabs)
        interaction_layout.addWidget(right_side)

        interaction_layout.setAlignment(Qt.AlignTop)
        interaction_layout.setStretch(0, 2)
        interaction_layout.setStretch(1, 3)
        main_layout.addLayout(interaction_layout)

        # Logging & Progress
        self.loggingField = QPlainTextEdit(readOnly=True)
        self.loggingField.setPlaceholderText("Logging output...")
        main_layout.addWidget(self.loggingField)
        self.progressBar = QProgressBar(value=0)
        main_layout.addWidget(self.progressBar)
        self.statusLabel = QLabel("Status: Not started")
        main_layout.addWidget(self.statusLabel)

        # attach syntax highlighter to colour ERROR: / WARNING:
        self.log_highlighter = LogHighlighter(self.loggingField.document())

        # File-table signals
        self.fileTable.fileSelected.connect(self._on_file_selected)
        self.fileTable.fileRenamed.connect(
            lambda original, new: self.loggingField.appendPlainText(
                f"Renamed {original}→{new}"
            )
        )
        self.fileTable.fileDeleted.connect(
            lambda filename: self.loggingField.appendPlainText(
                f"Deleted {filename}"
            )
        )
        self.fileTable.fileIgnored.connect(
            lambda filename, ign: self.loggingField.appendPlainText(
                f"{'Ignored' if ign else 'Unignored'} {filename}"
            )
        )
        self.fileTable.fileCopied.connect(self._on_file_copied)
        self.fileTable.openDiffRequested.connect(self._open_validation_dialog)

        self.setWindowTitle("Simulation Cockpit")

    def _on_file_selected(self, filename) -> None:
        self.settingsLabel.setText(f"Settings for: {filename}")
        model = self.models.get(filename)
        if model:
            self.settings_tabs.setModel(model)
        self.fileDirtyChanged.emit(model.is_dirty())
        self._emit_any_dirty()

    def is_simulation_running(self) -> bool:
        return self.batch_worker is not None and self.batch_worker.isRunning()

    def has_unsaved_changes(self) -> bool:
        return any(model.is_dirty() for model in self.models.values())

    def stop_simulation_and_wait(self) -> None:
        """Stop the batch worker and block until its thread has finished."""
        if self.is_simulation_running():
            self.batch_worker.stop()
            self.batch_worker.wait()

    def _validate_model(self, model) -> list:
        """Return list of jsonschema ValidationErrors for the model's current data."""
        schema = self._load_schema()
        if schema is None:
            return None
        validator = Draft7Validator(schema)
        return list(validator.iter_errors(model.data()))

    def _load_schema(self) -> dict | None:
        """Load and cache the JSON schema; read from disk only once."""
        if self._schema_cache is None:
            try:
                with open(SCHEMA_PATH, encoding="utf-8") as f:
                    self._schema_cache = json.load(f)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Schema Load Error",
                    f"Could not load JSON schema:\n{e}",
                )
                return None
        return self._schema_cache

    def _register_model(self, name, model) -> None:
        """Wire a FileModel's dirty signal into the table + cockpit, baseline it clean, and store it."""
        model.dirtyChanged.connect(
            lambda dirty, filename=name: self.fileTable.update_status(
                filename, dirty
            )
        )
        model.dirtyChanged.connect(
            lambda dirty, filename=name: self._on_model_dirty(filename, dirty)
        )
        model.dirtyChanged.connect(self.fileDirtyChanged)
        model.mark_clean()
        self.models[name] = model

    def _refresh_file_display(self, name, model) -> None:
        """Push a model's current dirty + validation state into the table row."""
        self.fileTable.update_status(name, model.is_dirty())
        errors = self._validate_model(model)
        if errors is not None:
            self.fileTable.set_validation_status(name, errors)

    def open_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select directory containing JSON files"
        )
        if not directory:
            return
        self.opened_directory = directory
        self.selectedDirLabel.setText(f"Selected Directory: {directory}")

        self.models.clear()
        for file in sorted(Path(directory).glob("*.json")):
            self._register_model(file.name, FileModel(str(file)))

        self.fileTable.load_directory(directory)
        for name, model in self.models.items():
            self._refresh_file_display(name, model)
        self._emit_any_dirty()

    def _generate_skeleton_from_schema(
        self,
        subschema: dict,
        schema_root: dict = None,
        external_defaults: dict = None,
        path: list | None = None,
        _depth: int = 0,
    ) -> Any:
        """
        Generate a skeleton value for the provided subschema.

        Priority (when choosing a value):
        1. `const` (if present)
        2. schema `default` (this function now PREFERS schema defaults)
        3. `enum` (first entry)
        4. `examples` (first entry)
        5. external_defaults (looked up by path), only if none of the above present
        6. type-based sensible fallback (object -> {}, array -> [], string -> "", integer -> minimum or 0, number -> minimum or 0.0, boolean -> False)

        Parameters
        ----------
        - subschema: the (sub-)schema to produce a value for
        - schema_root: full schema (used to resolve internal $ref). If None, subschema is treated as root.
        - external_defaults: nested dict of external defaults (optional). Will be consulted *after* schema defaults.
        - path: list of property names representing the location in the final document (used for external_defaults lookup)
        - _depth: recursion depth guard
        """
        # protect against runaway recursion
        if _depth > 40:
            return None

        if schema_root is None:
            schema_root = subschema

        if path is None:
            path = []

        # helper: resolve internal $ref (only support '#/...' style)
        def _resolve_ref(ref: str) -> Any:
            if not isinstance(ref, str) or not ref.startswith("#/"):
                return None
            parts = ref.lstrip("#/").split("/")
            node = schema_root
            for p in parts:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    return None
            return node

        # helper: lookup external defaults using the path; try exact path then nearest ancestor
        def _external_lookup(ex_defaults, pth) -> Any:
            if not ex_defaults:
                return None
            # try exact
            node = ex_defaults
            try:
                for k in pth:
                    node = node[k]
                return node
            except Exception:
                pass
            # fallback to nearest ancestor (Atoms -> top-level Atoms default)
            for i in range(len(pth) - 1, 0, -1):
                node = ex_defaults
                ok = True
                for k in pth[:i]:
                    if isinstance(node, dict) and k in node:
                        node = node[k]
                    else:
                        ok = False
                        break
                if ok:
                    return node
            return None

        # If there's a $ref: resolve and recurse (gives priority to referenced schema)
        if "$ref" in subschema:
            resolved = _resolve_ref(subschema["$ref"])
            if resolved is not None:
                return self._generate_skeleton_from_schema(
                    resolved,
                    schema_root=schema_root,
                    external_defaults=external_defaults,
                    path=path,
                    _depth=_depth + 1,
                )
            # unresolved $ref -> fallback to None
            return None

        # 1) const always wins
        if "const" in subschema:
            return subschema["const"]

        # 2) prefer schema default (the change you asked for)
        if "default" in subschema:
            return subschema["default"]

        # 3) enum -> pick first entry
        if (
            "enum" in subschema
            and isinstance(subschema["enum"], list)
            and subschema["enum"]
        ):
            return subschema["enum"][0]

        # 4) try examples
        if (
            "examples" in subschema
            and isinstance(subschema["examples"], list)
            and subschema["examples"]
        ):
            return subschema["examples"][0]

        # 5) handle oneOf / anyOf: prefer a branch with a default, else pick first branch
        for comb in ("oneOf", "anyOf"):
            if (
                comb in subschema
                and isinstance(subschema[comb], list)
                and subschema[comb]
            ):
                # try to find first branch which has a default or const
                for branch in subschema[comb]:
                    # Resolve $ref in branch if present for detection
                    b = branch
                    if isinstance(branch, dict) and "$ref" in branch:
                        resolved_b = _resolve_ref(branch["$ref"])
                        if resolved_b is not None:
                            b = resolved_b
                    if isinstance(b, dict) and (
                        "default" in b or "const" in b
                    ):
                        return self._generate_skeleton_from_schema(
                            b, schema_root, external_defaults, path, _depth + 1
                        )
                # else fallback to first branch
                return self._generate_skeleton_from_schema(
                    subschema[comb][0],
                    schema_root,
                    external_defaults,
                    path,
                    _depth + 1,
                )

        # If we reach here, no schema default/const/enum/examples were present.
        # Consult external defaults (if provided) BEFORE doing type fallbacks.
        ext = _external_lookup(external_defaults, path)
        if ext is not None:
            return ext

        # Determine type (could be list) or infer from properties/items
        t = subschema.get("type")
        if isinstance(t, list) and t:
            t = t[0]
        if t is None:
            if "properties" in subschema:
                t = "object"
            elif "items" in subschema:
                t = "array"

        # Object handling
        if t == "object":
            result = {}
            props = subschema.get("properties", {})
            required_props = set(subschema.get("required", []))
            # If properties present, recuse for each property
            for prop_name, prop_schema in props.items():
                result[prop_name] = self._generate_skeleton_from_schema(
                    prop_schema,
                    schema_root=schema_root,
                    external_defaults=external_defaults,
                    path=path + [prop_name],
                    _depth=_depth + 1,
                )
            # ensure required props exist (even if not in properties)
            for req in required_props:
                if req not in result:
                    result[req] = None
            return result

        # Array handling
        if t == "array":
            items_schema = subschema.get("items")
            min_items = int(subschema.get("minItems", 0))
            # Decide how many to create:
            # - If minItems > 0: create that many
            # - If minItems == 0 but items_schema has a default/object/array -> create 1 element for UX
            create_count = min_items
            if create_count == 0 and items_schema:
                # create 1 if item is object/array or has a default/const/enum/examples
                if isinstance(items_schema, dict) and (
                    "default" in items_schema
                    or "const" in items_schema
                    or "enum" in items_schema
                    or items_schema.get("type") in ("object", "array")
                ):
                    create_count = 1
            if create_count == 0:
                return []

            if isinstance(items_schema, dict):
                arr = []
                for i in range(create_count):
                    arr.append(
                        self._generate_skeleton_from_schema(
                            items_schema,
                            schema_root=schema_root,
                            external_defaults=external_defaults,
                            path=path + [str(i)],
                            _depth=_depth + 1,
                        )
                    )
                return arr
            elif isinstance(items_schema, list):
                # tuple-style items (each position may have its own schema)
                arr = []
                for idx, itschema in enumerate(items_schema[:create_count]):
                    arr.append(
                        self._generate_skeleton_from_schema(
                            itschema,
                            schema_root=schema_root,
                            external_defaults=external_defaults,
                            path=path + [str(idx)],
                            _depth=_depth + 1,
                        )
                    )
                return arr
            else:
                return []

        # Primitives
        if t == "string":
            return ""

        if t == "integer":
            if "minimum" in subschema:
                try:
                    return int(subschema["minimum"])
                except Exception:
                    pass
            return 0

        if t == "number":
            if "minimum" in subschema:
                try:
                    return float(subschema["minimum"])
                except Exception:
                    pass
            return 0.0

        if t == "boolean":
            return False

        # fallback
        return None

    def create_new_file(self) -> None:

        self._check_unsaved()

        # 1) Ensure a directory is set
        if not self.opened_directory:
            dir_choice = QFileDialog.getExistingDirectory(
                self, "Select directory to create new file"
            )
            if not dir_choice:
                return
            self.opened_directory = dir_choice
            self.selectedDirLabel.setText(f"Selected Directory: {dir_choice}")

        # 2) Prompt for file_name
        filename, ok = QInputDialog.getText(
            self, "New JSON file_name", "Enter new file_name (without .json):"
        )
        if not (ok and filename.strip()):
            return
        filename = filename.strip()
        if not filename.lower().endswith(".json"):
            filename += ".json"
        new_path = os.path.join(self.opened_directory, filename)

        # 3) Overwrite check
        if os.path.exists(new_path):
            resp = QMessageBox.question(
                self,
                "Overwrite?",
                f"A file named '{filename}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return

        # 4) Load schema & generate skeleton
        schema = self._load_schema()
        if schema is None:
            return
        skeleton = self._generate_skeleton_from_schema(schema)

        # 5) Write the new file
        try:
            with open(new_path, "w") as f:
                json.dump(skeleton, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write file:\n{e}")
            return

        # 6a) Create FileModel for the newly created file and register signals
        new_name = os.path.basename(new_path)
        model = FileModel(new_path)
        self._register_model(new_name, model)

        # 6b) Log and refresh UI
        self.loggingField.appendPlainText(f"Created: {new_path}")
        self.fileTable.refresh_table()
        self._refresh_file_display(new_name, model)

    def save_file(self) -> None:
        row = self.fileTable.table.currentRow()
        if row < 0:
            return
        filename = self.fileTable.table.item(row, 0).text()
        model = self.models.get(filename)
        if model and model.is_dirty():
            model.save()
            self.loggingField.appendPlainText(f"Saved {filename}")
            # re‐apply the clean highlight for this one row
            self.fileTable.update_status(filename, model.is_dirty())
        else:
            self.loggingField.appendPlainText(
                f"No changes to save for {filename}"
            )

    def save_all(self) -> None:
        count = 0
        for _filename, model in self.models.items():
            if model.is_dirty():
                model.save()
                count += 1
        self.loggingField.appendPlainText(f"Saved {count} file(s)")
        # re‐apply all statuses so the visuals match model states
        for filename, model in self.models.items():
            self.fileTable.update_status(filename, model.is_dirty())

    def discard_changes(self) -> None:
        row = self.fileTable.table.currentRow()
        if row < 0:
            return
        filename = self.fileTable.table.item(row, 0).text()
        model = self.models.get(filename)
        if model and model.is_dirty():
            model.reset()
            self.loggingField.appendPlainText(
                f"Discarded changes to {filename}"
            )
            # Reload the settings panel so it shows the restored values
            current_label = self.settingsLabel.text().replace(
                "Settings for: ", ""
            )
            if filename == current_label:
                self.settings_tabs.setModel(model)
            tbl = self.fileTable.table
            tbl.selectionModel().clearSelection()
            tbl.selectRow(row)

    def discard_all_changes(self) -> None:
        current_label = self.settingsLabel.text().replace("Settings for: ", "")
        for filename, model in self.models.items():
            if model.is_dirty():
                model.reset()
                self.loggingField.appendPlainText(
                    f"Discarded changes to {filename}"
                )
                if filename == current_label:
                    self.settings_tabs.setModel(model)
        self.fileTable.table.selectionModel().clearSelection()

    def run_simulation_from_file_table(self) -> None:
        if self.simulation_running_flag:
            QMessageBox.warning(
                self, "Simulation", "Simulation already running."
            )
            return

        self._check_unsaved()

        # Build queue of filenames
        tbl = self.fileTable.table
        self.simulation_queue = [
            tbl.item(r, 0).text()
            for r in range(tbl.rowCount())
            if tbl.item(r, 1).checkState() != Qt.Checked
        ]
        if not self.simulation_queue:
            return

        # Mark all queued files as pending (first will flip to simulating via fileStarted)
        for name in self.simulation_queue:
            self.fileTable.set_simulation_status(name, FileSimState.PENDING)

        # Start batch worker
        self.batch_worker = BatchSimulationWorker(
            self.opened_directory, self.simulation_queue
        )
        self.batch_worker.progressChanged.connect(self.progressBar.setValue)
        self.batch_worker.statusChanged.connect(self._on_status_update)
        self.batch_worker.fileStarted.connect(self._on_file_started)
        self.batch_worker.fileFinished.connect(self._on_file_finished)
        self.batch_worker.finished.connect(self._on_all_finished)
        self.simulation_running_flag = True
        self.simulationStateChanged.emit("running")
        self.batch_worker.start()

    def _on_file_started(self, filename: str, total_steps: int) -> None:
        self.fileTable.set_simulation_status(filename, FileSimState.SIMULATING)

    def _on_file_finished(self, filename: str) -> None:
        self.fileTable.set_simulation_status(filename, FileSimState.DONE)

    def _on_all_finished(self) -> None:
        for name in list(self.models.keys()):
            self.fileTable.set_simulation_status(name, None)
        self.progressBar.setValue(0)
        self.simulation_running_flag = False
        if self.batch_worker:
            self.batch_worker.deleteLater()
            self.batch_worker = None
        self.simulationStateChanged.emit("idle")

    def startCompilationAnimation(self) -> None:
        # If an animation timer already exists, try to stop and delete it.
        if hasattr(self, "compilingTimer") and self.compilingTimer is not None:
            try:
                self.compilingTimer.stop()
                self.compilingTimer.deleteLater()
            except RuntimeError:
                # The timer may already have been deleted.
                pass
        # Create a new timer.
        self.compilingTimer = QTimer(self)
        self.compilingTimer.setInterval(500)  # Update every 500ms
        self.compilingAnimationStep = 0
        self.compilingTimer.timeout.connect(self._update_compiling_status)
        self.compilingTimer.start()
        self.isCompilingAnimationActive = True

    def _update_compiling_status(self) -> None:
        # Only update if the animation is active.
        if self.isCompilingAnimationActive:
            dots = "." * ((self.compilingAnimationStep % 3) + 1)
            self.statusLabel.setText(
                f"Compiling{dots} (this may take a couple of minutes)"
            )
            self.compilingAnimationStep += 1

    def stopCompilationAnimation(self) -> None:
        self.isCompilingAnimationActive = False
        if hasattr(self, "compilingTimer") and self.compilingTimer is not None:
            try:
                self.compilingTimer.stop()
                self.compilingTimer.deleteLater()
            except RuntimeError:
                # The timer was already deleted.
                pass
            self.compilingTimer = None

    def _on_status_update(self, status) -> None:
        # Start/stop compilation animation
        if status == "Simulation instance created":
            self.startCompilationAnimation()
        if status == "Starting simulation...":
            self.stopCompilationAnimation()

        # Update status label
        self.statusLabel.setText(status)

        # If this is a "building" line, insert a separator for visibility
        # and always append it to the log (even if it's a short "Processing step" line).
        is_building = "Building" in status

        if is_building:
            # blank line for separation (so each build stands out)
            self.loggingField.appendPlainText("")

        # keep skipping frequent "Processing step" lines
        if not status.startswith("Processing step"):
            self.loggingField.appendPlainText(status)

        # auto-scroll to bottom for visibility
        v = self.loggingField.verticalScrollBar()
        v.setValue(v.maximum())

    def logMessage(self, message: str) -> None:
        """Append a message to the log box and optionally print to console."""
        if hasattr(self, "loggingField") and self.loggingField is not None:
            self.loggingField.appendPlainText(message)
        print(message)

    def _check_unsaved(self) -> None:
        # Save or discard unsaved changes
        dirty = [n for n, m in self.models.items() if m.is_dirty()]
        if dirty:
            resp = QMessageBox.question(
                self,
                f"Unsaved Changes in {len(dirty)} files.",
                "Save all changes?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if resp == QMessageBox.Cancel:
                return
            if resp == QMessageBox.Save:
                self.save_all()

    def _on_model_dirty(self, filename, dirty) -> None:
        """
        Called when *any* model becomes dirty/clean.
        We always need to:
         - re‐emit anyDirtyChanged
         - if the model that changed is currently selected, also re‐emit fileDirtyChanged
        """
        # 1) any‐model
        self._emit_any_dirty()

        # 2) selected‐model (only if that file is the one loaded in the tabs)
        current_label = self.settingsLabel.text().replace("Settings for: ", "")
        if filename == current_label:
            self.fileDirtyChanged.emit(dirty)

    def _emit_any_dirty(self) -> None:
        """Helper to recompute & emit whether at least one model is dirty."""
        any_dirty = any(m.is_dirty() for m in self.models.values())
        self.anyDirtyChanged.emit(any_dirty)

    def _open_validation_dialog(self, filename: str) -> None:
        model = self.models.get(filename)
        if model is None:
            return
        schema = self._load_schema()
        if schema is None:
            return
        from GUI.widgets.dialogs.validation_dialog import ValidationDiffDialog

        dlg = ValidationDiffDialog(model, schema, parent=self)
        dlg.exec_()
        # Re-validate and update status after dialog closes (user may have saved changes)
        self._refresh_file_display(filename, model)

    def resume_from_checkpoint(self) -> None:
        """Continue an interrupted run from its latest checkpoint (distinct from unpause)."""
        if self.simulation_running_flag:
            QMessageBox.warning(
                self, "Simulation", "Simulation already running."
            )
            return

        checkpoint_dir = self._find_latest_resumable_checkpoint()
        if checkpoint_dir is None:
            QMessageBox.information(
                self, "Resume run", "No resumable checkpoint found."
            )
            return

        # Empty file list is intentional: the worker repopulates directory/file_names from the
        # checkpoint meta for whole-batch resume (D-07).
        self.batch_worker = BatchSimulationWorker(
            self.opened_directory, [], resume_checkpoint_dir=checkpoint_dir
        )
        self.batch_worker.progressChanged.connect(self.progressBar.setValue)
        self.batch_worker.statusChanged.connect(self._on_status_update)
        self.batch_worker.fileStarted.connect(self._on_file_started)
        self.batch_worker.fileFinished.connect(self._on_file_finished)
        self.batch_worker.finished.connect(self._on_all_finished)
        self.simulation_running_flag = True
        self.simulationStateChanged.emit("running")
        self.batch_worker.start()

    def _find_latest_resumable_checkpoint(self) -> str | None:
        """Scan simulation_results batch folders NEWEST-FIRST and return the first dir with a
        resumable checkpoint, or None. A later CLEAN batch deletes its own checkpoint (D-08),
        so do not stop at the newest folder if it has none.
        """
        results_root = os.path.join(
            batch_worker.REPO_ROOT, "simulation_results"
        )
        if not os.path.isdir(results_root):
            return None
        entries = []
        for name in os.listdir(results_root):
            full = os.path.join(results_root, name)
            if not os.path.isdir(full):
                continue
            m = re.match(
                r"^(\d{2})_(\d{2})_(\d{2})_(\d+)$", name
            )  # DD_MM_YY_N
            if m:
                dd, mm, yy, n = (int(g) for g in m.groups())
                key = (yy, mm, dd, n)
            else:
                key = (-1, -1, -1, os.path.getmtime(full))
            entries.append((key, full))
        entries.sort(reverse=True)  # newest first
        for _, folder in entries:
            ckpt = checkpoint.find_resumable_checkpoint(folder)
            if ckpt is not None:
                return ckpt
        return None

    def has_resumable_checkpoint(self) -> bool:
        """True if a resumable checkpoint exists (used to gate the Resume-run action)."""
        return self._find_latest_resumable_checkpoint() is not None

    def pause_simulation(self) -> None:
        if self.batch_worker:
            self.batch_worker.pause()
            self.simulationStateChanged.emit("paused")

    def resume_simulation(self) -> None:
        if self.batch_worker:
            self.batch_worker.resume()
            self.simulationStateChanged.emit("running")

    def cancel_simulation(self) -> None:
        if not self.batch_worker:
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Cancel Simulation")
        msg.setText("What would you like to cancel?")
        msg.setIcon(QMessageBox.Question)
        msg.setInformativeText(
            "Cancel current: stop the running file — partial results are saved and the run stays "
            "resumable from its last checkpoint — then continue with any remaining queued files.\n"
            "Cancel all: stop everything, including remaining queued files.\n"
            "Continue: keep running."
        )
        btn_current = msg.addButton(
            "Cancel current", QMessageBox.DestructiveRole
        )
        btn_all = msg.addButton("Cancel all", QMessageBox.DestructiveRole)
        btn_keep = msg.addButton("Continue", QMessageBox.RejectRole)
        msg.setDefaultButton(btn_keep)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked == btn_current:
            self.batch_worker.stop_current()
            # resume if paused so the cleanup routine can run
            self.simulationStateChanged.emit("running")
        elif clicked == btn_all:
            self.batch_worker.stop()
            # _on_all_finished will emit "idle" once the thread finishes

    def _on_file_copied(self, original_name: str, copy_name: str) -> None:
        """After FileTable actually writes the new JSON copy on disk,
        create a FileModel for it and register all the same signals.
        """
        full_path = os.path.join(self.opened_directory, copy_name)
        model = FileModel(full_path)
        self._register_model(copy_name, model)
        self.fileTable.refresh_table()
        self._refresh_file_display(copy_name, model)

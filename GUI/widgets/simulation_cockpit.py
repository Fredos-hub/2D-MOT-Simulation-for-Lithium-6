import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPlainTextEdit, QProgressBar,QGroupBox, 
    QFileDialog, QInputDialog, QMessageBox,QSizePolicy, QStyledItemDelegate
)
from PyQt5.QtCore import Qt, pyqtSignal,QTimer

from GUI.widgets.file_table import FileTableWidget
from GUI.widgets.settings_tabs import SettingsTabsWidget
from GUI.file_model import FileModel
from src.batch_worker import BatchSimulationWorker
from jsonschema import Draft7Validator

schema_version = 1

SCHEMA_PATH = os.path.join('GUI/schema', f'schema_v{str(schema_version)}.json')



class AlignDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment =  Qt.AlignCenter


class SimulationCockpit(QWidget):
    fileDirtyChanged = pyqtSignal(bool)
    anyDirtyChanged  = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.opened_directory = None
        self.models = {}
        self.simulation_running_flag = False
        self.simulation_queue = []  # list of (row, filename)
        self.batch_worker = None
        self._init_ui()

    def _init_ui(self):
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
        self.settings_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
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

        # File-table signals
        self.fileTable.fileSelected.connect(self._on_file_selected)
        self.fileTable.fileRenamed.connect(lambda original,new: self.loggingField.appendPlainText(f"Renamed {original}→{new}"))
        self.fileTable.fileDeleted.connect(lambda filename: self.loggingField.appendPlainText(f"Deleted {filename}"))
        self.fileTable.fileIgnored.connect(lambda filename,ign: self.loggingField.appendPlainText(f"{'Ignored' if ign else 'Unignored'} {filename}"))
        self.fileTable.fileCopied.connect(self._on_file_copied)

        self.setWindowTitle("Simulation Cockpit")

    def _on_file_selected(self, filename):
        self.settingsLabel.setText(f"Settings for: {filename}")
        model = self.models.get(filename)
        if model:
            self.settings_tabs.setModel(model)
        # 1) selected‐model dirty
        self.fileDirtyChanged.emit(model.is_dirty())

        # 2) any‐model dirty
        self._emit_any_dirty()

        # load into settings panel
        self.settings_tabs.setModel(model)

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select directory containing JSON files")
        if not directory:
            return
        self.opened_directory = directory
        self.selectedDirLabel.setText(f"Selected Directory: {directory}")

        self.models.clear()
        for file in sorted(Path(directory).glob("*.json")):
            name = file.name
            model = FileModel(str(file))
            model.dirtyChanged.connect(lambda dirty,filename=name: self.fileTable.updateStatus(filename, dirty))
            model.dirtyChanged.connect(lambda dirty, filename=name: self._on_model_dirty(filename, dirty))
            model.mark_clean()
            model.dirtyChanged.connect(self.fileDirtyChanged)
            self.models[name] = model

        self.fileTable.load_directory(directory)
        for name, m in self.models.items():
            self.fileTable.updateStatus(name, m.is_dirty())
        self._emit_any_dirty()

    def _generate_skeleton_from_schema(self,
                                    subschema: dict,
                                    schema_root: dict = None,
                                    external_defaults: dict = None,
                                    path: list | None = None,
                                    _depth: int = 0):
        """
        Generate a skeleton value for the provided subschema.

        Priority (when choosing a value):
        1. `const` (if present)
        2. schema `default` (this function now PREFERS schema defaults)
        3. `enum` (first entry)
        4. `examples` (first entry)
        5. external_defaults (looked up by path), only if none of the above present
        6. type-based sensible fallback (object -> {}, array -> [], string -> "", integer -> minimum or 0, number -> minimum or 0.0, boolean -> False)

        Parameters:
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
        def _resolve_ref(ref: str):
            if not isinstance(ref, str) or not ref.startswith('#/'):
                return None
            parts = ref.lstrip('#/').split('/')
            node = schema_root
            for p in parts:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    return None
            return node

        # helper: lookup external defaults using the path; try exact path then nearest ancestor
        def _external_lookup(ex_defaults, pth):
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
        if '$ref' in subschema:
            resolved = _resolve_ref(subschema['$ref'])
            if resolved is not None:
                return self._generate_skeleton_from_schema(
                    resolved, schema_root=schema_root,
                    external_defaults=external_defaults,
                    path=path, _depth=_depth + 1
                )
            # unresolved $ref -> fallback to None
            return None

        # 1) const always wins
        if 'const' in subschema:
            return subschema['const']

        # 2) prefer schema default (the change you asked for)
        if 'default' in subschema:
            return subschema['default']

        # 3) enum -> pick first entry
        if 'enum' in subschema and isinstance(subschema['enum'], list) and subschema['enum']:
            return subschema['enum'][0]

        # 4) try examples
        if 'examples' in subschema and isinstance(subschema['examples'], list) and subschema['examples']:
            return subschema['examples'][0]

        # 5) handle oneOf / anyOf: prefer a branch with a default, else pick first branch
        for comb in ('oneOf', 'anyOf'):
            if comb in subschema and isinstance(subschema[comb], list) and subschema[comb]:
                # try to find first branch which has a default or const
                for branch in subschema[comb]:
                    # Resolve $ref in branch if present for detection
                    b = branch
                    if isinstance(branch, dict) and '$ref' in branch:
                        resolved_b = _resolve_ref(branch['$ref'])
                        if resolved_b is not None:
                            b = resolved_b
                    if isinstance(b, dict) and ('default' in b or 'const' in b):
                        return self._generate_skeleton_from_schema(b, schema_root, external_defaults, path, _depth+1)
                # else fallback to first branch
                return self._generate_skeleton_from_schema(subschema[comb][0], schema_root, external_defaults, path, _depth+1)

        # If we reach here, no schema default/const/enum/examples were present.
        # Consult external defaults (if provided) BEFORE doing type fallbacks.
        ext = _external_lookup(external_defaults, path)
        if ext is not None:
            return ext

        # Determine type (could be list) or infer from properties/items
        t = subschema.get('type')
        if isinstance(t, list) and t:
            t = t[0]
        if t is None:
            if 'properties' in subschema:
                t = 'object'
            elif 'items' in subschema:
                t = 'array'

        # Object handling
        if t == 'object':
            result = {}
            props = subschema.get('properties', {})
            required_props = set(subschema.get('required', []))
            # If properties present, recuse for each property
            for prop_name, prop_schema in props.items():
                result[prop_name] = self._generate_skeleton_from_schema(
                    prop_schema,
                    schema_root=schema_root,
                    external_defaults=external_defaults,
                    path=path + [prop_name],
                    _depth=_depth + 1
                )
            # ensure required props exist (even if not in properties)
            for req in required_props:
                if req not in result:
                    result[req] = None
            return result

        # Array handling
        if t == 'array':
            items_schema = subschema.get('items')
            min_items = int(subschema.get('minItems', 0))
            # Decide how many to create:
            # - If minItems > 0: create that many
            # - If minItems == 0 but items_schema has a default/object/array -> create 1 element for UX
            create_count = min_items
            if create_count == 0 and items_schema:
                # create 1 if item is object/array or has a default/const/enum/examples
                if (isinstance(items_schema, dict) and (
                        'default' in items_schema or 'const' in items_schema or 'enum' in items_schema or
                        items_schema.get('type') in ('object', 'array'))):
                    create_count = 1
            if create_count == 0:
                return []

            if isinstance(items_schema, dict):
                arr = []
                for i in range(create_count):
                    arr.append(self._generate_skeleton_from_schema(
                        items_schema,
                        schema_root=schema_root,
                        external_defaults=external_defaults,
                        path=path + [str(i)],
                        _depth=_depth + 1
                    ))
                return arr
            elif isinstance(items_schema, list):
                # tuple-style items (each position may have its own schema)
                arr = []
                for idx, itschema in enumerate(items_schema[:create_count]):
                    arr.append(self._generate_skeleton_from_schema(
                        itschema,
                        schema_root=schema_root,
                        external_defaults=external_defaults,
                        path=path + [str(idx)],
                        _depth=_depth + 1
                    ))
                return arr
            else:
                return []

        # Primitives
        if t == 'string':
            return ""

        if t == 'integer':
            if 'minimum' in subschema:
                try:
                    return int(subschema['minimum'])
                except Exception:
                    pass
            return 0

        if t == 'number':
            if 'minimum' in subschema:
                try:
                    return float(subschema['minimum'])
                except Exception:
                    pass
            return 0.0

        if t == 'boolean':
            return False

        # fallback
        return None


    def create_new_file(self):
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
        if not filename.lower().endswith('.json'):
            filename += '.json'
        new_path = os.path.join(self.opened_directory, filename)

        # 3) Overwrite check
        if os.path.exists(new_path):
            resp = QMessageBox.question(
                self, "Overwrite?",
                f"A file named '{filename}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if resp != QMessageBox.Yes:
                return

        # 4) Load schema & generate skeleton
        try:
            with open(SCHEMA_PATH, 'r') as sf:
                schema = json.load(sf)
        except Exception as e:
            QMessageBox.critical(self, "Schema Load Error",
                                f"Could not load JSON schema:\n{e}")
            return

        skeleton = self._generate_skeleton_from_schema(schema)

        # 5) Write the new file
        try:
            with open(new_path, 'w') as f:
                json.dump(skeleton, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write file:\n{e}")
            return

        # 6) Log and refresh
        self.loggingField.appendPlainText(f"Createdirty: {new_path}")
        self.fileTable.refresh_table()

    def save_file(self):
        row = self.fileTable.table.currentRow()
        if row < 0:
            return
        filename = self.fileTable.table.item(row, 0).text()
        model = self.models.get(filename)
        if model and model.is_dirty():
            model.save()
            self.loggingField.appendPlainText(f"Saved {filename}")
            # re‐apply the clean highlight for this one row
            self.fileTable.updateStatus(filename, model.is_dirty())
        else:
            self.loggingField.appendPlainText(f"No changes to save for {filename}")


    def save_all(self):
        count = 0
        for filename, model in self.models.items():
            if model.is_dirty():
                 model.save()
                 count += 1
        self.loggingField.appendPlainText(f"Saved {count} file(s)")
        # re‐apply all statuses so the visuals match model states
        for filename, model in self.models.items():
            self.fileTable.updateStatus(filename, model.is_dirty())

    def discard_changes(self):
        row = self.fileTable.table.currentRow()
        if row < 0:
            return
        filename = self.fileTable.table.item(row, 0).text()
        model = self.models.get(filename)
        if model and model.is_dirty():
            model.reset()
            self.loggingField.appendPlainText(f"Discarded changes to {filename}")

            tbl = self.fileTable.table
            tbl.selectionModel().clearSelection()    # remove all selection
            tbl.selectRow(row)                       # re‑select the same row




    def discard_all_changes(self):
        for filename,model in self.models.items():
            if model.is_dirty() == True:
                model.reset()
                self.loggingField.appendPlainText(f"Discarded changes to {filename}")
            tbl = self.fileTable.table
        tbl.selectionModel().clearSelection()    # remove all selection

   
        return


    def run_simulation_from_file_table(self):
        if self.simulation_running_flag:
            QMessageBox.warning(self, "Simulation", "Simulation already running.")
            return

        # Save or discard unsaved changes
        dirty = [n for n,m in self.models.items() if m.is_dirty()]
        if dirty:
            resp = QMessageBox.question(
                self, f"Unsaved Changes in {len(dirty)} files.",
                "Save all changes?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if resp == QMessageBox.Cancel:
                return
            if resp == QMessageBox.Save:
                self.save_all()

        # Build queue of filenames
        tbl = self.fileTable.table
        self.simulation_queue = [tbl.item(r,0).text()
                                 for r in range(tbl.rowCount())
                                 if tbl.item(r,1).checkState() != Qt.Checked]
        if not self.simulation_queue:
            return

        # Mark statuses
        for idx, name in enumerate(self.simulation_queue):
            row = next(r for r in range(tbl.rowCount()) if tbl.item(r,0).text()==name)
            tbl.item(row,2).setText("simulating" if idx==0 else "pending")

        # Start batch worker
        self.batch_worker = BatchSimulationWorker(self.opened_directory, self.simulation_queue)
        self.batch_worker.progressChanged.connect(self.progressBar.setValue)
        self.batch_worker.statusChanged.connect(self.handleStatusUpdate)
        self.batch_worker.fileFinished.connect(self._on_file_finished)
        self.batch_worker.finished.connect(self._on_all_finished)
        self.simulation_running_flag = True
        self.batch_worker.start()


    def _on_file_finished(self, filename):
        tbl = self.fileTable.table
        for row in range(tbl.rowCount()):
            if tbl.item(row,0).text() == filename:
                tbl.item(row,2).setText("done")
                break

    def _on_all_finished(self):
        tbl = self.fileTable.table
        for r in range(tbl.rowCount()):
            tbl.item(r,2).setText("")
        self.progressBar.setValue(0)
        self.simulation_running_flag = False
        if self.batch_worker:
            self.batch_worker.deleteLater()
            self.batch_worker = None

    def handleStatusUpdate(self, status):
        self.statusLabel.setText(status)
        if not status.startswith("Processing step"):
            self.loggingField.appendPlainText(status)

    def startCompilationAnimation(self):
        # If an animation timer already exists, try to stop and delete it.
        if hasattr(self, 'compilingTimer') and self.compilingTimer is not None:
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
        self.compilingTimer.timeout.connect(self.updateCompilingStatus)
        self.compilingTimer.start()
        self.isCompilingAnimationActive = True

    def updateCompilingStatus(self):
        # Only update if the animation is active.
        if self.isCompilingAnimationActive:
            dots = "." * ((self.compilingAnimationStep % 3) + 1)
            self.statusLabel.setText(f"Compiling{dots} (this may take a couple of minutes)")
            self.compilingAnimationStep += 1

    def stopCompilationAnimation(self):
        self.isCompilingAnimationActive = False
        if hasattr(self, 'compilingTimer') and self.compilingTimer is not None:
            try:
                self.compilingTimer.stop()
                self.compilingTimer.deleteLater()
            except RuntimeError:
                # The timer was already deleted.
                pass
            self.compilingTimer = None

    def handleStatusUpdate(self, status):
        # Only stop the compilation animation if we receive a clear signal
        # that compilation is finished, e.g., when the status is "Starting simulation..."
        if status == "Simulation instance created":
            self.startCompilationAnimation()
        if status == "Starting simulation...":
            self.stopCompilationAnimation()
        self.statusLabel.setText(status)
        if not status.startswith("Processing step"):
            self.loggingField.appendPlainText(status)


    def logMessage(self, message: str):
        """Append a message to the log box and optionally print to console."""
        if hasattr(self, 'loggingField') and self.loggingField is not None:
            self.loggingField.appendPlainText("\n" + message)
        print(message)



    def _on_model_dirty(self, filename, dirty):
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

    def _emit_any_dirty(self):
        """Helper to recompute & emit whether at least one model is dirty."""
        any_dirty = any(m.is_dirty() for m in self.models.values())
        self.anyDirtyChanged.emit(any_dirty)


    def _on_file_copied(self, original_name: str, copy_name: str):
        """After FileTable actually writes the new JSON copy on disk,
        create a FileModel for it and register all the same signals."""
        full_path = os.path.join(self.opened_directory, copy_name)
        # 1) build model
        model = FileModel(full_path)
        # 2) wire up the dirtyChanged → table/status callbacks
        model.dirtyChanged.connect(lambda dirty, filename=copy_name:
                                self.fileTable.updateStatus(filename, dirty))
        model.dirtyChanged.connect(lambda dirty, filename=copy_name:
                                self._on_model_dirty(filename, dirty))
        model.dirtyChanged.connect(self.fileDirtyChanged)
        # 3) mark it clean, add to dict
        model.mark_clean()
        self.models[copy_name] = model
        # 4) refresh UI
        self.fileTable.refresh_table()
        self.fileTable.updateStatus(copy_name, model.is_dirty())
                
import os
import shutil
from enum import StrEnum

from PyQt5.QtCore import QEvent, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class FileSimState(StrEnum):
    """Exclusive simulation-run states for a single file row."""

    PENDING = "pending"
    SIMULATING = "simulating"
    DONE = "done"


class ReadOnlyDelegate(QStyledItemDelegate):
    """A delegate that prevents editing of cells."""

    def createEditor(self, parent, option, index) -> None:
        return None


class CenteredCheckBoxDelegate(QStyledItemDelegate):
    """Radio-button-style indicator, centered in the cell."""

    def paint(self, painter, option, index) -> None:
        style = (
            option.widget.style() if option.widget else QApplication.style()
        )

        # Draw only the background panel (selection highlight, setBackground color) — no checkbox decoration
        bg_opt = QStyleOptionViewItem(option)
        self.initStyleOption(bg_opt, index)
        style.drawPrimitive(
            QStyle.PE_PanelItemViewItem, bg_opt, painter, option.widget
        )

        # Draw radio indicator, centered and clipped to the cell
        rb = QStyleOptionButton()
        rb.state = QStyle.State_Enabled
        rb.state |= (
            QStyle.State_On
            if index.data(Qt.CheckStateRole) == Qt.Checked
            else QStyle.State_Off
        )
        rb.rect = option.rect
        ind = style.subElementRect(
            QStyle.SE_RadioButtonIndicator, rb, option.widget
        )
        cx = option.rect.x() + (option.rect.width() - ind.width()) // 2
        cy = option.rect.y() + (option.rect.height() - ind.height()) // 2
        rb.rect = QRect(cx, cy, ind.width(), ind.height())
        painter.save()
        painter.setClipRect(option.rect)
        style.drawPrimitive(
            QStyle.PE_IndicatorRadioButton, rb, painter, option.widget
        )
        painter.restore()

    def editorEvent(self, event, model, option, index) -> bool:
        if not (index.flags() & Qt.ItemIsUserCheckable):
            return False
        if (
            event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
        ):
            current = index.data(Qt.CheckStateRole)
            new_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
            model.setData(index, new_state, Qt.CheckStateRole)
            return True
        return False


class _DraggableTable(QTableWidget):
    """QTableWidget with reliable single-row drag-and-drop reordering."""

    rowsReordered = pyqtSignal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDropIndicatorShown(True)

    def _drop_row(self, event) -> int:
        """Row index where the dragged row should land."""
        pos = event.pos()
        index = self.indexAt(pos)
        if not index.isValid():
            return self.rowCount()
        rect = self.visualRect(index)
        return index.row() + 1 if pos.y() >= rect.center().y() else index.row()

    def dropEvent(self, event) -> None:
        if event.source() is not self:
            super().dropEvent(event)
            return
        src_rows = sorted({i.row() for i in self.selectedIndexes()})
        if not src_rows:
            event.ignore()
            return
        drop = self._drop_row(event)
        # Clone the items so we can re-insert after removal.
        snapshots = []
        for r in src_rows:
            row_items = []
            for c in range(self.columnCount()):
                src = self.item(r, c)
                row_items.append(
                    QTableWidgetItem(src) if src else QTableWidgetItem()
                )
            snapshots.append(row_items)
        self.blockSignals(True)
        try:
            for r in reversed(src_rows):
                self.removeRow(r)
                if r < drop:
                    drop -= 1
            drop = max(0, min(drop, self.rowCount()))
            for i, items in enumerate(snapshots):
                self.insertRow(drop + i)
                for c, it in enumerate(items):
                    self.setItem(drop + i, c, it)
        finally:
            self.blockSignals(False)
        self.clearSelection()
        self.selectRow(drop)
        # We already moved the rows ourselves. Report a non-move action so
        # QAbstractItemView.startDrag() does NOT also call clearOrRemoveRows()
        # and delete the just-dropped row.
        event.setDropAction(Qt.IgnoreAction)
        event.accept()
        self.rowsReordered.emit()


class FileTableWidget(QWidget):
    """
    A widget that displays a directory of JSON files in a table with
    options to ignore, rename, delete files.
    """

    fileRenamed = pyqtSignal(str, str)  # old_name, new_name
    fileDeleted = pyqtSignal(str)  # file_name
    fileCopied = pyqtSignal(str, str)  # original, copy_name
    fileIgnored = pyqtSignal(str, bool)  # file_name, ignored_flag
    fileSelected = pyqtSignal(str)  # file_name
    openDiffRequested = pyqtSignal(str)  # file_name

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_dir = None
        self._dirty_state: set = set()  # filenames with unsaved changes
        self._validation_state: dict = {}  # filename -> [] (valid) | [errors] | None (unknown)
        self._sim_state: dict = {}  # filename -> FileSimState | None
        self._updating_display = False  # re-entrancy guard for itemChanged
        self._order: list = []  # session-only user-defined row order
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.table = _DraggableTable(0, 3, self)
        self.table.setHorizontalHeaderLabels(
            ["Loaded File", "Ignore", "Status"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(1, 70)
        # Delegates: read-only for name/status, centered checkbox for ignore
        self.table.setItemDelegateForColumn(0, ReadOnlyDelegate(self.table))
        self.table.setItemDelegateForColumn(
            1, CenteredCheckBoxDelegate(self.table)
        )
        self.table.setItemDelegateForColumn(2, ReadOnlyDelegate(self.table))

        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)

        # connect signals
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemDoubleClicked.connect(self._on_double_clicked)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.rowsReordered.connect(self._on_rows_reordered)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

    def load_directory(self, directory) -> None:
        if not os.path.isdir(directory):
            return
        self._current_dir = directory
        self.refresh_table()

    def update_status(self, filename, dirty: bool) -> None:
        """Called when the dirty state of a file changes."""
        if dirty:
            self._dirty_state.add(filename)
        else:
            self._dirty_state.discard(filename)
        row = self._find_row(filename)
        if row >= 0:
            self._refresh_row_display(row, filename)

    def set_validation_status(self, filename, errors) -> None:
        """
        Set validation result for a file.
        errors: list of jsonschema ValidationError objects ([] = valid, None = unknown).
        """
        self._validation_state[filename] = errors
        row = self._find_row(filename)
        if row >= 0:
            self._refresh_row_display(row, filename)

    def set_simulation_status(
        self, filename, state: FileSimState | None
    ) -> None:
        """Set the simulation run state (PENDING / SIMULATING / DONE / None to clear)."""
        self._sim_state[filename] = state
        row = self._find_row(filename)
        if row >= 0:
            self._refresh_row_display(row, filename)

    def _find_row(self, filename: str) -> int:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == filename:
                return row
        return -1

    def _refresh_row_display(self, row: int, filename: str) -> None:
        """
        Compute background + status text from all state layers (priority order):
          ignored > simulation state > dirty/validation
        """
        if self._updating_display:
            return
        self._updating_display = True
        try:
            self._do_refresh_row_display(row, filename)
        finally:
            self._updating_display = False

    def _do_refresh_row_display(self, row: int, filename: str) -> None:
        status_item = self.table.item(row, 2)
        if status_item is None:
            return

        ignore_item = self.table.item(row, 1)
        ignored = (
            ignore_item is not None and ignore_item.checkState() == Qt.Checked
        )

        sim = self._sim_state.get(filename)
        dirty = filename in self._dirty_state
        errors = self._validation_state.get(filename)

        if ignored:
            bg = QBrush(QColor(205, 205, 205))
            fg = QColor(90, 90, 90)
            if errors is None:
                text = "ignored"
            elif not errors:
                text = "ignored  ✓ valid"
            else:
                text = f"ignored  {len(errors)} error(s)"

        elif sim == FileSimState.SIMULATING:
            bg = QBrush(QColor(200, 230, 255))
            text = "⟳ simulating"
            fg = QColor(0, 70, 160)
        elif sim == FileSimState.PENDING:
            bg = QBrush(QColor(230, 230, 230))
            text = "pending"
            fg = QColor(90, 90, 90)
        elif sim == FileSimState.DONE:
            bg = QBrush(QColor(220, 255, 220))
            text = "✓ done"
            fg = QColor(0, 130, 0)

        else:
            # dirty/validation
            if errors is None:
                if dirty:
                    bg = QBrush(QColor(255, 210, 100))
                    text = "✎ unsaved"
                    fg = QColor(150, 70, 0)
                else:
                    bg = QBrush(Qt.white)
                    text = ""
                    fg = QColor(Qt.black)
            elif not errors:
                if dirty:
                    bg = QBrush(QColor(255, 215, 105))
                    text = "✎ unsaved  ✓ valid"
                    fg = QColor(150, 70, 0)
                else:
                    bg = QBrush(Qt.white)
                    text = "✓ valid"
                    fg = QColor(0, 140, 0)
            else:
                n = len(errors)
                if dirty:
                    bg = QBrush(QColor(255, 195, 110))
                    text = f"✎ unsaved  {n} error(s)"
                    fg = QColor(160, 0, 0)
                else:
                    bg = QBrush(QColor(255, 210, 210))
                    text = f"{n} error(s)"
                    fg = QColor(180, 0, 0)

        for col in range(self.table.columnCount()):
            cell = self.table.item(row, col)
            if cell:
                cell.setBackground(bg)

        status_item.setText(text)
        status_item.setForeground(QBrush(fg))
        status_item.setTextAlignment(Qt.AlignCenter)

    def _on_rows_reordered(self) -> None:
        """Sync session order list with the table's current row order."""
        self._order = [
            self.table.item(r, 0).text()
            for r in range(self.table.rowCount())
            if self.table.item(r, 0)
        ]

    def refresh_table(self) -> None:
        if not self._current_dir:
            return
        on_disk = [
            f
            for f in os.listdir(self._current_dir)
            if f.lower().endswith(".json")
        ]
        on_disk_set = set(on_disk)
        # Preserve user-defined order; append new files (alphabetical) at the end
        kept = [f for f in self._order if f in on_disk_set]
        new = sorted(f for f in on_disk if f not in kept)
        files = kept + new
        self._order = files[:]

        self.table.blockSignals(True)
        self.table.setRowCount(0)

        for fn in files:
            row = self.table.rowCount()
            self.table.insertRow(row)
            # Loaded File (read-only)
            item = QTableWidgetItem(fn)
            item.setData(Qt.UserRole, False)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)

            # Ignore checkbox cell
            ignore_item = QTableWidgetItem()
            ignore_item.setFlags(ignore_item.flags() | Qt.ItemIsUserCheckable)
            ignore_item.setFlags(ignore_item.flags() & ~Qt.ItemIsEditable)
            ignore_item.setCheckState(Qt.Unchecked)
            self.table.setItem(row, 1, ignore_item)

            # Status (read-only via delegate)
            status_item = QTableWidgetItem("")
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, status_item)

        self.table.blockSignals(False)

        # Restore stored dirty/validation display state
        for row in range(self.table.rowCount()):
            fn = self.table.item(row, 0).text()
            self._refresh_row_display(row, fn)

    def _on_double_clicked(self, item) -> None:
        row = item.row()
        fn_item = self.table.item(row, 0)
        if fn_item:
            self.openDiffRequested.emit(fn_item.text())

    def _on_selection_changed(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            fn = self.table.item(row, 0).text()
            self.fileSelected.emit(fn)

    def _on_item_changed(self, item) -> None:
        """React only to checkbox toggles in the Ignore column (col 1)."""
        if self._updating_display:
            return
        if item.column() != 1:
            return
        if not (item.flags() & Qt.ItemIsUserCheckable):
            return
        row = item.row()
        checked = item.checkState() == Qt.Checked
        name = self.table.item(row, 0).text()
        self._apply_ignore(row, checked)
        self.fileIgnored.emit(name, checked)

    def _apply_ignore(self, row: int, checked: bool) -> None:
        """Visual update for ignore state (does not emit fileIgnored)."""
        if self._updating_display:
            return
        self._updating_display = True
        try:
            name = self.table.item(row, 0).text()
            self._do_refresh_row_display(row, name)
        finally:
            self._updating_display = False

    def _show_context_menu(self, point) -> None:
        row = self.table.rowAt(point.y())
        if row < 0:
            return
        filename = self.table.item(row, 0).text()
        menu = QMenu(self)
        validate_action = QAction("View Validation Issues…", self)
        rename = QAction("Rename File", self)
        delete = QAction("Delete File", self)
        copy = QAction("Copy File", self)
        menu.addAction(validate_action)
        menu.addSeparator()
        menu.addAction(rename)
        menu.addAction(delete)
        menu.addAction(copy)
        validate_action.triggered.connect(
            lambda: self.openDiffRequested.emit(filename)
        )
        rename.triggered.connect(lambda: self._rename(row))
        delete.triggered.connect(lambda: self._delete(row))
        copy.triggered.connect(lambda: self._copy(row))
        menu.exec_(self.table.viewport().mapToGlobal(point))

    def _rename(self, row) -> None:
        old = self.table.item(row, 0).text()
        base = os.path.splitext(old)[0]
        new_name, ok = QInputDialog.getText(
            self, "Rename File", "New name (without .json):", text=base
        )
        if not (ok and new_name.strip()):
            return
        if not new_name.lower().endswith(".json"):
            new_name += ".json"
        old_path = os.path.join(self._current_dir, old)
        new_path = os.path.join(self._current_dir, new_name)
        if os.path.exists(new_path):
            QMessageBox.warning(
                self,
                "Cannot Rename",
                f"A file named '{new_name}' already exists.",
            )
            return
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rename: {e}")
            return
        if old in self._order:
            self._order[self._order.index(old)] = new_name
        self.fileRenamed.emit(old, new_name)
        self.refresh_table()

    def _delete(self, row) -> None:
        name = self.table.item(row, 0).text()
        path = os.path.join(self._current_dir, name)
        ans = QMessageBox.question(
            self,
            "Delete File?",
            f"Permanently delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        try:
            os.remove(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete: {e}")
            return
        self.fileDeleted.emit(name)
        self.refresh_table()

    def _copy(self, row) -> None:
        """Create a copy of the selected JSON file, appending '_copy' to its base name."""
        original = self.table.item(row, 0).text()
        base, ext = os.path.splitext(original)
        copy_name = f"{base}_copy{ext}"
        orig_path = os.path.join(self._current_dir, original)
        copy_path = os.path.join(self._current_dir, copy_name)
        # ensure unique copy name
        count = 1
        while os.path.exists(copy_path):
            copy_name = f"{base}_copy{count}{ext}"
            copy_path = os.path.join(self._current_dir, copy_name)
            count += 1
        try:
            shutil.copy(orig_path, copy_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy: {e}")
            return
        if original in self._order:
            self._order.insert(self._order.index(original) + 1, copy_name)
        self.fileCopied.emit(original, copy_name)
        self.refresh_table()

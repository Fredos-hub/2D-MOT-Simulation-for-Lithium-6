import os
import json
import shutil
from PyQt5.QtWidgets import (
    QWidget, QTableWidget, QTableWidgetItem, QPushButton, QCheckBox,
    QVBoxLayout, QMenu, QAction, QInputDialog, QMessageBox, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QStyledItemDelegate


class ReadOnlyDelegate(QStyledItemDelegate):
    """A delegate that prevents editing of cells."""
    def createEditor(self, parent, option, index):
        return None


class FileTableWidget(QWidget):
    """
    A widget that displays a directory of JSON files in a table with
    options to ignore, rename, delete files.
    """
    fileRenamed     = pyqtSignal(str, str)  # old_name, new_name
    fileDeleted     = pyqtSignal(str)       # file_name
    fileCopied      = pyqtSignal(str, str)   # original, copy_name
    fileIgnored     = pyqtSignal(str, bool) # file_name, ignored_flag
    fileSelected    = pyqtSignal(str)       # file_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_dir = None
        self._setup_ui()

    def _setup_ui(self):
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels([
            "Loaded File", "Ignore", "Status"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # Make Status column read-only via delegate
        self.table.setItemDelegateForColumn(2, ReadOnlyDelegate(self.table))
        self.table.setItemDelegateForColumn(0, ReadOnlyDelegate(self.table))
        #self.table.setItemDelegateForColumn(1, ReadOnlyDelegate(self.table))

        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)

        # connect signals
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        #self.table.cellChanged.connect(self._on_cell_changed)
        # replace the generic itemChanged for ignore‑toggles
        # with a focused handler on clicks in column 2:
        self.table.cellClicked.connect(self._on_ignore_clicked)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

    def load_directory(self, directory):
        if not os.path.isdir(directory):
            return
        self._current_dir = directory
        self.refresh_table()



    def updateStatus(self, filename, dirty: bool):
        """Update the 'Status' column for a given filename and highlight the row."""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == filename:
                # Color the entire row
                for col in range(self.table.columnCount()):
                    cell = self.table.item(row, col)
                    if cell:
                        if dirty:
                            # light yellow background for unsaved changes
                            cell.setBackground(QBrush(QColor(255, 255, 150)))
                        else:
                            # reset to default background
                            cell.setBackground(QBrush(Qt.white))
                # Update status text
                status_item = self.table.item(row, 2)
                status_item.setText("unsaved" if dirty else "")
                status_item.setTextAlignment(Qt.AlignCenter)
                break



    def refresh_table(self):
        if not self._current_dir:
            return
        files = [f for f in os.listdir(self._current_dir)
                 if f.lower().endswith('.json')]

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


    def _on_selection_changed(self):
        row = self.table.currentRow()
        if row >= 0:
            fn = self.table.item(row, 0).text()
            self.fileSelected.emit(fn)



    def _on_ignore_clicked(self, row: int, col: int):
        """Only for clicks in the Ignore column, toggle & emit."""
        if col != 1:
            return
        item = self.table.item(row, col)
        # flip it:
        new_checked = (item.checkState() != Qt.Checked)
        item.setCheckState(Qt.Checked if new_checked else Qt.Unchecked)
        # apply the row coloring & emit
        self._apply_ignore(row, new_checked)

    def _apply_ignore(self, row: int, checked: bool):
        """Common logic to gray out and emit fileIgnored."""
        bg = QBrush(Qt.darkGray) if checked else QBrush(Qt.white)
        for c in range(self.table.columnCount()):
            cell = self.table.item(row, c)
            if cell:
                cell.setBackground(bg)
        name = self.table.item(row, 0).text()
        self.fileIgnored.emit(name, checked)

    def _show_context_menu(self, point):
        row = self.table.rowAt(point.y())
        if row < 0:
            return
        menu = QMenu(self)
        rename = QAction("Rename File", self)
        delete = QAction("Delete File", self)
        copy = QAction("Copy File", self)
        menu.addAction(rename); menu.addAction(delete); menu.addAction(copy)
        rename.triggered.connect(lambda: self._rename(row))
        delete.triggered.connect(lambda: self._delete(row))
        copy.triggered.connect(lambda: self._copy(row))
        menu.exec_(self.table.viewport().mapToGlobal(point))

    def _rename(self, row):
        old = self.table.item(row, 0).text()
        base = os.path.splitext(old)[0]
        new_name, ok = QInputDialog.getText(
            self, "Rename File", "New name (without .json):", text=base
        )
        if not (ok and new_name.strip()):
            return
        if not new_name.lower().endswith('.json'):
            new_name += '.json'
        old_path = os.path.join(self._current_dir, old)
        new_path = os.path.join(self._current_dir, new_name)
        if os.path.exists(new_path):
            QMessageBox.warning(self, "Cannot Rename",
                                f"A file named '{new_name}' already exists.")
            return
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to rename: {e}")
            return
        self.fileRenamed.emit(old, new_name)
        self.refresh_table()

    def _delete(self, row):
        name = self.table.item(row, 0).text()
        path = os.path.join(self._current_dir, name)
        ans = QMessageBox.question(
            self, "Delete File?",
            f"Permanently delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if ans != QMessageBox.Yes:
            return
        try:
            os.remove(path)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to delete: {e}")
            return
        self.fileDeleted.emit(name)
        self.refresh_table()


    def _copy(self, row):
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
            QMessageBox.critical(self, "Error",
                                 f"Failed to copy: {e}")
            return
        self.fileCopied.emit(original, copy_name)
        self.refresh_table()

    

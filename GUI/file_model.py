# models/file_model.py

import json
import copy
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

class FileModel(QObject):
    """
    In-memory model of a JSON file, with change-tracking ("dirty") and
    nested get/set for arbitrary keys.
    """
    dirtyChanged = pyqtSignal(bool)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = Path(filepath)
        self._original = self._load_file()
        # make a deep copy for editing
        self._current = copy.deepcopy(self._original)
        self._dirty = False

    def _load_file(self):
        if not self.filepath.exists():
            return {}
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def is_dirty(self):
        return self._dirty

    def _update_dirty(self):
        dirty = (self._current != self._original)
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirtyChanged.emit(self._dirty)

    def get(self, *keys, default=None):
        """
        Safely drill into nested dicts:
            model.get('Simulation','max_step_number', default=100)
        """
        node = self._current
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default:
                return default
        return node
    def safe_get(self, section, key, default):
        try:
            return self.get(section, key, default=default)
        except Exception:
            return default
    def set(self, value, *keys):
        """
        Set a nested key, creating intermediate dicts as needed.
        Emits dirtyChanged if the data really changed.
        """
        node = self._current
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        self._update_dirty()

    def save(self):
        """
        Write _current back to the JSON file, update _original, clear dirty.
        """
        self.filepath.write_text(
            json.dumps(self._current, indent=4),
            encoding='utf-8'
        )
        self._original = copy.deepcopy(self._current)
        self._update_dirty()

    def reset(self):
        """
        Discard in-memory changes and restore from disk.
        """
        self._original = self._load_file()
        self._current = copy.deepcopy(self._original)
        self._update_dirty()

    def mark_clean(self):
        """
        Treat the in‐memory current state as the new baseline.
        Only emit dirtyChanged if we were previously dirty.
        """
        was_dirty = self._dirty
        # 1) Reset the baseline
        self._original = copy.deepcopy(self._current)
        # 2) Clear the internal flag
        self._dirty = False
        # 3) Notify listeners *only* if we really went from dirty→clean
        if was_dirty:
            self.dirtyChanged.emit(False)